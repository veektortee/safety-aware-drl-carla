"""Game-style checkpoint rings for the demo (display only, never part of what the policy sees).

Every SPACING_M metres along the planned route there is a checkpoint gate, plus a finish gate at the
end. The car "collects" a checkpoint when its route progress passes it.

- Overview (CARLA server window, top-down spectator): the next gates are drawn as ovals on the road
  with CARLA's HUD lines (draw_hud_line), which are not rendered into camera sensors, so the policy's
  camera image is unchanged. Collected gates flash green.
- Driver view (the camera window): the same gates are drawn as upright rings projected into the
  image, with a small HUD (checkpoints collected, speed, route progress). This goes through
  CarlaGymEnv.display_overlay, which only affects the window, not the observation.

Usage (demo/run_model_inference.py):
    rings = CheckpointRings(carla_env)
    obs, info = env.reset(...); rings.start_episode()
    ... each step: obs, ... = env.step(action); rings.update()
"""

import math
from typing import List, Optional

import cv2
import numpy as np

try:
    import carla
except ImportError:  # unit tests without the CARLA client
    carla = None

SPACING_M = 30.0          # distance between checkpoints along the route
SHOW_AHEAD = 3            # gates drawn ahead of the car
GATE_HALF_WIDTH_M = 2.6   # ring half width (across the lane)
GATE_HALF_HEIGHT_M = 1.8  # upright ring half height (driver view)
GROUND_HALF_LEN_M = 1.6   # ground oval half length (along the road, overview)
SEGMENTS = 32
FLASH_TICKS = 20          # how long a collected gate stays green / the banner shows (1 s at 20 FPS)

NEXT_BGR = (0, 200, 255)       # amber: the gate to drive through next
AHEAD_BGR = (0, 140, 255)      # orange: the ones after it
DONE_BGR = (60, 220, 60)       # green flash when collected
FINISH_BGR = (255, 255, 255)   # white finish gate


def checkpoint_positions(route_length_m: float, spacing_m: float = SPACING_M) -> List[float]:
    """Arc lengths of the gates: every spacing_m (none in the last 10 m) plus the finish."""
    if route_length_m <= 0:
        return []
    s = list(np.arange(spacing_m, route_length_m - 10.0, spacing_m))
    return [float(x) for x in s] + [float(route_length_m)]


def ring_points(center, yaw_rad: float, half_w: float, half_h: float, upright: bool,
                segments: int = SEGMENTS) -> np.ndarray:
    """(segments+1, 3) closed oval around `center`. Upright: in the plane across the road (a gate
    the car drives through); flat: lying on the road (seen from the top-down overview)."""
    cx, cy, cz = center
    fwd = np.array([math.cos(yaw_rad), math.sin(yaw_rad), 0.0])
    right = np.array([-math.sin(yaw_rad), math.cos(yaw_rad), 0.0])
    second = np.array([0.0, 0.0, 1.0]) if upright else fwd
    t = np.linspace(0.0, 2.0 * math.pi, segments + 1)
    return (np.array([cx, cy, cz])[None, :] + np.cos(t)[:, None] * half_w * right[None, :]
            + np.sin(t)[:, None] * half_h * second[None, :])


def project_to_image(points_world: np.ndarray, world_to_camera: np.ndarray, width: int, height: int,
                     fov_deg: float) -> np.ndarray:
    """World points -> pixel (u, v); rows behind the camera are NaN. CARLA camera axes: x forward,
    y right, z up."""
    pts = np.c_[points_world, np.ones(len(points_world))] @ world_to_camera.T
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    f = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        u = width / 2.0 + f * y / x
        v = height / 2.0 - f * z / x
    uv = np.stack([u, v], axis=1)
    uv[x < 0.5] = np.nan
    return uv


class CheckpointRings:
    """Checkpoint gates along CarlaGymEnv's route; see the module docstring."""

    def __init__(self, carla_env, spacing_m: float = SPACING_M, overview: bool = True):
        self.env = carla_env
        self.spacing_m = float(spacing_m)
        self.overview = bool(overview)
        self.gates: List[dict] = []
        self.collected = 0
        self._flash: List[tuple] = []        # (gate, ticks left)
        self._banner = 0
        carla_env.display_overlay = self.overlay

    # -- episode -------------------------------------------------------------------------------
    def start_episode(self):
        route = self.env.route
        self.gates = []
        for i, s in enumerate(checkpoint_positions(route.length_m, self.spacing_m)):
            x, y = route.point_at(s)
            self.gates.append({"s": s, "xy": (x, y), "z": route.z_at(s), "yaw": route.yaw_at(s),
                               "finish": i == len(checkpoint_positions(route.length_m, self.spacing_m)) - 1})
        self.collected = 0
        self._flash = []
        self._banner = 0

    def update(self):
        snap = getattr(self.env, "_snapshot", None)
        if not snap or not self.gates:
            return
        s = float(snap.get("s", 0.0))
        while self.collected < len(self.gates) and s + 1.0 >= self.gates[self.collected]["s"]:
            self._flash.append((self.gates[self.collected], FLASH_TICKS))
            self.collected += 1
            self._banner = FLASH_TICKS
        self._flash = [(g, n - 1) for g, n in self._flash if n > 1]
        self._banner = max(self._banner - 1, 0)
        if self.overview:
            self._draw_overview()

    def finish_episode(self, end_reason: str):
        """The env ends an episode at the route end a little before the finish gate's arc length:
        a finished route collects every remaining gate."""
        if end_reason == "endpoint" and self.gates:
            self.collected = len(self.gates)
            self._banner = FLASH_TICKS

    # -- overview (server window) --------------------------------------------------------------
    def _draw_overview(self):
        if carla is None:
            return
        debug = self.env.world.debug
        life = 0.12                                   # redrawn every tick (0.05 s)
        drawn = [(g, NEXT_BGR if k == 0 else AHEAD_BGR)
                 for k, g in enumerate(self.gates[self.collected:self.collected + SHOW_AHEAD])]
        drawn += [(g, DONE_BGR) for g, _ in self._flash]
        for gate, bgr in drawn:
            if gate["finish"] and bgr != DONE_BGR:
                bgr = FINISH_BGR
            color = carla.Color(r=bgr[2], g=bgr[1], b=bgr[0])
            pts = ring_points((*gate["xy"], gate["z"] + 0.3), gate["yaw"], GATE_HALF_WIDTH_M,
                              GROUND_HALF_LEN_M, upright=False)
            for a, b in zip(pts[:-1], pts[1:]):
                debug.draw_hud_line(carla.Location(*map(float, a)), carla.Location(*map(float, b)),
                                    thickness=0.25, color=color, life_time=life)

    # -- driver view (camera window) -----------------------------------------------------------
    def overlay(self, image: np.ndarray) -> np.ndarray:
        cam = getattr(self.env, "rgb_camera", None)
        if cam is None or not self.gates:
            return image
        h, w = image.shape[:2]
        try:
            fov = float(cam.attributes.get("fov", 90.0))
            world_to_cam = np.array(cam.get_transform().get_inverse_matrix())
        except Exception:
            return image
        ahead = self.gates[self.collected:self.collected + SHOW_AHEAD]
        layer = image.copy()
        for k, gate in reversed(list(enumerate(ahead))):            # far gates first
            bgr = FINISH_BGR if gate["finish"] else (NEXT_BGR if k == 0 else AHEAD_BGR)
            self._draw_gate(layer, gate, world_to_cam, w, h, fov, bgr, thick=6 if k == 0 else 3)
        for gate, _ in self._flash:
            self._draw_gate(layer, gate, world_to_cam, w, h, fov, DONE_BGR, thick=6)
        image = cv2.addWeighted(layer, 0.85, image, 0.15, 0.0)
        self._draw_hud(image)
        return image

    @staticmethod
    def _draw_gate(img, gate, world_to_cam, w, h, fov, bgr, thick):
        pts = ring_points((*gate["xy"], gate["z"] + GATE_HALF_HEIGHT_M + 0.1), gate["yaw"],
                          GATE_HALF_WIDTH_M, GATE_HALF_HEIGHT_M, upright=True)
        uv = project_to_image(pts, world_to_cam, w, h, fov)
        ok = ~np.isnan(uv).any(axis=1)
        if ok.sum() < 3:
            return
        poly = uv[ok].astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [poly], isClosed=bool(ok.all()), color=(20, 20, 20), thickness=thick + 4,
                      lineType=cv2.LINE_AA)
        cv2.polylines(img, [poly], isClosed=bool(ok.all()), color=bgr, thickness=thick, lineType=cv2.LINE_AA)

    def _draw_hud(self, img):
        snap = getattr(self.env, "_snapshot", None) or {}
        total = len(self.gates)
        speed = float(snap.get("speed", 0.0)) * 3.6
        route = getattr(self.env, "route", None)
        frac = min(max(float(snap.get("s", 0.0)) / route.length_m, 0.0), 1.0) if route and route.length_m else 0.0
        cv2.rectangle(img, (10, 10), (300, 92), (0, 0, 0), -1)
        cv2.putText(img, f"CHECKPOINTS {self.collected}/{total}", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    NEXT_BGR, 2, cv2.LINE_AA)
        cv2.putText(img, f"{speed:5.1f} km/h   route {frac:4.0%}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        if self._banner > 0:
            done = self.collected >= total
            text = "FINISH!" if done else "CHECKPOINT!"
            size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.4, 3)[0]
            org = ((img.shape[1] - size[0]) // 2, 140)          # below the HUD box
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 0, 0), 7, cv2.LINE_AA)
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_DUPLEX, 1.4, DONE_BGR if not done else FINISH_BGR, 3,
                        cv2.LINE_AA)
