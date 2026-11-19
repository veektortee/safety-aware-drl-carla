"""
Core CARLA library for the safety-aware SAC experiments (despite the file name this is not a test).

CarlaGymEnv (reward v3)
  * action  [steer, accel] in [-1, 1]^2; accel -> (throttle, brake) via commons.vehicle_model.accel_to_pedals
            (no deadzone, no mutual-exclusion hacks).
  * obs     {'rgb_data': (360, 640, 3) uint8 front camera (BGR), 'nav': (NAV_DIM,) float32}; 'nav' is built
            in commons.nav_layout.NAV_FEATURES order.
  * sensors RGB camera, LiDAR (one full ground/ego-filtered sweep per 20 Hz tick), collision and lane
            invasion. Every sensor message is queued with its simulator frame; after `frame = world.tick()`
            the env waits for the frame-matched camera image and LiDAR sweep and drains the event queues,
            keeping only events with episode_start_frame < event.frame <= frame. A crash from the previous
            episode therefore can never leak into the next one.
  * reset   brake + zero velocity, teleport, settle ticks in first gear, spectator, queue flush, start frame,
            then counters; spawn >= 50 m from the last crash and clear of vehicles/walkers. Ego role 'hero'.
  * route   commons.route.Route planned with CARLA's GlobalRoutePlanner; length by curriculum stage.
  * state   ONE snapshot per tick (route projection, lane status, lead obstacle, traffic light, stop sign,
            speed, speed limit) feeds the nav vector, the reward, the terminations, build_cbf_state() and
            expert_action().
  * reward  compute_reward_v3(): route progress (finite supply) + time cost + small capped shaping penalties
            + terminal events. Collision / red light / off-route / wrong-way terminate; timeout and
            "blocked without reason" truncate (bootstrap, no penalty).
  * curriculum: set_curriculum_stage() takes effect at the next reset (route length, crash penalty, traffic).

Also here: PipelineObservationWrapper (STT embedding of the camera), the CBFSafetyLayerWrapper re-export,
PolicyTrustScoreCallback and the env-unwrapping helpers used by the experiments.
Per-step diagnostics go through commons.debug.dprint (CARLA_DEBUG=1).
"""
import os
import sys
import math
import queue
import random
import time
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces

import torch

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import carla
except ImportError:
    raise RuntimeError("CARLA module not found. Install via CARLA PythonAPI")

from models.pipeline import Pipeline
from commons.vehicle_model import DT, B_MAX, BRAKE_DELAY_S, EGO_HALF_WIDTH, accel_to_pedals
from commons.nav_layout import NAV_FEATURES, NAV_DIM
from commons.overtake import CLEAR_AHEAD_M, CLEAR_BEHIND_M, OvertakeTracker, lane_clear
from commons.curriculum_stages import STAGES
from commons.debug import DEBUG, dprint
from commons.navigation import GlobalRoutePlanner
from commons.route import Route
from commons.perception import find_lead, lidar_static_gap, combine, ttc as time_to_collision
from commons.traffic_rules import (
    TrafficLightTracker,
    RedLightMonitor,
    StopSignMonitor,
    lane_invasion_is_illegal,
)
from commons.expert import ExpertDriver
from commons.cbf_wrapper import CBFSafetyLayerWrapper  # noqa: F401  (re-export; experiments import it from here)

from stable_baselines3.common.callbacks import BaseCallback


# ============================================================================
# Reward v3 configuration and pure helpers (unit-tested without CARLA)
# ============================================================================

class RewardConfigV3:
    """Reward / termination / reset constants. CarlaGymEnv inherits them as class attributes."""

    REWARD_VERSION = "3.0"

    # dense terms (per 0.05 s step)
    W_PROGRESS = 1.0              # * clip(ds / (L * DT), -1, 1)
    TIME_PENALTY = -0.2
    W_SPEEDING = -2.0             # * max(0, v / L - 1)
    W_LATERAL = -0.2              # * clip((|e_lane| - deadband) / range, 0, 1)
    LATERAL_DEADBAND_M = 0.5
    LATERAL_RANGE_M = 1.25
    W_HEADING = -0.1              # * min(|e_psi_lane| / (pi / 2), 1)
    DENSE_PENALTY_CAP = -0.5      # speeding + lateral + heading never below this per step

    # events
    ILLEGAL_LANE_PENALTY = -1.0   # per illegal lane invasion event
    STOP_SIGN_PENALTY = -10.0     # per stop-sign violation

    # terminal
    GOAL_REWARD = 50.0
    GOAL_REMAINING_M = 3.0
    PEDESTRIAN_CRASH_FACTOR = 2.0
    CRASH_SPEED_FACTOR_MAX = 1.5  # collision = -C * (0.5 + 0.5 * min(v / L, 1.5))
    OFF_ROUTE_EY_M = 3.5
    OFF_ROUTE_STEPS = 20          # 1 s
    WRONG_WAY_STEPS = 40          # 2 s

    # truncation
    BLOCKED_SPEED_MPS = 0.5
    BLOCKED_STEPS = 300           # 15 s
    BLOCKED_LEAD_GAP_M = 15.0
    BLOCKED_TL_GAP_M = 30.0
    BLOCKED_STOP_GAP_M = 10.0
    MIN_EPISODE_STEPS = 600
    EPISODE_TIME_SPEED_FRAC = 0.4  # per-episode limit = (route_length / (0.4 * v_limit) + LIGHT_WAIT_S) / DT
    LIGHT_WAIT_S = 60.0            # allowance for legally waiting at red lights (CARLA red phases are long)

    # speed limit
    SPEED_LIMIT_MIN_MPS = 5.0
    SPEED_LIMIT_MAX_MPS = 25.0

    # reset / spawn
    SPAWN_MIN_CRASH_DIST_M = 50.0
    SPAWN_MIN_VEHICLE_DIST_M = 10.0
    SPAWN_MIN_WALKER_DIST_M = 5.0
    SETTLE_TICKS = 5

    # sensors / route
    SENSOR_TIMEOUT_S = 2.0
    LIDAR_GROUND_Z_CUT = -1.3     # sensor at z = 1.7 m -> road at about -1.7 m in the sensor frame
    LIDAR_EGO_HALF_X = 2.6
    LIDAR_EGO_HALF_Y = 1.2
    ROUTE_SAMPLING_M = 2.0
    TTC_CAP_S = 100.0


REWARD_COMPONENT_KEYS = (
    "progress", "time", "speeding", "lateral", "heading",
    "illegal_lane", "stop_sign",
    "goal", "collision", "red_light", "off_route", "wrong_way",
    "steer_change", "overtake",          # reward 3.1 terms; always 0.0 when the env options are off
)

TERMINAL_REASONS = ("endpoint", "collision", "collision_pedestrian", "red_light", "off_route", "wrong_way")
TRUNCATION_REASONS = ("timeout", "blocked")


def wrap_angle(a: float) -> float:
    """Wrap an angle in radians to [-pi, pi)."""
    return (float(a) + math.pi) % (2.0 * math.pi) - math.pi


def clip_speed_limit(limit_kmh: float, cfg=RewardConfigV3) -> float:
    """Vehicle speed limit [km/h] -> m/s clipped to [SPEED_LIMIT_MIN_MPS, SPEED_LIMIT_MAX_MPS]."""
    return float(np.clip(float(limit_kmh) / 3.6, cfg.SPEED_LIMIT_MIN_MPS, cfg.SPEED_LIMIT_MAX_MPS))


def episode_step_limit(route_length_m: float, v_limit_mps: float, time_limit: int, cfg=RewardConfigV3) -> int:
    """clip(int((route_length / (0.4 * v_limit) + LIGHT_WAIT_S) / DT), 600, time_limit)."""
    drive_s = route_length_m / (cfg.EPISODE_TIME_SPEED_FRAC * max(float(v_limit_mps), 0.1))
    steps = int((drive_s + cfg.LIGHT_WAIT_S) / DT)
    return int(min(max(steps, cfg.MIN_EPISODE_STEPS), int(time_limit)))


def drain_frame_events(q: "queue.Queue", start_frame: int, end_frame: int) -> list:
    """Empty a sensor event queue of (frame, ...) tuples.

    Returns the events with start_frame < frame <= end_frame (in arrival order). Older events (e.g. the
    previous episode's crash delivered late on the callback thread) are dropped; events newer than
    end_frame (should not happen in synchronous mode) are put back for the next drain.
    """
    accepted, future = [], []
    while True:
        try:
            ev = q.get_nowait()
        except queue.Empty:
            break
        frame = ev[0]
        if frame <= start_frame:
            continue
        if frame <= end_frame:
            accepted.append(ev)
        else:
            future.append(ev)
    for ev in future:
        q.put(ev)
    return accepted


def wait_for_frame(q: "queue.Queue", frame: int, timeout: float):
    """Block until the queue yields a (frame, ...) message for `frame` (older ones are discarded).

    Returns the message or None on timeout.
    """
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return None
        try:
            msg = q.get(timeout=remaining)
        except queue.Empty:
            return None
        if msg[0] >= frame:
            return msg


def select_spawn_index(spawn_xyz, vehicle_xyz, walker_xyz, crash_xyz, order: Sequence[int],
                       min_vehicle_m: float, min_walker_m: float, min_crash_m: float) -> Tuple[int, bool]:
    """First spawn index in `order` that is >= min_vehicle_m from every vehicle, >= min_walker_m from
    every walker and >= min_crash_m from the last crash (crash_xyz None or min_crash_m <= 0: ignored).

    Returns (index, True); if none qualifies, (index with the best normalised clearance, False).
    """
    spawn_xyz = np.asarray(spawn_xyz, dtype=float).reshape(-1, 3)

    def min_dist(points):
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        if len(pts) == 0:
            return np.full(len(spawn_xyz), np.inf)
        return np.linalg.norm(spawn_xyz[:, None, :] - pts[None, :, :], axis=2).min(axis=1)

    dv = min_dist(vehicle_xyz)
    dw = min_dist(walker_xyz)
    use_crash = crash_xyz is not None and min_crash_m > 0.0
    dc = min_dist([crash_xyz]) if use_crash else np.full(len(spawn_xyz), np.inf)

    best, best_score = int(order[0]), -np.inf
    for i in order:
        i = int(i)
        if dv[i] >= min_vehicle_m and dw[i] >= min_walker_m and (not use_crash or dc[i] >= min_crash_m):
            return i, True
        score = min(dv[i] / min_vehicle_m, dw[i] / min_walker_m, dc[i] / min_crash_m if use_crash else np.inf)
        if score > best_score:
            best, best_score = i, score
    return best, False


def map_basename(path: str) -> str:
    """'/Game/Carla/Maps/Town03_Opt' or 'Carla/Maps/Town03_Opt' -> 'Town03_Opt'."""
    return str(path).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def pick_town_maps(available: Sequence[str]) -> List[str]:
    """One map per installed town (Town01, Town02, ..., Town10HD), preferring the layered '_Opt' build.

    `available` is client.get_available_maps(); test/empty maps (EmptyMap, OpenDriveMap) are skipped.
    """
    towns: Dict[str, str] = {}
    for path in available:
        name = map_basename(path)
        if not name.startswith("Town"):
            continue
        town = name[:-4] if name.endswith("_Opt") else name
        if town not in towns or name.endswith("_Opt"):
            towns[town] = name
    return [towns[t] for t in sorted(towns)]


# Towns left out of maps="all" (training rotation, eval rounds, demo); an explicit maps=[...] still loads them.
# Town01_Opt with S5 traffic (230 cars, ~250 walkers) failed inside CARLA's native client code (Traffic Manager
# / walker navigation run in-process during world.tick()) in 7 of 8 reproductions on 2026-11-19, within 1-4
# minutes: unbounded memory growth (75 GB in training on 2026-11-18) or an access violation, with hybrid
# physics on or off, with or without walkers, at 140 or 230 cars, with or without a map switch (the 8th run
# lasted 6.7 minutes without failing). Town10HD at S5 ran for hours.
UNSTABLE_TOWNS = ("Town01",)


def town_of(map_name: str) -> str:
    """'Carla/Maps/Town01_Opt' -> 'Town01'."""
    name = map_basename(map_name)
    return name[:-4] if name.endswith("_Opt") else name


def next_map(maps: Sequence[str], current: str) -> str:
    """The map after `current` in the rotation (the first one if `current` is not in it)."""
    names = [map_basename(m) for m in maps]
    cur = map_basename(current)
    return names[(names.index(cur) + 1) % len(names)] if cur in names else names[0]


def eval_map(maps: Sequence[str], slot: int, slots: int) -> str:
    """Map of evaluation episode `slot` of `slots`: contiguous blocks covering every map once per round."""
    slots = max(int(slots), 1)
    i = min(int(slot) * len(maps) // slots, len(maps) - 1)
    return map_basename(maps[i])


def vehicle_cap(n_spawn_points: int) -> int:
    """NPC vehicles a map can hold: its spawn points minus a reserve (>= 10, 10%) so the ego always
    finds a clear spawn point."""
    return max(0, int(n_spawn_points) - max(10, int(n_spawn_points) // 10))


# Traffic fill: CarlaGymEnv(traffic_fill=f) never puts more than share f of a town's capacity on the road -
# at most f * vehicle_cap(spawn points) NPC cars and f * the configured pedestrian maximum. The env default
# stays 1.0 (the original curriculum, which the 2q_cnn run in progress on 2026-11-19 keeps). Experiments
# started after it pass --traffic-fill, default TRAINING_TRAFFIC_FILL: at 100% (S5 filled every spawn point
# but the reserve) towns jammed solid and cars piled onto each other, which taught the agents little.
# Lower stages are unaffected while their counts are below the cap (S2 15/15, S3 60/60).
TRAINING_TRAFFIC_FILL = 0.7


def traffic_caps(n_spawn_points: int, max_pedestrians: int, fill: float = 1.0) -> Tuple[int, int]:
    """(NPC cars, walkers) a town may hold at the given fill."""
    return int(fill * vehicle_cap(n_spawn_points)), int(fill * max(0, int(max_pedestrians)))


# The training curriculum used by the experiments started after 2026-11-19 (5q CNN, 2q STT, 5q STT): 70%
# traffic, smooth steering (low-pass 0.6: per-step jitter -75%, time constant ~0.1 s; penalty
# -0.5 * (command change)^2: a full left-right flip costs -2 on that step, steady driving ~0),
# +5 per safe overtake (well below any crash penalty, C = 50..300) and the perception corridor on the ego.
TRAINING_STEER_SMOOTHING = 0.6
TRAINING_STEER_CHANGE_PENALTY = 0.5
TRAINING_OVERTAKE_BONUS = 5.0


def add_training_curriculum_args(parser):
    """Add the --traffic-fill / --steer-smoothing / --steer-change-penalty / --overtake-bonus /
    --route-corridor options; their defaults are the training curriculum above."""
    group = parser.add_argument_group("traffic and driving quality (defaults: the 70% curriculum)")
    group.add_argument("--traffic-fill", type=float, default=TRAINING_TRAFFIC_FILL,
                       help=f"Share of each town's car capacity and of --num-pedestrians the curriculum may use "
                            f"(default {TRAINING_TRAFFIC_FILL}; 1.0 = the original 100%% curriculum)")
    group.add_argument("--steer-smoothing", type=float, default=TRAINING_STEER_SMOOTHING,
                       help=f"Low-pass on the applied steering, 0 = off (default {TRAINING_STEER_SMOOTHING})")
    group.add_argument("--steer-change-penalty", type=float, default=TRAINING_STEER_CHANGE_PENALTY,
                       help=f"Reward -w * (steering command change)^2 per step, 0 = off "
                            f"(default {TRAINING_STEER_CHANGE_PENALTY})")
    group.add_argument("--overtake-bonus", type=float, default=TRAINING_OVERTAKE_BONUS,
                       help=f"Reward per safe overtake of a slow/queued car, 0 = off (default {TRAINING_OVERTAKE_BONUS})")
    group.add_argument("--route-corridor", action="store_true",
                       help="Old perception: lead corridor on the route centre line instead of on the ego")
    return group


def training_curriculum_kwargs(args) -> Dict:
    """CarlaGymEnv keyword arguments from the options added by add_training_curriculum_args."""
    return dict(traffic_fill=float(args.traffic_fill), steer_smoothing=float(args.steer_smoothing),
                steer_change_penalty=float(args.steer_change_penalty), overtake_bonus=float(args.overtake_bonus),
                lead_corridor_follows_ego=not bool(args.route_corridor))


# NPC physics guard. In dense traffic PhysX sometimes launches a car into the air or drops it through the
# road (seen on Town01_Opt at S5: cars 40 m up, 60 m under the map, 56 m/s launches). Perception takes
# lead vehicles from the actor list projected onto the route, so a car sunk under the road is a phantom
# obstacle the CBF brakes for. The env destroys such cars right after the tick that shows them
# (CarlaGymEnv._remove_broken_npcs) and replaces them at the next reset.
NPC_MAX_SPEED_MPS = 40.0        # no NPC legitimately drives this fast on these towns (highways are 90 km/h)
NPC_MAX_VERTICAL_MPS = 8.0      # launched or falling; driving over hills and kerbs stays well below
NPC_Z_BELOW_ROADS_M = 6.0       # this far below the map's lowest road = fell through
NPC_Z_ABOVE_ROADS_M = 8.0       # this far above the map's highest road = launched


def npc_physics_broken(vx: float, vy: float, vz: float, z: float, road_z_min: float, road_z_max: float) -> bool:
    """True for an NPC car the physics engine has launched, dropped through the road or sped up absurdly."""
    speed_sq = vx * vx + vy * vy + vz * vz
    if not math.isfinite(speed_sq) or not math.isfinite(z):
        return True
    return (speed_sq > NPC_MAX_SPEED_MPS ** 2 or abs(vz) > NPC_MAX_VERTICAL_MPS
            or z < road_z_min - NPC_Z_BELOW_ROADS_M or z > road_z_max + NPC_Z_ABOVE_ROADS_M)


def condition_flags(snap: dict, cfg=RewardConfigV3) -> Tuple[bool, bool, bool]:
    """(off_route, wrong_way, blocked) conditions of one snapshot (the env counts consecutive steps)."""
    off_route = (not snap["on_drivable"]) or (abs(snap["e_y"]) > cfg.OFF_ROUTE_EY_M and not snap["in_same_dir_lane"])
    wrong_way = bool(snap["in_opposite_lane"])
    light_reason = snap["tl_state"] in ("red", "yellow") and snap["tl_stop_gap"] < cfg.BLOCKED_TL_GAP_M
    stop_reason = snap["stop_gap"] < cfg.BLOCKED_STOP_GAP_M
    blocked = (abs(snap["speed"]) < cfg.BLOCKED_SPEED_MPS
               and snap["lead_gap"] > cfg.BLOCKED_LEAD_GAP_M
               and not light_reason and not stop_reason)
    return bool(off_route), wrong_way, bool(blocked)


def compute_reward_v3(snap: dict, prev: dict, cfg=RewardConfigV3):
    """Reward v3 for one step.

    snap: this step's snapshot. Required keys: speed, speed_limit, s, e_lane, e_psi_lane, remaining_m,
          crash_penalty, off_route_steps, wrong_way_steps, blocked_steps, episode_step, episode_limit;
          optional event keys: collision, collision_pedestrian, red_light_run, stop_violation,
          n_illegal_invasions.
    prev: the previous snapshot (keys s, speed).
    Returns (reward, components, terminated, truncated, reason); components always holds every key of
    REWARD_COMPONENT_KEYS and sums to reward.
    """
    L = max(float(snap["speed_limit"]), 1e-3)
    v = float(snap["speed"])
    ds = float(snap["s"]) - float(prev["s"])
    comps = dict.fromkeys(REWARD_COMPONENT_KEYS, 0.0)

    comps["progress"] = cfg.W_PROGRESS * float(np.clip(ds / (L * DT), -1.0, 1.0))
    comps["time"] = cfg.TIME_PENALTY

    speeding = cfg.W_SPEEDING * max(0.0, v / L - 1.0)
    lateral = cfg.W_LATERAL * float(np.clip((abs(float(snap["e_lane"])) - cfg.LATERAL_DEADBAND_M)
                                            / cfg.LATERAL_RANGE_M, 0.0, 1.0))
    heading = cfg.W_HEADING * min(abs(float(snap["e_psi_lane"])) / (math.pi / 2.0), 1.0)
    dense = speeding + lateral + heading
    if dense < cfg.DENSE_PENALTY_CAP:
        scale = cfg.DENSE_PENALTY_CAP / dense      # keep the split, cap the total
        speeding, lateral, heading = speeding * scale, lateral * scale, heading * scale
    comps["speeding"], comps["lateral"], comps["heading"] = speeding, lateral, heading

    comps["illegal_lane"] = cfg.ILLEGAL_LANE_PENALTY * int(snap.get("n_illegal_invasions", 0))
    if snap.get("stop_violation", False):
        comps["stop_sign"] = cfg.STOP_SIGN_PENALTY

    # reward 3.1 (env options; 0.0 when off): smooth steering and safe overtaking
    w_steer = float(getattr(cfg, "steer_change_penalty", 0.0) or 0.0)
    if w_steer > 0.0:
        d_steer = float(snap.get("cmd_steer", 0.0)) - float(snap.get("prev_cmd_steer", 0.0))
        comps["steer_change"] = -w_steer * d_steer * d_steer
    comps["overtake"] = float(snap.get("overtake_bonus", 0.0))

    C = float(snap["crash_penalty"])
    # severity uses the pre-impact speed: the impact itself already slows the car on the event tick
    v_impact = max(abs(v), abs(float(prev.get("speed", 0.0))))
    severity = 0.5 + 0.5 * min(v_impact / L, cfg.CRASH_SPEED_FACTOR_MAX)

    terminated, truncated, reason = False, False, None
    if snap.get("collision_pedestrian", False):
        comps["collision"] = -cfg.PEDESTRIAN_CRASH_FACTOR * C * severity
        reason = "collision_pedestrian"
    elif snap.get("collision", False):
        comps["collision"] = -C * severity
        reason = "collision"
    elif snap.get("red_light_run", False):
        comps["red_light"] = -C
        reason = "red_light"
    elif snap["off_route_steps"] >= cfg.OFF_ROUTE_STEPS:
        comps["off_route"] = -C
        reason = "off_route"
    elif snap["wrong_way_steps"] >= cfg.WRONG_WAY_STEPS:
        comps["wrong_way"] = -C
        reason = "wrong_way"
    elif float(snap["remaining_m"]) < cfg.GOAL_REMAINING_M:
        comps["goal"] = cfg.GOAL_REWARD
        reason = "endpoint"

    if reason is not None:
        terminated = True
    elif snap["episode_step"] >= snap["episode_limit"]:
        truncated, reason = True, "timeout"
    elif snap["blocked_steps"] >= cfg.BLOCKED_STEPS:
        truncated, reason = True, "blocked"

    reward = float(sum(comps.values()))
    return reward, comps, terminated, truncated, reason


def cbf_gap(lead_gap: float, lead_v: float, lead_kind: str, tl_state: str, tl_stop_gap: float,
            v: float) -> Tuple[float, float, str]:
    """(gap, v_lead, gap_kind) for the CBF: the lead obstacle, or the stop line of a red/yellow light
    when it is nearer and the car can still stop before it (v*BRAKE_DELAY_S + v^2/(2*B_MAX) < stop_gap)."""
    gap, v_lead, kind = float(lead_gap), float(lead_v), str(lead_kind)
    if tl_state in ("red", "yellow"):
        vv = max(float(v), 0.0)
        stoppable = vv * BRAKE_DELAY_S + vv * vv / (2.0 * B_MAX) < tl_stop_gap
        if stoppable and tl_stop_gap < gap:
            gap, v_lead, kind = float(tl_stop_gap), 0.0, "stop_line"
    return gap, v_lead, kind


def roadoption_name(option) -> str:
    """RoadOption -> 'left' | 'right' | 'straight' | 'follow' (LANEFOLLOW / CHANGELANE* / VOID)."""
    name = str(getattr(option, "name", option)).upper()
    if name.endswith("CHANGELANELEFT") or name.endswith("CHANGELANERIGHT"):
        return "follow"
    for key in ("LEFT", "RIGHT", "STRAIGHT"):
        if name.endswith(key):
            return key.lower()
    return "follow"


def build_nav_vector(snap: dict) -> np.ndarray:
    """The policy's navigation vector, exactly in commons.nav_layout.NAV_FEATURES order."""
    v = float(snap["speed"])
    L = max(float(snap["speed_limit"]), 1e-3)
    e_psi = float(snap["e_psi"])
    pts = np.asarray(snap["route_pts"], dtype=float).reshape(3, 2) / 20.0
    cmd = snap["cmd"]
    has_lead = snap["lead_kind"] != "none"
    tl = snap["tl_state"]
    light_gap = float(snap["tl_stop_gap"]) if tl != "none" else 100.0
    stop_gap = min(light_gap, float(snap["stop_gap"]))
    route_len = max(float(snap["route_length"]), 1e-3)

    values = {
        "speed_over_limit": float(np.clip(v / L, 0.0, 2.0)),
        "speed_limit_norm": L / 30.0,
        "lateral_error_norm": float(np.clip(float(snap["e_y"]) / 2.0, -2.0, 2.0)),
        "heading_err_sin": math.sin(e_psi),
        "heading_err_cos": math.cos(e_psi),
        "route_pt5_x": pts[0, 0],
        "route_pt5_y": pts[0, 1],
        "route_pt10_x": pts[1, 0],
        "route_pt10_y": pts[1, 1],
        "route_pt20_x": pts[2, 0],
        "route_pt20_y": pts[2, 1],
        "cmd_follow": float(cmd == "follow"),
        "cmd_left": float(cmd == "left"),
        "cmd_right": float(cmd == "right"),
        "cmd_straight": float(cmd == "straight"),
        "cmd_dist_norm": float(np.clip(float(snap["cmd_dist"]) / 50.0, 0.0, 1.0)),
        "lead_gap_norm": min(float(snap["lead_gap"]), 50.0) / 50.0,
        "lead_rel_speed_norm": float(np.clip((float(snap["lead_v"]) - v) / 10.0, -1.0, 1.0)) if has_lead else 0.0,
        "lead_is_pedestrian": float(snap["lead_kind"] == "walker"),
        "tl_red": float(tl == "red"),
        "tl_yellow": float(tl == "yellow"),
        "tl_green": float(tl == "green"),
        "stop_gap_norm": min(max(stop_gap, 0.0), 50.0) / 50.0,
        "prev_steer": float(snap["prev_steer"]),
        "prev_accel": float(snap["prev_accel"]),
        "route_remaining_frac": float(np.clip(float(snap["remaining_m"]) / route_len, 0.0, 1.0)),
    }
    nav = np.array([values[name] for name in NAV_FEATURES], dtype=np.float32)
    return np.nan_to_num(nav, nan=0.0, posinf=0.0, neginf=0.0)


# ============================================================================
# Custom CARLA Gym Environment (Direct API)
# ============================================================================

class CarlaGymEnv(RewardConfigV3, gym.Env):
    """CARLA driving env with route navigation, reward v3, frame-synchronised sensors and a curriculum."""

    metadata = {"render_modes": ["human"]}

    EGO_BLUEPRINTS = ("vehicle.tesla.model3", "vehicle.audi.a2", "vehicle.dodge.charger", "vehicle.bmw.grandtourer")
    TM_PORT_OFFSET = 6000   # Traffic Manager port = CARLA port + 6000 (2000 -> 8000), so parallel runs don't collide
    # Every map load gets a NEW Traffic Manager on the next port (8000, 8100, ... wrapping after 50), and the
    # old one is shut down before load_world(). The Traffic Manager runs inside this process. On 2026-11-19
    # one carried across the Town10HD -> Town01 switch launched or sank 6 cars in the first 7 s, and every
    # run with such a switch ended in the memory runaway (see UNSTABLE_TOWNS); one created on the already
    # loaded Town01 produced no launched or sunk car in 4 episodes.
    TM_PORT_STRIDE = 100
    TM_GENERATIONS = 50
    CAMERA_W = 640
    CAMERA_H = 360
    MAP_LOAD_TIMEOUT_S = 180.0

    def __init__(
        self,
        host: str = "localhost",
        port: int = 2000,
        timeout: float = 10.0,
        time_limit: int = 3000,
        render_mode: Optional[str] = None,
        num_npc_vehicles: int = 300,
        num_pedestrians: int = 300,
        show_sensor_data: bool = False,
        curriculum_stage: int = 0,
        seed: Optional[int] = None,
        maps="all",
        map_rotate_every: int = 10,
        traffic_fill: float = 1.0,
        steer_smoothing: float = 0.0,
        steer_change_penalty: float = 0.0,
        overtake_bonus: float = 0.0,
        lead_corridor_follows_ego: bool = False,
    ):
        """maps: 'all' = every installed town (one map per town, '_Opt' preferred), None = stay on the
        server's current map, or a list of map names. Training switches to the next map every
        `map_rotate_every` episodes; evaluation episodes pick their map with reset options
        {'map_slot': i, 'map_slots': n} (or {'map': name}). num_npc_vehicles / num_pedestrians cap
        the curriculum stage's counts; traffic_fill caps them further at that share of the town's car
        capacity and of num_pedestrians (default 1.0 = no extra cap; the experiments pass
        TRAINING_TRAFFIC_FILL = 0.7 through --traffic-fill).

        Driving-quality options (all off by default = the behaviour the 2q_cnn run was trained with):
          steer_smoothing k in [0, 1): applied steer = k * previous applied steer + (1 - k) * command
              (a first-order low-pass; prev_steer in the observation is the applied value, so the
              state stays Markov).
          steer_change_penalty w: reward term -w * (command_t - command_{t-1})^2 on the steering
              command the env receives, so the policy itself stops oscillating.
          overtake_bonus B: +B once per vehicle passed after a safe-overtake opportunity (see
              commons/overtake.py).
          lead_corridor_follows_ego: centre the perception corridor (lead vehicle, LiDAR gap) on the
              ego instead of the route centre line, so a lane change is seen correctly.
        Any of steer_change_penalty / overtake_bonus changes the reward: REWARD_VERSION becomes "3.1"."""
        self.host = host
        self.port = port
        self.timeout = timeout
        self.time_limit = int(time_limit)
        self.render_mode = render_mode
        self.num_npc_vehicles = int(num_npc_vehicles)
        self.num_pedestrians = int(num_pedestrians)
        self.traffic_fill = float(np.clip(float(traffic_fill), 0.0, 1.0))
        self.steer_smoothing = float(np.clip(float(steer_smoothing), 0.0, 0.95))
        self.steer_change_penalty = max(0.0, float(steer_change_penalty))
        self.overtake_bonus = max(0.0, float(overtake_bonus))
        self.lead_corridor_follows_ego = bool(lead_corridor_follows_ego)
        if self.steer_change_penalty > 0.0 or self.overtake_bonus > 0.0:
            self.REWARD_VERSION = "3.1"      # instance override: v3 + steering-change / overtaking terms
        self._overtake = OvertakeTracker(self.overtake_bonus)
        self._prev_cmd_steer = 0.0
        self.show_sensor_data = bool(show_sensor_data)
        self.display_overlay = None     # optional fn(BGR copy) -> BGR image for the camera window (demo HUD)
        self._maps_request = maps
        self.map_rotate_every = max(1, int(map_rotate_every))
        self._maps: List[str] = []
        self._episodes_on_map = 0
        self.curriculum_stage = int(np.clip(int(curriculum_stage), 0, len(STAGES) - 1))
        self._active_stage = self.curriculum_stage      # stage of the episode in progress
        self._seed = seed
        self._rng = random.Random(seed)

        # CARLA objects
        self.client = None
        self.world = None
        self.map = None
        self.blueprint_library = None
        self.traffic_manager = None
        self._tm_generation = 0             # n-th Traffic Manager of this env (a new one per map load)
        self.spectator = None

        # Actors (NPC / walker lists hold actor ids)
        self.ego_vehicle = None
        self.npc_vehicles: List[int] = []
        self.walkers: List[int] = []
        self.walker_controllers: List[int] = []
        self._population_counts = None

        # Sensors and their frame-stamped queues
        self.rgb_camera = None
        self.lidar_sensor = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self._q_rgb: "queue.Queue" = queue.Queue()
        self._q_lidar: "queue.Queue" = queue.Queue()
        self._q_collision: "queue.Queue" = queue.Queue()
        self._q_lane: "queue.Queue" = queue.Queue()
        self.rgb_data = None            # latest frame-matched camera image (H, W, 3) uint8 BGR
        self.lidar_points = None        # latest filtered sweep (N, 3) in the sensor frame
        self.lidar_transform = None     # sensor world transform of that sweep
        self._sensor_timeouts = 0
        self._frame = 0
        self._road_z_range = (-1e9, 1e9)   # lowest / highest road of the loaded map (NPC physics guard)
        self._npc_removed_on_map = 0
        self._npc_removed_pending = 0       # removed by the guard, not yet replaced

        # Episode state
        self.route = None
        self._episode_start_frame = 0
        self._episode_step = 0
        self._episode_limit = self.time_limit
        self._snapshot: Optional[dict] = None
        self._cbf_state: dict = self._default_cbf_state()
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._proj_hint = None
        self._off_route_steps = 0
        self._wrong_way_steps = 0
        self._blocked_steps = 0
        self._last_crash_xyz = None
        self._spawn_index = None
        self._ep = self._new_episode_stats()
        self._collision_count = 0           # also the duck-typing marker used by _find_carla_and_cbf
        self._ensemble_uncertainty = 0.0    # written by PolicyTrustScoreCallback
        self._trust_score = 1.0

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict({
            'rgb_data': spaces.Box(low=0, high=255, shape=(self.CAMERA_H, self.CAMERA_W, 3), dtype=np.uint8),
            'nav': spaces.Box(low=-np.inf, high=np.inf, shape=(NAV_DIM,), dtype=np.float32),
        })

        self._connect_carla()
        self._maps = self._resolve_maps(maps)
        if len(self._maps) > 1:
            print(f"[ENV] Map rotation: {', '.join(self._maps)} (every {self.map_rotate_every} episodes)")
        self._setup_world()
        self._init_world_objects()
        self._red_monitor = RedLightMonitor()
        self._expert = ExpertDriver()

        self._apply_stage_population()
        # Warm-up ticks so every sensor has delivered at least once before the first reset().
        self._tick()
        self._tick()

    # ------------------------------------------------------------------ setup
    def _resolve_maps(self, maps) -> List[str]:
        if maps is None:
            return [self.current_map_name]
        if isinstance(maps, str) and maps.lower() == "all":
            listing = getattr(self.client, "get_available_maps", None)
            found = pick_town_maps(listing()) if callable(listing) else []
            found = [m for m in found if town_of(m) not in UNSTABLE_TOWNS]
            return found or [self.current_map_name]
        names = [map_basename(m) for m in ([maps] if isinstance(maps, str) else maps)]
        return names or [self.current_map_name]

    @property
    def current_map_name(self) -> str:
        return map_basename(getattr(self.map, "name", "")) if self.map is not None else ""

    def _init_world_objects(self):
        """Ego, sensors, spawn points and the map-specific planners for the loaded world."""
        self._clear_map()
        self._spawn_points = list(self.map.get_spawn_points())
        self._spawn_xyz = np.array([[sp.location.x, sp.location.y, sp.location.z] for sp in self._spawn_points])
        road_z = [wp.transform.location.z for wp in self.map.generate_waypoints(10.0)] or [0.0]
        self._road_z_range = (float(min(road_z)), float(max(road_z)))
        self._npc_removed_on_map = 0
        self._npc_removed_pending = 0
        self._spawn_ego()
        self._attach_sensors()
        self._grp = GlobalRoutePlanner(self.map, self.ROUTE_SAMPLING_M)
        self._tl_tracker = TrafficLightTracker(self.world, self.map)
        self._stop_monitor = StopSignMonitor(self.world, self.map)

    def _release_world_actors(self):
        """Stop and destroy the sensors, the traffic and the ego (before a map change or on close)."""
        for sensor in (self.rgb_camera, self.lidar_sensor, self.collision_sensor, self.lane_invasion_sensor):
            try:
                if sensor is not None:
                    sensor.stop()
                    sensor.destroy()
            except Exception:
                pass
        self.rgb_camera = self.lidar_sensor = self.collision_sensor = self.lane_invasion_sensor = None
        try:
            self._destroy_population()
        except Exception:
            self.npc_vehicles, self.walkers, self.walker_controllers = [], [], []
            self._population_counts = None
        try:
            if self.ego_vehicle is not None:
                self.ego_vehicle.destroy()
        except Exception:
            pass
        self.ego_vehicle = None

    def _load_map(self, name: str):
        """Switch the server to another town and rebuild everything that belongs to the old one."""
        name = map_basename(name)
        t0 = time.time()
        print(f"[ENV] Loading map {name} (was {self.current_map_name})...")
        self._release_world_actors()
        # Load in asynchronous mode and wait until the server really reports the new town: in
        # synchronous mode load_world() can return the old world before the change took effect.
        try:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
        except Exception:
            pass
        self._retire_traffic_manager()   # never carry a Traffic Manager across load_world()
        self.client.set_timeout(self.MAP_LOAD_TIMEOUT_S)
        try:
            world = self.client.load_world(name)
            deadline = time.time() + self.MAP_LOAD_TIMEOUT_S
            while map_basename(world.get_map().name) != name:
                if time.time() > deadline:
                    raise RuntimeError(f"CARLA did not switch to map {name} within {self.MAP_LOAD_TIMEOUT_S:.0f} s")
                time.sleep(0.5)
                world = self.client.get_world()
        finally:
            self.client.set_timeout(self.timeout)
        self.world = world
        self.map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()
        self.spectator = self.world.get_spectator()
        self._tm_generation = (self._tm_generation + 1) % self.TM_GENERATIONS
        self._setup_world()              # a fresh Traffic Manager, built on the loaded map
        self.world.tick()
        for q in (self._q_rgb, self._q_lidar, self._q_collision, self._q_lane):   # frames of the old world
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
        self._init_world_objects()
        self._population_counts = None
        self._last_crash_xyz = None
        self._proj_hint = None
        self._episodes_on_map = 0
        self._tick()
        self._tick()
        print(f"[ENV] Map {self.current_map_name} ready in {time.time() - t0:.0f} s "
              f"({len(self._spawn_points)} spawn points)")

    def _target_map(self, options: Dict) -> Optional[str]:
        """The map this reset must switch to, or None to stay."""
        if options.get("map"):
            want = map_basename(options["map"])
        elif len(self._maps) < 2:
            return None
        elif options.get("map_slot") is not None:
            want = eval_map(self._maps, options["map_slot"], options.get("map_slots", 1))
        elif self._episodes_on_map >= self.map_rotate_every:
            want = next_map(self._maps, self.current_map_name)
        else:
            return None
        return None if want == self.current_map_name else want

    def _connect_carla(self):
        print(f"Connecting to CARLA at {self.host}:{self.port}...")
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(self.timeout)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()
        self.spectator = self.world.get_spectator()
        print(f"[OK] Connected to CARLA ({self.map.name})")

    def _retire_traffic_manager(self):
        """Shut the current Traffic Manager down (its cars are already destroyed) so none survives a map load."""
        tm, self.traffic_manager = self.traffic_manager, None
        if tm is None:
            return
        try:
            tm.set_synchronous_mode(False)
            tm.shut_down()
        except Exception as exc:  # noqa: BLE001 - a new one on a new port is created after the load anyway
            print(f"[ENV] WARNING could not shut down the old Traffic Manager: {exc}")

    def _setup_world(self):
        # TM must be synchronous before the world is, or NPCs tick on a different clock.
        tm_port = self.port + self.TM_PORT_OFFSET + self.TM_PORT_STRIDE * self._tm_generation
        self.traffic_manager = self.client.get_trafficmanager(tm_port)
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        self.traffic_manager.set_hybrid_physics_mode(True)   # needs an actor with role_name 'hero'
        if self._seed is not None:
            self.traffic_manager.set_random_device_seed(int(self._seed))
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = DT
        self.world.apply_settings(settings)
        print("[OK] Synchronous mode enabled (20 FPS), TM in sync mode")

    def _clear_map(self):
        actors = self.world.get_actors()
        for ctrl in actors.filter('controller.ai.walker'):
            try:
                ctrl.stop()
            except Exception:
                pass
        ids = [a.id for a in actors
               if a.type_id.startswith(('vehicle', 'walker', 'sensor', 'controller'))]
        if ids:
            self.client.apply_batch_sync([carla.command.DestroyActor(i) for i in ids], False)
        print(f"[OK] Map cleared ({len(ids)} actors removed)")

    def _spawn_ego(self):
        ego_bp = None
        for name in self.EGO_BLUEPRINTS:
            found = self.blueprint_library.filter(name)
            if len(found) > 0:
                ego_bp = found[0]
                break
        if ego_bp is None:
            ego_bp = self.blueprint_library.filter("vehicle.*")[0]
        if ego_bp.has_attribute('role_name'):
            ego_bp.set_attribute('role_name', 'hero')   # TM hybrid physics freezes every car without a hero

        order = list(range(len(self._spawn_points)))
        self._rng.shuffle(order)
        for i in order:
            self.ego_vehicle = self.world.try_spawn_actor(ego_bp, self._spawn_points[i])
            if self.ego_vehicle is not None:
                break
        if self.ego_vehicle is None:
            raise RuntimeError("Could not spawn the ego vehicle: no free spawn point")
        print(f"[OK] Ego vehicle spawned ({ego_bp.id}, role_name=hero)")

    def _attach_sensors(self):
        bp = self.blueprint_library.find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(self.CAMERA_W))
        bp.set_attribute('image_size_y', str(self.CAMERA_H))
        bp.set_attribute('fov', '90')
        self.rgb_camera = self.world.spawn_actor(
            bp, carla.Transform(carla.Location(x=0.8, z=1.7)), attach_to=self.ego_vehicle)
        self.rgb_camera.listen(self._on_rgb_image)

        bp = self.blueprint_library.find('sensor.lidar.ray_cast')
        bp.set_attribute('channels', '32')
        bp.set_attribute('range', '50')
        bp.set_attribute('points_per_second', '56000')
        # One full 360-degree sweep per 20 Hz tick (at 10 Hz every other tick held only half a scan).
        bp.set_attribute('rotation_frequency', '20')
        self.lidar_sensor = self.world.spawn_actor(
            bp, carla.Transform(carla.Location(z=1.7)), attach_to=self.ego_vehicle)
        self.lidar_sensor.listen(self._on_lidar_data)

        bp = self.blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=self.ego_vehicle)
        self.collision_sensor.listen(self._on_collision)

        bp = self.blueprint_library.find('sensor.other.lane_invasion')
        self.lane_invasion_sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=self.ego_vehicle)
        self.lane_invasion_sensor.listen(self._on_lane_invasion)
        print("[OK] Sensors attached (RGB camera, 20 Hz LiDAR, collision, lane invasion)")

    # ------------------------------------------------------------ callbacks
    # Callbacks run on the CARLA client thread: they only convert and enqueue, stamped with event.frame.
    def _on_rgb_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        self._q_rgb.put((image.frame, array[:, :, :3].copy()))

    def _on_lidar_data(self, point_cloud):
        data = np.frombuffer(point_cloud.raw_data, dtype=np.float32).reshape(-1, 4)
        points = data[:, :3]
        # Drop ground returns (sensor 1.7 m up: road at z ~ -1.7) and the ego car's own body.
        keep = points[:, 2] > self.LIDAR_GROUND_Z_CUT
        keep &= ~((np.abs(points[:, 0]) < self.LIDAR_EGO_HALF_X) & (np.abs(points[:, 1]) < self.LIDAR_EGO_HALF_Y))
        self._q_lidar.put((point_cloud.frame, points[keep].copy(), point_cloud.transform))

    def _on_collision(self, event):
        other = event.other_actor
        type_id = other.type_id if other is not None else "unknown"
        other_id = other.id if other is not None else None
        imp = event.normal_impulse
        impulse = math.sqrt(imp.x ** 2 + imp.y ** 2 + imp.z ** 2)
        self._q_collision.put((event.frame, type_id, other_id, impulse))

    def _on_lane_invasion(self, event):
        self._q_lane.put((event.frame, list(event.crossed_lane_markings)))

    # ------------------------------------------------------------- ticking
    def _tick(self) -> int:
        """world.tick(), remove physics-broken NPC cars, then wait for the frame-matched camera image and
        LiDAR sweep."""
        frame = self.world.tick()
        self._frame = frame
        self._remove_broken_npcs()
        msg = wait_for_frame(self._q_rgb, frame, self.SENSOR_TIMEOUT_S)
        if msg is not None:
            self.rgb_data = msg[1]
        else:
            self._on_sensor_timeout("camera", frame)
        msg = wait_for_frame(self._q_lidar, frame, self.SENSOR_TIMEOUT_S)
        if msg is not None:
            self.lidar_points, self.lidar_transform = msg[1], msg[2]
        else:
            self._on_sensor_timeout("lidar", frame)
        return frame

    def _adjacent_lane_clear(self, actors, snap) -> bool:
        """True if a same-direction Driving lane beside the ego may legally be entered here (lane marking)
        and holds no vehicle from CLEAR_BEHIND_M behind to CLEAR_AHEAD_M ahead (commons/overtake.py)."""
        ego = self.ego_vehicle
        ego_loc = ego.get_location()
        wp = self.map.get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None or wp.is_junction:
            return False
        sides = []
        if wp.lane_change in (carla.LaneChange.Left, carla.LaneChange.Both):
            sides.append(wp.get_left_lane())
        if wp.lane_change in (carla.LaneChange.Right, carla.LaneChange.Both):
            sides.append(wp.get_right_lane())
        sides = [w for w in sides
                 if w is not None and w.lane_type == carla.LaneType.Driving and w.lane_id * wp.lane_id > 0]
        if not sides:
            return False
        s_ego = float(snap["s"])
        near = [a for a in actors.filter("vehicle.*")
                if a.id != ego.id and a.get_location().distance(ego_loc) < CLEAR_AHEAD_M + 25.0]
        vehicles = []
        if near:
            xy = np.array([[a.get_location().x, a.get_location().y] for a in near], dtype=float)
            s_v, l_v = self.route.project_points(xy, s_ego - CLEAR_BEHIND_M - 10.0, s_ego + CLEAR_AHEAD_M + 10.0)
            vehicles = [(float(s), float(l), float(a.bounding_box.extent.x)) for a, s, l in zip(near, s_v, l_v)]
        for adj in sides:
            loc = adj.transform.location
            _, l_adj = self.route.project_points(np.array([[loc.x, loc.y]], dtype=float), s_ego - 5.0, s_ego + 5.0)
            if lane_clear(vehicles, s_ego, float(l_adj[0]), 0.5 * float(adj.lane_width)):
                return True
        return False

    def _overtake_step(self, snap, actors, n_illegal: int, crashed: bool) -> float:
        """Safe-overtake bonus earned on this step (commons/overtake.OvertakeTracker); never raises."""
        route = self.route

        def lead_front_s(actor_id):
            a = actors.find(int(actor_id))
            if a is None:
                return None
            loc = a.get_location()
            s, _ = route.project_points(np.array([[loc.x, loc.y]], dtype=float), snap["s"] - 60.0, snap["s"] + 60.0)
            return float(s[0]) + float(a.bounding_box.extent.x)

        try:
            return self._overtake.update(snap, self._episode_step, n_illegal, crashed,
                                         lambda: self._adjacent_lane_clear(actors, snap), lead_front_s)
        except Exception as exc:  # noqa: BLE001 - a reward helper must never end the episode
            if not getattr(self, "_overtake_warned", False):
                print(f"[ENV] WARNING overtake check failed ({exc}); no bonus on such steps")
                self._overtake_warned = True
            return 0.0

    def _remove_broken_npcs(self) -> int:
        """Destroy NPC cars that the last tick shows launched, under the map or absurdly fast (see
        npc_physics_broken), so they are no phantom obstacles and the Traffic Manager stops driving them.
        Returns how many."""
        if not self.npc_vehicles:
            return 0
        snapshot = self.world.get_snapshot()
        z_min, z_max = self._road_z_range
        broken = []
        for vid in self.npc_vehicles:
            actor = snapshot.find(vid)
            if actor is None:
                continue
            v = actor.get_velocity()
            if npc_physics_broken(v.x, v.y, v.z, actor.get_transform().location.z, z_min, z_max):
                broken.append(vid)
        if not broken:
            return 0
        try:
            self.client.apply_batch_sync([carla.command.DestroyActor(i) for i in broken], False)
        except Exception as exc:  # noqa: BLE001 - never let the guard end an episode
            print(f"[ENV] WARNING could not destroy broken NPC cars {broken}: {exc}")
        gone = set(broken)
        self.npc_vehicles = [i for i in self.npc_vehicles if i not in gone]
        self._npc_removed_on_map += len(broken)
        self._npc_removed_pending += len(broken)   # replaced at the next reset
        print(f"[ENV] removed {len(broken)} NPC car(s) with broken physics (launched, under the map or "
              f"> {NPC_MAX_SPEED_MPS:.0f} m/s); {self._npc_removed_on_map} on {self.current_map_name} so far")
        return len(broken)

    def _on_sensor_timeout(self, name: str, frame: int):
        self._sensor_timeouts += 1
        dprint(f"[ENV] {name} frame {frame} not received within {self.SENSOR_TIMEOUT_S}s, using the last one")
        if self._sensor_timeouts % 100 == 1:
            print(f"[ENV] WARNING sensor timeout #{self._sensor_timeouts} ({name}); using the last message")

    def _flush_event_queues(self):
        for q in (self._q_collision, self._q_lane):
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

    # ---------------------------------------------------------- population
    def _population_target(self, stage: int) -> Tuple[int, int]:
        """Stage counts, capped by the constructor's maxima and by the traffic fill of the town
        (TRAFFIC_FILL of its car capacity and of the pedestrian maximum)."""
        cfg = STAGES[stage]
        n_npc = self.num_npc_vehicles if cfg.get("num_npc") is None else min(int(cfg["num_npc"]), self.num_npc_vehicles)
        n_walk = (self.num_pedestrians if cfg.get("num_walkers") is None
                  else min(int(cfg["num_walkers"]), self.num_pedestrians))
        car_cap, walker_cap = traffic_caps(len(self._spawn_points), self.num_pedestrians, self.traffic_fill)
        return min(n_npc, car_cap), min(n_walk, walker_cap)

    def _apply_stage_population(self):
        """(Re)spawn NPC vehicles and walkers when the active stage needs different counts."""
        target = self._population_target(self._active_stage)
        if target == self._population_counts:
            if self._npc_removed_pending > 0:   # replace the cars the physics guard removed
                missing, before = self._npc_removed_pending, len(self.npc_vehicles)
                self._npc_removed_pending = 0
                self._spawn_npcs(missing)
                print(f"[ENV] replaced {len(self.npc_vehicles) - before}/{missing} removed NPC cars "
                      f"({len(self.npc_vehicles)}/{target[0]} vehicles)")
            return
        self._destroy_population()
        self._spawn_npcs(target[0])
        self._spawn_walkers(target[1])
        self._population_counts = target
        print(f"[ENV] Traffic for stage {self._active_stage} ({STAGES[self._active_stage]['name']}) on "
              f"{self.current_map_name}: {len(self.npc_vehicles)}/{target[0]} vehicles, "
              f"{len(self.walkers)}/{target[1]} walkers")

    def _spawn_npcs(self, n: int):
        if n <= 0:
            return
        SpawnActor, SetAutopilot, FutureActor = (carla.command.SpawnActor, carla.command.SetAutopilot,
                                                 carla.command.FutureActor)
        bps = [bp for bp in self.blueprint_library.filter('vehicle.*')
               if not bp.has_attribute('number_of_wheels') or int(bp.get_attribute('number_of_wheels')) == 4]
        ego_loc = self.ego_vehicle.get_location()
        spawn_points = [sp for sp in self._spawn_points if sp.location.distance(ego_loc) > 15.0]
        self._rng.shuffle(spawn_points)
        tm_port = self.traffic_manager.get_port()
        batch = []
        for sp in spawn_points[:n]:
            bp = self._rng.choice(bps)
            if bp.has_attribute('color'):
                bp.set_attribute('color', self._rng.choice(bp.get_attribute('color').recommended_values))
            if bp.has_attribute('role_name'):
                bp.set_attribute('role_name', 'autopilot')
            batch.append(SpawnActor(bp, sp).then(SetAutopilot(FutureActor, True, tm_port)))
        for resp in self.client.apply_batch_sync(batch, True):
            if not resp.error:
                self.npc_vehicles.append(resp.actor_id)

    def _spawn_walkers(self, n: int):
        if n <= 0:
            return
        SpawnActor = carla.command.SpawnActor
        walker_bps = self.blueprint_library.filter('walker.pedestrian.*')
        ctrl_bp = self.blueprint_library.find('controller.ai.walker')
        batch, speeds = [], []
        for _ in range(n):
            loc = self.world.get_random_location_from_navigation()
            if loc is None:
                continue
            bp = self._rng.choice(walker_bps)
            if bp.has_attribute('is_invincible'):
                bp.set_attribute('is_invincible', 'false')
            speed = 1.4
            if bp.has_attribute('speed'):
                values = bp.get_attribute('speed').recommended_values
                if len(values) > 1:
                    speed = float(values[1])    # walking speed
            batch.append(SpawnActor(bp, carla.Transform(loc)))
            speeds.append(speed)
        walker_ids, walker_speeds = [], []
        for resp, speed in zip(self.client.apply_batch_sync(batch, True), speeds):
            if not resp.error:
                walker_ids.append(resp.actor_id)
                walker_speeds.append(speed)

        ctrl_batch = [SpawnActor(ctrl_bp, carla.Transform(), wid) for wid in walker_ids]
        orphans, ctrl_speeds = [], {}
        for resp, wid, speed in zip(self.client.apply_batch_sync(ctrl_batch, True), walker_ids, walker_speeds):
            if resp.error:
                orphans.append(wid)
            else:
                self.walkers.append(wid)
                self.walker_controllers.append(resp.actor_id)
                ctrl_speeds[resp.actor_id] = speed
        if orphans:
            self.client.apply_batch_sync([carla.command.DestroyActor(i) for i in orphans], False)

        self._tick()   # controllers need one tick before start()
        for ctrl in self.world.get_actors(self.walker_controllers):
            ctrl.start()
            target = self.world.get_random_location_from_navigation()
            if target is not None:
                ctrl.go_to_location(target)
            ctrl.set_max_speed(ctrl_speeds.get(ctrl.id, 1.4))

    def _destroy_population(self):
        if self.walker_controllers:
            for ctrl in self.world.get_actors(self.walker_controllers):
                try:
                    ctrl.stop()
                except Exception:
                    pass
        ids = list(self.walker_controllers) + list(self.walkers) + list(self.npc_vehicles)
        if ids:
            self.client.apply_batch_sync([carla.command.DestroyActor(i) for i in ids], True)
        self.npc_vehicles, self.walkers, self.walker_controllers = [], [], []
        self._population_counts = None
        self._npc_removed_pending = 0

    # --------------------------------------------------------------- reset
    def _select_spawn(self, actors, spawn_index: Optional[int]) -> int:
        ego_id = self.ego_vehicle.id
        vehicles = [(l.x, l.y, l.z) for a in actors.filter('vehicle.*') if a.id != ego_id
                    for l in (a.get_location(),)]
        walkers = [(l.x, l.y, l.z) for a in actors.filter('walker.pedestrian.*') for l in (a.get_location(),)]
        n = len(self._spawn_points)
        if spawn_index is not None:
            # deterministic (evaluation): the requested index, or the next clear one after it
            start = int(spawn_index) % n
            order = [(start + k) % n for k in range(n)]
            idx, ok = select_spawn_index(self._spawn_xyz, vehicles, walkers, None, order,
                                         self.SPAWN_MIN_VEHICLE_DIST_M, self.SPAWN_MIN_WALKER_DIST_M, 0.0)
        else:
            order = list(range(n))
            self._rng.shuffle(order)
            idx, ok = select_spawn_index(self._spawn_xyz, vehicles, walkers, self._last_crash_xyz, order,
                                         self.SPAWN_MIN_VEHICLE_DIST_M, self.SPAWN_MIN_WALKER_DIST_M,
                                         self.SPAWN_MIN_CRASH_DIST_M)
        if not ok:
            dprint(f"[ENV] no spawn point meets the clearance rules, using the best one ({idx})")
        return idx

    def _teleport_ego(self, transform):
        """Brake + zero velocity, teleport, tick, zero velocity, settle in first gear."""
        ego = self.ego_vehicle
        zero = carla.Vector3D(0.0, 0.0, 0.0)
        # manual first gear during the settle avoids the first-throttle gearbox delay (CARLA issue #1640)
        hold = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, hand_brake=False,
                                    manual_gear_shift=True, gear=1)
        ego.apply_control(hold)
        ego.set_target_velocity(zero)
        ego.set_target_angular_velocity(zero)
        ego.set_transform(transform)       # TeleportPhysics keeps velocity; the new pose shows after a tick
        self._tick()
        ego.set_target_velocity(zero)
        ego.set_target_angular_velocity(zero)
        for _ in range(self.SETTLE_TICKS):
            ego.apply_control(hold)
            self._tick()
        ego.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, manual_gear_shift=False))
        self._move_spectator()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Start a new episode.

        options: 'spawn_index' (int, deterministic spawn point) and 'route_seed' (int, deterministic
        destination) for evaluation; 'map_slot' / 'map_slots' (evaluation map) or 'map' (a map name).
        A pending curriculum stage change and a due map switch are applied here.
        """
        super().reset(seed=seed)
        if seed is not None:
            self._rng.seed(seed)
        options = dict(options or {})

        target = self._target_map(options)
        if target is not None:
            self._load_map(target)
        if options.get("map_slot") is None:
            self._episodes_on_map += 1          # training episodes drive the rotation; eval ones do not

        self._active_stage = self.curriculum_stage
        stage = STAGES[self._active_stage]
        self._apply_stage_population()

        self._spawn_index = self._select_spawn(self.world.get_actors(), options.get('spawn_index'))
        self._teleport_ego(self._spawn_points[self._spawn_index])

        # Everything sensed so far belongs to the previous episode or the teleport.
        self._flush_event_queues()
        self._episode_start_frame = self._frame

        # Only now clear the per-episode state.
        self._episode_step = 0
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._prev_cmd_steer = 0.0
        self._overtake.reset()
        self._proj_hint = None
        self._off_route_steps = 0
        self._wrong_way_steps = 0
        self._blocked_steps = 0
        self._collision_count = 0
        self._ep = self._new_episode_stats()
        self._red_monitor.reset()
        self._stop_monitor.reset()

        route_seed = options.get('route_seed')
        rng = random.Random(int(route_seed)) if route_seed is not None else self._rng
        self.route = Route.plan(self._grp, self.map, self.ego_vehicle.get_location(), self._spawn_points,
                                float(stage['route_min_m']), float(stage['route_max_m']), rng)

        snap = self._build_snapshot(self.world.get_actors())
        self._episode_limit = episode_step_limit(self.route.length_m, snap['speed_limit'], self.time_limit, self)
        snap.update(self._event_defaults())
        snap.update(episode_step=0, episode_limit=self._episode_limit,
                    crash_penalty=float(stage['crash_penalty']))
        self._set_snapshot(snap)
        self._ep["max_s"] = snap["s"]

        info = {
            "spawn_index": self._spawn_index,
            "map": self.current_map_name,
            "route_length_m": float(self.route.length_m),
            "episode_limit": self._episode_limit,
            "curriculum_stage": self._active_stage,
            "speed_limit_mps": snap["speed_limit"],
        }
        dprint(f"[ENV] reset: stage={self._active_stage} spawn={self._spawn_index} "
               f"route={self.route.length_m:.0f}m limit={self._episode_limit} steps v0={snap['speed']:.2f}")
        return self._make_obs(snap), info

    # ------------------------------------------------------------ snapshot
    @staticmethod
    def _event_defaults() -> dict:
        return dict(collision=False, collision_pedestrian=False, n_invasions=0, n_illegal_invasions=0,
                    off_route_steps=0, wrong_way_steps=0, blocked_steps=0)

    def _build_snapshot(self, actors) -> dict:
        """The single per-tick state used by the observation, reward, terminations, CBF and expert."""
        ego = self.ego_vehicle
        route = self.route
        tf = ego.get_transform()
        loc, yaw = tf.location, tf.rotation.yaw
        vel = ego.get_velocity()
        fwd = tf.get_forward_vector()
        v = float(vel.x * fwd.x + vel.y * fwd.y + vel.z * fwd.z)    # signed forward speed
        # carla.Waypoint has no speed limit; the vehicle's limit follows the speed signs it passes
        L = clip_speed_limit(ego.get_speed_limit(), self)

        proj = route.project(loc, yaw, hint_idx=self._proj_hint)
        self._proj_hint = proj.idx
        lane = route.lane_status(loc, yaw, self.map)

        offset = float(proj.e_y) if self.lead_corridor_follows_ego else 0.0   # corridor centre [m, + = right]
        lead_actor = find_lead(actors, ego, route, proj, lateral_offset=offset)
        if self.lidar_points is not None and self.lidar_transform is not None:
            static_gap = float(lidar_static_gap(self.lidar_points, self.lidar_transform, route, proj,
                                                lateral_offset=offset))
        else:
            static_gap = 100.0
        lead = combine(lead_actor, static_gap)

        tl = self._tl_tracker.query(ego, route, proj)
        stop_gap, stop_violation = self._stop_monitor.update(ego, route, proj, v)
        red_run = bool(self._red_monitor.update(tl, proj, v))
        y_left, y_right = route.lane_bounds(proj.s, self.map, EGO_HALF_WIDTH)
        pts = route.points_ahead_ego(proj.s, tf, (5.0, 10.0, 20.0))
        cmd, cmd_dist = route.next_maneuver(proj.s, 50.0)

        legal_lane = bool(lane.on_drivable) and bool(lane.in_same_dir_lane)
        e_lane = float(lane.e_lane) if legal_lane else float(proj.e_y)
        e_psi_lane = wrap_angle(lane.lane_heading_err) if legal_lane else wrap_angle(proj.e_psi)
        ttc_lead = min(float(time_to_collision(lead.gap_m, v, lead.v_lead_mps)), self.TTC_CAP_S)

        return {
            "frame": self._frame,
            "x": float(loc.x), "y": float(loc.y), "z": float(loc.z), "yaw": float(yaw),
            "speed": v, "speed_limit": L,
            "s": float(proj.s), "e_y": float(proj.e_y), "e_psi": float(proj.e_psi), "kappa": float(proj.kappa),
            "proj_idx": int(proj.idx), "lane_width": float(proj.lane_width),
            "remaining_m": float(proj.remaining_m), "route_length": float(route.length_m),
            "on_drivable": bool(lane.on_drivable), "in_route_lane": bool(lane.in_route_lane),
            "in_same_dir_lane": bool(lane.in_same_dir_lane), "in_opposite_lane": bool(lane.in_opposite_lane),
            "is_junction": bool(lane.is_junction),
            "e_lane": e_lane, "e_psi_lane": e_psi_lane,
            "lead_gap": float(lead.gap_m), "lead_v": float(lead.v_lead_mps), "lead_kind": str(lead.kind),
            "lead_id": lead.actor_id, "ttc": ttc_lead,
            "tl_state": str(tl.state), "tl_stop_gap": float(tl.stop_gap_m), "tl_id": tl.light_id,
            "stop_gap": float(stop_gap), "stop_violation": bool(stop_violation), "red_light_run": red_run,
            "y_max_left": float(y_left), "y_max_right": float(y_right),
            "route_pts": np.asarray(pts, dtype=float).reshape(3, 2),
            "cmd": roadoption_name(cmd), "cmd_dist": float(cmd_dist),
            "prev_steer": float(self._prev_action[0]), "prev_accel": float(self._prev_action[1]),
            # objects for the expert / lane-invasion legality (not used by the pure reward)
            "_transform": tf, "_proj": proj, "_lane": lane, "_lead": lead, "_tl": tl,
        }

    def _set_snapshot(self, snap: dict):
        self._snapshot = snap
        self._cbf_state = self._cbf_state_from(snap)

    def _cbf_state_from(self, snap: dict) -> dict:
        gap, v_lead, kind = cbf_gap(snap["lead_gap"], snap["lead_v"], snap["lead_kind"],
                                    snap["tl_state"], snap["tl_stop_gap"], snap["speed"])
        ttc_s = min(float(time_to_collision(gap, snap["speed"], v_lead)), self.TTC_CAP_S)
        return {
            "speed": snap["speed"],
            "speed_limit": snap["speed_limit"],
            "gap": gap,
            "v_lead": v_lead,
            "gap_kind": kind,
            "e_y": snap["e_y"],
            "e_psi": snap["e_psi"],
            "kappa_ref": snap["kappa"],
            "y_max_left": snap["y_max_left"],
            "y_max_right": snap["y_max_right"],
            "d_collision": gap,
            "ttc": ttc_s,
            "lane_offset": snap["e_y"],     # legacy key
        }

    def _default_cbf_state(self) -> dict:
        return {"speed": 0.0, "speed_limit": 8.33, "gap": 100.0, "v_lead": 0.0, "gap_kind": "none",
                "e_y": 0.0, "e_psi": 0.0, "kappa_ref": 0.0, "y_max_left": 1.5, "y_max_right": 1.5,
                "d_collision": 100.0, "ttc": self.TTC_CAP_S, "lane_offset": 0.0}

    def build_cbf_state(self, current_speed: Optional[float] = None) -> Dict:
        """CBF input for the action about to be applied: the snapshot from the end of the previous
        step/reset (the wrapper calls this BEFORE env.step). `current_speed` is ignored (legacy)."""
        return dict(self._cbf_state)

    def _make_obs(self, snap: dict) -> Dict:
        rgb = self.rgb_data
        if rgb is None or rgb.shape != (self.CAMERA_H, self.CAMERA_W, 3):
            rgb = np.zeros((self.CAMERA_H, self.CAMERA_W, 3), dtype=np.uint8)
        return {'rgb_data': rgb, 'nav': build_nav_vector(snap)}

    def expert_action(self) -> np.ndarray:
        """Autopilot action [steer, accel] for the current snapshot (replay-buffer warm-up)."""
        s = self._snapshot
        if s is None or self.route is None:
            return np.zeros(2, dtype=np.float32)
        a = self._expert.act(s["_transform"], s["speed"], self.route, s["_proj"], s["_lead"], s["_tl"],
                             s["stop_gap"], s["speed_limit"])
        return np.clip(np.asarray(a, dtype=np.float32).reshape(2), -1.0, 1.0)

    # ---------------------------------------------------------------- step
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        if self.route is None or self._snapshot is None:
            raise RuntimeError("CarlaGymEnv.step() called before reset()")
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        cmd_steer = float(np.clip(a[0], -1.0, 1.0))
        accel = float(np.clip(a[1], -1.0, 1.0))
        k = self.steer_smoothing          # 0.0 = off (applied steer = command)
        steer = float(k * float(self._prev_action[0]) + (1.0 - k) * cmd_steer) if k > 0.0 else cmd_steer
        throttle, brake = accel_to_pedals(accel)
        self.ego_vehicle.apply_control(carla.VehicleControl(throttle=float(throttle), steer=steer,
                                                            brake=float(brake)))
        self._prev_action = np.array([steer, accel], dtype=np.float32)   # executed (post-CBF) action
        prev_cmd_steer, self._prev_cmd_steer = self._prev_cmd_steer, cmd_steer

        prev = self._snapshot
        frame = self._tick()
        self._episode_step += 1

        collisions = drain_frame_events(self._q_collision, self._episode_start_frame, frame)
        invasions = drain_frame_events(self._q_lane, self._episode_start_frame, frame)
        actors = self.world.get_actors()                       # the one get_actors() call of this tick
        snap = self._build_snapshot(actors)

        n_illegal = sum(1 for ev in invasions if lane_invasion_is_illegal(ev[1], prev["_lane"], snap["_lane"]))
        collision_ped = any(str(ev[1]).startswith("walker") for ev in collisions)

        off_route, wrong_way, blocked = condition_flags(snap, self)
        self._off_route_steps = self._off_route_steps + 1 if off_route else 0
        self._wrong_way_steps = self._wrong_way_steps + 1 if wrong_way else 0
        self._blocked_steps = self._blocked_steps + 1 if blocked else 0
        snap.update(
            collision=bool(collisions), collision_pedestrian=collision_ped,
            n_invasions=len(invasions), n_illegal_invasions=n_illegal,
            off_route_steps=self._off_route_steps, wrong_way_steps=self._wrong_way_steps,
            blocked_steps=self._blocked_steps,
            episode_step=self._episode_step, episode_limit=self._episode_limit,
            crash_penalty=float(STAGES[self._active_stage]["crash_penalty"]),
            cmd_steer=cmd_steer, prev_cmd_steer=prev_cmd_steer,
        )
        if self.overtake_bonus > 0.0:
            snap["overtake_bonus"] = self._overtake_step(snap, actors, n_illegal, bool(collisions))

        reward, comps, terminated, truncated, reason = compute_reward_v3(snap, prev, self)
        self._accumulate(snap, prev, reward, comps, collisions, collision_ped, len(invasions), n_illegal)
        self._set_snapshot(snap)
        obs = self._make_obs(snap)

        completion = 1.0 if reason == "endpoint" else self._completion_frac()
        info = {
            "reward_components": comps,
            "terminated_reason": reason if terminated else None,
            "truncated_reason": reason if truncated else None,
            "speed_mps": snap["speed"],
            "speed_limit_mps": snap["speed_limit"],
            "route_completion_frac": completion,
            "curriculum_stage": self._active_stage,
            "episode_step": self._episode_step,
            "lead_gap_m": snap["lead_gap"],
            "ttc_s": snap["ttc"],
            "tl_state": snap["tl_state"],
        }
        if terminated or truncated:
            ep = self._episode_summary(reason, terminated, completion)
            info["episode_metrics"] = ep    # canonical key (SB3's Monitor overwrites info['episode'])
            info["episode"] = ep            # same dict object
            if reason in ("collision", "collision_pedestrian"):
                self._last_crash_xyz = (snap["x"], snap["y"], snap["z"])

        if self.render_mode == "human":
            self.render()
        self._show_camera()
        if DEBUG:
            dprint(f"[ENV] t={self._episode_step:4d} v={snap['speed']:5.2f}/{snap['speed_limit']:4.1f} "
                   f"u=[{steer:+.2f},{accel:+.2f}] s={snap['s']:6.1f}/{snap['route_length']:.0f} "
                   f"ey={snap['e_y']:+.2f} epsi={snap['e_psi']:+.2f} "
                   f"gap={snap['lead_gap']:5.1f}({snap['lead_kind']}) "
                   f"tl={snap['tl_state']} r={reward:+.3f}" + (f" END={reason}" if reason else ""))
        return obs, float(reward), bool(terminated), bool(truncated), info

    # ------------------------------------------------------- episode stats
    @staticmethod
    def _new_episode_stats() -> dict:
        return dict(ret=0.0, distance=0.0, speed_sum=0.0, collisions=0, collisions_ped=0, lane_inv=0,
                    illegal=0, red=0, stop=0, speeding_steps=0, min_ttc=RewardConfigV3.TTC_CAP_S,
                    min_gap=100.0, max_s=0.0, comps=dict.fromkeys(REWARD_COMPONENT_KEYS, 0.0),
                    t0=time.time())

    def _accumulate(self, snap, prev, reward, comps, collisions, collision_ped, n_inv, n_illegal):
        ep = self._ep
        ep["ret"] += reward
        ep["distance"] += math.hypot(snap["x"] - prev["x"], snap["y"] - prev["y"])
        ep["speed_sum"] += abs(snap["speed"])
        if collisions:
            ep["collisions"] += 1
            self._collision_count += 1
        if collision_ped:
            ep["collisions_ped"] += 1
        ep["lane_inv"] += n_inv
        ep["illegal"] += n_illegal
        ep["red"] += int(snap["red_light_run"])
        ep["stop"] += int(snap["stop_violation"])
        ep["speeding_steps"] += int(snap["speed"] > snap["speed_limit"])
        ep["min_ttc"] = min(ep["min_ttc"], snap["ttc"])
        ep["min_gap"] = min(ep["min_gap"], snap["lead_gap"])
        ep["max_s"] = max(ep["max_s"], snap["s"])
        for k, val in comps.items():
            ep["comps"][k] = ep["comps"].get(k, 0.0) + val

    def _completion_frac(self) -> float:
        if self.route is None or self.route.length_m <= 0:
            return 0.0
        return float(np.clip(self._ep["max_s"] / self.route.length_m, 0.0, 1.0))

    def _episode_summary(self, reason: str, terminated: bool, completion: float) -> dict:
        ep = self._ep
        n = max(self._episode_step, 1)
        ret = float(ep["ret"])
        wall = time.time() - ep["t0"]
        return {
            "return": ret,
            "length_steps": int(self._episode_step),
            "distance_m": float(ep["distance"]),
            "mean_speed_mps": float(ep["speed_sum"] / n),
            "route_length_m": float(self.route.length_m),
            "route_completion_frac": float(completion),
            "end_reason": reason,
            "terminated": bool(terminated),
            "collisions": int(ep["collisions"]),
            "collisions_pedestrian": int(ep["collisions_ped"]),
            "lane_invasions": int(ep["lane_inv"]),
            "illegal_lane_invasions": int(ep["illegal"]),
            "red_light_violations": int(ep["red"]),
            "stop_sign_violations": int(ep["stop"]),
            "speeding_frac": float(ep["speeding_steps"] / n),
            "min_ttc_s": float(ep["min_ttc"]),
            "min_gap_m": float(ep["min_gap"]),
            "curriculum_stage": int(self._active_stage),
            "reward_components": {k: float(v) for k, v in ep["comps"].items()},
            "spawn_index": self._spawn_index,
            "map": self.current_map_name,
            # SB3 Monitor-compatible keys (its _dump_logs reads 'r' and 'l')
            "r": ret,
            "l": int(self._episode_step),
            "t": round(wall, 6),
        }

    # ------------------------------------------------------------ curriculum
    def set_curriculum_stage(self, stage: int):
        """Select a curriculum stage; it takes effect at the next reset()."""
        stage = int(np.clip(int(stage), 0, len(STAGES) - 1))
        if stage != self.curriculum_stage:
            print(f"[ENV] Curriculum stage {self.curriculum_stage} -> {stage} ({STAGES[stage]['name']}), "
                  f"applied at the next reset")
        self.curriculum_stage = stage

    # ------------------------------------------------------------- display
    def _move_spectator(self):
        if self.ego_vehicle and self.spectator:
            transform = self.ego_vehicle.get_transform()
            self.spectator.set_transform(carla.Transform(transform.location + carla.Location(z=30),
                                                         carla.Rotation(pitch=-90)))

    def render(self):
        """Spectator follows the ego vehicle from above."""
        self._move_spectator()

    def _show_camera(self):
        if self.show_sensor_data and self.rgb_data is not None:
            image = self.rgb_data
            if self.display_overlay is not None:     # display only: the observation is never touched
                image = self.display_overlay(self.rgb_data.copy())
            cv2.imshow('CARLA RGB Camera', image)    # BGR, as cv2 expects
            cv2.waitKey(1)

    def close(self):
        """Destroy sensors and actors and hand the server back in asynchronous mode."""
        print("Destroying CARLA actors and sensors...")
        self._release_world_actors()
        try:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
            self.traffic_manager.set_synchronous_mode(False)
        except Exception:
            pass
        if self.show_sensor_data:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        print("[OK] All actors and sensors destroyed")

    @property
    def safety_metrics(self) -> Dict:
        """Running per-episode safety counters (for ad-hoc logging)."""
        ep = self._ep
        return {
            'collisions': ep["collisions"],
            'collisions_pedestrian': ep["collisions_ped"],
            'lane_invasions': ep["lane_inv"],
            'illegal_lane_invasions': ep["illegal"],
            'red_light_violations': ep["red"],
            'stop_sign_violations': ep["stop"],
            'route_completion_frac': self._completion_frac(),
            'episode_step': self._episode_step,
            'curriculum_stage': self._active_stage,
            'ensemble_uncertainty': self._ensemble_uncertainty,
            'trust_score': self._trust_score,
        }


# ============================================================================
# Observation Preprocessing Wrapper (Pipeline Integration)
# ============================================================================

class PipelineObservationWrapper(gym.ObservationWrapper):
    """
    Converts raw CARLA observations to pipeline embeddings (512-dim tensor).
    Supports loading pretrained encoder while keeping transformer trainable.
    """

    def __init__(
        self,
        env: gym.Env,
        embed_dim: int = 512,
        num_frames: int = 8,
        encoder_path: Optional[str] = None
    ):
        super().__init__(env)

        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.encoder_path = encoder_path

        # Initialize pipeline with ImageNet-pretrained ResNet50
        # (not loading feature extractor weights from encoder_path)
        print(f"[PIPELINE] Initializing with ImageNet-pretrained ResNet50")
        self.pipeline = Pipeline.from_defaults(
            num_frames=num_frames,
            embed_dim=embed_dim,
            use_timesformer=False,
            fe_weights_path=None,  # Use default ImageNet weights
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.pipeline.eval()

        # Load pretrained SpatioTemporalEncoder weights if provided
        if encoder_path is not None:
            print(f"[PIPELINE] Loading pretrained SpatioTemporalEncoder from: {encoder_path}")
            try:
                state_dict = torch.load(encoder_path, map_location=self.pipeline.device)
                # Handle both direct state_dict and checkpoint format
                if isinstance(state_dict, dict) and 'st_encoder_state' in state_dict:
                    self.pipeline.st_encoder.load_state_dict(state_dict['st_encoder_state'])
                else:
                    # Assume it's direct state_dict
                    self.pipeline.st_encoder.load_state_dict(state_dict)
                print("[PIPELINE] ✓ Loaded SpatioTemporalEncoder weights")
            except Exception as e:
                print(f"[PIPELINE] Error loading checkpoint: {e}")
        else:
            print("[PIPELINE] Training SpatioTemporalEncoder from scratch")

        # Freeze feature extractor (ResNet50)
        print("[PIPELINE] Freezing ResNet50 feature extractor weights")
        self.pipeline.feature_extractor.requires_grad = False
        for param in self.pipeline.feature_extractor.parameters():
            param.requires_grad = False

        # Ensure transformer modules are trainable
        print("[PIPELINE] Enabling training for SpatioTemporalEncoder + StackedHierarchicalTransformer")
        for param in self.pipeline.st_encoder.parameters():
            param.requires_grad = True
        for param in self.pipeline.stacked_transformer.parameters():
            param.requires_grad = True

        # Frame buffer
        self.frame_buffer = deque(maxlen=num_frames)
        for _ in range(num_frames):
            self.frame_buffer.append(np.zeros((360, 640, 3), dtype=np.uint8))

        # Update observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(embed_dim,), dtype=np.float32
        )

    def observation(self, obs: Dict) -> np.ndarray:
        """Convert observation dict to pipeline embedding"""
        rgb_frame = obs.get('rgb_data')
        if rgb_frame is None or rgb_frame.size == 0:
            rgb_frame = np.zeros((360, 640, 3), dtype=np.uint8)

        # Ensure BGR format and correct shape
        if rgb_frame.dtype != np.uint8:
            rgb_frame = (rgb_frame * 255).astype(np.uint8)
        if rgb_frame.ndim == 2:
            rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_GRAY2BGR)

        self.frame_buffer.append(rgb_frame)
        frames = list(self.frame_buffer)

        # Process through pipeline
        try:
            with torch.no_grad():
                embedding, _ = self.pipeline.process_sequence(frames)
                embedding = embedding.cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"Error in pipeline: {e}")
            embedding = np.zeros(self.embed_dim, dtype=np.float32)

        return embedding


# ============================================================================
# Env Unwrapping Utilities
# ============================================================================
# SB3 auto-wraps the env in DummyVecEnv (and Monitor), so model.get_env().unwrapped is the VecEnv shell,
# not CarlaGymEnv. Go through .envs[0] and then the .env chain.

def _unwrap_vec_env(model):
    """
    Return the raw gym.Env (index-0 worker) from an SB3 VecEnv.

    SB3 always wraps user envs in DummyVecEnv / SubprocVecEnv.
    DummyVecEnv exposes its workers via .envs[].
    """
    vec_env = model.get_env()
    if vec_env is None:
        return None
    # DummyVecEnv (most common during single-env training)
    if hasattr(vec_env, "envs"):
        return vec_env.envs[0]
    return vec_env


def _find_carla_and_cbf(model):
    """
    Walk the full wrapper chain and return (CarlaGymEnv, CBFSafetyLayerWrapper).
    Either value is None if not found.

    Wrapper stack:
        DummyVecEnv
          └─ Monitor (added by SB3)
               └─ observation wrapper (CNN image+nav / STT embedding+nav)
                    └─ CBFSafetyLayerWrapper      ← ActionWrapper
                         └─ CarlaGymEnv            ← base env
    """
    base = _unwrap_vec_env(model)
    if base is None:
        return None, None

    carla_env = None
    cbf_wrapper = None
    current = base

    while current is not None:
        # Match CBFSafetyLayerWrapper by isinstance, with duck-type fallback
        if isinstance(current, CBFSafetyLayerWrapper):
            cbf_wrapper = current
        elif hasattr(current, "cbf_layer") and hasattr(current, "correction_count"):
            cbf_wrapper = current

        # Match CarlaGymEnv by isinstance, with duck-type fallback
        if isinstance(current, CarlaGymEnv):
            carla_env = current
        elif hasattr(current, "_collision_count") and hasattr(current, "ego_vehicle"):
            carla_env = current

        current = getattr(current, "env", None)

    return carla_env, cbf_wrapper


def _get_policy_metric_observation(callback) -> Optional[np.ndarray]:
    """
    Return the freshest observation available for live policy metric evaluation.

    For off-policy algorithms this is usually `new_obs` from the callback locals.
    """
    observation = callback.locals.get("new_obs")
    if observation is None:
        observation = getattr(callback.model, "_last_obs", None)
    return observation


def _get_step_info(callback, env_idx: int = 0) -> Dict:
    """Return the info dict for the current VecEnv step."""
    infos = callback.locals.get("infos")
    if infos is None or len(infos) <= env_idx:
        return {}
    info = infos[env_idx]
    return info if isinstance(info, dict) else {}


def _compute_live_policy_metrics(model, observation) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute ensemble uncertainty and trust directly from the current policy.

    This uses the critic ensemble on the actor's deterministic action, which keeps
    the logged trust signal aligned with the live policy used by the safety layer.
    """
    if observation is None:
        return None, None

    try:
        obs_tensor, _ = model.policy.obs_to_tensor(observation)
        with torch.no_grad():
            policy_action = model.actor(obs_tensor, deterministic=True)
            _, uncertainty, trust_score = model.critic.predict(obs_tensor, policy_action)

        return (
            float(uncertainty.mean().detach().cpu().item()),
            float(trust_score.mean().detach().cpu().item()),
        )
    except Exception:
        return None, None


# ============================================================================
# Live Policy Trust Callback
# ============================================================================

class PolicyTrustScoreCallback(BaseCallback):
    """
    Periodically push live ensemble trust metrics from the actor/critic into the CBF wrapper.
    """

    def __init__(self, update_freq: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.update_freq = max(1, int(update_freq))
        self.current_trust = 1.0
        self.current_uncertainty = 0.0

    def _on_step(self) -> bool:
        if self.num_timesteps % self.update_freq != 0:
            return True

        observation = _get_policy_metric_observation(self)
        uncertainty, trust_score = _compute_live_policy_metrics(self.model, observation)
        if uncertainty is None or trust_score is None:
            return True

        self.current_uncertainty = uncertainty
        self.current_trust = trust_score

        carla_env, cbf_wrapper = _find_carla_and_cbf(self.model)
        if cbf_wrapper is not None:
            cbf_wrapper.set_trust_score(trust_score)
            cbf_wrapper.ensemble_uncertainty = uncertainty

        if carla_env is not None:
            carla_env._ensemble_uncertainty = uncertainty
            carla_env._trust_score = trust_score

        self.logger.record("cbf/trust_score", self.current_trust)
        self.logger.record("cbf/ensemble_uncertainty", self.current_uncertainty)

        return True


# ============================================================================
# Environment factory
# ============================================================================

def create_carla_env(
    host: str = "localhost",
    port: int = 2000,
    time_limit: int = 3000,
    render: bool = False,
    num_npc: int = 20,
    num_pedestrians: int = 30,
    show_sensor_data: bool = False,
    curriculum_stage: int = 0,
    seed: Optional[int] = None,
    **cbf_kwargs,
) -> gym.Env:
    """CarlaGymEnv -> CBFSafetyLayerWrapper (observation wrappers are added by the experiments)."""
    env = CarlaGymEnv(
        host=host,
        port=port,
        time_limit=time_limit,
        render_mode="human" if render else None,
        num_npc_vehicles=num_npc,
        num_pedestrians=num_pedestrians,
        show_sensor_data=show_sensor_data,
        curriculum_stage=curriculum_stage,
        seed=seed,
    )
    return CBFSafetyLayerWrapper(env, **cbf_kwargs)
