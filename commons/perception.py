"""Obstacle perception in the route frame: lead actor, LiDAR static obstacles, time to collision.

Everything is measured along the (curved) route corridor rather than straight ahead of the ego,
so at junction turns the corridor follows the turn instead of running into roadside buildings.
Gaps are bumper-to-bumper: from the ego front bumper (EGO_HALF_LENGTH ahead of the ego centre,
measured along the route) to the nearest point of the obstacle inside the corridor.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from commons.vehicle_model import EGO_HALF_LENGTH, EGO_HALF_WIDTH

NO_GAP = 100.0                 # reported gap when nothing is in the corridor
LEAD_CORRIDOR_MARGIN = 0.4     # actor corridor half width = EGO_HALF_WIDTH + this
LIDAR_CORRIDOR_MARGIN = 0.3    # LiDAR corridor half width = EGO_HALF_WIDTH + this
WALKER_PREDICT_S = 2.0         # walkers are swept along their lateral (route-normal) motion for this long
COMBINE_TOL_M = 1.0            # a LiDAR return must be this much nearer than the actor to override it
EGO_BOX_MARGIN = 0.2           # LiDAR returns inside the ego box (+ margin, sensor frame) are dropped
LIDAR_CEILING_Z = 1.3          # sensor frame: drop returns higher than this (3.0 m above ground for a 1.7 m mount)


@dataclass
class LeadInfo:
    gap_m: float          # bumper-to-bumper distance along the route corridor; 100.0 if none
    v_lead_mps: float     # lead speed projected on the route tangent (signed; 0 for statics)
    kind: str             # 'vehicle' | 'walker' | 'static' | 'none'
    actor_id: int | None


def _no_lead() -> LeadInfo:
    return LeadInfo(NO_GAP, 0.0, "none", None)


def _convex_hull(pts: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain; pts (N, 2) -> hull vertices in order (degenerate inputs allowed)."""
    p = np.unique(np.round(pts, 6), axis=0)
    if len(p) <= 2:
        return p
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for q in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    for q in p[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return np.array(lower[:-1] + upper[:-1])


def _strip_s_range(hull: np.ndarray, w: float):
    """(min s, max s) of the polygon `hull` (columns s, l) intersected with the strip |l| <= w, or None."""
    cands = []
    n = len(hull)
    for k in range(n):
        s0, l0 = hull[k]
        s1, l1 = hull[(k + 1) % n]
        if abs(l0) <= w:
            cands.append(s0)
        for c in (-w, w):
            if (l0 - c) * (l1 - c) < 0.0:
                cands.append(s0 + (c - l0) / (l1 - l0) * (s1 - s0))
    if not cands:
        return None
    return min(cands), max(cands)


def find_lead(actors, ego, route, proj, horizon_m=50.0, corridor_half_width=None) -> LeadInfo:
    """Nearest vehicle / pedestrian in the route corridor ahead of the ego.

    actors: a world.get_actors() snapshot (only vehicle.* and walker.pedestrian.* are used; the
    ego is skipped by id). Each actor's bounding box (bounding_box.get_world_vertices) is projected
    into the route (s, l) frame and intersected with the corridor |l| < corridor_half_width
    (default EGO_HALF_WIDTH + 0.4). Walkers are additionally swept along their velocity component
    normal to the route for 2 s (a pedestrian stepping towards the lane is caught early; one
    walking away is not inflated). An actor counts if part of it is ahead of the ego centre and
    its nearest corridor point is less than horizon_m ahead of the ego centre.
    gap = nearest corridor point s - (s_ego + EGO_HALF_LENGTH), clipped at 0.
    """
    w = (EGO_HALF_WIDTH + LEAD_CORRIDOR_MARGIN) if corridor_half_width is None else float(corridor_half_width)
    ego_id = getattr(ego, "id", None)
    ego_loc = ego.get_transform().location
    s_ego = float(proj.s)
    front = s_ego + EGO_HALF_LENGTH
    r2 = (horizon_m + EGO_HALF_LENGTH + 10.0) ** 2

    s_from, s_to = s_ego - EGO_HALF_LENGTH - 5.0, front + horizon_m + 5.0
    near = []
    for a in actors:
        tid = getattr(a, "type_id", "") or ""
        if tid.startswith("vehicle."):
            kind = "vehicle"
        elif tid.startswith("walker.pedestrian"):
            kind = "walker"
        else:
            continue
        if ego_id is not None and a.id == ego_id:
            continue
        tf = a.get_transform()
        loc = tf.location
        if (loc.x - ego_loc.x) ** 2 + (loc.y - ego_loc.y) ** 2 > r2:
            continue
        near.append((a, kind, tf))
    if not near:
        return _no_lead()
    # cheap pre-filter on the actor centres (boxes are < 7 m half length, walker sweep < 6 m)
    _, l_c = route.project_points(np.array([[tf.location.x, tf.location.y] for _, _, tf in near]), s_from, s_to)
    cands = []
    for (a, kind, tf), lc in zip(near, l_c):
        if abs(lc) > w + 12.0:
            continue
        verts = a.bounding_box.get_world_vertices(tf)
        cands.append((a, kind, np.array([[v.x, v.y] for v in verts], dtype=float)))
    if not cands:
        return _no_lead()

    all_xy = np.concatenate([c[2] for c in cands], axis=0)
    s_all, l_all = route.project_points(all_xy, s_from, s_to)

    best = _no_lead()
    start = 0
    for a, kind, xy in cands:
        n = len(xy)
        s_i, l_i = s_all[start:start + n], l_all[start:start + n]
        start += n
        if np.min(np.abs(l_i)) > w + 15.0:   # far off the corridor (also: beyond the searched route part)
            continue
        vel = a.get_velocity()
        yaw_r = route.yaw_at(float(np.mean(s_i)))
        c, sn = math.cos(yaw_r), math.sin(yaw_r)
        v_t = vel.x * c + vel.y * sn
        pts = np.stack([s_i, l_i], axis=1)
        if kind == "walker":
            v_n = -vel.x * sn + vel.y * c
            pts = np.vstack([pts, pts + np.array([0.0, v_n * WALKER_PREDICT_S])])
        rng = _strip_s_range(_convex_hull(pts), w)
        if rng is None:
            continue
        s_lo, s_hi = rng
        if s_hi <= s_ego or s_lo - s_ego >= horizon_m:
            continue
        gap = max(0.0, s_lo - front)
        if gap < best.gap_m:
            best = LeadInfo(float(gap), float(v_t), kind, int(a.id))
    return best


def lidar_static_gap(points_sensor, sensor_transform, route, proj, horizon_m=40.0,
                     ground_z_cut=-1.3, ceiling_z=LIDAR_CEILING_Z) -> float:
    """Gap [m] from the ego front bumper to the nearest LiDAR return inside the curved route corridor.

    points_sensor: (N, 3) or (N, 4) LiDAR points in the sensor frame (x forward, y right, z up;
    (3, N) / (4, N) is accepted too). Returns with z <= ground_z_cut or z >= ceiling_z (sensor
    frame), inside the ego box (sensor assumed above the ego centre) or out of range are dropped;
    the rest are transformed to world with sensor_transform, projected onto the local route part
    ahead of the ego and kept if |l| < EGO_HALF_WIDTH + 0.3. A second, road-relative ground test
    (height above the route's road surface at the point, compared with the sensor's own height
    above the road) removes ground returns on grade changes. 100.0 if nothing remains.
    """
    P = np.asarray(points_sensor)
    if P.ndim != 2 or P.size == 0:
        return NO_GAP
    if P.shape[1] not in (3, 4) and P.shape[0] in (3, 4):
        P = P.T
    P = P[:, :3].astype(float, copy=False)
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    r_max = horizon_m + EGO_HALF_LENGTH + 2.0
    in_ego_box = (np.abs(x) < EGO_HALF_LENGTH + EGO_BOX_MARGIN) & (np.abs(y) < EGO_HALF_WIDTH + EGO_BOX_MARGIN)
    keep = (z > ground_z_cut) & (z < ceiling_z) & ~in_ego_box & (x * x + y * y < r_max * r_max)
    if not np.any(keep):
        return NO_GAP
    P = P[keep]

    M = np.asarray(sensor_transform.get_matrix(), dtype=float)
    W = P @ M[:3, :3].T + M[:3, 3]
    s_ego = float(proj.s)
    front = s_ego + EGO_HALF_LENGTH
    s, l = route.project_points(W[:, :2], s_ego - 1.0, front + horizon_m + 2.0)
    sel = (np.abs(l) < EGO_HALF_WIDTH + LIDAR_CORRIDOR_MARGIN) & (s > s_ego) & (s - front < horizon_m)
    if not np.any(sel):
        return NO_GAP
    s, Wz = s[sel], W[sel, 2]
    sensor_h = float(sensor_transform.location.z) - route.z_at(s_ego)
    above_road = Wz - route.z_at(s)
    s = s[above_road > sensor_h + ground_z_cut]
    if s.size == 0:
        return NO_GAP
    return float(max(0.0, float(np.min(s)) - front))


def combine(lead_actor: LeadInfo, static_gap: float) -> LeadInfo:
    """Nearest of the actor lead and the LiDAR gap.

    The LiDAR also sees the lead actor itself (usually at the same or a slightly larger gap than
    its bounding box), so the static return only wins when it is more than COMBINE_TOL_M nearer;
    this keeps the actor's velocity for car following.
    """
    static_gap = float(static_gap)
    if static_gap < NO_GAP and static_gap < lead_actor.gap_m - COMBINE_TOL_M:
        return LeadInfo(static_gap, 0.0, "static", None)
    return lead_actor


def ttc(gap_m, v_ego, v_lead) -> float:
    """Time to collision [s] = gap / closing speed; inf if not closing."""
    closing = float(v_ego) - float(v_lead)
    if closing <= 1e-3:
        return float("inf")
    return max(0.0, float(gap_m)) / closing
