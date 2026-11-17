"""Traffic rules along the route: traffic lights, stop signs, lane-marking legality, durations.

Distances are measured along the route from the ego FRONT bumper (EGO_HALF_LENGTH ahead of the
ego centre). Stop lines are matched to the route once per (route, tracker) and cached.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from commons.vehicle_model import EGO_HALF_LENGTH

try:
    import carla
except ImportError:  # pragma: no cover
    carla = None

NO_GAP = 100.0
_LC_RIGHT = 1   # carla.LaneChange.Right
_LC_LEFT = 2    # carla.LaneChange.Left
# Leaderboard RunningRedLightTest: the violation is the vehicle's tail point (0.8 * half length
# behind the centre) crossing the stop line.
TAIL_OFFSET_M = 0.8 * EGO_HALF_LENGTH


def _wrap(a: float) -> float:
    return math.pi - ((math.pi - a) % (2.0 * math.pi))


def _state_name(state) -> str:
    """carla.TrafficLightState (or a plain string) -> 'red' | 'yellow' | 'green' | 'none'."""
    name = getattr(state, "name", None) or str(state)
    name = name.split(".")[-1].lower()
    return name if name in ("red", "yellow", "green") else "none"


def _marking_name(t) -> str:
    """carla.LaneMarkingType (or a plain string) -> its name, e.g. 'SolidBroken'."""
    name = getattr(t, "name", None) or str(t)
    return name.split(".")[-1]


@dataclass
class TrafficLightInfo:
    state: str            # 'none' | 'green' | 'yellow' | 'red'
    stop_gap_m: float     # distance along the route from the ego front bumper to the light's stop line; 100.0 if none
    light_id: int | None


def _no_light() -> TrafficLightInfo:
    return TrafficLightInfo("none", NO_GAP, None)


class TrafficLightTracker:
    """Traffic lights whose stop line lies on the route lanes ahead.

    Lights and their stop waypoints (tl.get_stop_waypoints(); trigger-volume fallback walked to
    the junction entry like the leaderboard) are cached at construction, so build one tracker per
    world / map load. A stop waypoint belongs to the route when it projects onto the route with a
    heading within 45 deg and lies on a route lane ((road_id, lane_id) used by the route and
    |l| < lane width) or within half a lane width of the route centre.

    query() reports the nearest light ahead. A light whose line the front bumper has already
    passed is still reported, with a NEGATIVE stop_gap_m, while it is red and the ego tail has
    not cleared the line (so RedLightMonitor can see the crossing); green / yellow lights already
    entered are skipped. Lights in state Off / Unknown are ignored.
    """

    PASS_MARGIN_M = EGO_HALF_LENGTH + TAIL_OFFSET_M + 2.0
    MATCH_HEADING_RAD = math.radians(45.0)

    def __init__(self, world, wmap):
        self._wmap = wmap
        self._lights = []   # (actor, id, [(x, y, yaw_rad, (road_id, lane_id))])
        try:
            actors = list(world.get_actors().filter("*traffic_light*"))
        except Exception:
            actors = []
        for tl in actors:
            try:
                wps = list(tl.get_stop_waypoints())
            except Exception:
                wps = []
            if not wps:
                wps = self._trigger_stop_waypoints(tl, wmap)
            stops = []
            for wp in wps:
                loc = wp.transform.location
                stops.append((float(loc.x), float(loc.y), math.radians(wp.transform.rotation.yaw),
                              (int(wp.road_id), int(wp.lane_id))))
            if stops:
                self._lights.append((tl, int(tl.id), stops))
        self._route = None
        self._stops = []    # sorted [(s_stop, actor, light_id)]

    @staticmethod
    def _trigger_stop_waypoints(tl, wmap):
        """Leaderboard-style stop positions: trigger volume sampled along x, advanced to the junction."""
        try:
            base_tf = tl.get_transform()
            tv = tl.trigger_volume
            area_loc = base_tf.transform(tv.location)
            ext = tv.extent
            yaw = math.radians(base_tf.rotation.yaw)
            out, seen = [], set()
            for x in np.arange(-0.9 * ext.x, 0.9 * ext.x + 1e-6, 1.0):
                pt = carla.Location(x=area_loc.x + x * math.cos(yaw), y=area_loc.y + x * math.sin(yaw), z=area_loc.z)
                wp = wmap.get_waypoint(pt)
                if wp is None:
                    continue
                key = (wp.road_id, wp.lane_id)
                if key in seen:
                    continue
                seen.add(key)
                for _ in range(200):
                    if wp.is_junction:
                        break
                    nxt = wp.next(0.5)
                    if not nxt or nxt[0].is_junction:
                        break
                    wp = nxt[0]
                out.append(wp)
            return out
        except Exception:
            return []

    def _match_route(self, route):
        xy, yaw, meta = [], [], []
        for tl, lid, stops in self._lights:
            for x, y, wyaw, key in stops:
                xy.append((x, y))
                yaw.append(wyaw)
                meta.append((tl, lid, key))
        found = []
        if xy:
            s, l = route.project_points(np.array(xy))
            keys = route.lane_keys
            for k in range(len(xy)):
                lw = route.lane_width_at(float(s[k]))
                if abs(_wrap(yaw[k] - route.yaw_at(float(s[k])))) > self.MATCH_HEADING_RAD:
                    continue
                tl, lid, key = meta[k]
                if (key in keys and abs(l[k]) < lw) or abs(l[k]) < 0.5 * lw:
                    found.append((float(s[k]), abs(float(l[k])), tl, lid))
        # one entry per light and route pass: merge entries of the same light within 10 m
        found.sort(key=lambda e: e[0])
        merged = []
        for e in found:
            dup = next((i for i, m in enumerate(merged) if m[3] == e[3] and abs(m[0] - e[0]) < 10.0), None)
            if dup is None:
                merged.append(e)
            elif e[1] < merged[dup][1]:
                merged[dup] = e
        self._stops = sorted([(m[0], m[2], m[3]) for m in merged], key=lambda e: e[0])
        self._route = route

    def route_stop_lines(self, route):
        """[(s_stop, light_id)] matched on `route` (for debugging / tests)."""
        if route is not self._route:
            self._match_route(route)
        return [(s, lid) for s, _, lid in self._stops]

    def query(self, ego, route, proj, max_dist=50.0) -> TrafficLightInfo:
        if route is not self._route:
            self._match_route(route)
        front = float(proj.s) + EGO_HALF_LENGTH
        for s_stop, tl, lid in self._stops:
            gap = s_stop - front
            if gap > max_dist:
                break
            if gap < -self.PASS_MARGIN_M:
                continue
            try:
                state = _state_name(tl.get_state())
            except Exception:
                state = _state_name(getattr(tl, "state", "none"))
            if state == "none":
                continue
            if gap < 0.0 and state != "red":
                continue
            return TrafficLightInfo(state, float(gap), lid)
        return _no_light()


class RedLightMonitor:
    """Detects running a red light, leaderboard style but without punishing a legal yellow entry.

    A violation is reported on the step the ego's tail point (0.8 * EGO_HALF_LENGTH behind the
    centre) crosses the stop line while the light is red, provided the light was also red when
    the front bumper crossed the line. Each light is reported at most once per episode.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._prev = None          # (light_id, front gap, route s) of the previous step
        self._front_state = {}     # light_id -> light state when the front bumper crossed its line
        self._flagged = set()

    def update(self, tl: TrafficLightInfo, proj, v) -> bool:
        """Needs the tracker's report every step. TrafficLightTracker keeps reporting a red light
        until well after the tail crossing (PASS_MARGIN_M), and a light that is not red when it
        drops out of the report cannot produce a violation, so consecutive reports suffice."""
        s = float(proj.s)
        cur = None
        if tl.light_id is not None and tl.state != "none" and tl.stop_gap_m < NO_GAP:
            cur = (tl.light_id, float(tl.stop_gap_m), s)
        violated = False
        if cur is not None and self._prev is not None and self._prev[0] == cur[0]:
            lid, gap = cur[0], cur[1]
            _, pgap, ps = self._prev
            if -1.0 <= s - ps < 10.0:          # ignore relocalisation jumps
                if pgap > 0.0 >= gap:
                    self._front_state[lid] = tl.state
                ptail = pgap + EGO_HALF_LENGTH + TAIL_OFFSET_M
                tail = gap + EGO_HALF_LENGTH + TAIL_OFFSET_M
                if (ptail > 0.0 >= tail and tl.state == "red" and self._front_state.get(lid) == "red"
                        and lid not in self._flagged):
                    self._flagged.add(lid)
                    violated = True
        self._prev = cur
        return violated


class StopSignMonitor:
    """Stop signs on the route and the leaderboard RunningStopTest rule.

    Stop-sign actors (world.get_actors().filter('traffic.stop')) are cached at construction. A sign
    applies to the route when route points (sampled every 0.5 m, outside junctions) fall inside its
    trigger volume and the route enters a junction within 15 m after leaving the volume; the stop
    line is placed at that junction entry. The stop counts as made when the speed drops below
    0.1 m/s while the front bumper is within 4 m before the line (up to EGO_HALF_LENGTH past it).
    A violation is reported when the ego centre crosses the line without having stopped.
    Completed stops are no longer reported, so the next stop line (or 100) is returned.
    """

    STOP_SPEED = 0.1
    ZONE_M = 4.0
    JUNCTION_SEARCH_M = 15.0

    def __init__(self, world, wmap):
        self._wmap = wmap
        self._signs = []    # (id, cx, cy, cz, ex, ey, ez, yaw_rad)
        try:
            actors = list(world.get_actors().filter("traffic.stop"))
        except Exception:
            actors = []
        for a in actors:
            try:
                tf = a.get_transform()
                tv = a.trigger_volume
                c = tf.transform(tv.location)
                tv_yaw = tv.rotation.yaw if hasattr(tv, "rotation") else 0.0
                yaw = math.radians(tf.rotation.yaw + tv_yaw)
                e = tv.extent
                self._signs.append((int(a.id), c.x, c.y, c.z, e.x, e.y, e.z, yaw))
            except Exception:
                continue
        self._route = None
        self._stops = []    # sorted [(s_stop, sign_id)]
        self._state = []    # per stop: dict(seen, completed, done)

    def _match_route(self, route):
        stops = []
        if self._signs and route.length_m > 0.0:
            s_samp = np.arange(0.0, route.length_m + 1e-6, 0.5)
            xy = route.points_at(s_samp)
            z = route.z_at(s_samp)
            vs, vj = route.vertex_s, route.vertex_is_junction
            near = np.clip(np.searchsorted(vs, s_samp), 0, len(vs) - 1)
            junc_samp = vj[near] | vj[np.clip(near - 1, 0, len(vs) - 1)]
            for sid, cx, cy, cz, ex, ey, ez, yaw in self._signs:
                dx, dy = xy[:, 0] - cx, xy[:, 1] - cy
                lx = dx * math.cos(yaw) + dy * math.sin(yaw)
                ly = -dx * math.sin(yaw) + dy * math.cos(yaw)
                inside = ((np.abs(lx) <= ex + 0.5) & (np.abs(ly) <= ey + 0.5)
                          & (np.abs(z - cz) <= ez + 3.0) & ~junc_samp)
                idx = np.nonzero(inside)[0]
                if idx.size == 0:
                    continue
                runs = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)
                for run in runs:
                    s_end = s_samp[run[-1]]
                    j = int(np.searchsorted(vs, s_end, side="right"))
                    while j < len(vs) and not vj[j] and vs[j] - s_end <= self.JUNCTION_SEARCH_M:
                        j += 1
                    if j >= len(vs) or not vj[j] or vs[j] - s_end > self.JUNCTION_SEARCH_M:
                        continue    # not heading into a junction: not our stop
                    s_stop = max(s_end, 0.5 * (vs[j - 1] + vs[j])) if j > 0 else vs[j]
                    stops.append((float(s_stop), sid))
        stops.sort(key=lambda e: e[0])
        dedup = []
        for e in stops:
            if not dedup or e[0] - dedup[-1][0] > 3.0:
                dedup.append(e)
        self._stops = dedup
        self._state = [dict(seen=False, completed=False, done=False) for _ in dedup]
        self._route = route

    def route_stop_lines(self, route):
        """[(s_stop, sign_id)] matched on `route` (for debugging / tests)."""
        if route is not self._route:
            self._match_route(route)
        return list(self._stops)

    def reset(self):
        self._route = None
        self._stops = []
        self._state = []

    def update(self, ego, route, proj, v) -> tuple:
        if route is not self._route:
            self._match_route(route)
        front = float(proj.s) + EGO_HALF_LENGTH
        v = float(v)
        violated = False
        gap_out = NO_GAP
        for (s_stop, _sid), st in zip(self._stops, self._state):
            if st["done"]:
                continue
            gap = s_stop - front
            if gap > NO_GAP:
                break
            if gap >= 0.0:
                st["seen"] = True
            if -EGO_HALF_LENGTH <= gap <= self.ZONE_M and v < self.STOP_SPEED:
                st["completed"] = st["done"] = True
                continue
            if gap < -EGO_HALF_LENGTH:
                st["done"] = True
                if st["seen"] and not st["completed"]:
                    violated = True
                continue
            if gap_out >= NO_GAP:
                gap_out = max(0.0, gap)
        return gap_out, violated


def _crossing_side(before, after) -> str:
    """'left' or 'right': which marking was crossed, inferred from the lane offsets.

    The lane-invasion event does not say which side. If the ego centre changed lanes between the
    two statuses the offset flips sign (left lane change: before < 0, after > 0); otherwise the
    side the centre is on in its lane tells which marking a corner touched.
    """
    if before is not None and after is not None:
        b, a = float(before.e_lane), float(after.e_lane)
        if b * a < 0.0 and abs(a - b) > 1.0:
            return "left" if b < 0.0 else "right"
        return "left" if a < 0.0 else "right"
    ref = after if after is not None else before
    return "left" if ref is not None and float(ref.e_lane) < 0.0 else "right"


def lane_invasion_is_illegal(crossed_markings, lane_status_before, lane_status_after) -> bool:
    """True if a lane-invasion event broke a rule.

    Illegal: the ego (after the event) is in an opposite-direction lane; any crossed marking of
    type Solid, SolidSolid, Curb or Grass; SolidBroken / BrokenSolid crossed from the solid side,
    i.e. towards a side the lane's lane_change permission does not allow (permission relative to
    the travel direction, from LaneStatus.lane_change of the lane the ego came from). Broken,
    BrokenBroken, BottsDots, Other and NONE markings are legal to cross.
    """
    if lane_status_after is not None and lane_status_after.in_opposite_lane:
        return True
    markings = list(crossed_markings or [])
    if not markings:
        return False
    side = None
    for m in markings:
        name = _marking_name(getattr(m, "type", m))
        if name in ("Solid", "SolidSolid", "Curb", "Grass"):
            return True
        if name in ("SolidBroken", "BrokenSolid"):
            if side is None:
                side = _crossing_side(lane_status_before, lane_status_after)
            ref = lane_status_before if lane_status_before is not None else lane_status_after
            perm = getattr(ref, "lane_change", None) if ref is not None else None
            if perm is None:
                perm = int(getattr(m, "lane_change", 3))
            perm = int(perm)
            if not perm & (_LC_LEFT if side == "left" else _LC_RIGHT):
                return True
    return False


class DurationMonitor:
    """Generic "condition true for N consecutive steps"."""

    def __init__(self, steps: int):
        self.steps = int(steps)
        self.count = 0

    def update(self, condition: bool) -> bool:
        self.count = self.count + 1 if condition else 0
        return self.count >= self.steps

    def reset(self):
        self.count = 0
