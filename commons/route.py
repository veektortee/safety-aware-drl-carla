"""Global route for one episode: planning, Frenet-style projection and lane queries.

Conventions (CARLA): x forward, y to the RIGHT, z up, yaw clockwise. In world 2-D coordinates a
heading psi has forward vector (cos psi, sin psi) and right vector (-sin psi, cos psi).

- `e_y`   > 0: ego is right of the route centre line
- `e_psi` > 0: ego is yawed clockwise (to the right) of the route tangent
- `kappa` > 0: the route turns right (heading increases with arc length)

The core geometry (`project`, `point_at`, `points_ahead_ego`, `next_maneuver`, `project_points`)
only reads `waypoint.transform` (location + rotation), `waypoint.lane_width` and
`waypoint.is_junction`, so it can be unit-tested with fake waypoints. `lane_bounds` and
`lane_status` additionally use the carla Map / Waypoint lane API.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from commons.navigation.road_option import RoadOption

try:  # carla is only needed for the lane queries (enum values); geometry works without it.
    import carla
except ImportError:  # pragma: no cover
    carla = None

# Waypoints closer than this to the previous kept one are dropped (GRP repeats edge end points).
DEDUP_MIN_M = 0.5
# project(): local search window around the hint segment.
BACK_WINDOW_M = 8.0      # the projection never moves more than this (+ one segment) backwards
FWD_WINDOW_M = 30.0
HEADING_COST_M = 1.0     # cost added for heading mismatch: HEADING_COST_M * (1 - cos(dpsi))
GLOBAL_TIE_M = 0.5       # global search: earliest pass whose cost is within this of the best
GLOBAL_CLUSTER_M = 10.0
# A waypoint yaw that disagrees with the polyline by more than this is replaced by the polyline heading.
TANGENT_FALLBACK_RAD = math.radians(45.0)
LANE_BOUND_MARGIN_M = 0.2
MAX_KINK_DEG = 50.0      # Route.plan rejects routes with a sharper polyline kink (GRP artefact)
_MANEUVERS = (int(RoadOption.LEFT), int(RoadOption.RIGHT), int(RoadOption.STRAIGHT))
# carla.LaneChange bit values (Right = 1, Left = 2, Both = 3)
_LC_RIGHT = 1
_LC_LEFT = 2


def wrap_angle(a):
    """Wrap an angle [rad] (scalar or array) to (-pi, pi]."""
    if np.ndim(a):
        return math.pi - np.mod(math.pi - np.asarray(a, dtype=float), 2.0 * math.pi)
    return math.pi - ((math.pi - float(a)) % (2.0 * math.pi))


def _wrap_deg(a: float) -> float:
    return math.degrees(wrap_angle(math.radians(a)))


def _as_location(obj):
    """Accept a carla.Location, a carla.Transform (spawn point) or a Waypoint."""
    if hasattr(obj, "location"):
        return obj.location
    if hasattr(obj, "transform"):
        return obj.transform.location
    return obj


def _to_option(o) -> RoadOption:
    try:
        return RoadOption(int(o))
    except (TypeError, ValueError):
        return RoadOption.VOID


@dataclass
class RouteProjection:
    s: float            # arc length of the ego's projection on the route [m]
    e_y: float          # signed lateral error [m], + = right of route centre line
    e_psi: float        # heading error [rad] in (-pi, pi], + = clockwise of route tangent
    kappa: float        # route curvature at s [1/m], + = route turning right
    idx: int            # index of the route segment (use as hint next step)
    lane_width: float   # route lane width at s [m]
    remaining_m: float  # length - s


@dataclass
class LaneStatus:
    on_drivable: bool          # ego centre is on a Driving (or Parking) lane (project_to_road=False)
    in_route_lane: bool
    in_same_dir_lane: bool     # a Driving lane whose yaw is within 90 deg of the ego yaw
    in_opposite_lane: bool     # a Driving lane whose yaw differs by > 120 deg, outside junctions
    e_lane: float              # signed offset from the centre of the lane the ego is in [m]
    lane_heading_err: float    # rad, ego yaw - that lane's yaw
    is_junction: bool
    # Extra (not in the original contract, defaulted so positional construction still works):
    # carla.LaneChange permission of the lane the ego is in, relative to its travel direction
    # (1 = Right, 2 = Left, 3 = Both, 0 = none). Used by traffic_rules.lane_invasion_is_illegal.
    lane_change: int = 3


class Route:
    """Polyline route through CARLA waypoints (~2 m apart) with per-waypoint RoadOptions."""

    def __init__(self, waypoints, options):
        wps, opts = self._clean(list(waypoints), list(options))
        if not wps:
            raise ValueError("Route needs at least one waypoint")
        self.waypoints = wps
        self.options = opts

        xy = np.array([[w.transform.location.x, w.transform.location.y] for w in wps], dtype=float)
        z = np.array([w.transform.location.z for w in wps], dtype=float)
        yaw = np.radians([w.transform.rotation.yaw for w in wps]).astype(float)
        lw = np.array([float(getattr(w, "lane_width", 3.5) or 3.5) for w in wps], dtype=float)
        junc = np.array([bool(getattr(w, "is_junction", False)) for w in wps], dtype=bool)
        if len(wps) == 1:  # synthesise a second vertex 2 m ahead so the geometry is defined
            xy = np.vstack([xy, xy[0] + 2.0 * np.array([math.cos(yaw[0]), math.sin(yaw[0])])])
            z, yaw, lw, junc = (np.append(a, a[0]) for a in (z, yaw, lw, junc))
            self.waypoints = wps + [wps[0]]
            self.options = opts + [opts[0]]

        self._xy, self._z, self._lw, self._junc = xy, z, lw, junc
        seg = xy[1:] - xy[:-1]
        seg_len = np.hypot(seg[:, 0], seg[:, 1])
        seg_len = np.maximum(seg_len, 1e-6)
        self._seg = seg
        self._len = seg_len
        self._len2 = seg_len ** 2
        self._u = seg / seg_len[:, None]
        self._r = np.stack([-self._u[:, 1], self._u[:, 0]], axis=1)   # right normal
        self._seg_yaw = np.arctan2(seg[:, 1], seg[:, 0])
        self._s = np.concatenate([[0.0], np.cumsum(seg_len)])
        self._nseg = len(seg_len)
        self.length_m = float(self._s[-1])
        # Largest heading change between consecutive segments [deg]. GRP lane-change artefacts
        # (sideways jumps onto another road, zig-zags inside junctions) show up as kinks > 50 deg.
        self.max_kink_deg = float(np.degrees(np.max(np.abs(wrap_angle(self._seg_yaw[1:] - self._seg_yaw[:-1]))))) \
            if self._nseg > 1 else 0.0

        # Tangent yaw per vertex: the waypoint's lane yaw unless it disagrees with the polyline.
        n = len(xy)
        poly_yaw = np.empty(n)
        poly_yaw[0] = self._seg_yaw[0]
        poly_yaw[-1] = self._seg_yaw[-1]
        if n > 2:
            poly_yaw[1:-1] = self._seg_yaw[:-1] + 0.5 * wrap_angle(self._seg_yaw[1:] - self._seg_yaw[:-1])
        bad = np.abs(wrap_angle(yaw - poly_yaw)) > TANGENT_FALLBACK_RAD
        self._yaw = np.where(bad, poly_yaw, yaw)

        # Curvature per vertex from the tangent yaws (central differences), lightly smoothed.
        kappa = np.zeros(n)
        if n >= 3:
            kappa[1:-1] = wrap_angle(self._yaw[2:] - self._yaw[:-2]) / np.maximum(self._s[2:] - self._s[:-2], 1e-3)
        if n >= 2:
            kappa[0] = wrap_angle(self._yaw[1] - self._yaw[0]) / max(self._s[1] - self._s[0], 1e-3)
            kappa[-1] = wrap_angle(self._yaw[-1] - self._yaw[-2]) / max(self._s[-1] - self._s[-2], 1e-3)
        if n >= 3:
            padded = np.concatenate([[kappa[0]], kappa, [kappa[-1]]])
            kappa = 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]
        self._kappa = kappa

        self._is_maneuver = np.array([int(o) in _MANEUVERS for o in self.options], dtype=bool)
        self._lane_keys = set()
        for w in self.waypoints:
            key = self._lane_key(w)
            if key is not None:
                self._lane_keys.add(key)
        self._bounds_cache = {}
        self._last_proj = None

    # ------------------------------------------------------------------ read-only views
    @property
    def vertex_xy(self) -> np.ndarray:
        """(N, 2) world x, y of the route vertices."""
        return self._xy

    @property
    def vertex_s(self) -> np.ndarray:
        """(N,) arc length of each vertex [m]."""
        return self._s

    @property
    def vertex_is_junction(self) -> np.ndarray:
        """(N,) waypoint.is_junction per vertex."""
        return self._junc

    @property
    def lane_keys(self) -> set:
        """{(road_id, lane_id)} of the lanes the route uses."""
        return self._lane_keys

    # ------------------------------------------------------------------ construction
    @staticmethod
    def _lane_key(wp):
        try:
            return int(wp.road_id), int(wp.lane_id)
        except (AttributeError, TypeError, ValueError):
            return None

    @staticmethod
    def _clean(wps, opts):
        """Drop near-duplicate and backward-pointing waypoints (keeps options aligned)."""
        if len(opts) < len(wps):
            opts = opts + [RoadOption.LANEFOLLOW] * (len(wps) - len(opts))
        out_w, out_o = [], []
        for w, o in zip(wps, opts):
            if w is None:
                continue
            if out_w:
                p = out_w[-1].transform.location
                q = w.transform.location
                dx, dy = q.x - p.x, q.y - p.y
                if math.hypot(dx, dy) < DEDUP_MIN_M:
                    continue
                # behind both the previous point's and its own heading -> a backward glitch
                yp = math.radians(out_w[-1].transform.rotation.yaw)
                yq = math.radians(w.transform.rotation.yaw)
                if (dx * math.cos(yp) + dy * math.sin(yp) < 0.0) and (dx * math.cos(yq) + dy * math.sin(yq) < 0.0):
                    continue
            out_w.append(w)
            out_o.append(_to_option(o))
        return out_w, out_o

    @classmethod
    def from_waypoints(cls, waypoints, options) -> "Route":
        return cls(waypoints, options)

    @classmethod
    def from_trace(cls, trace) -> "Route":
        """From GlobalRoutePlanner.trace_route output: list of (waypoint, RoadOption)."""
        return cls([w for w, _ in trace], [o for _, o in trace])

    @classmethod
    def plan(cls, grp, wmap, start, spawn_points, min_len_m: float, max_len_m: float, rng,
             max_tries: int = 30) -> "Route":
        """Route from `start` to a random spawn point; first one with length in [min, max] wins.

        Routes with GRP lane-change glitches (a kink > MAX_KINK_DEG between consecutive segments)
        are not accepted. Destinations are drawn from the spawn points whose straight-line distance
        can give a route in range (all of them if none can). Falls back to the traced route closest
        to the range (glitch-free ones first) if it is at least half the minimum length, else to a
        plain lane-follow route from the start (random branch at junctions). Never returns an empty
        or degenerate route (on big maps a failed trace once gave a 2 m "route" = a free goal).
        """
        start_loc = _as_location(start)
        cands = list(spawn_points or [])
        in_reach = []
        for p in cands:
            loc = _as_location(p)
            if 0.5 * min_len_m <= math.hypot(loc.x - start_loc.x, loc.y - start_loc.y) <= max_len_m:
                in_reach.append(p)
        cands = in_reach or cands
        best, best_err = None, float("inf")
        for _ in range(max_tries):
            if not cands:
                break
            dest = _as_location(rng.choice(cands))
            d = math.hypot(dest.x - start_loc.x, dest.y - start_loc.y)
            if d < 5.0:
                continue
            if d > max_len_m and best is not None:   # route length >= straight-line distance
                continue
            try:
                trace = grp.trace_route(start_loc, dest)
                if len(trace) < 2:
                    continue
                route = cls.from_trace(trace)
            except Exception:   # no path / start not localisable
                continue
            length = route.length_m
            clean = route.max_kink_deg <= MAX_KINK_DEG
            if clean and min_len_m <= length <= max_len_m:
                return route
            err = max(0.0, min_len_m - length, length - max_len_m) + (0.0 if clean else 1e4)
            if err < best_err:
                best, best_err = route, err
        if best is not None and best.length_m >= 0.5 * min_len_m:
            return best
        return cls._lane_follow_route(wmap, start_loc, min_len_m, max_len_m, rng)

    @classmethod
    def _lane_follow_route(cls, wmap, start_loc, min_len_m, max_len_m, rng, step=2.0):
        wp = wmap.get_waypoint(start_loc)
        wps, opts = [wp], [RoadOption.LANEFOLLOW]
        target = 0.5 * (min_len_m + max_len_m)
        total = 0.0
        while total < target and len(wps) < 5000:
            nxt = wp.next(step)
            if not nxt:
                break
            wp = nxt[0] if len(nxt) == 1 else rng.choice(list(nxt))
            total += step
            wps.append(wp)
            opts.append(RoadOption.LANEFOLLOW)
        return cls(wps, opts)

    # ------------------------------------------------------------------ geometry
    def _seg_range(self, s_from, s_to):
        """Segment index range [lo, hi) overlapping the arc-length interval [s_from, s_to]."""
        lo = 0 if s_from is None else int(np.searchsorted(self._s, s_from, side="right")) - 1
        hi = self._nseg if s_to is None else int(np.searchsorted(self._s, s_to, side="left")) + 1
        lo = min(max(lo, 0), self._nseg - 1)
        hi = min(max(hi, lo + 1), self._nseg)
        return lo, hi

    def _interp_vertex(self, arr, i, t):
        return float(arr[i] + (arr[i + 1] - arr[i]) * t)

    def project(self, location, yaw_deg: float, hint_idx=None) -> RouteProjection:
        """Project the ego onto the route.

        With `hint_idx` (the previous projection's idx) only a window from 8 m behind to 30 m
        ahead of that segment is searched and the result never lies more than 8 m behind the
        hint segment's start, so it cannot jump back to an overlapping earlier part of the route.
        Without a hint the whole route is searched, preferring the earliest matching pass.
        """
        px, py = float(location.x), float(location.y)
        yaw = math.radians(float(yaw_deg))
        if hint_idx is None:
            lo, hi = 0, self._nseg
            s_min = None
        else:
            h = int(min(max(int(hint_idx), 0), self._nseg - 1))
            s_min = self._s[h] - BACK_WINDOW_M
            lo, hi = self._seg_range(s_min, self._s[h] + FWD_WINDOW_M)
            hi = max(hi, min(h + 1, self._nseg))

        A = self._xy[lo:hi]
        D = self._seg[lo:hi]
        t = ((px - A[:, 0]) * D[:, 0] + (py - A[:, 1]) * D[:, 1]) / self._len2[lo:hi]
        t = np.clip(t, 0.0, 1.0)
        if s_min is not None:
            t_min = np.clip((s_min - self._s[lo:hi]) / self._len[lo:hi], 0.0, 1.0)
            t = np.maximum(t, t_min)
        fx = A[:, 0] + t * D[:, 0]
        fy = A[:, 1] + t * D[:, 1]
        dist = np.hypot(px - fx, py - fy)
        cost = dist + HEADING_COST_M * (1.0 - np.cos(yaw - self._seg_yaw[lo:hi]))

        if hint_idx is None:
            best = float(cost.min())
            first = int(np.argmax(cost <= best + GLOBAL_TIE_M))
            s_first = self._s[lo + first]
            cluster = np.nonzero(self._s[lo:hi][first:] <= s_first + GLOBAL_CLUSTER_M)[0] + first
            k = int(cluster[np.argmin(cost[cluster])])
        else:
            k = int(np.argmin(cost))
        i = lo + k
        tk = float(t[k])

        s = float(self._s[i] + tk * self._len[i])
        e_y = float((px - fx[k]) * self._r[i, 0] + (py - fy[k]) * self._r[i, 1])
        route_yaw = float(self._yaw[i] + wrap_angle(self._yaw[i + 1] - self._yaw[i]) * tk)
        e_psi = float(wrap_angle(yaw - route_yaw))
        proj = RouteProjection(
            s=s,
            e_y=e_y,
            e_psi=e_psi,
            kappa=self._interp_vertex(self._kappa, i, tk),
            idx=i,
            lane_width=self._interp_vertex(self._lw, i, tk),
            remaining_m=float(self.length_m - s),
        )
        self._last_proj = proj
        return proj

    def points_at(self, s) -> np.ndarray:
        """Vectorised point_at: (M, 2) world (x, y) at arc lengths s."""
        return self._points_at(s)

    def _points_at(self, s):
        """Vectorised world (x, y) at arc lengths s (extrapolated along the end segments)."""
        s = np.atleast_1d(np.asarray(s, dtype=float))
        i = np.clip(np.searchsorted(self._s, s, side="right") - 1, 0, self._nseg - 1)
        t = (s - self._s[i]) / self._len[i]
        # inside the route t is in [0, 1]; before the start / after the end it extrapolates
        return self._xy[i] + t[:, None] * self._seg[i]

    def point_at(self, s: float) -> tuple:
        """World (x, y) at arc length s; beyond the ends the end segments are extrapolated."""
        p = self._points_at(s)[0]
        return float(p[0]), float(p[1])

    def yaw_at(self, s):
        """Route tangent yaw [rad] at arc length(s) s (scalar or array)."""
        s_arr = np.clip(np.atleast_1d(np.asarray(s, dtype=float)), 0.0, self.length_m)
        i = np.clip(np.searchsorted(self._s, s_arr, side="right") - 1, 0, self._nseg - 1)
        t = np.clip((s_arr - self._s[i]) / self._len[i], 0.0, 1.0)
        out = wrap_angle(self._yaw[i] + wrap_angle(self._yaw[i + 1] - self._yaw[i]) * t)
        return float(out[0]) if np.ndim(s) == 0 else out

    def z_at(self, s):
        """Road-surface z of the route at arc length(s) s."""
        out = np.interp(np.asarray(s, dtype=float), self._s, self._z)
        return float(out) if np.ndim(s) == 0 else out

    def lane_width_at(self, s: float) -> float:
        return float(np.interp(s, self._s, self._lw))

    def is_junction_at(self, s: float) -> bool:
        return bool(self._junc[self.nearest_vertex(s)])

    def nearest_vertex(self, s: float) -> int:
        i = int(np.searchsorted(self._s, s))
        if i <= 0:
            return 0
        if i >= len(self._s):
            return len(self._s) - 1
        return i if (self._s[i] - s) < (s - self._s[i - 1]) else i - 1

    def points_ahead_ego(self, s: float, ego_transform, distances=(5.0, 10.0, 20.0)) -> np.ndarray:
        """Route points at arc length s + d, in the ego frame (x forward, y right) [m]; shape (len(d), 2)."""
        pts = self._points_at(np.asarray(distances, dtype=float) + float(s))
        loc = ego_transform.location
        psi = math.radians(ego_transform.rotation.yaw)
        c, sn = math.cos(psi), math.sin(psi)
        dx = pts[:, 0] - loc.x
        dy = pts[:, 1] - loc.y
        return np.stack([dx * c + dy * sn, -dx * sn + dy * c], axis=1)

    def next_maneuver(self, s: float, horizon_m: float = 50.0) -> tuple:
        """First LEFT/RIGHT/STRAIGHT option at or ahead of s within horizon -> (option, distance [m]).

        Distance is 0 while the ego is inside the manoeuvre. Otherwise (LANEFOLLOW, horizon_m).
        """
        i = int(np.clip(np.searchsorted(self._s, s, side="right") - 1, 0, len(self._s) - 1))
        idx = np.nonzero(self._is_maneuver[i:])[0]
        if idx.size:
            j = i + int(idx[0])
            dist = max(0.0, float(self._s[j] - s))
            if dist <= horizon_m:
                return self.options[j], dist
        return RoadOption.LANEFOLLOW, float(horizon_m)

    def curvature_profile(self, s: float, horizon_m: float):
        """(distance ahead [m], curvature [1/m]) of the route vertices in [s, s + horizon]."""
        lo = int(np.searchsorted(self._s, s, side="left"))
        hi = int(np.searchsorted(self._s, s + horizon_m, side="right"))
        k_here = float(np.interp(s, self._s, self._kappa))
        d = np.concatenate([[0.0], self._s[lo:hi] - s])
        k = np.concatenate([[k_here], self._kappa[lo:hi]])
        return d, k

    def project_points(self, xy, s_from=None, s_to=None):
        """Vectorised projection of world points (M, 2) onto the route part [s_from, s_to].

        Returns (s, l): arc length of the nearest point on that part of the polyline and the signed
        Euclidean distance to it (+ = right). Points beyond the ends of the part get a large |l|.
        """
        P = np.asarray(xy, dtype=float).reshape(-1, 2)
        if P.shape[0] == 0:
            return np.zeros(0), np.zeros(0)
        lo, hi = self._seg_range(s_from, s_to)
        A = self._xy[lo:hi]
        D = self._seg[lo:hi]
        rel = P[:, None, :] - A[None, :, :]                                    # (M, K, 2)
        t = np.clip(np.einsum("mkj,kj->mk", rel, D) / self._len2[lo:hi], 0.0, 1.0)
        diff = rel - t[:, :, None] * D[None, :, :]
        d2 = np.einsum("mkj,mkj->mk", diff, diff)
        k = np.argmin(d2, axis=1)
        m = np.arange(P.shape[0])
        seg = lo + k
        s = self._s[seg] + t[m, k] * self._len[seg]
        side = diff[m, k, 0] * self._r[seg, 0] + diff[m, k, 1] * self._r[seg, 1]
        l = np.where(side >= 0.0, 1.0, -1.0) * np.sqrt(d2[m, k])
        return s, l

    # ------------------------------------------------------------------ lane queries (carla Map)
    def _adjacent_extension(self, wp, side: str) -> float:
        """Width of the adjacent lane on `side` if the ego may legally use it, else 0."""
        try:
            if bool(wp.is_junction):
                return 0.0
            perm = int(wp.lane_change)
            if not perm & (_LC_LEFT if side == "left" else _LC_RIGHT):
                return 0.0
            adj = wp.get_left_lane() if side == "left" else wp.get_right_lane()
            if adj is None:
                return 0.0
            driving = carla.LaneType.Driving if carla is not None else "Driving"
            if adj.lane_type != driving:
                return 0.0
            if abs(_wrap_deg(adj.transform.rotation.yaw - wp.transform.rotation.yaw)) >= 90.0:
                return 0.0
            return float(adj.lane_width)
        except (AttributeError, RuntimeError, TypeError):
            return 0.0

    def lane_bounds(self, s: float, wmap, ego_half_width: float) -> tuple:
        """(y_left, y_right) > 0: lateral free space [m] from the route centre to the usable edge.

        Each side = lane_width/2 - ego_half_width + 0.2, plus the adjacent lane's width when that
        lane is a Driving lane with the same travel direction (< 90 deg) and the marking allows
        the crossing (waypoint.lane_change). Inside junctions only the route lane counts.
        `wmap` is accepted for interface stability; the route waypoints carry the lane links.
        """
        base = max(0.05, 0.5 * self.lane_width_at(s) - float(ego_half_width) + LANE_BOUND_MARGIN_M)
        i = self.nearest_vertex(s)
        ext = self._bounds_cache.get(i)
        if ext is None:
            wp = self.waypoints[i]
            ext = (self._adjacent_extension(wp, "left"), self._adjacent_extension(wp, "right"))
            self._bounds_cache[i] = ext
        return base + ext[0], base + ext[1]

    def lane_status(self, location, yaw_deg: float, wmap, proj=None) -> LaneStatus:
        """Which lane the ego centre is in, relative to the route and its travel direction.

        `proj` (optional, defaults to the last result of `project`) is used inside junctions,
        where overlapping junction lanes make the map's lane choice arbitrary: there the ego counts
        as in the route lane when |e_y| < lane_width / 2.
        """
        lane_types = carla.LaneType.Driving | carla.LaneType.Parking
        wp = wmap.get_waypoint(location, project_to_road=False, lane_type=lane_types)
        on_drivable = wp is not None
        ref = wp if on_drivable else wmap.get_waypoint(location, project_to_road=True,
                                                       lane_type=carla.LaneType.Driving)
        if proj is None:
            proj = self._last_proj
        if ref is None:
            e = float(proj.e_y) if proj is not None else 0.0
            return LaneStatus(False, False, False, False, e, 0.0, False, 0)

        lane_tf = ref.transform
        dpsi_deg = _wrap_deg(float(yaw_deg) - lane_tf.rotation.yaw)
        is_junction = bool(ref.is_junction)
        is_driving = on_drivable and ref.lane_type == carla.LaneType.Driving
        in_same = bool(is_driving and abs(dpsi_deg) < 90.0)
        in_opp = bool(is_driving and abs(dpsi_deg) > 120.0 and not is_junction)

        lane_yaw = math.radians(lane_tf.rotation.yaw)
        dx = location.x - lane_tf.location.x
        dy = location.y - lane_tf.location.y
        e_lane = float(-dx * math.sin(lane_yaw) + dy * math.cos(lane_yaw))

        in_route = bool(on_drivable and self._lane_key(ref) in self._lane_keys)
        if on_drivable and not in_route and is_junction and proj is not None:
            in_route = abs(proj.e_y) < 0.5 * proj.lane_width
        try:
            lane_change = int(ref.lane_change)
        except (AttributeError, TypeError, ValueError):
            lane_change = 3
        return LaneStatus(
            on_drivable=bool(on_drivable),
            in_route_lane=in_route,
            in_same_dir_lane=in_same,
            in_opposite_lane=in_opp,
            e_lane=e_lane,
            lane_heading_err=math.radians(dpsi_deg),
            is_junction=is_junction,
            lane_change=lane_change,
        )
