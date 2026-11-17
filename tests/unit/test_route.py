"""Unit tests for commons.route and the bundled commons.navigation planner (no CARLA server).

Run: python -m pytest tests/unit/test_route.py   or   python tests/unit/test_route.py
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _nav_fakes import (FakeStraightMap, FakeWaypoint, polyline_waypoints, right_turn_route,  # noqa: E402
                        run_tests, straight_waypoints)

import carla  # noqa: E402
import numpy as np  # noqa: E402

from commons.navigation import GlobalRoutePlanner, RoadOption  # noqa: E402
from commons.route import LaneStatus, Route, RouteProjection, wrap_angle  # noqa: E402
from commons.vehicle_model import EGO_HALF_WIDTH  # noqa: E402

XODR = os.environ.get("CARLA_XODR", r"D:\WORK\Carla\CarlaUE4\Content\Carla\Maps\OpenDrive\Town10HD.xodr")


def _loc(x, y, z=0.0):
    return carla.Location(x=x, y=y, z=z)


def _straight_route(length=100.0, **kw):
    wps = straight_waypoints(length, **kw)
    return Route.from_waypoints(wps, [RoadOption.LANEFOLLOW] * len(wps))


def test_wrap_angle():
    assert abs(wrap_angle(3 * math.pi) - math.pi) < 1e-9
    assert abs(wrap_angle(-math.pi) - math.pi) < 1e-9          # (-pi, pi]
    assert abs(wrap_angle(0.5) - 0.5) < 1e-12
    arr = wrap_angle(np.array([-3 * math.pi / 2, 2 * math.pi]))
    assert np.allclose(arr, [math.pi / 2, 0.0])


def test_from_waypoints_dedups_and_measures():
    wps = straight_waypoints(100.0)
    dup = []
    for w in wps:
        dup += [w, FakeWaypoint(w.transform.location.x + 0.1, 0.0, 0.0)]   # near-duplicates
    route = Route.from_waypoints(dup, [RoadOption.LANEFOLLOW] * len(dup))
    assert len(route.waypoints) == len(wps)
    assert len(route.options) == len(route.waypoints)
    assert abs(route.length_m - 100.0) < 1e-6
    assert route.max_kink_deg < 1e-6
    single = Route.from_waypoints(wps[:1], [RoadOption.LANEFOLLOW])     # never degenerate
    assert single.length_m > 1.0


def test_project_straight():
    route = _straight_route(100.0)
    p = route.project(_loc(37.3, 1.2), 10.0)
    assert isinstance(p, RouteProjection)
    assert abs(p.s - 37.3) < 1e-6
    assert abs(p.e_y - 1.2) < 1e-6                     # + = right of the centre line
    assert abs(p.e_psi - math.radians(10.0)) < 1e-6    # + = yawed clockwise
    assert abs(p.kappa) < 1e-9
    assert abs(p.remaining_m - 62.7) < 1e-6
    assert abs(p.lane_width - 3.5) < 1e-9
    q = route.project(_loc(37.3, -0.8), -20.0, hint_idx=p.idx)
    assert q.e_y < 0 and q.e_psi < 0
    # e_psi wraps into (-pi, pi]
    r = route.project(_loc(50.0, 0.0), 180.0, hint_idx=q.idx)
    assert abs(abs(r.e_psi) - math.pi) < 1e-6


def test_project_curvature_sign():
    wps, opts, t0, t1 = right_turn_route(radius=10.0, direction=1)
    route = Route.from_waypoints(wps, opts)
    mid = 0.5 * (t0 + t1)
    x, y = route.point_at(mid)
    p = route.project(_loc(x, y), math.degrees(route.yaw_at(mid)))
    assert abs(p.s - mid) < 0.3
    assert 0.07 < p.kappa < 0.13, p.kappa              # right turn -> kappa ~ +1/R
    wps, opts, t0, t1 = right_turn_route(radius=10.0, direction=-1)
    route = Route.from_waypoints(wps, opts)
    x, y = route.point_at(mid)
    p = route.project(_loc(x, y), math.degrees(route.yaw_at(mid)))
    assert -0.13 < p.kappa < -0.07, p.kappa
    # a point outside the right turn (towards the left, -y side before turning) has e_y < 0
    p2 = route.project(_loc(x - 1.0, y - 1.0), math.degrees(route.yaw_at(mid)))
    assert p2.e_y != 0.0


def test_project_hint_handles_self_overlap():
    # the route passes (20, 0) twice: first eastbound at s=20, later northbound (+y) at s=100
    pts = [(0, 0), (40, 0), (40, -20), (20, -20), (20, 20), (20, 40)]
    wps = polyline_waypoints(pts)
    route = Route.from_waypoints(wps, [RoadOption.LANEFOLLOW] * len(wps))
    first = route.project(_loc(19.0, 0.3), 0.0)                       # global search, eastbound
    assert abs(first.s - 19.0) < 0.5
    # walk along the route with hints; at the crossing the projection must stay on the 2nd pass
    idx = None
    prev_s = 0.0
    for s in np.arange(0.0, route.length_m, 0.5):
        x, y = route.point_at(s)
        p = route.project(_loc(x, y), math.degrees(route.yaw_at(s)), idx)
        assert abs(p.s - s) < 0.6, (s, p.s)
        assert p.s >= prev_s - 1e-6
        idx, prev_s = p.idx, p.s
    hint_late = route.project(_loc(20.0, -2.0), 90.0).idx              # around s = 98
    p = route.project(_loc(20.3, 0.0), 0.0, hint_idx=hint_late)        # even with an eastbound yaw
    assert p.s > 90.0, p.s
    # far behind the hint: never more than ~10 m backwards
    p = route.project(_loc(5.0, 0.0), 0.0, hint_idx=hint_late)
    assert p.s >= route.vertex_s[hint_late] - 10.0
    # early hint: does not jump ahead to the later pass
    p = route.project(_loc(20.0, 0.2), 90.0, hint_idx=route.project(_loc(15.0, 0.0), 0.0).idx)
    assert p.s < 30.0, p.s


def test_point_at_and_points_ahead_ego():
    route = _straight_route(100.0)
    assert route.point_at(12.5) == (12.5, 0.0)
    assert route.point_at(105.0) == (105.0, 0.0)       # extrapolated past the end
    tf = carla.Transform(_loc(10.0, 0.0), carla.Rotation(yaw=0.0))
    pts = route.points_ahead_ego(10.0, tf)
    assert pts.shape == (3, 2)
    assert np.allclose(pts, [[5, 0], [10, 0], [20, 0]], atol=1e-6)
    # ego yawed 90 deg clockwise (facing +y): the route (+x) is to its LEFT -> y < 0
    tf = carla.Transform(_loc(10.0, 0.0), carla.Rotation(yaw=90.0))
    pts = route.points_ahead_ego(10.0, tf, (5.0,))
    assert np.allclose(pts, [[0.0, -5.0]], atol=1e-5)
    # right turn ahead: the 20 m point is to the right (y > 0)
    wps, opts, t0, _ = right_turn_route()
    route = Route.from_waypoints(wps, opts)
    tf = carla.Transform(_loc(t0 - 2.0, 0.0), carla.Rotation(yaw=0.0))
    pts = route.points_ahead_ego(t0 - 2.0, tf)
    assert pts[2, 1] > 2.0 and abs(pts[0, 1]) < 0.5


def test_next_maneuver():
    wps, opts, t0, t1 = right_turn_route(l1=40.0)
    route = Route.from_waypoints(wps, opts)
    opt, d = route.next_maneuver(0.0, 50.0)
    assert opt == RoadOption.RIGHT and abs(d - 40.0) < 2.1
    opt, d = route.next_maneuver(0.0, 20.0)
    assert opt == RoadOption.LANEFOLLOW and d == 20.0
    opt, d = route.next_maneuver(0.5 * (t0 + t1), 50.0)
    assert opt == RoadOption.RIGHT and d == 0.0
    opt, d = route.next_maneuver(t1 + 5.0, 50.0)
    assert opt == RoadOption.LANEFOLLOW and d == 50.0


def _with_neighbours(wps, left=None, right=None, lane_change=carla.LaneChange.NONE, junction=False):
    for w in wps:
        w.lane_change = lane_change
        w.is_junction = junction
        loc = w.transform.location
        if left is not None:
            w._left = FakeWaypoint(loc.x, loc.y - 3.5, left[0], lane_type=left[1], lane_width=left[2])
        if right is not None:
            w._right = FakeWaypoint(loc.x, loc.y + 3.5, right[0], lane_type=right[1], lane_width=right[2])
    return wps


def test_lane_bounds():
    base = 0.5 * 3.5 - EGO_HALF_WIDTH + 0.2
    drv = carla.LaneType.Driving
    # left: same direction, crossing allowed; right: opposite direction
    wps = _with_neighbours(straight_waypoints(50.0), left=(0.0, drv, 3.0), right=(180.0, drv, 3.5),
                           lane_change=carla.LaneChange.Both)
    route = Route.from_waypoints(wps, [RoadOption.LANEFOLLOW] * len(wps))
    yl, yr = route.lane_bounds(20.0, None, EGO_HALF_WIDTH)
    assert abs(yl - (base + 3.0)) < 1e-9 and abs(yr - base) < 1e-9
    # crossing not permitted by the marking
    wps = _with_neighbours(straight_waypoints(50.0), left=(0.0, drv, 3.5), right=(0.0, drv, 3.5),
                           lane_change=carla.LaneChange.Right)
    route = Route.from_waypoints(wps, [RoadOption.LANEFOLLOW] * len(wps))
    yl, yr = route.lane_bounds(20.0, None, EGO_HALF_WIDTH)
    assert abs(yl - base) < 1e-9 and abs(yr - (base + 3.5)) < 1e-9
    # neighbour is not a driving lane / we are in a junction
    wps = _with_neighbours(straight_waypoints(50.0), right=(0.0, carla.LaneType.Sidewalk, 2.0),
                           lane_change=carla.LaneChange.Both)
    route = Route.from_waypoints(wps, [RoadOption.LANEFOLLOW] * len(wps))
    assert abs(route.lane_bounds(20.0, None, EGO_HALF_WIDTH)[1] - base) < 1e-9
    wps = _with_neighbours(straight_waypoints(50.0), right=(0.0, drv, 3.5), lane_change=carla.LaneChange.Both,
                           junction=True)
    route = Route.from_waypoints(wps, [RoadOption.LANEFOLLOW] * len(wps))
    assert route.lane_bounds(20.0, None, EGO_HALF_WIDTH) == (base, base)


def _lane_map(junction_x=None):
    drv = carla.LaneType.Driving
    return FakeStraightMap([
        dict(lane_id=-1, y=0.0, yaw=0.0, lane_type=drv, width=3.5, lane_change=carla.LaneChange.Right),
        dict(lane_id=-2, y=3.5, yaw=0.0, lane_type=drv, width=3.5, lane_change=carla.LaneChange.Left),
        dict(lane_id=1, y=-3.5, yaw=180.0, lane_type=drv, width=3.5),
        dict(lane_id=-3, y=6.5, yaw=0.0, lane_type=carla.LaneType.Sidewalk, width=2.5),
    ], junction_x)


def test_lane_status():
    route = _straight_route(200.0, lane_id=-1, road_id=1)
    wmap = _lane_map()
    st = route.lane_status(_loc(10.0, 0.3), 2.0, wmap)
    assert isinstance(st, LaneStatus)
    assert st.on_drivable and st.in_route_lane and st.in_same_dir_lane and not st.in_opposite_lane
    assert abs(st.e_lane - 0.3) < 1e-6 and abs(st.lane_heading_err - math.radians(2.0)) < 1e-9
    assert st.lane_change == int(carla.LaneChange.Right)
    st = route.lane_status(_loc(10.0, 3.2), 0.0, wmap)                 # overtaking lane, same direction
    assert st.on_drivable and not st.in_route_lane and st.in_same_dir_lane
    assert abs(st.e_lane + 0.3) < 1e-6
    st = route.lane_status(_loc(10.0, -3.5), 0.0, wmap)                # opposite lane
    assert st.in_opposite_lane and not st.in_same_dir_lane
    assert abs(abs(st.lane_heading_err) - math.pi) < 1e-6
    st = route.lane_status(_loc(10.0, 6.5), 0.0, wmap)                 # sidewalk
    assert not st.on_drivable and not st.in_route_lane and not st.in_same_dir_lane
    # inside a junction the opposite-lane flag is suspended
    st = route.lane_status(_loc(10.0, -3.5), 0.0, _lane_map(junction_x=(0.0, 50.0)))
    assert st.is_junction and not st.in_opposite_lane


class _FakeGRP:
    """trace_route: a straight 2 m-sampled line from origin to destination (or a kinked one)."""

    def __init__(self, kinked_dest_x=()):
        self.kinked_dest_x = set(kinked_dest_x)
        self.calls = 0

    def trace_route(self, origin, destination):
        self.calls += 1
        pts = [(origin.x, origin.y), (destination.x, destination.y)]
        if round(destination.x) in self.kinked_dest_x:   # sideways jump at the start
            pts = [(origin.x, origin.y), (origin.x, origin.y + 3.5), (destination.x, destination.y + 3.5)]
        wps = polyline_waypoints(pts)
        return [(w, RoadOption.LANEFOLLOW) for w in wps]


def test_plan_length_window_and_fallbacks():
    start = _loc(0.0, 0.0)
    spawns = [carla.Transform(_loc(x, 0.0), carla.Rotation()) for x in (50.0, 120.0, 250.0, 330.0, 600.0)]
    for seed in range(5):
        r = Route.plan(_FakeGRP(), None, start, spawns, 200.0, 400.0, random.Random(seed))
        assert 200.0 <= r.length_m <= 400.0, r.length_m
    # kinked routes are never accepted when a clean one exists
    r = Route.plan(_FakeGRP(kinked_dest_x=(250,)), None, start, spawns, 200.0, 400.0, random.Random(0))
    assert r.max_kink_deg <= 50.0 and 200.0 <= r.length_m <= 400.0
    # nothing in range -> closest to the range
    r = Route.plan(_FakeGRP(), None, start, spawns[:2], 200.0, 400.0, random.Random(0))
    assert abs(r.length_m - 120.0) < 1e-6

    class Broken:
        def trace_route(self, a, b):
            raise RuntimeError("no path")

    wmap = _lane_map()
    r = Route.plan(Broken(), wmap, start, spawns, 80.0, 150.0, random.Random(0))
    assert len(r.waypoints) >= 2 and 80.0 <= r.length_m <= 150.0     # lane-follow fallback
    r = Route.plan(Broken(), wmap, start, [], 80.0, 150.0, random.Random(0))
    assert r.length_m > 0.0


def test_bundled_planner_on_real_map():
    """GlobalRoutePlanner + Route on an offline OpenDRIVE map (skipped if the .xodr is absent)."""
    if not os.path.exists(XODR):
        print("  (skipped: no OpenDRIVE file at %s; set CARLA_XODR)" % XODR)
        return
    wmap = carla.Map("Town10HD", open(XODR).read())
    grp = GlobalRoutePlanner(wmap, 2.0)
    spawns = [w.transform for w in wmap.generate_waypoints(20.0) if not w.is_junction]
    rng = random.Random(3)
    for _ in range(5):
        start = rng.choice(spawns)
        route = Route.plan(grp, wmap, start.location, spawns, 200.0, 400.0, rng)
        assert 200.0 <= route.length_m <= 400.0 and route.max_kink_deg <= 50.0
        assert all(isinstance(o, RoadOption) for o in route.options)
        idx, prev = None, -1.0
        for s in np.arange(0.0, route.length_m, 2.0):
            x, y = route.point_at(s)
            yaw = math.degrees(route.yaw_at(s))
            p = route.project(_loc(x, y), yaw, idx)
            assert p.s >= prev - 1e-6 and abs(p.e_y) < 0.3
            idx, prev = p.idx, p.s
            if int(s) % 20 == 0:
                st = route.lane_status(_loc(x, y), yaw, wmap)
                assert st.on_drivable and (st.in_route_lane or st.is_junction)
                assert st.in_same_dir_lane or st.is_junction
                yl, yr = route.lane_bounds(p.s, wmap, EGO_HALF_WIDTH)
                assert yl > 0.0 and yr > 0.0


if __name__ == "__main__":
    sys.exit(run_tests(dict(globals())))
