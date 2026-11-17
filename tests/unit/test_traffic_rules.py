"""Unit tests for commons.traffic_rules (no CARLA server).

Run: python -m pytest tests/unit/test_traffic_rules.py   or   python tests/unit/test_traffic_rules.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _nav_fakes import (FakeMarking, FakeStopSign, FakeTrafficLight, FakeWaypoint, FakeWorld,  # noqa: E402
                        run_tests, straight_waypoints)

import carla  # noqa: E402
import numpy as np  # noqa: E402

from commons.navigation import RoadOption  # noqa: E402
from commons.route import LaneStatus, Route  # noqa: E402
from commons.traffic_rules import (DurationMonitor, RedLightMonitor, StopSignMonitor, TrafficLightInfo,  # noqa: E402
                                   TrafficLightTracker, lane_invasion_is_illegal)
from commons.vehicle_model import EGO_HALF_LENGTH  # noqa: E402

RED, YELLOW, GREEN, OFF = (carla.TrafficLightState.Red, carla.TrafficLightState.Yellow,
                           carla.TrafficLightState.Green, carla.TrafficLightState.Off)


def _route(wps):
    return Route.from_waypoints(wps, [RoadOption.LANEFOLLOW] * len(wps))


def _proj(route, s, hint=None):
    x, y = route.point_at(s)
    return route.project(carla.Location(x=x, y=y), 0.0, hint)


def _light_world():
    lights = {
        "A": FakeTrafficLight(11, [FakeWaypoint(50.0, 0.0, 0.0, road_id=1, lane_id=-1)]),       # our lane
        "B": FakeTrafficLight(12, [FakeWaypoint(50.0, -3.5, 180.0, road_id=1, lane_id=1)]),     # opposite
        "C": FakeTrafficLight(13, [FakeWaypoint(45.0, 10.0, 90.0, road_id=7, lane_id=-1)]),     # crossing road
        "D": FakeTrafficLight(14, [FakeWaypoint(150.0, 3.5, 0.0, road_id=1, lane_id=-2)]),      # other lane only
        "E": FakeTrafficLight(15, [FakeWaypoint(120.0, 0.3, 0.0, road_id=9, lane_id=-1)]),      # geometric match
    }
    return FakeWorld(lights.values()), lights


def test_tracker_matches_route_stop_lines():
    world, lights = _light_world()
    route = _route(straight_waypoints(200.0, road_id=1, lane_id=-1))
    tracker = TrafficLightTracker(world, None)
    lines = tracker.route_stop_lines(route)
    assert [lid for _, lid in lines] == [11, 15], lines
    assert abs(lines[0][0] - 50.0) < 1e-6


def test_tracker_query_states_and_gaps():
    world, lights = _light_world()
    route = _route(straight_waypoints(200.0, road_id=1, lane_id=-1))
    tracker = TrafficLightTracker(world, None)
    info = tracker.query(None, route, _proj(route, 20.0))
    assert isinstance(info, TrafficLightInfo)
    assert info.state == "red" and info.light_id == 11
    assert abs(info.stop_gap_m - (50.0 - 20.0 - EGO_HALF_LENGTH)) < 1e-6
    lights["A"].state = GREEN
    assert tracker.query(None, route, _proj(route, 20.0)).state == "green"
    lights["A"].state = OFF                     # dark light -> next one (E, beyond 50 m -> none)
    assert tracker.query(None, route, _proj(route, 20.0)).state == "none"
    assert tracker.query(None, route, _proj(route, 20.0), max_dist=120.0).light_id == 15
    lights["A"].state = RED
    far = tracker.query(None, route, _proj(route, 0.0), max_dist=20.0)
    assert far.state == "none" and far.stop_gap_m == 100.0 and far.light_id is None
    # front bumper over the line: still reported while red (negative gap), skipped once green
    info = tracker.query(None, route, _proj(route, 49.0))
    assert info.light_id == 11 and -EGO_HALF_LENGTH < info.stop_gap_m < 0.0
    lights["A"].state = GREEN
    assert tracker.query(None, route, _proj(route, 49.0)).light_id != 11
    # a new route object triggers a re-match
    route2 = _route(straight_waypoints(40.0, road_id=1, lane_id=-1))
    assert tracker.route_stop_lines(route2) == []


def _drive_through(states_by_s, s_from=30.0, s_to=70.0, step=0.5, stop_at=None):
    """Drive along the A-light route; states_by_s(s) -> state of light A. Returns violation s list."""
    world, lights = _light_world()
    route = _route(straight_waypoints(200.0, road_id=1, lane_id=-1))
    tracker = TrafficLightTracker(world, None)
    mon = RedLightMonitor()
    hits, hint = [], None
    for s in np.arange(s_from, s_to, step):
        if stop_at is not None:
            s = min(s, stop_at)
        lights["A"].state = states_by_s(s)
        proj = _proj(route, s, hint)
        hint = proj.idx
        tl = tracker.query(None, route, proj)
        if mon.update(tl, proj, 5.0):
            hits.append(float(s))
    return hits


def test_red_light_monitor():
    tail_cross = 50.0 + 0.8 * EGO_HALF_LENGTH
    hits = _drive_through(lambda s: RED)
    assert len(hits) == 1 and tail_cross <= hits[0] < tail_cross + 0.51, hits
    # entered on green / yellow (front over the line at s >= 47.6), light turns red while crossing: legal
    assert _drive_through(lambda s: GREEN if s < 48.5 else RED) == []
    assert _drive_through(lambda s: YELLOW if s < 48.5 else RED) == []
    # red already on the step the front bumper crosses: violation
    assert len(_drive_through(lambda s: GREEN if s < 48.0 else RED)) == 1
    # front crossed on red, then green before the tail crossed: legal (as in the leaderboard)
    assert _drive_through(lambda s: RED if s < 49.0 else GREEN) == []
    # stopped before the line
    assert _drive_through(lambda s: RED, stop_at=46.0) == []
    mon = RedLightMonitor()
    mon.reset()
    assert mon.update(TrafficLightInfo("none", 100.0, None), _proj(_route(straight_waypoints(20.0)), 1.0), 0.0) is False


def _junction_route(reverse=False):
    wps = straight_waypoints(120.0, road_id=1, lane_id=-1)
    for w in wps:
        w.is_junction = 40.0 <= w.transform.location.x <= 60.0
    if reverse:
        wps = [FakeWaypoint(w.transform.location.x, 0.0, 180.0, is_junction=w.is_junction) for w in wps[::-1]]
    return _route(wps)


def _sign_world():
    # sign on the kerb at x = 36; its trigger volume covers our lane just before the junction
    return FakeWorld([FakeStopSign(21, 36.0, 5.0, volume_center=(0.0, -5.0), extent=(2.0, 1.5, 1.0))])


def test_stop_sign_matching():
    mon = StopSignMonitor(_sign_world(), None)
    lines = mon.route_stop_lines(_junction_route())
    assert len(lines) == 1 and abs(lines[0][0] - 39.0) < 0.6 and lines[0][1] == 21, lines
    mon2 = StopSignMonitor(_sign_world(), None)
    assert mon2.route_stop_lines(_junction_route(reverse=True)) == []       # leaving the junction
    wps = straight_waypoints(120.0, y0=20.0)
    assert StopSignMonitor(_sign_world(), None).route_stop_lines(_route(wps)) == []   # elsewhere


def _run_stop(speed_profile):
    route = _junction_route()
    mon = StopSignMonitor(_sign_world(), None)
    out, hint = [], None
    for s, v in speed_profile:
        proj = _proj(route, s, hint)
        hint = proj.idx
        out.append(mon.update(None, route, proj, v))
    return out


def test_stop_sign_violation_and_completion():
    # rolls through at 5 m/s
    res = _run_stop([(s, 5.0) for s in np.arange(10.0, 60.0, 0.5)])
    assert sum(v for _, v in res) == 1
    gap0 = res[0][0]
    assert abs(gap0 - (39.0 - 10.0 - EGO_HALF_LENGTH)) < 0.6
    # stops 1 m before the line, then goes: no violation, and the line is no longer reported
    prof = [(s, 5.0) for s in np.arange(10.0, 35.5, 0.5)] + [(35.5, 0.0)] + [(s, 3.0) for s in np.arange(36.0, 60.0, 0.5)]
    res = _run_stop(prof)
    assert sum(v for _, v in res) == 0
    assert res[-1][0] == 100.0 and res[len(prof) - 10][0] == 100.0
    # stopped too early (outside the 4 m zone) does not count
    prof = [(s, 5.0) for s in np.arange(10.0, 28.0, 0.5)] + [(28.0, 0.0)] + [(s, 3.0) for s in np.arange(28.5, 60.0, 0.5)]
    assert sum(v for _, v in _run_stop(prof)) == 1


def _ls(e_lane, opposite=False, lane_change=3, same=True):
    return LaneStatus(True, False, same, opposite, e_lane, 0.0, False, lane_change)


def test_lane_invasion_rules():
    LMT = carla.LaneMarkingType
    broken = [FakeMarking(LMT.Broken)]
    # legal overtaking: left lane change across a broken line into a same-direction lane
    assert lane_invasion_is_illegal(broken, _ls(-1.6), _ls(1.7)) is False
    # ... but not into an opposite-direction lane
    assert lane_invasion_is_illegal(broken, _ls(-1.6), _ls(-1.5, opposite=True)) is True
    for t in (LMT.Solid, LMT.SolidSolid, LMT.Curb, LMT.Grass):
        assert lane_invasion_is_illegal([FakeMarking(t)], _ls(1.2), _ls(1.4)) is True, t
    for t in (LMT.BrokenBroken, LMT.BottsDots, LMT.NONE, LMT.Other):
        assert lane_invasion_is_illegal([FakeMarking(t)], _ls(1.2), _ls(1.4)) is False, t
    # SolidBroken: allowed only towards the side the lane permits (Left = 2, Right = 1)
    sb = [FakeMarking(LMT.SolidBroken)]
    assert lane_invasion_is_illegal(sb, _ls(-1.6, lane_change=2), _ls(1.7)) is False     # left, permitted
    assert lane_invasion_is_illegal(sb, _ls(1.6, lane_change=2), _ls(-1.7)) is True      # right, not permitted
    assert lane_invasion_is_illegal(sb, _ls(1.3, lane_change=1), _ls(1.5, lane_change=1)) is False  # corner, right
    assert lane_invasion_is_illegal(sb, _ls(-1.3, lane_change=0), _ls(-1.5, lane_change=0)) is True
    # markings given as plain names, and empty events
    assert lane_invasion_is_illegal(["Solid"], _ls(1.0), _ls(1.0)) is True
    assert lane_invasion_is_illegal([], _ls(1.0), _ls(1.0)) is False
    assert lane_invasion_is_illegal(broken, None, _ls(1.0)) is False


def test_duration_monitor():
    m = DurationMonitor(3)
    assert [m.update(c) for c in (False, True, True, True, True)] == [False, False, False, True, True]
    assert [m.update(c) for c in (False, True, True)] == [False, False, False]
    m.reset()
    assert m.count == 0
    assert [m.update(True) for _ in range(3)] == [False, False, True]


if __name__ == "__main__":
    sys.exit(run_tests(dict(globals())))
