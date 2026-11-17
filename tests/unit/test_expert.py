"""Closed-loop tests of commons.expert.ExpertDriver on a kinematic car (no CARLA server).

The car is a kinematic bicycle with the measured AccelModel and a 0.3 s pedal delay; obstacles,
lights and stop signs go through the real commons.perception / commons.traffic_rules code.
Run: python -m pytest tests/unit/test_expert.py   or   python tests/unit/test_expert.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _nav_fakes import (FakeActor, FakeStopSign, FakeTrafficLight, FakeWaypoint, FakeWorld, KinematicCar,  # noqa: E402
                        right_turn_route, run_tests, straight_waypoints)

import carla  # noqa: E402
import numpy as np  # noqa: E402

from commons.expert import ExpertDriver  # noqa: E402
from commons.navigation import RoadOption  # noqa: E402
from commons.perception import LeadInfo, find_lead  # noqa: E402
from commons.route import Route  # noqa: E402
from commons.traffic_rules import RedLightMonitor, StopSignMonitor, TrafficLightInfo, TrafficLightTracker  # noqa: E402
from commons.vehicle_model import DT, EGO_HALF_LENGTH  # noqa: E402

PEDAL_DELAY_STEPS = 6


def _route(wps, opts=None):
    return Route.from_waypoints(wps, opts or [RoadOption.LANEFOLLOW] * len(wps))


def simulate(route, car, seconds, speed_limit=10.0, actors=(), world=None, on_step=None):
    expert = ExpertDriver()
    tracker = TrafficLightTracker(world, None) if world is not None else None
    stops = StopSignMonitor(world, None) if world is not None else None
    red_mon = RedLightMonitor()
    hint, log = None, []
    for k in range(int(round(seconds / DT))):
        if on_step is not None:
            on_step(k * DT)
        tf = car.transform
        proj = route.project(tf.location, tf.rotation.yaw, hint)
        hint = proj.idx
        lead = find_lead(list(actors) + [car], car, route, proj)
        tl = tracker.query(car, route, proj) if tracker else TrafficLightInfo("none", 100.0, None)
        stop_gap, stop_viol = stops.update(car, route, proj, car.v) if stops else (100.0, False)
        red = red_mon.update(tl, proj, car.v)
        a = expert.act(tf, car.v, route, proj, lead, tl, stop_gap, speed_limit)
        assert a.shape == (2,) and a.dtype == np.float32 and np.all(np.abs(a) <= 1.0)
        log.append(dict(t=k * DT, s=proj.s, e_y=proj.e_y, v=car.v, gap=lead.gap_m, tl_gap=tl.stop_gap_m,
                        tl_state=tl.state, stop_gap=stop_gap, stop_viol=stop_viol, red=red, a=a.copy()))
        car.step(a)
    return log


def _col(log, key):
    return np.array([r[key] for r in log])


def test_cruise_respects_limit_and_converges_laterally():
    route = _route(straight_waypoints(600.0))
    car = KinematicCar(0.0, 1.0, 5.0, pedal_delay_steps=PEDAL_DELAY_STEPS)   # start off-centre and yawed
    log = simulate(route, car, 30.0, speed_limit=10.0)
    v, t, e_y = _col(log, "v"), _col(log, "t"), _col(log, "e_y")
    assert v.max() <= 10.0, v.max()
    assert np.all((v[t > 10.0] > 8.0) & (v[t > 10.0] <= 10.0)), (v[t > 10.0].min(), v[t > 10.0].max())
    assert np.all(np.abs(e_y[t > 6.0]) < 0.2), np.abs(e_y[t > 6.0]).max()


def _turn_check(direction, radius, limit=12.0):
    wps, opts, t0, t1 = right_turn_route(l1=60.0, radius=radius, l2=80.0, direction=direction)
    route = _route(wps, opts)
    car = KinematicCar(0.0, 0.0, 0.0, pedal_delay_steps=PEDAL_DELAY_STEPS)
    log = simulate(route, car, 18.0, speed_limit=limit)     # ends before the route end (goal)
    s, e_y, v = _col(log, "s"), _col(log, "e_y"), _col(log, "v")
    in_turn = (s > t0) & (s < t1)
    assert in_turn.any() and s[-1] > t1 + 20.0
    assert np.abs(e_y).max() < 0.7, np.abs(e_y).max()
    assert v[in_turn].max() <= math.sqrt(2.5 * radius) + 0.7, v[in_turn].max()


def test_follows_right_and_left_turns():
    _turn_check(+1, 10.0)
    _turn_check(-1, 15.0)


def test_stops_behind_static_vehicle():
    route = _route(straight_waypoints(300.0))
    parked = FakeActor(1, "vehicle.audi.a2", 100.0, 0.0)
    car = KinematicCar(0.0, 0.0, 0.0, pedal_delay_steps=PEDAL_DELAY_STEPS)
    log = simulate(route, car, 30.0, speed_limit=12.0, actors=[parked])
    gap, v = _col(log, "gap"), _col(log, "v")
    assert v[-1] < 0.1
    assert 2.5 <= gap[-1] <= 5.5, gap[-1]
    assert gap.min() > 1.0


def test_follows_moving_lead_without_collision():
    route = _route(straight_waypoints(600.0))
    lead = FakeActor(1, "vehicle.audi.a2", 40.0, 0.0, vel=(5.0, 0.0))

    def move(t):
        lead.set_transform(carla.Transform(carla.Location(x=40.0 + 5.0 * t, y=0.0), carla.Rotation()))

    car = KinematicCar(0.0, 0.0, 0.0, pedal_delay_steps=PEDAL_DELAY_STEPS)
    log = simulate(route, car, 40.0, speed_limit=12.0, actors=[lead], on_step=move)
    gap, v, t = _col(log, "gap"), _col(log, "v"), _col(log, "t")
    assert gap.min() > 3.0, gap.min()
    late = t > 30.0
    assert np.all(np.abs(v[late] - 5.0) < 1.0), (v[late].min(), v[late].max())
    assert np.all((gap[late] > 6.0) & (gap[late] < 25.0)), (gap[late].min(), gap[late].max())


def test_red_light_stop_then_go():
    route = _route(straight_waypoints(300.0, road_id=1, lane_id=-1))
    light = FakeTrafficLight(11, [FakeWaypoint(100.0, 0.0, 0.0, road_id=1, lane_id=-1)])
    world = FakeWorld([light])

    def lights(t):
        light.state = carla.TrafficLightState.Red if t < 20.0 else carla.TrafficLightState.Green

    car = KinematicCar(0.0, 0.0, 0.0, pedal_delay_steps=PEDAL_DELAY_STEPS)
    log = simulate(route, car, 40.0, speed_limit=12.0, world=world, on_step=lights)
    t, v, s, tl_gap = _col(log, "t"), _col(log, "v"), _col(log, "s"), _col(log, "tl_gap")
    waiting = (t > 15.0) & (t < 20.0)
    assert np.all(v[waiting] < 0.1)
    front_at_stop = s[waiting][-1] + EGO_HALF_LENGTH
    assert 97.0 <= front_at_stop <= 100.0, front_at_stop
    assert s[-1] > 150.0
    assert not any(r["red"] for r in log)


def test_yellow_light_decisions():
    route = _route(straight_waypoints(300.0))
    car = KinematicCar(50.0, 0.0, 0.0)
    proj = route.project(car.transform.location, 0.0)
    none = LeadInfo(100.0, 0.0, "none", None)
    exp = ExpertDriver()
    # too close to stop comfortably at 10 m/s: keep going
    a = exp.act(car.transform, 10.0, route, proj, none, TrafficLightInfo("yellow", 5.0, 3), 100.0, 12.0)
    assert a[1] > -0.05, a
    # plenty of room: slow down for it
    v_t, _ = exp.target_speed(10.0, route, proj, none, TrafficLightInfo("yellow", 20.0, 3), 100.0, 12.0)
    assert v_t < 10.0
    # red, stopped with the bumper slightly over the line: hold
    a = exp.act(car.transform, 0.0, route, proj, none, TrafficLightInfo("red", -0.5, 3), 100.0, 12.0)
    assert a[1] < 0.0


def test_stop_sign_full_stop_then_go():
    wps = straight_waypoints(160.0, road_id=1, lane_id=-1)
    for w in wps:
        w.is_junction = 80.0 <= w.transform.location.x <= 100.0
    route = _route(wps)
    world = FakeWorld([FakeStopSign(21, 76.0, 5.0, volume_center=(0.0, -5.0), extent=(2.0, 1.5, 1.0))])
    car = KinematicCar(0.0, 0.0, 0.0, pedal_delay_steps=PEDAL_DELAY_STEPS)
    log = simulate(route, car, 40.0, speed_limit=10.0, world=world)
    s, v = _col(log, "s"), _col(log, "v")
    stop_line = 79.0
    zone = (s + EGO_HALF_LENGTH > stop_line - 4.0) & (s + EGO_HALF_LENGTH < stop_line)
    assert v[zone].min() < 0.1
    assert not any(r["stop_viol"] for r in log)
    assert s[-1] > 120.0


def test_hold_and_output_range():
    route = _route(straight_waypoints(100.0))
    car = KinematicCar(10.0, 0.0, 0.0)
    proj = route.project(car.transform.location, 0.0)
    exp = ExpertDriver()
    a = exp.act(car.transform, 0.0, route, proj, LeadInfo(1.0, 0.0, "static", None),
                TrafficLightInfo("none", 100.0, None), 100.0, 10.0)
    assert a.dtype == np.float32 and a.shape == (2,) and a[1] < 0.0
    a = exp.act(car.transform, 3.0, route, proj, LeadInfo(100.0, 0.0, "none", None),
                TrafficLightInfo("none", 100.0, None), 100.0, 10.0)
    assert a[1] > 0.0 and abs(a[0]) < 1e-3


if __name__ == "__main__":
    sys.exit(run_tests(dict(globals())))
