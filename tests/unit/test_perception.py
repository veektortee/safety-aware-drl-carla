"""Unit tests for commons.perception (no CARLA server).

Run: python -m pytest tests/unit/test_perception.py   or   python tests/unit/test_perception.py
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _nav_fakes import FakeActor, KinematicCar, FakeWaypoint, right_turn_route, run_tests, straight_waypoints, walker  # noqa: E402,E501

import carla  # noqa: E402
import numpy as np  # noqa: E402

from commons.navigation import RoadOption  # noqa: E402
from commons.perception import LeadInfo, NO_GAP, combine, find_lead, lidar_static_gap, ttc  # noqa: E402
from commons.route import Route  # noqa: E402
from commons.vehicle_model import EGO_HALF_LENGTH  # noqa: E402


def _route(wps, opts=None):
    return Route.from_waypoints(wps, opts or [RoadOption.LANEFOLLOW] * len(wps))


def _ego_and_proj(route, x, y, yaw=0.0):
    ego = KinematicCar(x, y, yaw)
    proj = route.project(ego.transform.location, yaw)
    return ego, proj


def _sensor_tf(x, y, yaw=0.0, z=1.7):
    return carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(yaw=yaw))


def test_find_lead_vehicle_ahead():
    route = _route(straight_waypoints(200.0))
    ego, proj = _ego_and_proj(route, 10.0, 0.0)
    lead = FakeActor(1, "vehicle.audi.a2", 30.0, 0.0, vel=(5.0, 0.0))
    info = find_lead([ego, lead], ego, route, proj)
    assert isinstance(info, LeadInfo)
    assert info.kind == "vehicle" and info.actor_id == 1
    assert abs(info.gap_m - (30.0 - 2.3 - (10.0 + EGO_HALF_LENGTH))) < 1e-3
    assert abs(info.v_lead_mps - 5.0) < 1e-6


def test_find_lead_ignores_irrelevant_actors():
    route = _route(straight_waypoints(200.0))
    ego, proj = _ego_and_proj(route, 10.0, 0.0)
    actors = [
        ego,
        FakeActor(2, "vehicle.audi.a2", 30.0, 3.5),          # adjacent lane
        FakeActor(3, "vehicle.audi.a2", 2.0, 0.0),           # behind
        FakeActor(4, "vehicle.audi.a2", 75.0, 0.0),          # beyond the 50 m horizon
        FakeActor(5, "traffic.traffic_light", 20.0, 0.0),    # not a vehicle / walker
        FakeActor(6, "sensor.lidar.ray_cast", 20.0, 0.0),
        walker(7, 25.0, 4.0),                                # standing on the sidewalk
        walker(8, 30.0, 4.0, vel=(0.0, 1.5)),                # walking away from the road
    ]
    info = find_lead(actors, ego, route, proj)
    assert info.kind == "none" and info.gap_m == NO_GAP and info.actor_id is None


def test_find_lead_crossing_vehicle_and_walker():
    route = _route(straight_waypoints(200.0))
    ego, proj = _ego_and_proj(route, 10.0, 0.0)
    # perpendicular vehicle spanning the corridor: no corner is inside |l| < w, the box is
    crossing = FakeActor(1, "vehicle.audi.a2", 40.0, 0.0, yaw_deg=90.0, vel=(0.0, 6.0))
    info = find_lead([crossing], ego, route, proj)
    assert info.kind == "vehicle"
    assert abs(info.gap_m - (40.0 - 1.0 - 12.4)) < 1e-3 and abs(info.v_lead_mps) < 1e-6
    # pedestrian on the kerb stepping towards the lane: swept 2 s along its lateral motion
    ped = walker(9, 25.0, 4.0, vel=(0.0, -1.5))
    info = find_lead([ped], ego, route, proj)
    assert info.kind == "walker" and info.actor_id == 9
    assert abs(info.gap_m - (25.0 - 0.25 - 12.4)) < 1e-3
    # nearest of several
    info = find_lead([crossing, ped, FakeActor(3, "vehicle.x", 60.0, 0.0)], ego, route, proj)
    assert info.actor_id == 9


def test_find_lead_curved_corridor():
    wps, opts, t0, t1 = right_turn_route(l1=40.0, radius=10.0)
    route = _route(wps, opts)
    ego, proj = _ego_and_proj(route, 30.0, 0.0)
    # car parked straight ahead of the junction where the route has already turned right
    off_route = FakeActor(1, "vehicle.x", 40.0 + 14.0, 0.0)
    assert find_lead([off_route], ego, route, proj).kind == "none"
    # car standing on the turn path, 45 deg into the turn
    th = math.radians(45.0)
    x, y = 40.0 + 10.0 * math.sin(th), 10.0 - 10.0 * math.cos(th)
    on_turn = FakeActor(2, "vehicle.x", x, y, yaw_deg=45.0)
    info = find_lead([off_route, on_turn], ego, route, proj)
    assert info.actor_id == 2
    s_car = 40.0 + 10.0 * th
    assert abs(info.gap_m - (s_car - 2.3 - (30.0 + EGO_HALF_LENGTH))) < 0.6


def test_lidar_gap_basic_filters():
    route = _route(straight_waypoints(200.0))
    _, proj = _ego_and_proj(route, 10.0, 0.0)
    tf = _sensor_tf(10.0, 0.0)
    obstacle = np.array([[20.0, 0.0, -1.0]])                 # 0.7 m above ground, 20 m ahead of the sensor
    assert abs(lidar_static_gap(obstacle, tf, route, proj) - (30.0 - 12.4)) < 1e-6
    junk = np.array([
        [15.0, 0.0, -1.7],      # ground
        [1.0, 0.5, -0.5],       # ego body
        [15.0, 3.0, -1.0],      # outside the corridor (adjacent lane)
        [15.0, 0.0, 2.0],       # overhead (above the ceiling)
        [-15.0, 0.0, -1.0],     # behind
        [60.0, 0.0, -1.0],      # beyond the horizon
    ])
    assert lidar_static_gap(junk, tf, route, proj) == NO_GAP
    both = np.vstack([junk, obstacle, [[35.0, 0.5, -0.5]]])
    assert abs(lidar_static_gap(both, tf, route, proj) - 17.6) < 1e-6
    # (3, N) and (N, 4) layouts are accepted
    assert abs(lidar_static_gap(both.T, tf, route, proj) - 17.6) < 1e-6
    with_i = np.hstack([both, np.ones((len(both), 1))]).astype(np.float32)
    assert abs(lidar_static_gap(with_i, tf, route, proj) - 17.6) < 1e-4
    assert lidar_static_gap(np.zeros((0, 3)), tf, route, proj) == NO_GAP


def test_lidar_gap_rotated_frame():
    # route heading +y (yaw 90); sensor-frame x forward maps to world +y
    route = _route(straight_waypoints(200.0, yaw_deg=90.0))
    _, proj = _ego_and_proj(route, 0.0, 10.0, yaw=90.0)
    tf = _sensor_tf(0.0, 10.0, yaw=90.0)
    pts = np.array([[20.0, 0.0, -1.0],       # straight ahead -> world (0, 30)
                    [10.0, 3.0, -1.0]])      # 3 m to the right -> world (-3, 20): off corridor
    assert abs(lidar_static_gap(pts, tf, route, proj) - (30.0 - 10.0 - EGO_HALF_LENGTH)) < 1e-5


def test_lidar_gap_curved_corridor_ignores_roadside_building():
    wps, opts, t0, t1 = right_turn_route(l1=40.0, radius=10.0)
    route = _route(wps, opts)
    _, proj = _ego_and_proj(route, 30.0, 0.0)
    tf = _sensor_tf(30.0, 0.0)
    building = np.array([[25.0, y, -0.7] for y in np.linspace(-1.0, 1.0, 9)])   # world x = 55, straight ahead
    assert lidar_static_gap(building, tf, route, proj) == NO_GAP
    th = math.radians(45.0)
    wx, wy = 40.0 + 10.0 * math.sin(th), 10.0 - 10.0 * math.cos(th)
    on_turn = np.array([[wx - 30.0, wy, -1.0]])
    gap = lidar_static_gap(np.vstack([building, on_turn]), tf, route, proj)
    assert abs(gap - (40.0 + 10.0 * th - 32.4)) < 0.5, gap


def test_lidar_gap_grade_change_is_not_an_obstacle():
    # flat for 20 m, then a 10 % upgrade: ground returns on the slope rise above the sensor-frame cut
    wps = straight_waypoints(200.0)
    for w in wps:
        x = w.transform.location.x
        w.transform.location.z = max(0.0, x - 20.0) * 0.10
    route = _route(wps)
    _, proj = _ego_and_proj(route, 10.0, 0.0)
    tf = _sensor_tf(10.0, 0.0, z=1.7)
    slope_ground = np.array([[x - 10.0, 0.0, max(0.0, x - 20.0) * 0.10 - 1.7] for x in (30.0, 35.0, 40.0)])
    assert np.all(slope_ground[:, 2] > -1.3)                   # would pass the plain z cut
    assert lidar_static_gap(slope_ground, tf, route, proj) == NO_GAP
    box_on_slope = np.array([[30.0, 0.0, 2.0 + 0.8 - 1.7]])     # 0.8 m above the slope at x = 40
    assert abs(lidar_static_gap(box_on_slope, tf, route, proj) - (40.0 - 12.4)) < 1e-6


def test_lidar_gap_is_fast():
    route = _route(straight_waypoints(400.0))
    _, proj = _ego_and_proj(route, 100.0, 0.0)
    tf = _sensor_tf(100.0, 0.0)
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(-50, 50, 3000), rng.uniform(-50, 50, 3000), rng.uniform(-2, 1, 3000)])
    lidar_static_gap(pts, tf, route, proj)
    t = time.perf_counter()
    for _ in range(20):
        lidar_static_gap(pts, tf, route, proj)
    per_call = (time.perf_counter() - t) / 20
    assert per_call < 0.02, per_call


def test_combine():
    actor = LeadInfo(20.0, 4.0, "vehicle", 7)
    assert combine(actor, 19.5) is actor                     # LiDAR return on the same car
    c = combine(actor, 12.0)
    assert c.kind == "static" and c.gap_m == 12.0 and c.v_lead_mps == 0.0 and c.actor_id is None
    none = LeadInfo(NO_GAP, 0.0, "none", None)
    assert combine(none, NO_GAP).kind == "none"
    assert combine(none, 30.0).kind == "static"


def test_ttc():
    assert abs(ttc(20.0, 10.0, 5.0) - 4.0) < 1e-9
    assert ttc(20.0, 5.0, 5.0) == float("inf")
    assert ttc(20.0, 3.0, 5.0) == float("inf")
    assert ttc(-1.0, 5.0, 0.0) == 0.0


if __name__ == "__main__":
    sys.exit(run_tests(dict(globals())))
