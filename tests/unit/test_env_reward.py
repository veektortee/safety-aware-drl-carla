"""Offline tests for the pure parts of tests/pipeline_carla_test.py (reward v3, terminations, nav vector,
frame-filtered event draining, spawn selection). No CARLA server is needed.

Run:  python tests/unit/test_env_reward.py      (or python -m pytest tests/unit/test_env_reward.py)

The env module imports the NAV / CBF modules owned by other branches (commons.navigation, commons.route,
commons.perception, commons.traffic_rules, commons.expert, commons.cbf_wrapper). Until those branches are
merged, the ones that do not exist yet are replaced by empty stand-ins below; the pure functions under test
do not use them.
"""
import enum
import importlib
import math
import os
import queue
import sys
import types

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

_OTHER_BRANCH_MODULES = {
    "commons.navigation": ["GlobalRoutePlanner", "RoadOption"],
    "commons.route": ["Route", "RouteProjection", "LaneStatus"],
    "commons.perception": ["LeadInfo", "find_lead", "lidar_static_gap", "combine", "ttc"],
    "commons.traffic_rules": ["TrafficLightInfo", "TrafficLightTracker", "RedLightMonitor", "StopSignMonitor",
                              "lane_invasion_is_illegal", "DurationMonitor"],
    "commons.expert": ["ExpertDriver"],
    "commons.cbf_wrapper": ["CBFSafetyLayerWrapper"],
}


def _stub_missing_modules():
    stubbed = []
    for name, attrs in _OTHER_BRANCH_MODULES.items():
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as exc:
            if exc.name != name:
                raise
            mod = types.ModuleType(name)
            for attr in attrs:
                setattr(mod, attr, type(attr, (), {}))
            sys.modules[name] = mod
            stubbed.append(name)
    return stubbed


STUBBED = _stub_missing_modules()

import numpy as np  # noqa: E402

import tests.pipeline_carla_test as env_mod  # noqa: E402
from commons.curriculum_stages import STAGES  # noqa: E402
from commons.nav_layout import NAV_DIM, NAV_FEATURES, NAV_INDEX  # noqa: E402
from commons.vehicle_model import DT  # noqa: E402

cfg = env_mod.RewardConfigV3
compute = env_mod.compute_reward_v3
L = 10.0          # speed limit used by the synthetic snapshots [m/s]
C = 100.0         # crash penalty used by the synthetic snapshots


def snap(**kw):
    """Synthetic snapshot: on the route centre, at the speed limit, 100 m to go, no events."""
    s = dict(speed=L, speed_limit=L, s=L * DT, e_lane=0.0, e_psi_lane=0.0, e_y=0.0, e_psi=0.0,
             remaining_m=100.0, crash_penalty=C, off_route_steps=0, wrong_way_steps=0, blocked_steps=0,
             episode_step=10, episode_limit=1000, collision=False, collision_pedestrian=False,
             red_light_run=False, stop_violation=False, n_illegal_invasions=0)
    s.update(kw)
    return s


PREV = dict(s=0.0, speed=L)
PREV_STOPPED = dict(s=0.0, speed=0.0)


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol


def check_components(reward, comps):
    assert set(comps) == set(env_mod.REWARD_COMPONENT_KEYS), comps.keys()
    assert close(sum(comps.values()), reward), (sum(comps.values()), reward)


# --------------------------------------------------------------------------- dense reward
def test_driving_at_limit_earns_0p8():
    r, comps, term, trunc, reason = compute(snap(), PREV, cfg)
    check_components(r, comps)
    assert close(r, 0.8), r
    assert not term and not trunc and reason is None


def test_standing_still_costs_0p2():
    r, comps, term, trunc, reason = compute(snap(speed=0.0, s=0.0), PREV_STOPPED, cfg)
    check_components(r, comps)
    assert close(r, -0.2), r
    assert not term and not trunc


def test_slower_than_limit_earns_less():
    r_half, *_ = compute(snap(speed=0.5 * L, s=0.5 * L * DT), PREV, cfg)
    assert close(r_half, 0.3), r_half
    assert r_half < 0.8


def test_speeding_never_beats_the_limit():
    r_limit, *_ = compute(snap(), PREV, cfg)
    last = r_limit
    for factor in (1.01, 1.05, 1.1, 1.2, 1.5, 2.0, 3.0):
        v = factor * L
        r, comps, *_ = compute(snap(speed=v, s=v * DT), PREV, cfg)
        check_components(r, comps)
        assert r < r_limit, (factor, r, r_limit)
        assert r <= last + 1e-9, (factor, r, last)
        last = r


def test_reversing_progress_is_negative():
    r, comps, *_ = compute(snap(speed=-L, s=-L * DT), dict(s=0.0, speed=0.0), cfg)
    assert close(comps["progress"], -1.0)
    assert close(r, -1.2), r


def test_progress_clipped_on_projection_jump():
    r, comps, *_ = compute(snap(s=25.0), PREV, cfg)
    assert close(comps["progress"], 1.0)


def test_lateral_and_heading_terms():
    # inside the 0.5 m deadband: nothing
    _, comps, *_ = compute(snap(e_lane=0.4), PREV, cfg)
    assert comps["lateral"] == 0.0
    # half way through the 1.25 m ramp
    _, comps, *_ = compute(snap(e_lane=-(0.5 + 0.625)), PREV, cfg)
    assert close(comps["lateral"], -0.1)
    # saturated
    _, comps, *_ = compute(snap(e_lane=3.0), PREV, cfg)
    assert close(comps["lateral"], -0.2)
    _, comps, *_ = compute(snap(e_psi_lane=math.pi / 4), PREV, cfg)
    assert close(comps["heading"], -0.05)
    _, comps, *_ = compute(snap(e_psi_lane=-math.pi), PREV, cfg)
    assert close(comps["heading"], -0.1)


def test_dense_penalties_capped_at_minus_0p5():
    v = 3.0 * L
    r, comps, *_ = compute(snap(speed=v, s=v * DT, e_lane=5.0, e_psi_lane=math.pi), PREV, cfg)
    check_components(r, comps)
    dense = comps["speeding"] + comps["lateral"] + comps["heading"]
    assert close(dense, -0.5), dense
    assert comps["speeding"] < 0 and comps["lateral"] < 0 and comps["heading"] < 0
    # the split is kept: uncapped values -4.0 / -0.2 / -0.1
    assert close(comps["speeding"] / comps["lateral"], 4.0 / 0.2)
    # below the cap nothing is rescaled
    _, comps, *_ = compute(snap(e_lane=5.0, e_psi_lane=math.pi), PREV, cfg)
    assert close(comps["lateral"] + comps["heading"], -0.3)


def test_illegal_lane_change_and_stop_sign_events():
    r, comps, term, trunc, _ = compute(snap(n_illegal_invasions=2), PREV, cfg)
    assert close(comps["illegal_lane"], -2.0) and close(r, 0.8 - 2.0)
    r, comps, term, trunc, _ = compute(snap(stop_violation=True), PREV, cfg)
    assert close(comps["stop_sign"], -10.0) and close(r, 0.8 - 10.0)
    assert not term and not trunc


# --------------------------------------------------------------------------- terminal events
def test_goal():
    r, comps, term, trunc, reason = compute(snap(remaining_m=2.9), PREV, cfg)
    check_components(r, comps)
    assert term and not trunc and reason == "endpoint"
    assert close(comps["goal"], 50.0) and close(r, 50.8)
    _, _, term, _, _ = compute(snap(remaining_m=3.0), PREV, cfg)
    assert not term


def test_collision_scales_with_impact_speed():
    r, comps, term, trunc, reason = compute(snap(collision=True), PREV, cfg)
    check_components(r, comps)
    assert term and not trunc and reason == "collision"
    assert close(comps["collision"], -C)                      # v = L -> 0.5 + 0.5
    _, comps, *_ = compute(snap(collision=True, speed=0.0, s=0.0), PREV_STOPPED, cfg)
    assert close(comps["collision"], -0.5 * C)                # standstill -> 0.5 C
    v = 2.5 * L
    _, comps, *_ = compute(snap(collision=True, speed=v, s=v * DT), PREV, cfg)
    assert close(comps["collision"], -1.25 * C)               # capped at min(v/L, 1.5)
    # the impact slows the car on the event tick: the pre-impact speed counts
    _, comps, *_ = compute(snap(collision=True, speed=2.0), dict(s=0.0, speed=L), cfg)
    assert close(comps["collision"], -C)


def test_pedestrian_collision_doubles():
    r, comps, term, trunc, reason = compute(snap(collision=True, collision_pedestrian=True), PREV, cfg)
    check_components(r, comps)
    assert term and reason == "collision_pedestrian"
    assert close(comps["collision"], -2.0 * C)


def test_crash_penalty_follows_stage():
    for stage in STAGES:
        _, comps, *_ = compute(snap(collision=True, crash_penalty=stage["crash_penalty"]), PREV, cfg)
        assert close(comps["collision"], -stage["crash_penalty"])


def test_red_light():
    r, comps, term, trunc, reason = compute(snap(red_light_run=True), PREV, cfg)
    check_components(r, comps)
    assert term and not trunc and reason == "red_light" and close(comps["red_light"], -C)


def test_off_route_needs_20_steps():
    _, _, term, _, _ = compute(snap(off_route_steps=19), PREV, cfg)
    assert not term
    r, comps, term, trunc, reason = compute(snap(off_route_steps=20), PREV, cfg)
    check_components(r, comps)
    assert term and not trunc and reason == "off_route" and close(comps["off_route"], -C)


def test_wrong_way_needs_40_steps():
    _, _, term, _, _ = compute(snap(wrong_way_steps=39), PREV, cfg)
    assert not term
    r, comps, term, trunc, reason = compute(snap(wrong_way_steps=40), PREV, cfg)
    check_components(r, comps)
    assert term and not trunc and reason == "wrong_way" and close(comps["wrong_way"], -C)


def test_collision_has_priority_over_goal():
    _, comps, term, _, reason = compute(snap(collision=True, remaining_m=1.0), PREV, cfg)
    assert term and reason == "collision" and comps["goal"] == 0.0


# --------------------------------------------------------------------------- truncation
def test_blocked_truncates_without_penalty():
    _, _, term, trunc, _ = compute(snap(speed=0.0, s=0.0, blocked_steps=299), PREV_STOPPED, cfg)
    assert not term and not trunc
    r, comps, term, trunc, reason = compute(snap(speed=0.0, s=0.0, blocked_steps=300), PREV_STOPPED, cfg)
    check_components(r, comps)
    assert trunc and not term and reason == "blocked"
    assert close(r, -0.2), r
    for key in ("goal", "collision", "red_light", "off_route", "wrong_way"):
        assert comps[key] == 0.0


def test_timeout_truncates():
    r, comps, term, trunc, reason = compute(snap(episode_step=1000, episode_limit=1000), PREV, cfg)
    assert trunc and not term and reason == "timeout" and close(r, 0.8)
    _, _, term, trunc, _ = compute(snap(episode_step=999, episode_limit=1000), PREV, cfg)
    assert not trunc


def test_termination_beats_truncation():
    _, _, term, trunc, reason = compute(snap(collision=True, episode_step=1000, episode_limit=1000,
                                             blocked_steps=400), PREV, cfg)
    assert term and not trunc and reason == "collision"


# --------------------------------------------------------------------------- condition flags
def flag_snap(**kw):
    s = dict(on_drivable=True, e_y=0.0, in_same_dir_lane=True, in_opposite_lane=False, speed=0.0,
             lead_gap=100.0, tl_state="none", tl_stop_gap=100.0, stop_gap=100.0)
    s.update(kw)
    return s


def test_condition_flags_off_route_and_wrong_way():
    assert env_mod.condition_flags(flag_snap(on_drivable=False), cfg)[0]
    assert not env_mod.condition_flags(flag_snap(e_y=4.0, in_same_dir_lane=True), cfg)[0]   # legal overtaking lane
    assert env_mod.condition_flags(flag_snap(e_y=-4.0, in_same_dir_lane=False), cfg)[0]
    assert not env_mod.condition_flags(flag_snap(e_y=3.0, in_same_dir_lane=False), cfg)[0]
    assert env_mod.condition_flags(flag_snap(in_opposite_lane=True), cfg)[1]
    assert not env_mod.condition_flags(flag_snap(), cfg)[1]


def test_condition_flags_blocked():
    blocked = lambda **kw: env_mod.condition_flags(flag_snap(**kw), cfg)[2]  # noqa: E731
    assert blocked()
    assert blocked(speed=-0.3)
    assert not blocked(speed=1.0)
    assert not blocked(lead_gap=10.0)                          # queued behind something
    assert not blocked(tl_state="red", tl_stop_gap=20.0)       # waiting at a red light
    assert not blocked(tl_state="yellow", tl_stop_gap=29.0)
    assert blocked(tl_state="green", tl_stop_gap=20.0)
    assert blocked(tl_state="red", tl_stop_gap=40.0)
    assert not blocked(stop_gap=5.0)                           # at a stop sign


# --------------------------------------------------------------------------- nav vector
def nav_snap(**kw):
    s = dict(speed=5.0, speed_limit=10.0, e_y=1.0, e_psi=0.3,
             route_pts=np.array([[5.0, 0.0], [10.0, 1.0], [20.0, 4.0]]), cmd="left", cmd_dist=25.0,
             lead_gap=20.0, lead_v=3.0, lead_kind="vehicle", tl_state="red", tl_stop_gap=30.0, stop_gap=100.0,
             prev_steer=0.1, prev_accel=-0.4, remaining_m=150.0, route_length=300.0)
    s.update(kw)
    return s


def test_nav_vector_layout():
    assert NAV_DIM == 26 == len(NAV_FEATURES)
    nav = env_mod.build_nav_vector(nav_snap())
    assert nav.shape == (NAV_DIM,) and nav.dtype == np.float32
    f = lambda name: float(nav[NAV_INDEX[name]])  # noqa: E731
    assert close(f("speed_over_limit"), 0.5, 1e-6)
    assert close(f("speed_limit_norm"), 10.0 / 30.0, 1e-6)
    assert close(f("lateral_error_norm"), 0.5, 1e-6)
    assert close(f("heading_err_sin"), math.sin(0.3), 1e-6)
    assert close(f("heading_err_cos"), math.cos(0.3), 1e-6)
    assert close(f("route_pt5_x"), 0.25, 1e-6) and close(f("route_pt10_y"), 0.05, 1e-6)
    assert close(f("route_pt20_x"), 1.0, 1e-6) and close(f("route_pt20_y"), 0.2, 1e-6)
    assert [f("cmd_follow"), f("cmd_left"), f("cmd_right"), f("cmd_straight")] == [0.0, 1.0, 0.0, 0.0]
    assert close(f("cmd_dist_norm"), 0.5, 1e-6)
    assert close(f("lead_gap_norm"), 0.4, 1e-6)
    assert close(f("lead_rel_speed_norm"), -0.2, 1e-6)
    assert f("lead_is_pedestrian") == 0.0
    assert [f("tl_red"), f("tl_yellow"), f("tl_green")] == [1.0, 0.0, 0.0]
    assert close(f("stop_gap_norm"), 0.6, 1e-6)
    assert close(f("prev_steer"), 0.1, 1e-6) and close(f("prev_accel"), -0.4, 1e-6)
    assert close(f("route_remaining_frac"), 0.5, 1e-6)


def test_nav_vector_edge_cases():
    f = lambda nav, name: float(nav[NAV_INDEX[name]])  # noqa: E731
    nav = env_mod.build_nav_vector(nav_snap(lead_kind="none", lead_gap=100.0, tl_state="none", stop_gap=100.0,
                                            cmd="follow", cmd_dist=50.0, speed=-2.0, e_y=10.0))
    assert f(nav, "lead_rel_speed_norm") == 0.0 and f(nav, "lead_gap_norm") == 1.0
    assert [f(nav, "tl_red"), f(nav, "tl_yellow"), f(nav, "tl_green")] == [0.0, 0.0, 0.0]
    assert f(nav, "stop_gap_norm") == 1.0                      # light gap ignored when there is no light
    assert f(nav, "cmd_follow") == 1.0 and f(nav, "cmd_dist_norm") == 1.0
    assert f(nav, "speed_over_limit") == 0.0 and f(nav, "lateral_error_norm") == 2.0
    nav = env_mod.build_nav_vector(nav_snap(lead_kind="walker", tl_state="none", stop_gap=8.0))
    assert f(nav, "lead_is_pedestrian") == 1.0 and close(f(nav, "stop_gap_norm"), 0.16, 1e-6)
    for cmd, name in (("right", "cmd_right"), ("straight", "cmd_straight")):
        nav = env_mod.build_nav_vector(nav_snap(cmd=cmd))
        assert f(nav, name) == 1.0 and f(nav, "cmd_follow") == 0.0


def test_roadoption_names():
    class RoadOption(enum.IntEnum):   # same members as commons.navigation.RoadOption
        VOID = -1
        LEFT = 1
        RIGHT = 2
        STRAIGHT = 3
        LANEFOLLOW = 4
        CHANGELANELEFT = 5
        CHANGELANERIGHT = 6
    expect = {RoadOption.VOID: "follow", RoadOption.LEFT: "left", RoadOption.RIGHT: "right",
              RoadOption.STRAIGHT: "straight", RoadOption.LANEFOLLOW: "follow",
              RoadOption.CHANGELANELEFT: "follow", RoadOption.CHANGELANERIGHT: "follow"}
    for opt, name in expect.items():
        assert env_mod.roadoption_name(opt) == name, (opt, env_mod.roadoption_name(opt))


# --------------------------------------------------------------------------- misc helpers
def test_speed_limit_and_episode_limit():
    assert close(env_mod.clip_speed_limit(30.0), 30.0 / 3.6)
    assert env_mod.clip_speed_limit(0.0) == 5.0
    assert env_mod.clip_speed_limit(200.0) == 25.0
    assert abs(env_mod.episode_step_limit(200.0, 30.0 / 3.6, 4000) - 2400) <= 1   # (200 / (0.4 * 8.33) + 60) / 0.05
    assert env_mod.episode_step_limit(80.0, 25.0, 3000) == 1360                    # (80 / 10 + 60) / 0.05
    assert env_mod.episode_step_limit(10.0, 25.0, 3000) >= 600
    assert env_mod.episode_step_limit(5000.0, 5.0, 3000) == 3000


def test_drain_ignores_previous_episode_events():
    """The confirmed reset bug: the old crash's events (one per contact frame) arrive after reset()."""
    q = queue.Queue()
    for frame in range(95, 101):                    # previous episode's crash, delivered late
        q.put((frame, "vehicle.audi.a2", 7, 1000.0))
    q.put((104, "static.pole", 9, 10.0))            # during the settle ticks
    start_frame = 106                               # recorded after the settle ticks
    assert env_mod.drain_frame_events(q, start_frame, 107) == []
    q.put((107, "vehicle.audi.a2", 7, 500.0))
    q.put((108, "vehicle.audi.a2", 7, 500.0))       # newer than the current frame -> kept for later
    events = env_mod.drain_frame_events(q, start_frame, 107)
    assert [e[0] for e in events] == [107]
    assert [e[0] for e in env_mod.drain_frame_events(q, start_frame, 108)] == [108]
    assert q.empty()


def test_wait_for_frame():
    q = queue.Queue()
    for frame in (5, 6, 7):
        q.put((frame, f"img{frame}"))
    assert env_mod.wait_for_frame(q, 7, 0.1) == (7, "img7")
    assert env_mod.wait_for_frame(q, 8, 0.05) is None


def test_select_spawn_index():
    spawns = [(20.0 * i, 0.0, 0.0) for i in range(6)]      # x = 0, 20, ..., 100
    order = list(range(6))
    idx, ok = env_mod.select_spawn_index(spawns, [(0.0, 3.0, 0.0)], [], None, order, 10.0, 5.0, 50.0)
    assert (idx, ok) == (1, True)
    idx, ok = env_mod.select_spawn_index(spawns, [], [], (20.0, 0.0, 0.0), order, 10.0, 5.0, 50.0)
    assert (idx, ok) == (4, True)                          # first point >= 50 m from the crash
    idx, ok = env_mod.select_spawn_index(spawns, [], [(80.0, 2.0, 0.0)], (20.0, 0.0, 0.0), order, 10.0, 5.0, 50.0)
    assert (idx, ok) == (5, True)                          # walker 2 m from index 4
    idx, ok = env_mod.select_spawn_index(spawns, [(x, 0.0, 0.0) for x, _, _ in spawns], [], None, [3, 0], 10.0,
                                         5.0, 50.0)
    assert not ok and idx in (3, 0)
    # deterministic evaluation order ignores the crash rule (min_crash_m = 0)
    idx, ok = env_mod.select_spawn_index(spawns, [], [], (40.0, 0.0, 0.0), [2, 3, 4, 5, 0, 1], 10.0, 5.0, 0.0)
    assert (idx, ok) == (2, True)


def test_cbf_gap_stop_line():
    # red light 20 m ahead, 5 m/s: stoppable (1.75 + 2.4 m < 20 m) -> the stop line is the obstacle
    assert env_mod.cbf_gap(100.0, 0.0, "none", "red", 20.0, 5.0) == (20.0, 0.0, "stop_line")
    # 20 m/s cannot stop within 20 m -> keep the lead (do not demand an impossible stop)
    assert env_mod.cbf_gap(100.0, 0.0, "none", "red", 20.0, 20.0) == (100.0, 0.0, "none")
    assert env_mod.cbf_gap(100.0, 0.0, "none", "green", 20.0, 5.0) == (100.0, 0.0, "none")
    assert env_mod.cbf_gap(12.0, 4.0, "vehicle", "yellow", 20.0, 5.0) == (12.0, 4.0, "vehicle")


# --------------------------------------------------------------------------- offline env (fake CARLA world)
# Runs the real CarlaGymEnv __init__ / reset / step against a fake world: a straight 100 m route along +x,
# a point-mass ego, and the NAV/CBF/perception functions replaced by simple fakes. It exercises the glue
# (reset sequence, frame filtering, snapshot, info dicts) without a CARLA server.
import carla  # noqa: E402
from types import SimpleNamespace  # noqa: E402


class _Opt(enum.IntEnum):
    LANEFOLLOW = 4


class FakeActor:
    def __init__(self, actor_id, type_id, x, y=0.0):
        self.id, self.type_id = actor_id, type_id
        self._loc = carla.Location(x=x, y=y, z=0.0)

    def get_location(self):
        return self._loc


class FakeActorList(list):
    def filter(self, pattern):
        prefix = pattern.rstrip("*")
        return FakeActorList(a for a in self if a.type_id.startswith(prefix))


class FakeEgo:
    id = 1
    type_id = "vehicle.tesla.model3"

    def __init__(self):
        self.x, self.y, self.v = 0.0, 0.0, 0.0
        self.controls, self.zeroed, self._pending = [], 0, None

    def apply_control(self, control):
        self.controls.append(control)

    def set_target_velocity(self, vel):
        self.zeroed += 1
        self.v = 0.0

    def set_target_angular_velocity(self, w):
        pass

    def set_transform(self, tf):
        self._pending = tf          # like TeleportPhysics: visible after the next tick

    def get_transform(self):
        return carla.Transform(carla.Location(x=self.x, y=self.y, z=0.0), carla.Rotation(yaw=0.0))

    def get_location(self):
        return carla.Location(x=self.x, y=self.y, z=0.0)

    def get_velocity(self):
        return carla.Vector3D(self.v, 0.0, 0.0)

    def get_speed_limit(self):
        return 30.0


class FakeWorld:
    def __init__(self, env):
        self.env, self.ego, self.frame = env, FakeEgo(), 1000
        self.others = FakeActorList()
        self.scheduled = {}         # frame -> [(queue_attr, event_tuple_without_frame)]

    def tick(self):
        ego = self.ego
        if ego._pending is not None:
            ego.x, ego.y, ego._pending = ego._pending.location.x, ego._pending.location.y, None
        c = ego.controls[-1] if ego.controls else carla.VehicleControl()
        accel = 4.0 * c.throttle - 6.0 * c.brake
        ego.v = max(0.0, ego.v + accel * DT)
        ego.x += ego.v * DT
        self.frame += 1
        self.env._q_rgb.put((self.frame, np.zeros((360, 640, 3), np.uint8)))
        self.env._q_lidar.put((self.frame, np.zeros((0, 3), np.float32), carla.Transform()))
        for q_attr, ev in self.scheduled.pop(self.frame, []):
            getattr(self.env, q_attr).put((self.frame,) + tuple(ev))
        return self.frame

    def get_actors(self, ids=None):
        return FakeActorList([self.ego] + list(self.others))


class FakeRoute:
    def __init__(self, x0, length=100.0):
        self.x0, self.length_m = x0, length

    def project(self, loc, yaw, hint_idx=None):
        s = loc.x - self.x0
        return SimpleNamespace(s=s, e_y=loc.y, e_psi=0.0, kappa=0.0, idx=max(int(s // 2), 0),
                               lane_width=3.5, remaining_m=self.length_m - s)

    def lane_status(self, loc, yaw, wmap):
        return SimpleNamespace(on_drivable=True, in_route_lane=True, in_same_dir_lane=True,
                               in_opposite_lane=False, e_lane=loc.y, lane_heading_err=0.0, is_junction=False)

    def lane_bounds(self, s, wmap, half_width):
        return 1.5, 1.5

    def points_ahead_ego(self, s, tf, distances=(5.0, 10.0, 20.0)):
        return np.array([[d, 0.0] for d in distances])

    def next_maneuver(self, s, horizon_m=50.0):
        return _Opt.LANEFOLLOW, horizon_m


def _none_lead():
    return SimpleNamespace(gap_m=100.0, v_lead_mps=0.0, kind="none", actor_id=None)


_FAKES = dict(
    GlobalRoutePlanner=lambda wmap, res: object(),
    Route=SimpleNamespace(plan=lambda grp, wmap, start, sps, mn, mx, rng: FakeRoute(start.x)),
    find_lead=lambda actors, ego, route, proj, *a, **k: _none_lead(),
    lidar_static_gap=lambda pts, tf, route, proj, *a, **k: 100.0,
    combine=lambda lead, gap: lead,
    time_to_collision=lambda gap, v, vl: float("inf"),
    TrafficLightTracker=lambda world, wmap: SimpleNamespace(
        query=lambda ego, route, proj, max_dist=50.0: SimpleNamespace(state="none", stop_gap_m=100.0, light_id=None)),
    RedLightMonitor=lambda: SimpleNamespace(update=lambda tl, proj, v: False, reset=lambda: None),
    StopSignMonitor=lambda world, wmap: SimpleNamespace(update=lambda ego, route, proj, v: (100.0, False),
                                                        reset=lambda: None),
    lane_invasion_is_illegal=lambda markings, before, after: any(m == "Solid" for m in markings),
    ExpertDriver=lambda: SimpleNamespace(act=lambda *a: np.array([0.0, 0.5])),
)


class OfflineEnv(env_mod.CarlaGymEnv):
    def _connect_carla(self):
        self.world = FakeWorld(self)
        self.map = SimpleNamespace(get_spawn_points=lambda: [
            carla.Transform(carla.Location(x=100.0 * i, y=0.0, z=0.3)) for i in range(6)])
        self.client = SimpleNamespace(apply_batch_sync=lambda batch, tick=False: [])
        self.spectator = SimpleNamespace(set_transform=lambda tf: setattr(self, "_spectator_tf", tf))

    def _setup_world(self):
        pass

    def _clear_map(self):
        pass

    def _spawn_ego(self):
        self.ego_vehicle = self.world.ego

    def _attach_sensors(self):
        pass


def _with_fakes(fn):
    def run():
        saved = {k: getattr(env_mod, k) for k in _FAKES}
        try:
            for k, v in _FAKES.items():
                setattr(env_mod, k, v)
            fn()
        finally:
            for k, v in saved.items():
                setattr(env_mod, k, v)
    run.__name__ = fn.__name__
    return run


CBF_KEYS = {"speed", "speed_limit", "gap", "v_lead", "gap_kind", "e_y", "e_psi", "kappa_ref",
            "y_max_left", "y_max_right", "d_collision", "ttc"}
EPISODE_KEYS = {"return", "length_steps", "distance_m", "mean_speed_mps", "route_length_m",
                "route_completion_frac", "end_reason", "terminated", "collisions", "collisions_pedestrian",
                "lane_invasions", "illegal_lane_invasions", "red_light_violations", "stop_sign_violations",
                "speeding_frac", "min_ttc_s", "min_gap_m", "curriculum_stage", "reward_components", "r", "l", "t"}
STEP_KEYS = {"reward_components", "terminated_reason", "truncated_reason", "speed_mps", "speed_limit_mps",
             "route_completion_frac", "curriculum_stage"}


@_with_fakes
def test_offline_env_reset_sequence_and_spaces():
    env = OfflineEnv(num_npc_vehicles=0, num_pedestrians=0, seed=0)
    assert env.action_space.shape == (2,) and float(env.action_space.low[1]) == -1.0
    assert env.observation_space["nav"].shape == (NAV_DIM,)
    env.world.ego.v = 0.53                     # leftover speed from the previous episode
    obs, info = env.reset()
    assert obs["rgb_data"].shape == (360, 640, 3) and obs["nav"].shape == (NAV_DIM,)
    assert env.observation_space.contains(obs)
    assert env.world.ego.v == 0.0 and env.world.ego.zeroed >= 2
    ctrls = env.world.ego.controls
    settle = [c for c in ctrls if c.manual_gear_shift]
    assert len(settle) >= env.SETTLE_TICKS and all(c.gear == 1 and c.brake == 1.0 for c in settle)
    assert not ctrls[-1].manual_gear_shift      # automatic gearbox again for normal control
    assert env._episode_start_frame == env.world.frame
    assert hasattr(env, "_spectator_tf")
    assert set(env.build_cbf_state()) >= CBF_KEYS
    assert info["curriculum_stage"] == 0 and info["episode_limit"] >= 600
    assert close(info["speed_limit_mps"], 30.0 / 3.6)


@_with_fakes
def test_offline_env_stale_crash_is_ignored_and_real_crash_terminates():
    env = OfflineEnv(num_npc_vehicles=0, num_pedestrians=0, seed=0)
    env.reset()
    start = env._episode_start_frame
    for frame in range(start - 5, start + 1):  # previous crash, delivered after reset()
        env._q_collision.put((frame, "vehicle.audi.a2", 7, 900.0))
    obs, r, term, trunc, info = env.step(np.array([0.0, 1.0], np.float32))
    assert not term and not trunc and info["terminated_reason"] is None, info
    assert set(info) >= STEP_KEYS
    for _ in range(20):
        obs, r, term, trunc, info = env.step(np.array([0.0, 1.0], np.float32))
    assert r > 0.0 and info["speed_mps"] > 1.0
    env.world.scheduled[env.world.frame + 1] = [("_q_collision", ("vehicle.audi.a2", 7, 900.0))]
    obs, r, term, trunc, info = env.step(np.array([0.0, 1.0], np.float32))
    assert term and not trunc and info["terminated_reason"] == "collision"
    ep = info["episode_metrics"]
    assert ep is info["episode"] and set(ep) >= EPISODE_KEYS
    assert ep["end_reason"] == "collision" and ep["terminated"] and ep["collisions"] == 1
    assert ep["l"] == ep["length_steps"] == 22 and close(ep["r"], ep["return"])
    assert close(sum(ep["reward_components"].values()), ep["return"], 1e-6)
    crash_x = env._last_crash_xyz[0]
    env.reset()
    assert abs(env.world.ego.x - crash_x) >= 50.0      # respawn away from the crash


@_with_fakes
def test_offline_env_actions_nav_and_events():
    env = OfflineEnv(num_npc_vehicles=0, num_pedestrians=0, seed=1)
    env.reset(options={"spawn_index": 3, "route_seed": 1})
    assert env._spawn_index == 3 and close(env.world.ego.x, 300.0)
    obs, *_ = env.step(np.array([0.3, -0.5], np.float32))
    c = env.world.ego.controls[-1]
    assert close(c.steer, 0.3) and c.throttle == 0.0 and close(c.brake, 0.5)
    assert close(float(obs["nav"][NAV_INDEX["prev_steer"]]), 0.3) and close(float(obs["nav"][NAV_INDEX["prev_accel"]]), -0.5)
    env.step(np.array([0.0, 0.05], np.float32))          # small throttle is applied as is (no deadzone)
    c = env.world.ego.controls[-1]
    assert close(c.throttle, 0.05) and c.brake == 0.0
    env.world.scheduled[env.world.frame + 1] = [("_q_lane", (["Solid"],)), ("_q_lane", (["Broken"],))]
    _, r, _, _, info = env.step(np.array([0.0, 0.5], np.float32))
    assert close(info["reward_components"]["illegal_lane"], -1.0)
    a = env.expert_action()
    assert a.shape == (2,) and a.dtype == np.float32


@_with_fakes
def test_offline_env_goal_and_blocked():
    env = OfflineEnv(num_npc_vehicles=0, num_pedestrians=0, seed=2)
    env.reset()
    for _ in range(400):
        _, r, term, trunc, info = env.step(np.array([0.0, 1.0], np.float32))
        if term or trunc:
            break
    assert term and info["terminated_reason"] == "endpoint"
    assert info["episode_metrics"]["route_completion_frac"] == 1.0
    assert info["reward_components"]["goal"] == 50.0
    env.reset()
    for step in range(1, 400):
        _, r, term, trunc, info = env.step(np.array([0.0, -1.0], np.float32))
        if term or trunc:
            break
    assert trunc and not term and step == 300 and info["truncated_reason"] == "blocked"
    assert info["episode_metrics"]["end_reason"] == "blocked" and not info["episode_metrics"]["terminated"]


@_with_fakes
def test_offline_env_curriculum_stage_change():
    env = OfflineEnv(num_npc_vehicles=40, num_pedestrians=20, seed=3)
    calls = []
    env._spawn_npcs = lambda n: calls.append(("npc", n))
    env._spawn_walkers = lambda n: calls.append(("walker", n))
    env.reset()
    assert calls == []                                   # stage 0: still the empty world
    env.set_curriculum_stage(2)
    assert env._active_stage == 0                        # applies at the next reset
    _, info = env.reset()
    # the fake map has 6 spawn points: no room for NPC cars (spawn-point reserve), walkers as staged
    assert calls == [("npc", 0), ("walker", 15)] and info["curriculum_stage"] == 2
    original_cap = env_mod.vehicle_cap
    env_mod.vehicle_cap = lambda n: 1000                 # a big map
    try:
        env.set_curriculum_stage(3)
        env.reset()
        assert calls[-2:] == [("npc", 40), ("walker", 20)]   # stage 60/60 capped by the constructor's 40/20
        env.reset()
        assert len(calls) == 4                               # unchanged counts: no respawn
    finally:
        env_mod.vehicle_cap = original_cap
    env.world.scheduled[env.world.frame + 1] = [("_q_collision", ("walker.pedestrian.0001", 9, 50.0))]
    _, r, term, _, info = env.step(np.array([0.0, 0.0], np.float32))
    assert term and info["terminated_reason"] == "collision_pedestrian"
    assert close(info["reward_components"]["collision"], -2.0 * STAGES[3]["crash_penalty"] * 0.5)


@_with_fakes
def test_offline_sensor_callbacks():
    env = OfflineEnv(num_npc_vehicles=0, num_pedestrians=0, seed=4)
    for q in (env._q_rgb, env._q_lidar, env._q_collision, env._q_lane):
        while not q.empty():
            q.get_nowait()
    bgra = np.zeros((360, 640, 4), np.uint8)
    bgra[..., 0], bgra[..., 2], bgra[..., 3] = 10, 200, 255
    env._on_rgb_image(SimpleNamespace(raw_data=bgra.tobytes(), height=360, width=640, frame=77))
    frame, img = env._q_rgb.get_nowait()
    assert frame == 77 and img.shape == (360, 640, 3) and img[0, 0, 0] == 10 and img[0, 0, 2] == 200
    pts = np.array([[10.0, 0.0, -1.7, 1.0],     # road return       -> dropped
                    [1.0, 0.5, 0.0, 1.0],       # ego body          -> dropped
                    [10.0, 0.5, -0.5, 1.0],     # obstacle ahead    -> kept
                    [-20.0, 3.0, 1.0, 1.0]],    # behind, off-axis  -> kept (perception decides)
                   dtype=np.float32)
    tf = carla.Transform(carla.Location(z=1.7))
    env._on_lidar_data(SimpleNamespace(raw_data=pts.tobytes(), frame=78, transform=tf))
    frame, kept, sensor_tf = env._q_lidar.get_nowait()
    assert frame == 78 and kept.shape == (2, 3) and sensor_tf is tf
    assert np.allclose(kept[:, 0], [10.0, -20.0])
    env._on_collision(SimpleNamespace(frame=79, other_actor=SimpleNamespace(type_id="walker.pedestrian.0003", id=5),
                                      normal_impulse=carla.Vector3D(3.0, 4.0, 0.0)))
    assert env._q_collision.get_nowait() == (79, "walker.pedestrian.0003", 5, 5.0)
    env._on_lane_invasion(SimpleNamespace(frame=80, crossed_lane_markings=("m1", "m2")))
    assert env._q_lane.get_nowait() == (80, ["m1", "m2"])


def _run_all():
    tests = [(n, fn) for n, fn in sorted(globals().items()) if n.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    if STUBBED:
        print(f"(stand-ins used for modules not on this branch yet: {', '.join(STUBBED)})")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
