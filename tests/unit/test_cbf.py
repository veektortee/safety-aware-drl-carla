"""Offline unit tests for the discrete-time CBF safety layer, its gym wrapper and the critic trust score.

Run with either
    python -m pytest tests/unit/test_cbf.py -q
    python tests/unit/test_cbf.py            (plain python; also prints the closed-loop stopping table)
No CARLA connection is made.
"""
import math
import os
import sys
import tempfile
from collections import deque

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# The repository root has an __init__.py that does `import examples` (a package that no longer exists).
# pytest (rootdir = repo root) imports that __init__.py when it sets up the root package node, which
# would error every test. Register an empty stand-in so `python -m pytest` works from the repo root.
try:
    import examples  # noqa: F401
except ImportError:
    import types
    sys.modules["examples"] = types.ModuleType("examples")

import numpy as np  # noqa: E402

from commons.cbfQP_layer import CBFSafetyLayer, _solve_1d  # noqa: E402
from commons.vehicle_model import AccelModel, STEER_GAIN  # noqa: E402

try:
    import pytest
    _skip = pytest.skip
except ImportError:  # plain-python runner
    class _Skip(Exception):
        pass

    def _skip(msg):
        raise _Skip(msg)


DT = 0.05
FREE = dict(speed=8.0, speed_limit=13.9, gap=100.0, v_lead=0.0, gap_kind="none", e_y=0.0, e_psi=0.0,
            kappa_ref=0.0, y_max_left=1.0, y_max_right=1.0, d_collision=100.0, ttc=math.inf)


def _state(**kw):
    s = dict(FREE)
    s.update(kw)
    if "gap" in kw and "d_collision" not in kw:
        s["d_collision"] = kw["gap"]
    return s


# ---------------------------------------------------------------------------- closed-loop 1-D simulator

def simulate(v0, gap0, v_lead0=0.0, lead_decel=0.0, lead_brake_t=0.0, trust=1.0, v_lim=25.0, kind="vehicle",
             T=40.0, brake_delay_s=0.35, delay_mode="brake", policy_u=1.0):
    """Ego follows a policy that always demands `policy_u` (full throttle) through the CBF.

    Plant: commons.vehicle_model.AccelModel plus an actuator delay of brake_delay_s (7 steps):
      delay_mode='brake' (measured CARLA behaviour): throttle acts immediately; a brake command takes
                         effect brake_delay_s after it was issued (the car coasts in between);
      delay_mode='all'   (stress test): the whole command is a pure transport delay, i.e. the throttle
                         also keeps acting for brake_delay_s after the CBF cut it.
    The lead keeps v_lead0 and brakes at lead_decel from t = lead_brake_t until it stops.
    """
    cbf = CBFSafetyLayer()
    model = AccelModel()
    n = int(round(brake_delay_s / DT))
    pending = deque([policy_u] * n)
    v, xe, xl, vl = float(v0), 0.0, float(gap0), float(v_lead0)
    min_gap, first_active, v_at_onset, n_infeasible = gap0, None, None, 0
    for k in range(int(T / DT)):
        t = k * DT
        gap = xl - xe
        st = _state(speed=v, speed_limit=v_lim, gap=gap, v_lead=vl, gap_kind=kind)
        u = cbf.compute_safe_action(np.array([0.0, policy_u], dtype=np.float32), st, trust)
        info = cbf.last_info
        n_infeasible += info["infeasible"]
        if first_active is None and "collision" in info["active_rows"]:
            first_active, v_at_onset = gap, v
        cmd = float(u[1])
        pending.append(cmd)
        delayed = pending.popleft()
        if delay_mode == "all":
            applied = delayed
        elif cmd >= 0.0:
            applied = cmd
        else:
            applied = delayed if delayed < 0.0 else 0.0
        a = model.accel(applied, v)
        v_new = max(0.0, v + a * DT)
        xe += 0.5 * (v + v_new) * DT
        v = v_new
        vl_new = max(0.0, vl - lead_decel * DT) if (lead_decel > 0.0 and t >= lead_brake_t) else vl
        xl += 0.5 * (vl + vl_new) * DT
        vl = vl_new
        min_gap = min(min_gap, xl - xe)
    return dict(min_gap=min_gap, final_gap=xl - xe, v_end=v, first_active_gap=first_active,
                v_at_onset=v_at_onset, n_infeasible=n_infeasible)


STATIONARY_CASES = [(v0, v_lim, gap0) for v0 in (8.0, 12.0) for v_lim in (v0, 25.0) for gap0 in (40.0, 60.0, 100.0)]
LEAD_CASES = [(gap0, tb) for gap0 in (15.0, 20.0, 30.0) for tb in (0.0, 3.0)]


# ---------------------------------------------------------------------------- tests: layer

def test_free_road_passthrough_bit_exact():
    rng = np.random.default_rng(0)
    for dtype in (np.float32, np.float64):
        cbf = CBFSafetyLayer()
        for _ in range(1000):
            u = np.array([rng.uniform(-0.1, 0.1), rng.uniform(-1.0, 1.0)], dtype=dtype)
            out = cbf.compute_safe_action(u, FREE, trust_score=1.0)
            assert out.dtype == u.dtype and out.shape == u.shape
            assert np.array_equal(out, u), (u, out)
            assert out is not u
            assert cbf.last_info["active"] is False and cbf.last_info["active_rows"] == []
            assert cbf.correction_magnitude == 0.0
        assert cbf.correction_count == 0


def test_free_road_low_trust_still_passthrough():
    cbf = CBFSafetyLayer()
    u = np.array([0.05, 0.8], dtype=np.float32)
    for tau in (0.0, 0.3, 1.0):
        assert np.array_equal(cbf.compute_safe_action(u, FREE, trust_score=tau), u)


def test_closed_loop_stationary_obstacle_stops_with_margin():
    for v0, v_lim, gap0 in STATIONARY_CASES:
        r = simulate(v0, gap0, v_lim=v_lim)
        assert r["v_end"] == 0.0, (v0, v_lim, gap0, r)
        assert r["min_gap"] >= 1.0, (v0, v_lim, gap0, r)
        assert r["n_infeasible"] == 0, (v0, v_lim, gap0, r)


def test_closed_loop_lead_braking_6mps2():
    for gap0, tb in LEAD_CASES:
        r = simulate(10.0, gap0, v_lead0=10.0, lead_decel=6.0, lead_brake_t=tb, v_lim=13.9)
        assert r["v_end"] == 0.0, (gap0, tb, r)
        assert r["min_gap"] >= 1.0, (gap0, tb, r)


def test_closed_loop_full_transport_delay_never_collides():
    """Stress test: throttle ALSO lags 0.35 s. The loop oscillates but must never close the gap."""
    for v0, v_lim, gap0 in STATIONARY_CASES:
        r = simulate(v0, gap0, v_lim=v_lim, delay_mode="all")
        assert r["min_gap"] > 0.5, (v0, v_lim, gap0, r)
    for gap0, tb in LEAD_CASES:
        r = simulate(10.0, gap0, v_lead0=10.0, lead_decel=6.0, lead_brake_t=tb, v_lim=13.9, delay_mode="all")
        assert r["min_gap"] > 0.5, (gap0, tb, r)


def test_closed_loop_walker_and_stop_line_standoff():
    walker = simulate(12.0, 60.0, kind="walker")
    vehicle = simulate(12.0, 60.0, kind="vehicle")
    stop_line = simulate(12.0, 60.0, kind="stop_line")
    assert walker["min_gap"] >= 2.5 and walker["min_gap"] > vehicle["min_gap"] > stop_line["min_gap"] > 0.5


def test_lane_right_drift_steers_left():
    cbf = CBFSafetyLayer()
    st = _state(speed=8.0, e_y=0.6, e_psi=0.08)          # right of centre, heading further right
    u = np.array([0.05, 0.2])
    out = cbf.compute_safe_action(u, st)
    assert out[0] < u[0], out
    assert "lane_right" in cbf.last_info["active_rows"]
    assert out[1] == u[1]                                 # longitudinal axis untouched (bit-exact)


def test_lane_left_drift_steers_right():
    cbf = CBFSafetyLayer()
    st = _state(speed=8.0, e_y=-0.6, e_psi=-0.08)
    u = np.array([-0.05, 0.2])
    out = cbf.compute_safe_action(u, st)
    assert out[0] > u[0], out
    assert "lane_left" in cbf.last_info["active_rows"]
    assert out[1] == u[1]


def test_lane_curve_tracking_not_flagged():
    """Following a right curve with the matching steer must not trigger the lane rows."""
    cbf = CBFSafetyLayer()
    kappa = 0.05
    st = _state(speed=6.0, kappa_ref=kappa)
    u = np.array([kappa / STEER_GAIN, 0.2])
    assert np.array_equal(cbf.compute_safe_action(u, st), u)


def test_no_lane_rows_below_1mps():
    cbf = CBFSafetyLayer()
    st = _state(speed=0.8, e_y=1.5, e_psi=0.4)           # outside the lane and heading out
    u = np.array([0.6, 0.3])
    out = cbf.compute_safe_action(u, st)
    assert np.array_equal(out, u)
    assert not any(r.startswith("lane") for r in cbf.last_info["active_rows"])


def test_lane_closed_loop_holds_bound():
    for v in (3.0, 6.0, 10.0, 14.0):
        for s_cmd in (0.3, -0.3, 1.0):
            cbf = CBFSafetyLayer()
            e_y = e_psi = 0.0
            worst = 0.0
            for _ in range(400):
                st = _state(speed=v, speed_limit=20.0, e_y=e_y, e_psi=e_psi, y_max_left=0.87, y_max_right=0.87)
                s = float(cbf.compute_safe_action(np.array([s_cmd, 0.3]), st)[0])
                e_y += v * DT * math.sin(e_psi)
                e_psi += v * DT * STEER_GAIN * s
                worst = max(worst, abs(e_y))
            assert worst <= 0.87 + 1e-6, (v, s_cmd, worst)


def test_speed_row_reduces_accel():
    cbf = CBFSafetyLayer()
    st = _state(speed=15.5, speed_limit=13.9)            # above limit + dv
    u = np.array([0.0, 0.8])
    out = cbf.compute_safe_action(u, st)
    assert out[1] < u[1]
    assert cbf.last_info["active_rows"] == ["speed"]
    assert cbf.last_info["a_safe"] < cbf.last_info["a_ref"]
    assert out[0] == u[0]


def test_speed_row_inactive_under_limit():
    cbf = CBFSafetyLayer()
    st = _state(speed=10.0, speed_limit=13.9)
    u = np.array([0.0, 1.0])
    assert np.array_equal(cbf.compute_safe_action(u, st), u)


def test_infeasible_full_brake():
    cbf = CBFSafetyLayer()
    st = _state(speed=10.0, gap=3.0, gap_kind="vehicle")
    u = np.array([0.03, 0.5])
    out = cbf.compute_safe_action(u, st)
    assert out[1] == -1.0
    assert cbf.last_info["infeasible"] is True and cbf.last_info["active"] is True
    assert "collision" in cbf.last_info["active_rows"]
    assert out[0] == u[0]                                 # lateral solution kept
    assert cbf.collision_prevented is False
    assert cbf.infeasible_count == 1


def test_standstill_inside_d0_holds_brake_without_infeasible_flag():
    cbf = CBFSafetyLayer()
    out = cbf.compute_safe_action(np.array([0.0, 0.7]), _state(speed=0.0, gap=1.2, gap_kind="vehicle"))
    assert out[1] == -1.0 and cbf.last_info["hold"] is True and cbf.last_info["infeasible"] is False


def test_collision_row_brakes_and_reports_prevention():
    cbf = CBFSafetyLayer()
    st = _state(speed=10.0, gap=25.0, gap_kind="vehicle")
    u = np.array([0.0, 1.0])
    out = cbf.compute_safe_action(u, st)
    info = cbf.last_info
    assert "collision" in info["active_rows"] and not info["infeasible"]
    assert out[1] < u[1] and info["a_safe"] <= info["U_col"] + 1e-9
    assert cbf.collision_prevented is True
    assert abs(info["correction"] - abs(float(out[1]) - 1.0)) < 1e-9


def test_lower_trust_brakes_earlier_monotone():
    """Largest gap at which the collision row engages (v = 10 m/s, full throttle) grows as trust drops."""
    gaps = np.arange(99.0, 0.0, -0.05)
    onset = []
    for tau in (1.0, 0.75, 0.5, 0.25, 0.0):
        cbf = CBFSafetyLayer()
        first = None
        for g in gaps:
            cbf.compute_safe_action(np.array([0.0, 1.0]), _state(speed=10.0, gap=float(g), gap_kind="vehicle"), tau)
            if "collision" in cbf.last_info["active_rows"]:
                first = float(g)
                break
        onset.append(first)
    assert all(b >= a for a, b in zip(onset, onset[1:])), onset
    assert onset[-1] > onset[0] + 1.0, onset
    # closed loop: lower trust also stops further away
    stops = [simulate(12.0, 60.0, trust=tau)["final_gap"] for tau in (1.0, 0.5, 0.0)]
    assert stops[0] < stops[1] < stops[2], stops


def test_oncoming_lead_more_conservative_than_stationary():
    cbf = CBFSafetyLayer()
    cbf.compute_safe_action(np.array([0.0, 1.0]), _state(speed=8.0, gap=40.0, gap_kind="vehicle", v_lead=0.0))
    u_static = cbf.last_info["U_col"]
    cbf.compute_safe_action(np.array([0.0, 1.0]), _state(speed=8.0, gap=40.0, gap_kind="vehicle", v_lead=-5.0))
    u_oncoming = cbf.last_info["U_col"]
    cbf.compute_safe_action(np.array([0.0, 1.0]), _state(speed=8.0, gap=40.0, gap_kind="vehicle", v_lead=8.0))
    u_moving = cbf.last_info["U_col"]
    assert u_oncoming < u_static < u_moving


def test_old_style_state_does_not_crash_and_brakes():
    cbf = CBFSafetyLayer()
    u = np.array([0.0, 0.6], dtype=np.float32)
    old_far = {"d_collision": 100.0, "speed": 8.0, "speed_limit": 13.9, "lane_offset": 0.4}
    assert np.array_equal(cbf.compute_safe_action(u, old_far), u)
    old_near = {"d_collision": 15.0, "speed": 8.0, "speed_limit": 13.9, "lane_offset": 0.4}
    out = cbf.compute_safe_action(u, old_near)
    assert out[1] < u[1] and "collision" in cbf.last_info["active_rows"]
    old_tiny = {"d_collision": 2.0, "speed": 8.0, "speed_limit": 13.9, "lane_offset": -0.4}
    assert cbf.compute_safe_action(u, old_tiny)[1] == -1.0
    # completely empty / garbage states must not crash either
    cbf.compute_safe_action(u, {})
    cbf.compute_safe_action(u, {"speed": float("nan"), "gap": float("inf"), "e_y": None, "gap_kind": "???"})


def test_output_never_increases_accel_and_unconstrained_axis_is_exact():
    rng = np.random.default_rng(1)
    cbf = CBFSafetyLayer()
    for _ in range(3000):
        st = _state(speed=float(rng.uniform(0.0, 20.0)), speed_limit=float(rng.uniform(5.0, 20.0)),
                    gap=float(rng.choice([100.0, rng.uniform(0.0, 80.0)])), v_lead=float(rng.uniform(-5.0, 15.0)),
                    gap_kind=str(rng.choice(["vehicle", "walker", "static", "stop_line"])),
                    e_y=float(rng.uniform(-1.5, 1.5)), e_psi=float(rng.uniform(-0.5, 0.5)),
                    kappa_ref=float(rng.uniform(-0.05, 0.05)), y_max_left=float(rng.uniform(0.5, 4.0)),
                    y_max_right=float(rng.uniform(0.5, 4.0)))
        u = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
        out = cbf.compute_safe_action(u, st, trust_score=float(rng.uniform(0.0, 1.0)))
        rows = cbf.last_info["active_rows"]
        assert np.all(np.abs(out) <= 1.0)
        assert out[1] <= u[1]
        if not ({"collision", "speed"} & set(rows)):
            assert out[1] == u[1]
        if not ({"lane_left", "lane_right"} & set(rows)):
            assert out[0] == u[0]
        if not rows:
            assert np.array_equal(out, u)


def test_closed_form_matches_osqp():
    try:
        import osqp  # noqa: F401
    except ImportError:
        _skip("osqp not installed")
    rng = np.random.default_rng(2)
    cf, qp = CBFSafetyLayer(), CBFSafetyLayer(solver="osqp")
    compared = 0
    for _ in range(400):
        st = _state(speed=float(rng.uniform(1.0, 16.0)), speed_limit=float(rng.uniform(5.0, 16.0)),
                    gap=float(rng.uniform(5.0, 60.0)), gap_kind="vehicle", v_lead=float(rng.uniform(0.0, 10.0)),
                    e_y=float(rng.uniform(-1.2, 1.2)), e_psi=float(rng.uniform(-0.3, 0.3)),
                    kappa_ref=float(rng.uniform(-0.03, 0.03)), y_max_left=float(rng.uniform(0.3, 1.5)),
                    y_max_right=float(rng.uniform(0.3, 1.5)))
        u = rng.uniform(-1.0, 1.0, size=2)
        a = cf.compute_safe_action(u, st)
        b = qp.compute_safe_action(u, st)
        assert cf.last_info["active_rows"] == qp.last_info["active_rows"]
        if cf.last_info["active"]:
            compared += 1
            assert np.allclose(a, b, atol=2e-3), (st, u, a, b)
    assert compared > 50 and qp.osqp_failures == 0


def test_solve_1d_matches_brute_force():
    rng = np.random.default_rng(3)
    for _ in range(200):
        rows = [(float(rng.uniform(-2, 2)), float(rng.uniform(-1, 1)), float(rng.uniform(0, 100)),
                 float(rng.uniform(0, 10))) for _ in range(int(rng.integers(0, 4)))]
        ref, w = float(rng.uniform(-2, 2)), float(rng.uniform(0.1, 30))
        x = _solve_1d(ref, w, rows, -1.0, 1.0)

        def f(z):
            return w * (z - ref) ** 2 + sum(q * max(0.0, g * z - b) ** 2 + c * max(0.0, g * z - b)
                                            for g, b, q, c in rows)
        grid = np.linspace(-1.0, 1.0, 20001)
        assert f(x) <= min(f(z) for z in grid) + 1e-9


def test_legacy_kwargs_and_unknown_kwargs():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        CBFSafetyLayer(alpha=1.0, d_min=1.2, y_max=1.9, v_max=25.0)
    try:
        CBFSafetyLayer(not_a_param=1)
    except TypeError:
        pass
    else:
        raise AssertionError("unknown parameter accepted")


# ---------------------------------------------------------------------------- tests: wrapper

def _fake_env_cls():
    import gymnasium as gym
    from gymnasium import spaces

    class FakeCarlaEnv(gym.Env):
        def __init__(self, state, episode_len=5):
            self.observation_space = spaces.Box(-1.0, 1.0, (1,), np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
            self.state = dict(state)
            self.episode_len = episode_len
            self.executed = []
            self.t = 0

        def build_cbf_state(self):
            return dict(self.state)

        def reset(self, seed=None, options=None):
            self.t = 0
            return np.zeros(1, np.float32), {}

        def step(self, action):
            self.t += 1
            self.executed.append(np.array(action, copy=True))
            info = {"reward_components": {"progress": 1.0}}
            truncated = self.t >= self.episode_len
            if truncated:
                ep = {"return": float(self.t), "r": float(self.t), "l": self.t, "t": 0.0,
                      "length_steps": self.t, "end_reason": "timeout"}
                info["episode_metrics"] = ep
                info["episode"] = ep
            return np.zeros(1, np.float32), 1.0, False, truncated, info

    return FakeCarlaEnv


def test_wrapper_free_road_no_penalty():
    from commons.cbf_wrapper import CBFSafetyLayerWrapper
    env = CBFSafetyLayerWrapper(_fake_env_cls()(FREE))
    env.reset()
    a = np.array([0.02, 0.5], dtype=np.float32)
    _, r, _, _, info = env.step(a)
    assert r == 1.0 and info["cbf"]["active"] is False and info["cbf"]["penalty"] == 0.0
    assert np.array_equal(env.env.executed[-1], a)


def test_wrapper_penalty_info_and_episode_metrics():
    from commons.cbf_wrapper import CBFSafetyLayerWrapper
    st = _state(speed=10.0, gap=25.0, gap_kind="vehicle")
    env = CBFSafetyLayerWrapper(_fake_env_cls()(st, episode_len=4), intervention_penalty=0.2)
    env.reset()
    env.set_trust_score(0.8)
    penalties = []
    for k in range(4):
        a = np.array([0.0, 1.0], dtype=np.float32)
        _, r, term, trunc, info = env.step(a)
        safe = env.env.executed[-1]
        du2 = float(np.sum((safe.astype(np.float64) - a.astype(np.float64)) ** 2))
        assert info["cbf"]["active"] is True and "collision" in info["cbf"]["active_rows"]
        assert abs(info["cbf"]["penalty"] - 0.2 * min(1.0, du2)) < 1e-12 and info["cbf"]["penalty"] > 0.0
        assert abs(r - (1.0 - info["cbf"]["penalty"])) < 1e-12
        assert info["cbf"]["trust"] == 0.8
        penalties.append(info["cbf"]["penalty"])
    ep = info["episode_metrics"]
    assert info["episode"] is ep
    assert ep["cbf_interventions"] == 4 and ep["cbf_intervention_frac"] == 1.0
    assert ep["cbf_infeasible"] == 0
    assert abs(ep["cbf_penalty_sum"] - sum(penalties)) < 1e-12 and ep["cbf_penalty_sum"] > 0.0
    assert abs(ep["trust_mean"] - 0.8) < 1e-12
    assert ep["cbf_rows"]["collision"] == 4
    assert ep["return"] == 4.0 and ep["r"] == 4.0          # env return untouched
    # backward-compatible attributes
    assert env.correction_count == 4 and env.episode_corrections == 4 and env.last_correction_mag > 0.0
    assert hasattr(env, "cbf_layer") and hasattr(env, "ensemble_uncertainty")
    env.reset()
    assert env.episode_corrections == 0 and env.correction_count == 4 and env.cbf_layer.correction_count == 0


def test_wrapper_episode_fallback_key_and_infeasible_count():
    from commons.cbf_wrapper import CBFSafetyLayerWrapper
    FakeEnv = _fake_env_cls()

    class OldEpisodeKeyEnv(FakeEnv):
        def step(self, action):
            obs, r, term, trunc, info = super().step(action)
            info.pop("episode_metrics", None)
            return obs, r, term, trunc, info

    env = CBFSafetyLayerWrapper(OldEpisodeKeyEnv(_state(speed=10.0, gap=3.0, gap_kind="vehicle"), episode_len=2))
    env.reset()
    for _ in range(2):
        _, _, _, _, info = env.step(np.array([0.0, 0.5], dtype=np.float32))
        assert info["cbf"]["infeasible"] is True
    assert info["episode"]["cbf_infeasible"] == 2


def test_wrapper_trust_clipping_and_flags():
    import warnings
    from commons.cbf_wrapper import CBFSafetyLayerWrapper
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env = CBFSafetyLayerWrapper(_fake_env_cls()(FREE), alpha=1.0, correction_penalty=0.003)
    env.set_trust_score(1.7)
    assert env.trust_score == 1.0
    env.set_trust_score(-0.3)
    assert env.trust_score == 0.0
    env.set_trust_score(float("nan"))
    assert env.trust_score == 0.0
    env.set_trust_score(0.42)
    assert env.trust_score == 0.42
    env.use_trust_score = False
    env.reset()
    _, _, _, _, info = env.step(np.array([0.0, 0.1], dtype=np.float32))
    assert info["cbf"]["trust"] == 1.0


def test_wrapper_custom_cbf_params_pass_through():
    from commons.cbf_wrapper import CBFSafetyLayerWrapper
    env = CBFSafetyLayerWrapper(_fake_env_cls()(FREE), gamma_c_gain=0.1, d0_vehicle=2.5)
    assert env.cbf_layer.gamma_c_gain == 0.1 and env.cbf_layer.d0_vehicle == 2.5


# ---------------------------------------------------------------------------- tests: critic trust score

def test_ensemble_trust_formula_and_old_kwargs_load():
    try:
        import torch as th
        import gymnasium as gym
        from stable_baselines3 import SAC
        from stable_baselines3.sac.policies import EnsembleCritic
    except Exception as exc:  # pragma: no cover
        _skip(f"torch / SB3 not importable: {exc}")
    from gymnasium import spaces

    # formula
    critic = EnsembleCritic(spaces.Box(-1, 1, (3,)), spaces.Box(-1, 1, (2,)), [8], th.nn.Flatten(), 3,
                            n_critics=5)
    assert critic.trust_lambda == 2.0
    obs, act = th.randn(4, 3), th.randn(4, 2)
    q, var, trust = critic.ensemble_forward(obs, act)
    expected = th.exp(-2.0 * q.std(dim=1, keepdim=True) / (q.mean(dim=1, keepdim=True).abs() + 1.0))
    assert th.allclose(var, q.var(dim=1, keepdim=True)) and th.allclose(trust, expected, atol=1e-6)
    assert bool(((trust > 0) & (trust <= 1)).all())
    mean_q, unc, tr = critic.predict(obs, act)
    assert mean_q.shape == unc.shape == tr.shape == (4, 1)

    # models saved with trust_lambda=0.01 in policy_kwargs still load
    class Tiny(gym.Env):
        observation_space = spaces.Box(-1.0, 1.0, (3,), np.float32)
        action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)

        def reset(self, seed=None, options=None):
            return np.zeros(3, np.float32), {}

        def step(self, action):
            return np.zeros(3, np.float32), 0.0, False, True, {}

    model = SAC("MlpPolicy", Tiny(), policy_kwargs=dict(net_arch=[8], n_critics=2, trust_lambda=0.01),
                buffer_size=10, learning_starts=0, device="cpu")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "old_model.zip")
        model.save(path)
        loaded = SAC.load(path, device="cpu")
    assert loaded.critic.trust_lambda == 0.01
    _, _, t = loaded.critic.predict(th.zeros(1, 3), th.zeros(1, 2))
    assert 0.0 < float(t) <= 1.0


# ---------------------------------------------------------------------------- plain-python runner

def closed_loop_report():
    print("\nClosed-loop stopping table (policy = full throttle, AccelModel + 0.35 s brake delay)")
    print("  onset = first step the collision row engages; all-delay = throttle also lags 0.35 s (stress test)")
    print(f"{'scenario':42s} {'min gap [m]':>11s} {'onset gap':>10s} {'v@onset':>8s} "
          f"{'all-delay min gap':>18s} {'all-delay infeasible steps':>27s}")
    rows = [(f"static v0={v0:g} v_lim={vl:g} gap0={g:g}", dict(v0=v0, gap0=g, v_lim=vl))
            for v0, vl, g in STATIONARY_CASES]
    rows += [(f"lead 10 m/s brakes 6 m/s2 at t={tb:g}s gap0={g:g}",
              dict(v0=10.0, gap0=g, v_lead0=10.0, lead_decel=6.0, lead_brake_t=tb, v_lim=13.9))
             for g, tb in LEAD_CASES]
    rows += [(f"static v0=12 walker", dict(v0=12.0, gap0=60.0, kind="walker")),
             (f"static v0=12 stop_line", dict(v0=12.0, gap0=60.0, kind="stop_line"))]
    rows += [(f"static v0=12 gap0=60 trust={t:g}", dict(v0=12.0, gap0=60.0, trust=t)) for t in (0.5, 0.0)]
    for name, kw in rows:
        r = simulate(**kw)
        r_all = simulate(delay_mode="all", **kw)
        onset = f"{r['first_active_gap']:.1f}" if r["first_active_gap"] is not None else "-"
        v_on = f"{r['v_at_onset']:.1f}" if r["v_at_onset"] is not None else "-"
        print(f"{name:42s} {r['min_gap']:11.2f} {onset:>10s} {v_on:>8s} "
              f"{r_all['min_gap']:18.2f} {r_all['n_infeasible']:27d}")


def _run_all():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = skipped = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except KeyboardInterrupt:
            raise
        except BaseException as exc:  # noqa: B902 - pytest.skip raises a BaseException subclass
            if type(exc).__name__ in ("_Skip", "Skipped"):
                skipped += 1
                print(f"SKIP  {name}: {exc}")
            else:
                failed += 1
                print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed - skipped} passed, {failed} failed, {skipped} skipped")
    closed_loop_report()
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
