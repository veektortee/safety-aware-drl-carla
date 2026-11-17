"""Discrete-time Control Barrier Function (CBF) safety filter for u = [steer, accel] in [-1, 1]^2.

The filter solves, at every control step (dt = 0.05 s),

    min_{a, s, slacks}  W_a (a - a_ref)^2 + W_s (s - s_ref)^2 + sum_j (Q_j slack_j^2 + c_j slack_j)
    s.t.  a <= U_col                                   collision   (HARD, exact one-step bound)
          a <= U_v + slack_v                           speed limit (soft)
          k_s s - slack_R <= B_R                       right lane edge (soft, lookahead HOCBF)
          -k_s s - slack_L <= B_L                      left lane edge  (soft, lookahead HOCBF)
          -b_max <= a,  -1 <= s <= 1,  slacks >= 0

where a [m/s^2] is the longitudinal acceleration, s the steering command, a_ref the acceleration the
actor's command produces (vehicle_model.AccelModel) and s_ref the actor's steering. Rows are only
"active" when the actor's own command violates them; if no row is active the actor's action is returned
unchanged (bit-exact copy). There is no post-hoc rate limiting and no blending toward a fallback action.

Conventions (CARLA): steer > 0 turns right, e_y > 0 = right of the route centre, e_psi > 0 = yawed
clockwise of the route tangent, kappa_ref > 0 = route turning right.

Collision row (discrete CBF h(k+1) >= (1 - gamma_c) h(k) on the stopping-point barrier)
    h_c = d + vL^2/(2 b_L) - d0 - rho v - v^2/(2 b_e)
  The lead terms cancel exactly when the lead brakes at b_L, so the one-step condition is quadratic in a:
    q2 a^2 + q1 a <= R,  q2 = dt^2/(2 b_e),  q1 = dt (dt/2 + rho + v/b_e),  R = gamma_c h_c - v dt
    U_col = 2R / (q1 + sqrt(q1^2 + 4 q2 R))            (numerically stable root)
  q1^2 + 4 q2 R < 0 or U_col < -b_max  ->  infeasible  ->  full brake (u_accel = -1).
  Oncoming lead (v_lead < 0): vL = 0 and d_eff = d - |v_lead| (rho + v/b_e).
  gap_kind 'walker' uses d0_walker, 'stop_line' uses d0_stop_line and vL = 0. gap >= gap_none means no
  obstacle within the perception horizon and disables the row (a 'none' kind with a smaller gap is
  treated as 'static': the gap value is the authority).

Speed row:  a <= max(gamma_v (v_lim + dv - v)/dt, -speed_decel_max) + slack_v

Lane rows (discrete HOCBF == lookahead barrier on y_p = e_y + T_p v sin(e_psi), T_p = dt/gamma1):
    k_s = T_p v^2 dt cos(e_psi) G,   drift = v dt sin(e_psi) - T_p v^2 dt cos(e_psi) kappa_ref
    right:  k_s s - slack_R <= gamma2 (y_R - y_p) - drift
    left:  -k_s s - slack_L <= gamma2 (y_L + y_p) + drift
  G = vehicle_model.STEER_GAIN (yaw_rate = G v s). Skipped below lane_min_speed and for
  |e_psi| > lane_max_heading (linearisation invalid, e.g. wrong-way). Both rows are divided by k_s
  evaluated at max(v, lane_norm_speed), so their slacks are in steering units (the penalties mean the
  same at every speed) and the rows soften smoothly at walking pace instead of demanding full lock.

Trust tau in [0, 1] (critic-ensemble confidence) only TIGHTENS the filter. The formulas below are the
source of truth (the nominal "gamma_c = 0.2 / gamma2 = 0.2" of the design notes is the tau = 0.5 value):
    gamma_c = gamma_c_min + gamma_c_gain tau       (0.1 .. 0.3)
    gamma2  = gamma2_min  + gamma2_gain  tau       (0.1 .. 0.3)
    d0     += (1 - tau) d0_trust_margin            (+0 .. +1.0 m)
    rho    += (1 - tau) rho_trust_margin           (+0 .. +0.2 s)
    y_L, y_R -= (1 - tau) lane_trust_margin        (-0 .. -0.2 m)

The problem is separable (a: collision + speed, s: lane rows), so it is solved exactly in closed form
(1-D convex piecewise-quadratic minimisation, no solver tolerance noise). `solver='osqp'` solves the
same QP with OSQP instead (reference implementation, used by the unit tests as a cross-check).
"""
from __future__ import annotations

import math
import warnings
from typing import Optional

import numpy as np

from commons import vehicle_model as vm
from commons.debug import dprint

ROW_NAMES = ("collision", "speed", "lane_left", "lane_right")

DEFAULT_PARAMS = dict(
    # longitudinal capability (vehicle_model)
    b_max=vm.B_MAX,                      # deceleration the layer may rely on [m/s^2]
    b_e=vm.B_DESIGN,                     # deceleration the barrier plans with (< b_max: recursive feasibility)
    a_max=vm.A_MAX,                      # design acceleration; rows are upper bounds so it never cuts the actor
    b_L=8.0,                             # worst-case lead deceleration [m/s^2]
    rho=vm.BRAKE_DELAY_S + 0.1,          # reaction/brake-delay time in the barrier [s]
    # collision row
    d0_vehicle=2.0,                      # standstill distance to vehicles / static obstacles [m]
    d0_walker=3.0,                       # ... to pedestrians [m]
    d0_stop_line=1.0,                    # ... to a red/yellow stop line [m]
    gamma_c_min=0.1,                     # gamma_c = gamma_c_min + gamma_c_gain * trust
    gamma_c_gain=0.2,
    d0_trust_margin=1.0,                 # d0 += (1 - trust) * d0_trust_margin
    rho_trust_margin=0.2,                # rho += (1 - trust) * rho_trust_margin
    gap_none=100.0,                      # gap >= gap_none means "nothing ahead" (row disabled)
    v_stop=0.1,                          # below this speed an unsatisfiable collision row is a "hold", not "infeasible"
    # speed row
    dv=1.0,                              # tolerated excess over the speed limit [m/s]
    gamma_v=0.1,
    speed_decel_max=vm.COAST_DECEL,      # the soft speed row never asks for more than this deceleration [m/s^2]
    # lane rows
    gamma1=0.0625,                       # T_p = dt / gamma1 = 0.8 s lookahead
    gamma2_min=0.1,                      # gamma2 = gamma2_min + gamma2_gain * trust
    gamma2_gain=0.2,
    lane_trust_margin=0.2,               # lane bounds shrink by (1 - trust) * lane_trust_margin [m]
    lane_min_speed=1.0,                  # no lane rows below this speed [m/s]
    lane_norm_speed=4.0,                 # rows are normalised to steering units above this speed [m/s]
    lane_max_heading=math.radians(60.0),  # no lane rows for |e_psi| above this [rad]
    steer_gain=vm.STEER_GAIN,            # yaw_rate = steer_gain * v * steer
    y_max_default=1.75 - vm.EGO_HALF_WIDTH + 0.2,  # used if e_y is given but y_max_left/right are not
    # QP weights
    w_a=1.0 / 3.0 ** 2,                  # 1 / (3 m/s^2)^2
    w_s=1.0 / 0.2 ** 2,                  # 1 / 0.2^2
    slack_v_quad=1e2,
    slack_v_lin=10.0,
    slack_lane_quad=1e3,
    slack_lane_lin=1e2,
    solver="closed_form",                # 'closed_form' | 'osqp'
)

# Keyword arguments of the pre-rewrite layer (3-D action, ad-hoc Lie derivatives). Accepted and ignored
# with a warning so stale call sites do not crash.
LEGACY_PARAMS = frozenset({
    "alpha", "d_min", "y_max", "v_max", "vehicle_width", "v_nominal",
    "max_steering_rate", "max_accel_change", "alpha_lane",
})

_KNOWN_KINDS = ("vehicle", "walker", "static", "stop_line", "none")


def _finite(x, default):
    """float(x) if it is a finite number, else default."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def _solve_1d(ref: float, w: float, rows, lo: float, hi: float) -> float:
    """Exact argmin over x in [lo, hi] of
        w (x - ref)^2 + sum_i [Q_i max(0, g_i x - b_i)^2 + c_i max(0, g_i x - b_i)]
    for rows = [(g_i, b_i, Q_i, c_i)] with w > 0, Q_i, c_i >= 0. The objective is convex and a single
    quadratic between consecutive breakpoints b_i/g_i, so the minimum is the best clipped stationary point.
    """
    if hi <= lo:
        return lo
    pts = {lo, hi}
    for g, b, _, _ in rows:
        if g != 0.0:
            x = b / g
            if lo < x < hi:
                pts.add(x)
    pts = sorted(pts)

    def f(x):
        val = w * (x - ref) ** 2
        for g, b, q, c in rows:
            viol = g * x - b
            if viol > 0.0:
                val += q * viol * viol + c * viol
        return val

    best_x, best_f = lo, math.inf
    for x0, x1 in zip(pts[:-1], pts[1:]):
        xm = 0.5 * (x0 + x1)
        qa, qb = w, -2.0 * w * ref
        for g, b, q, c in rows:
            if g * xm - b > 0.0:
                qa += q * g * g
                qb += g * (c - 2.0 * q * b)
        x = min(max(-qb / (2.0 * qa), x0), x1)
        fx = f(x)
        if fx < best_f:
            best_x, best_f = x, fx
    return best_x


class CBFSafetyLayer:
    """Discrete-time CBF safety filter. See the module docstring for the math.

    Public API:
        compute_safe_action(u_actor, state, trust_score=1.0, verbose=False) -> np.ndarray
        last_info: dict  (active, active_rows, infeasible, correction, h_col, U_col, a_ref, a_safe, ...)
        correction_magnitude: float      ||u_safe - u_actor|| of the last call
        collision_prevented: bool        collision row was active and the output satisfies it
        reset_metrics()

    `state` keys (CarlaGymEnv.build_cbf_state): speed, speed_limit, gap, v_lead, gap_kind, e_y, e_psi,
    kappa_ref, y_max_left, y_max_right, d_collision (= gap, backward compat), ttc. Every key is optional:
    gap falls back to d_collision, v_lead to 0 (stationary: conservative), a missing speed_limit disables
    the speed row and a missing e_y disables the lane rows (the old unsigned `lane_offset` is ignored).
    """

    def __init__(self, dt: float = vm.DT, accel_model: Optional[vm.AccelModel] = None, **params):
        legacy = sorted(k for k in params if k in LEGACY_PARAMS)
        for k in legacy:
            params.pop(k)
        if legacy:
            warnings.warn(f"CBFSafetyLayer: ignoring legacy parameters {legacy} (layer was rewritten)",
                          stacklevel=2)
        unknown = sorted(set(params) - set(DEFAULT_PARAMS))
        if unknown:
            raise TypeError(f"CBFSafetyLayer: unknown parameters {unknown}; valid: {sorted(DEFAULT_PARAMS)}")
        p = dict(DEFAULT_PARAMS)
        p.update(params)
        if p["solver"] not in ("closed_form", "osqp"):
            raise ValueError(f"CBFSafetyLayer: solver must be 'closed_form' or 'osqp', got {p['solver']!r}")
        if not 0.0 < p["b_e"] < p["b_max"]:
            raise ValueError("CBFSafetyLayer: need 0 < b_e < b_max (recursive feasibility margin)")
        self.dt = float(dt)
        self.accel_model = accel_model if accel_model is not None else vm.AccelModel()
        self.params = p
        for k, v in p.items():
            setattr(self, k, v)

        self.correction_magnitude = 0.0
        self.collision_prevented = False
        self.avoidance_efficiency = 0.0
        self.last_info = self._blank_info()
        self.total_steps = 0
        self.total_corrections = 0
        self.osqp_failures = 0
        self.reset_metrics()

    # ------------------------------------------------------------------ public API

    def get_params(self) -> dict:
        """Copy of the effective parameters (for run_config logging)."""
        return dict(self.params, dt=self.dt)

    def reset_metrics(self):
        """Reset the per-episode counters (called by the wrapper on env.reset())."""
        self.correction_count = 0
        self.infeasible_count = 0
        self.steps = 0
        self.row_counts = {name: 0 for name in ROW_NAMES}
        self.constraint_violations = {"collision": 0, "lane": 0, "speed": 0}   # legacy view of row_counts
        self.correction_magnitude = 0.0
        self.collision_prevented = False
        self.avoidance_efficiency = 0.0

    def get_correction_magnitude(self) -> float:
        return self.correction_magnitude

    def compute_safe_action(self, u_actor, state: dict, trust_score: float = 1.0,
                            verbose: bool = False) -> np.ndarray:
        """Return the minimally corrected [steer, accel] (same shape/dtype as u_actor)."""
        u_in = np.asarray(u_actor)
        dtype = u_in.dtype if np.issubdtype(u_in.dtype, np.floating) else np.float64
        out = np.array(u_in, dtype=dtype, copy=True)
        flat = out.reshape(-1)
        if flat.size < 2:
            raise ValueError(f"CBFSafetyLayer: expected [steer, accel], got shape {u_in.shape}")
        s_ref = float(flat[0])
        u_acc = float(flat[1])
        tau = float(np.clip(_finite(trust_score, 0.0), 0.0, 1.0))   # non-finite trust -> most conservative

        st = self._parse_state(state if state is not None else {})
        v = st["speed"]
        a_ref = self.accel_model.accel(u_acc, v)

        col = self._collision_row(st, tau)
        U_v = self._speed_bound(st)
        lane = self._lane_rows(st, tau)

        # ---- which rows does the actor's own command violate?
        active_rows = []
        hold = False
        infeasible = False
        if col is not None and not col["feasible"]:
            if v < self.v_stop:
                hold = True
            else:
                infeasible = True
            col_active = u_acc > -1.0
        else:
            col_active = col is not None and a_ref > col["U"]
        speed_active = U_v is not None and a_ref > U_v
        right_active = lane is not None and lane["k_s"] * s_ref > lane["B_R"]
        left_active = lane is not None and -lane["k_s"] * s_ref > lane["B_L"]
        if col_active:
            active_rows.append("collision")
        if speed_active:
            active_rows.append("speed")
        if left_active:
            active_rows.append("lane_left")
        if right_active:
            active_rows.append("lane_right")
        lon_active = col_active or speed_active
        lat_active = left_active or right_active

        # ---- solve
        a_safe = a_ref
        if lon_active or lat_active:
            if infeasible or hold:
                a_lon = None
            else:
                a_lon = self._a_bounds_and_rows(a_ref, col, U_v)
            if self.solver == "osqp":
                a_sol, s_sol = self._solve_osqp(a_ref, s_ref, a_lon, lane)
            else:
                a_sol = _solve_1d(a_ref, self.w_a, a_lon[2], a_lon[0], a_lon[1]) if (lon_active and a_lon) else a_ref
                s_sol = self._solve_steer(s_ref, lane) if lat_active else s_ref

            if lon_active:
                if a_lon is None:                       # infeasible / standstill hold
                    flat[1] = -1.0
                    a_safe = self.accel_model.accel(-1.0, v)
                else:
                    a_safe = a_sol
                    flat[1] = min(self.accel_model.inverse(a_safe, v), u_acc)
            if lat_active:
                flat[0] = float(np.clip(s_sol, -1.0, 1.0))

        correction = float(np.linalg.norm(out.reshape(-1).astype(np.float64) - u_in.reshape(-1).astype(np.float64)))
        active = bool(active_rows)
        self.collision_prevented = bool(col_active and col is not None and col["feasible"])
        self.correction_magnitude = correction
        if col is not None and math.isfinite(col["h"]):
            self.avoidance_efficiency = float(np.clip(col["h"] / max(col["d0"], 1e-6), 0.0, 1.0))
        else:
            self.avoidance_efficiency = 1.0

        self.steps += 1
        self.total_steps += 1
        if active:
            self.correction_count += 1
            self.total_corrections += 1
        if infeasible:
            self.infeasible_count += 1
        for name in active_rows:
            self.row_counts[name] += 1
        if col_active:
            self.constraint_violations["collision"] += 1
        if speed_active:
            self.constraint_violations["speed"] += 1
        if lat_active:
            self.constraint_violations["lane"] += 1

        self.last_info = {
            "active": active,
            "active_rows": active_rows,
            "infeasible": infeasible,
            "hold": hold,
            "correction": correction,
            "h_col": col["h"] if col is not None else math.inf,
            "U_col": col["U"] if col is not None else math.inf,
            "U_speed": U_v if U_v is not None else math.inf,
            "h_lane_left": lane["h_L"] if lane is not None else math.inf,
            "h_lane_right": lane["h_R"] if lane is not None else math.inf,
            "a_ref": float(a_ref),
            "a_safe": float(a_safe),
            "trust": tau,
            "gamma_c": self.gamma_c_min + self.gamma_c_gain * tau,
        }
        if active:
            msg = (f"[CBF] rows={active_rows} infeasible={infeasible} hold={hold} v={v:.2f} gap={st['gap']:.2f} "
                   f"h_col={self.last_info['h_col']:.2f} U_col={self.last_info['U_col']:.2f} "
                   f"a_ref={a_ref:.2f}->{a_safe:.2f} u=[{s_ref:.3f},{u_acc:.3f}]->[{flat[0]:.3f},{flat[1]:.3f}]")
            if verbose:
                print(msg)
            else:
                dprint(msg)
        return out

    # ------------------------------------------------------------------ rows

    def _parse_state(self, state: dict) -> dict:
        gap_raw = state.get("gap", state.get("d_collision", None))
        if gap_raw is None:
            gap = self.gap_none
        else:
            gap = _finite(gap_raw, self.gap_none)       # inf / nan -> "nothing ahead"
            gap = max(0.0, min(gap, self.gap_none))
        kind = state.get("gap_kind", None)
        if kind not in _KNOWN_KINDS:
            kind = "vehicle"
        if gap >= self.gap_none:
            kind = "none"                               # the gap value is the authority
        elif kind == "none":
            kind = "static"
        e_y_raw = state.get("e_y", None)
        e_y = None if e_y_raw is None else _finite(e_y_raw, None)
        y_l = _finite(state.get("y_max_left", self.y_max_default), self.y_max_default)
        y_r = _finite(state.get("y_max_right", self.y_max_default), self.y_max_default)
        v_lim = state.get("speed_limit", None)
        v_lim = None if v_lim is None else _finite(v_lim, None)
        if v_lim is not None and v_lim <= 0.0:
            v_lim = None
        return dict(
            speed=_finite(state.get("speed", 0.0), 0.0),
            speed_limit=v_lim,
            gap=gap,
            v_lead=_finite(state.get("v_lead", 0.0), 0.0),
            gap_kind=kind,
            e_y=e_y,
            e_psi=_finite(state.get("e_psi", 0.0), 0.0),
            kappa_ref=_finite(state.get("kappa_ref", 0.0), 0.0),
            y_max_left=y_l,
            y_max_right=y_r,
        )

    def _collision_row(self, st: dict, tau: float) -> Optional[dict]:
        """Exact one-step bound a <= U_col of the discrete CBF on the stopping-point barrier."""
        kind = st["gap_kind"]
        d = st["gap"]
        if kind == "none" or d >= self.gap_none:
            return None
        v = max(0.0, st["speed"])
        dt = self.dt
        d0 = {"walker": self.d0_walker, "stop_line": self.d0_stop_line}.get(kind, self.d0_vehicle)
        d0 += (1.0 - tau) * self.d0_trust_margin
        rho = self.rho + (1.0 - tau) * self.rho_trust_margin
        gamma = self.gamma_c_min + self.gamma_c_gain * tau
        b_e = self.b_e
        v_lead = st["v_lead"]
        if kind == "stop_line":
            vL = 0.0
        elif v_lead >= 0.0:
            vL = v_lead
        else:                                   # oncoming: it keeps closing while we stop
            vL = 0.0
            d = d - abs(v_lead) * (rho + v / b_e)
        h = d + vL * vL / (2.0 * self.b_L) - d0 - rho * v - v * v / (2.0 * b_e)
        q2 = dt * dt / (2.0 * b_e)
        q1 = dt * (0.5 * dt + rho + v / b_e)
        R = gamma * h - v * dt
        disc = q1 * q1 + 4.0 * q2 * R
        if disc < 0.0:
            U = -math.inf
        else:
            U = 2.0 * R / (q1 + math.sqrt(disc))
        feasible = U >= -self.b_max
        return dict(h=float(h), U=float(U), feasible=bool(feasible), d0=d0, gamma=gamma)

    def _speed_bound(self, st: dict) -> Optional[float]:
        v_lim = st["speed_limit"]
        if v_lim is None:
            return None
        U = self.gamma_v * (v_lim + self.dv - st["speed"]) / self.dt
        return float(max(U, -self.speed_decel_max))

    def _lane_rows(self, st: dict, tau: float) -> Optional[dict]:
        if st["e_y"] is None:
            return None
        v = st["speed"]
        if v < self.lane_min_speed:
            return None
        e_psi = st["e_psi"]
        if abs(e_psi) > self.lane_max_heading:
            return None
        dt = self.dt
        c, s = math.cos(e_psi), math.sin(e_psi)
        T_p = dt / self.gamma1
        gamma2 = self.gamma2_min + self.gamma2_gain * tau
        margin = (1.0 - tau) * self.lane_trust_margin
        y_r = st["y_max_right"] - margin
        y_l = st["y_max_left"] - margin
        y_p = st["e_y"] + T_p * v * s
        k_s = T_p * v * v * dt * c * self.steer_gain
        drift = v * dt * s - T_p * v * v * dt * c * st["kappa_ref"]
        B_R = gamma2 * (y_r - y_p) - drift
        B_L = gamma2 * (y_l + y_p) + drift
        # Express both rows in steering units so the slack penalties mean the same at every speed
        # (k_s grows with v^2). Below lane_norm_speed the divisor is frozen, so at walking pace the rows
        # soften instead of demanding full steering lock for a few centimetres of lookahead error.
        v_n = max(v, self.lane_norm_speed)
        scale = T_p * v_n * v_n * dt * c * self.steer_gain
        return dict(
            k_s=k_s, B_R=B_R, B_L=B_L,
            g=k_s / scale, b_R=B_R / scale, b_L=B_L / scale,
            h_R=y_r - y_p,
            h_L=y_l + y_p,
            y_p=y_p,
        )

    # ------------------------------------------------------------------ solvers

    def _a_bounds_and_rows(self, a_ref: float, col: Optional[dict], U_v: Optional[float]):
        """(lo, hi, soft_rows) of the longitudinal sub-problem (feasible case)."""
        lo = min(-self.b_max, a_ref)            # never reduce braking the actor already asks for
        hi = max(self.a_max, a_ref)             # rows are upper bounds: the box never cuts the actor
        if col is not None:
            hi = min(hi, col["U"])
        hi = max(hi, lo)
        rows = []
        if U_v is not None:
            rows.append((1.0, U_v, self.slack_v_quad, self.slack_v_lin))
        return lo, hi, rows

    def _steer_rows(self, lane: dict):
        return [
            (lane["g"], lane["b_R"], self.slack_lane_quad, self.slack_lane_lin),
            (-lane["g"], lane["b_L"], self.slack_lane_quad, self.slack_lane_lin),
        ]

    def _solve_steer(self, s_ref: float, lane: dict) -> float:
        lo, hi = min(-1.0, s_ref), max(1.0, s_ref)
        return _solve_1d(s_ref, self.w_s, self._steer_rows(lane), lo, hi)

    def _solve_osqp(self, a_ref: float, s_ref: float, a_lon, lane: Optional[dict]):
        """Same QP with OSQP: x = [a, s, slack_v, slack_L, slack_R]. Falls back to the closed form."""
        try:
            import osqp
            import scipy.sparse as sp

            inf = np.inf
            if a_lon is None:                   # longitudinal part is decided (full brake); solve lateral only
                a_lo, a_hi, U_v = a_ref, a_ref, inf
            else:
                a_lo, a_hi, rows = a_lon
                U_v = rows[0][1] if rows else inf
            k_s = lane["g"] if lane is not None else 0.0          # normalised rows (steering units)
            B_R = lane["b_R"] if lane is not None else inf
            B_L = lane["b_L"] if lane is not None else inf
            P = sp.diags(2.0 * np.array([self.w_a, self.w_s, self.slack_v_quad,
                                         self.slack_lane_quad, self.slack_lane_quad])).tocsc()
            q = np.array([-2.0 * self.w_a * a_ref, -2.0 * self.w_s * s_ref,
                          self.slack_v_lin, self.slack_lane_lin, self.slack_lane_lin])
            A = np.array([
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, -1.0, 0.0, 0.0],      # speed:  a - slack_v <= U_v
                [0.0, -k_s, 0.0, -1.0, 0.0],     # left:  -k_s s - slack_L <= B_L
                [0.0, k_s, 0.0, 0.0, -1.0],      # right:  k_s s - slack_R <= B_R
            ])
            lo = np.array([a_lo, min(-1.0, s_ref), 0.0, 0.0, 0.0, -inf, -inf, -inf])
            hi = np.array([a_hi, max(1.0, s_ref), inf, inf, inf, U_v, B_L, B_R])
            prob = osqp.OSQP()
            settings = dict(verbose=False, eps_abs=1e-9, eps_rel=1e-9, max_iter=200000)
            try:                                # osqp >= 1.0 renamed polish -> polishing
                prob.setup(P=P, q=q, A=sp.csc_matrix(A), l=lo, u=hi, polishing=False, **settings)
            except (TypeError, ValueError):
                prob = osqp.OSQP()
                prob.setup(P=P, q=q, A=sp.csc_matrix(A), l=lo, u=hi, polish=False, **settings)
            try:
                res = prob.solve(raise_error=False)
            except TypeError:                   # osqp < 1.0
                res = prob.solve()
            if str(res.info.status).lower() != "solved":
                raise RuntimeError(f"OSQP status {res.info.status}")
            return float(res.x[0]), float(res.x[1])
        except Exception as exc:                # pragma: no cover - defensive
            self.osqp_failures += 1
            dprint(f"[CBF] OSQP failed ({exc}); using the closed-form solution")
            a_sol = _solve_1d(a_ref, self.w_a, a_lon[2], a_lon[0], a_lon[1]) if a_lon else a_ref
            s_sol = self._solve_steer(s_ref, lane) if lane is not None else s_ref
            return a_sol, s_sol

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _blank_info() -> dict:
        return {
            "active": False, "active_rows": [], "infeasible": False, "hold": False, "correction": 0.0,
            "h_col": math.inf, "U_col": math.inf, "U_speed": math.inf,
            "h_lane_left": math.inf, "h_lane_right": math.inf,
            "a_ref": 0.0, "a_safe": 0.0, "trust": 1.0, "gamma_c": None,
        }
