"""Rule-based expert driver used to pre-fill the SAC replay buffer with good driving.

Lateral: pure pursuit on the route from the rear axle (WHEELBASE, MAX_STEER_DEG), lookahead
lookahead_base + lookahead_gain * v. Longitudinal: a target speed that is the minimum of

- the cruise speed CRUISE_FRAC * speed limit,
- the curve speed sqrt(A_LAT_MAX / |kappa|) of every route vertex ahead, reachable with target_decel,
- a stopping ramp sqrt(2 * target_decel * max(0, gap - stop_offset - BRAKE_DELAY_S * v)) for the lead
  obstacle (stop LEAD_STOP_M behind it), a red / yellow light that can still be stopped for
  (stop LINE_STOP_M before the line) and the next stop sign (full stop; StopSignMonitor then reports
  the following stop line, so the expert drives on),

tracked by a PI controller whose acceleration demand is turned into the action through
AccelModel.inverse (so holding speed gets the throttle needed against the ~3 m/s^2 coast-down).
When the ramp alone is too gentle to stop in time, the exact constant deceleration v^2 / (2 d) to
the stop point is applied instead.
"""
from __future__ import annotations

import math

import numpy as np

from commons.perception import NO_GAP
from commons.vehicle_model import (A_MAX, B_MAX, BRAKE_DELAY_S, DT, EGO_HALF_LENGTH, FULL_BRAKE_DECEL,
                                   MAX_STEER_DEG, WHEELBASE, AccelModel)


class ExpertDriver:
    CRUISE_FRAC = 0.9          # cruise at 90 % of the limit (no speeding from overshoot)
    A_LAT_MAX = 2.5            # m/s^2, curve speed limit
    LEAD_STOP_M = 4.0          # stop this far behind the lead obstacle
    LINE_STOP_M = 1.0          # stop this far before a stop line (front bumper)
    KP = 1.5                   # 1/s
    KI = 0.4                   # 1/s^2
    I_MAX = 1.5                # m/s^2
    HOLD_ACCEL_CMD = -0.4      # brake command while holding at standstill

    def __init__(self, accel_model=None, target_decel=3.0, lookahead_base=4.0, lookahead_gain=0.4):
        self.accel_model = accel_model if accel_model is not None else AccelModel()
        self.target_decel = float(target_decel)
        self.lookahead_base = float(lookahead_base)
        self.lookahead_gain = float(lookahead_gain)
        self._integral = 0.0
        self.last_target_speed = 0.0
        self._route = None

    def reset(self):
        """Clear the integrator (also done automatically when act() sees a new route object)."""
        self._integral = 0.0
        self.last_target_speed = 0.0

    # ------------------------------------------------------------------ lateral
    def steer(self, ego_transform, v, route, proj) -> float:
        """Pure pursuit: target point `lookahead` metres of route ahead of the rear axle."""
        lookahead = self.lookahead_base + self.lookahead_gain * max(0.0, float(v))
        tx, ty = route.point_at(float(proj.s) - 0.5 * WHEELBASE + lookahead)
        psi = math.radians(ego_transform.rotation.yaw)
        c, s = math.cos(psi), math.sin(psi)
        loc = ego_transform.location
        rx, ry = loc.x - 0.5 * WHEELBASE * c, loc.y - 0.5 * WHEELBASE * s   # rear axle
        dx, dy = tx - rx, ty - ry
        x_l = dx * c + dy * s
        y_l = -dx * s + dy * c
        ld2 = max(x_l * x_l + y_l * y_l, 1e-6)
        curvature = 2.0 * y_l / ld2
        delta = math.atan(WHEELBASE * curvature)
        return float(np.clip(delta / math.radians(MAX_STEER_DEG), -1.0, 1.0))

    # ------------------------------------------------------------------ longitudinal
    def _ramp(self, gap_to_stop_point: float, v: float) -> float:
        d = gap_to_stop_point - BRAKE_DELAY_S * v
        return math.sqrt(2.0 * self.target_decel * max(0.0, d))

    def target_speed(self, v, route, proj, lead, tl, stop_gap_m, speed_limit_mps):
        """(target speed [m/s], constraints) where constraints = [(distance [m], speed there [m/s])]:
        stop points (speed 0) and curve vertices (curve speed) that act() must be able to meet."""
        v = max(0.0, float(v))
        v_t = self.CRUISE_FRAC * max(0.0, float(speed_limit_mps))
        constraints = []

        horizon = max(30.0, v * v / (2.0 * self.target_decel) + 20.0)
        d, k = route.curvature_profile(float(proj.s), horizon)
        v_curve = np.sqrt(self.A_LAT_MAX / np.maximum(np.abs(k), 1e-4))
        v_t = min(v_t, float(np.min(np.sqrt(v_curve ** 2 + 2.0 * self.target_decel * np.maximum(0.0, d - BRAKE_DELAY_S * v)))))
        slow = v_curve < v
        constraints.extend(zip(d[slow].tolist(), v_curve[slow].tolist()))

        if lead is not None and lead.gap_m < NO_GAP:
            g = float(lead.gap_m) - self.LEAD_STOP_M
            v_t = min(v_t, self._ramp(g, v))
            constraints.append((g, 0.0))

        if tl is not None and tl.state in ("red", "yellow") and tl.stop_gap_m < NO_GAP:
            gap = float(tl.stop_gap_m)
            stoppable = v * BRAKE_DELAY_S + v * v / (2.0 * B_MAX) < gap
            # already at / slightly over the line on red: hold instead of driving into the junction
            hold = tl.state == "red" and v < 2.0 and gap > -1.5 * EGO_HALF_LENGTH
            if stoppable or hold:
                g = gap - self.LINE_STOP_M
                v_t = min(v_t, self._ramp(g, v))
                constraints.append((g, 0.0))

        if stop_gap_m is not None and stop_gap_m < NO_GAP:
            g = float(stop_gap_m) - self.LINE_STOP_M
            v_t = min(v_t, self._ramp(g, v))
            constraints.append((g, 0.0))
        return max(0.0, v_t), constraints

    def act(self, ego_transform, v, route, proj, lead, tl, stop_gap_m: float,
            speed_limit_mps: float) -> np.ndarray:
        """[steer, accel] in [-1, 1] (accel >= 0 throttle, < 0 brake)."""
        if route is not self._route:      # new episode / route: start the integrator fresh
            self.reset()
            self._route = route
        v = float(v)
        steer = self.steer(ego_transform, v, route, proj)
        v_t, constraints = self.target_speed(v, route, proj, lead, tl, stop_gap_m, speed_limit_mps)
        self.last_target_speed = v_t

        if v_t < 0.05 and v < 0.3:
            self._integral = 0.0
            return np.array([steer, self.HOLD_ACCEL_CMD], dtype=np.float32)

        err = v_t - v
        a = self.KP * err + self._integral
        if err < -1.0:
            self._integral = 0.0                      # decelerating: drop the cruise integrator
        elif -FULL_BRAKE_DECEL < a < A_MAX:           # anti-windup
            self._integral = float(np.clip(self._integral + self.KI * err * DT, -self.I_MAX, self.I_MAX))

        # Exact constant deceleration that meets a constraint, once the ramp is no longer enough
        # (a P controller alone lags a decelerating profile by target_decel / KP ~ 2 m/s).
        v_pos = max(0.0, v)
        for g, v_end in constraints:
            if v_pos > v_end + 0.1:
                g_eff = max(g - 0.5 * BRAKE_DELAY_S * v_pos, 0.05)
                a_need = (v_end * v_end - v_pos * v_pos) / (2.0 * g_eff)
                if a_need < -0.8 * self.target_decel:
                    a = min(a, a_need)

        a = float(np.clip(a, -FULL_BRAKE_DECEL, A_MAX))
        u = self.accel_model.inverse(a, v_pos)
        return np.array([steer, float(np.clip(u, -1.0, 1.0))], dtype=np.float32)
