"""Measured ego-vehicle model shared by the env, the CBF safety layer and the expert driver.

All numbers were identified in CARLA 0.9.15 (Town10HD, sync mode, dt = 0.05 s) for
`vehicle.tesla.model3`, the ego blueprint used by CarlaGymEnv:

- coast-down (no pedals) decelerates at about 3.0 m/s^2 between 4 and 11 m/s
- full brake decelerates at about 5.7 m/s^2 (brake 0.5: 4.4, brake 0.2: 3.6)
- full throttle accelerates at about 7 m/s^2; throttle 0.3 barely moves the car
- brake response delay about 0.3-0.4 s
- yaw rate ~= STEER_GAIN * v * steer, with STEER_GAIN ~= 0.35-0.45 1/m per unit steer
- wheelbase 3.0 m, front max steer angle 70 deg, bounding-box half extents 2.40 x 1.08 m

The policy's longitudinal action is a single value u in [-1, 1]:
u >= 0 -> throttle = u, brake = 0;  u < 0 -> throttle = 0, brake = -u.
`AccelModel` maps (u, v) to an expected acceleration and back.
"""
from __future__ import annotations

import numpy as np

DT = 0.05
WHEELBASE = 3.0
EGO_HALF_LENGTH = 2.40
EGO_HALF_WIDTH = 1.08
MAX_STEER_DEG = 70.0
STEER_GAIN = 0.36            # yaw_rate [rad/s] ~= STEER_GAIN * v [m/s] * steer [-1..1]

FULL_BRAKE_DECEL = 5.7       # measured
B_MAX = 0.9 * FULL_BRAKE_DECEL   # 5.13 m/s^2, deceleration the safety layer may rely on
B_DESIGN = 0.7 * B_MAX           # 3.59 m/s^2, deceleration the CBF plans with (recursive feasibility margin)
A_MAX = 3.0                  # design acceleration limit used by the CBF / expert (car can do ~7)
COAST_DECEL = 3.0            # measured
BRAKE_DELAY_S = 0.35         # measured response delay of brake application

# Acceleration table a(u, v) [m/s^2]: rows = u grid, columns = speed grid (bin centres).
_SPEED_GRID = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0])
_U_GRID = np.array([-1.0, -0.5, -0.2, 0.0, 0.3, 0.5, 0.7, 1.0])
_ACCEL_TABLE = np.array([
    [-5.7, -5.7, -5.7, -5.7, -5.7, -5.7, -5.7],   # u = -1.0 (full brake)
    [-4.4, -4.4, -4.4, -4.4, -4.4, -4.4, -4.4],   # u = -0.5
    [-3.6, -3.6, -3.6, -3.6, -3.6, -3.6, -3.6],   # u = -0.2
    [-1.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0],   # u =  0.0 (coast; weaker near standstill)
    [0.9, 0.96, 0.51, 0.1, -0.3, -0.6, -0.9],     # u =  0.3
    [3.1, 2.7, 1.54, 1.39, 1.06, 0.8, 0.5],       # u =  0.5
    [4.5, 4.6, 3.46, 3.56, 3.45, 3.18, 2.67],     # u =  0.7
    [7.0, 7.0, 7.0, 7.0, 7.0, 6.8, 6.5],          # u =  1.0 (full throttle)
])


def accel_to_pedals(u: float) -> tuple[float, float]:
    """Map the single longitudinal action u in [-1, 1] to (throttle, brake)."""
    u = float(np.clip(u, -1.0, 1.0))
    return (u, 0.0) if u >= 0.0 else (0.0, -u)


def pedals_to_accel_cmd(throttle: float, brake: float) -> float:
    """Inverse of accel_to_pedals (brake wins when both are pressed)."""
    return -float(brake) if brake > 0.0 else float(throttle)


class AccelModel:
    """Bilinear interpolation of the measured acceleration table, monotone in u."""

    def __init__(self, table: np.ndarray = _ACCEL_TABLE):
        self.table = np.asarray(table, dtype=float)
        self._u_fine = np.linspace(-1.0, 1.0, 401)

    def accel(self, u: float, v: float) -> float:
        """Expected acceleration [m/s^2] for longitudinal action u at speed v [m/s]."""
        u = float(np.clip(u, -1.0, 1.0))
        v = float(np.clip(v, _SPEED_GRID[0], _SPEED_GRID[-1]))
        col = np.array([np.interp(v, _SPEED_GRID, row) for row in self.table])
        return float(np.interp(u, _U_GRID, col))

    def inverse(self, a_des: float, v: float) -> float:
        """Smallest-magnitude u in [-1, 1] whose expected acceleration is closest to a_des."""
        v = float(np.clip(v, _SPEED_GRID[0], _SPEED_GRID[-1]))
        col = np.array([np.interp(v, _SPEED_GRID, row) for row in self.table])
        a_fine = np.interp(self._u_fine, _U_GRID, col)
        if a_des <= a_fine[0]:
            return -1.0
        if a_des >= a_fine[-1]:
            return 1.0
        return float(np.interp(a_des, a_fine, self._u_fine))

    def bounds(self, v: float) -> tuple[float, float]:
        """(min, max) acceleration reachable at speed v."""
        return self.accel(-1.0, v), self.accel(1.0, v)


if __name__ == "__main__":
    m = AccelModel()
    for v in (1.0, 5.0, 9.0, 13.0):
        us = np.linspace(-1, 1, 21)
        acc = [m.accel(u, v) for u in us]
        assert all(b >= a - 1e-9 for a, b in zip(acc, acc[1:])), f"not monotone at v={v}"
        for a_des in (-5.0, -3.0, 0.0, 1.0, 3.0):
            u = m.inverse(a_des, v)
            assert abs(m.accel(u, v) - np.clip(a_des, *m.bounds(v))) < 0.05, (v, a_des, u)
    print("vehicle_model self-test OK")
