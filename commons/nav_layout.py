"""Layout of the navigation vector `obs['nav']` that CarlaGymEnv adds next to the camera image.

The env builds the vector in exactly this order; policies/wrappers must read NAV_DIM from here
(or from env.observation_space['nav'].shape) instead of hard-coding a size.
All values are normalised to roughly [-1, 1] (or [0, 1]); conventions follow CARLA:
x forward, y to the RIGHT, yaw increasing clockwise.
"""

NAV_FEATURES = [
    "speed_over_limit",      # v / v_limit, clipped to [0, 2]
    "speed_limit_norm",      # v_limit / 30 m/s
    "lateral_error_norm",    # e_y / 2 m, clipped to [-2, 2]; + = right of the route lane centre
    "heading_err_sin",       # sin(e_psi); + = vehicle yawed clockwise (right) of the route tangent
    "heading_err_cos",       # cos(e_psi)
    "route_pt5_x",           # route point 5 m ahead (arc length), ego frame, / 20 m
    "route_pt5_y",
    "route_pt10_x",          # route point 10 m ahead
    "route_pt10_y",
    "route_pt20_x",          # route point 20 m ahead
    "route_pt20_y",
    "cmd_follow",            # one-hot of the next manoeuvre within 50 m (CARLA RoadOption):
    "cmd_left",              #   LANEFOLLOW / CHANGELANE* / VOID -> follow
    "cmd_right",
    "cmd_straight",
    "cmd_dist_norm",         # distance to that manoeuvre / 50 m, clipped to [0, 1] (1 if none)
    "lead_gap_norm",         # gap to nearest obstacle in the route corridor (vehicle/walker/static), min(gap, 50) / 50
    "lead_rel_speed_norm",   # (v_lead - v) / 10, clipped to [-1, 1]; 0 if no lead
    "lead_is_pedestrian",    # 1 if the nearest obstacle is a walker
    "tl_red",                # traffic light affecting the route lane ahead (all three 0 = none)
    "tl_yellow",
    "tl_green",
    "stop_gap_norm",         # distance to the stop line of that light or of a stop sign ahead, min(d, 50) / 50 (1 if none)
    "prev_steer",            # previous executed action (after the CBF)
    "prev_accel",
    "route_remaining_frac",  # remaining route length / total route length
]

NAV_DIM = len(NAV_FEATURES)
NAV_INDEX = {name: i for i, name in enumerate(NAV_FEATURES)}
