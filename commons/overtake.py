"""Safe-overtake reward (opt-in: CarlaGymEnv(overtake_bonus=B)).

Pays +B once for passing a slow or queued vehicle through a clear, legal lane of the same direction
instead of waiting behind it. The pieces:

  * is_queued_behind(snap): a vehicle (not a pedestrian) is the lead, closer than QUEUE_LEAD_GAP_M and
    slower than max(QUEUE_MIN_SPEED_MPS, QUEUE_LEAD_SPEED_FRAC * speed limit); the ego is in a legal
    same-direction lane outside a junction; and the queue is not one for a red/yellow light or a stop
    sign (waiting there is correct, so passing is never rewarded).
  * lane_clear(...): no vehicle in the adjacent lane from CLEAR_BEHIND_M behind to CLEAR_AHEAD_M ahead
    of the ego (route frame, vehicle length included).
  * OvertakeTracker: when the ego is queued and an adjacent lane is legal and clear, it arms on that
    lead vehicle. It pays when that vehicle's front is PASS_MARGIN_M behind the ego's rear while the
    ego is on a legal same-direction lane, within ARM_TIMEOUT_STEPS, with no collision and no illegal
    lane crossing in between. Each vehicle pays at most once per episode, so weaving earns nothing.

The bonus stays small next to the crash penalty (C = 50..300), so the agent overtakes only when it can
do it safely; the CBF and the collision terms are unchanged.
"""

from typing import Callable, Iterable, Optional, Tuple

QUEUE_LEAD_GAP_M = 20.0
QUEUE_LEAD_SPEED_FRAC = 0.5
QUEUE_MIN_SPEED_MPS = 2.0
LIGHT_QUEUE_MARGIN_M = 30.0     # a red/yellow light or stop line within lead_gap + this = waiting for it
CLEAR_BEHIND_M = 12.0
CLEAR_AHEAD_M = 25.0
LANE_CLEAR_MARGIN_M = 0.3       # lateral margin added to the adjacent lane's half width
PASS_MARGIN_M = 2.0
ARM_TIMEOUT_STEPS = 400         # 20 s at 20 Hz to complete the pass
EGO_HALF_LENGTH_M = 2.4         # Tesla Model 3 (ego) half length


def is_queued_behind(snap: dict) -> bool:
    """True when the ego waits behind a slow vehicle for no traffic-control reason (see module doc)."""
    if snap.get("lead_kind") != "vehicle" or snap.get("lead_id") is None:
        return False
    if snap.get("is_junction") or not snap.get("in_same_dir_lane") or not snap.get("on_drivable"):
        return False
    gap = float(snap["lead_gap"])
    if gap > QUEUE_LEAD_GAP_M:
        return False
    slow = max(QUEUE_MIN_SPEED_MPS, QUEUE_LEAD_SPEED_FRAC * float(snap["speed_limit"]))
    if float(snap["lead_v"]) > slow:
        return False
    if snap.get("tl_state") in ("red", "yellow") and float(snap.get("tl_stop_gap", 1e9)) < gap + LIGHT_QUEUE_MARGIN_M:
        return False
    if float(snap.get("stop_gap", 1e9)) < gap + LIGHT_QUEUE_MARGIN_M:
        return False
    return True


def lane_clear(vehicles_sl: Iterable[Tuple[float, float, float]], s_ego: float, l_center: float,
               half_width: float) -> bool:
    """No vehicle (s, l, half_length) in the route-frame box |l - l_center| < half_width + margin,
    s_ego - CLEAR_BEHIND_M <= s +/- half_length <= s_ego + CLEAR_AHEAD_M."""
    w = float(half_width) + LANE_CLEAR_MARGIN_M
    lo, hi = float(s_ego) - CLEAR_BEHIND_M, float(s_ego) + CLEAR_AHEAD_M
    for s, l, half_len in vehicles_sl:
        if abs(float(l) - float(l_center)) < w and float(s) + half_len >= lo and float(s) - half_len <= hi:
            return False
    return True


class OvertakeTracker:
    """Per-episode state of the safe-overtake reward (bonus <= 0 disables it)."""

    def __init__(self, bonus: float):
        self.bonus = float(bonus)
        self.reset()

    def reset(self):
        self.armed: Optional[Tuple[int, int]] = None     # (lead actor id, step it was armed)
        self.paid = set()
        self.illegal = False
        self.passes = 0

    def update(self, snap: dict, step: int, illegal_invasions: int, crashed: bool,
               adjacent_clear: Callable[[], bool], lead_front_s: Callable[[int], Optional[float]]) -> float:
        """The bonus earned on this step (0.0 almost always).

        adjacent_clear(): True if a legal same-direction adjacent lane is clear (only called while the
        ego is queued and nothing is armed). lead_front_s(actor_id): route s of that vehicle's front
        bumper, or None if it no longer exists."""
        if self.bonus <= 0.0:
            return 0.0
        if self.armed is not None:
            lead_id, armed_step = self.armed
            if illegal_invasions > 0:
                self.illegal = True
            if crashed or step - armed_step > ARM_TIMEOUT_STEPS:
                self.armed = None
                return 0.0
            front = lead_front_s(lead_id)
            if front is None:
                self.armed = None
                return 0.0
            passed = front < float(snap["s"]) - EGO_HALF_LENGTH_M - PASS_MARGIN_M
            if passed and snap.get("in_same_dir_lane") and snap.get("on_drivable"):
                self.armed = None
                if self.illegal or lead_id in self.paid:
                    return 0.0
                self.paid.add(lead_id)
                self.passes += 1
                return self.bonus
            return 0.0
        if is_queued_behind(snap) and int(snap["lead_id"]) not in self.paid and adjacent_clear():
            self.armed = (int(snap["lead_id"]), int(step))
            self.illegal = False
        return 0.0
