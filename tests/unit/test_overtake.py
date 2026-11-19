"""Offline tests for the safe-overtake reward (commons/overtake.py). No CARLA server needed.

Run:  python tests/unit/test_overtake.py      or      python -m pytest tests/unit/test_overtake.py
"""

import os
import sys

sys.dont_write_bytecode = True  # tracked __pycache__/*.pyc files must not change
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from commons.overtake import (ARM_TIMEOUT_STEPS, EGO_HALF_LENGTH_M, PASS_MARGIN_M, OvertakeTracker,  # noqa: E402
                              is_queued_behind, lane_clear)


def queued(**over):
    """Ego at s=100, 8 m behind a car crawling at 1 m/s, no light or stop sign near."""
    snap = dict(lead_kind="vehicle", lead_id=7, lead_gap=8.0, lead_v=1.0, speed_limit=8.33, speed=0.5,
                is_junction=False, in_same_dir_lane=True, on_drivable=True,
                tl_state="none", tl_stop_gap=100.0, stop_gap=100.0, s=100.0)
    snap.update(over)
    return snap


def test_queued_behind_a_slow_car_but_not_for_lights_signs_walkers_or_junctions():
    assert is_queued_behind(queued())
    assert not is_queued_behind(queued(lead_kind="walker")), "never overtake pedestrians"
    assert not is_queued_behind(queued(lead_gap=35.0)), "lead too far away"
    assert not is_queued_behind(queued(lead_v=6.0)), "lead moving at a normal speed"
    assert not is_queued_behind(queued(tl_state="red", tl_stop_gap=20.0)), "queue for a red light"
    assert not is_queued_behind(queued(tl_state="yellow", tl_stop_gap=30.0)), "queue for a yellow light"
    assert is_queued_behind(queued(tl_state="red", tl_stop_gap=80.0)), "a light far beyond the queue"
    assert not is_queued_behind(queued(stop_gap=15.0)), "queue for a stop sign"
    assert not is_queued_behind(queued(is_junction=True)), "no overtaking inside junctions"
    assert not is_queued_behind(queued(in_same_dir_lane=False)), "ego not in a legal lane"


def test_lane_clear_box():
    # adjacent lane centred at l = -3.5, half width 1.75; ego at s = 100
    assert lane_clear([], 100.0, -3.5, 1.75)
    assert not lane_clear([(110.0, -3.4, 2.3)], 100.0, -3.5, 1.75), "car beside/ahead in that lane"
    assert not lane_clear([(90.0, -3.6, 2.3)], 100.0, -3.5, 1.75), "car close behind in that lane"
    assert lane_clear([(140.0, -3.5, 2.3)], 100.0, -3.5, 1.75), "beyond 25 m ahead"
    assert lane_clear([(80.0, -3.5, 2.3)], 100.0, -3.5, 1.75), "more than 12 m behind"
    assert lane_clear([(108.0, 0.0, 2.3)], 100.0, -3.5, 1.75), "the queued car in the ego lane"


def test_tracker_pays_once_for_a_completed_pass():
    t = OvertakeTracker(5.0)
    assert t.update(queued(), 1, 0, False, lambda: True, lambda _id: 110.0) == 0.0     # arms on car 7
    assert t.armed == (7, 1)
    behind = 100.0 - EGO_HALF_LENGTH_M - PASS_MARGIN_M - 1.0
    assert t.update(queued(s=104.0), 2, 0, False, lambda: True, lambda _id: 110.0) == 0.0, "not passed yet"
    assert t.update(queued(s=110.0), 3, 0, False, lambda: True, lambda _id: behind + 10.0) == 5.0, "passed"
    assert t.passes == 1 and t.armed is None
    # the same car can never pay again (no weaving behind it for more bonuses)
    assert t.update(queued(), 4, 0, False, lambda: True, lambda _id: 110.0) == 0.0
    assert t.armed is None


def test_tracker_needs_a_clear_lane_and_pays_nothing_after_a_crash_illegal_crossing_or_timeout():
    t = OvertakeTracker(5.0)
    assert t.update(queued(), 1, 0, False, lambda: False, lambda _id: 110.0) == 0.0
    assert t.armed is None, "adjacent lane occupied: no opportunity"

    for event in ("crash", "illegal", "timeout", "gone"):
        t = OvertakeTracker(5.0)
        t.update(queued(), 1, 0, False, lambda: True, lambda _id: 110.0)
        assert t.armed is not None
        passed_front = 50.0
        if event == "crash":
            assert t.update(queued(s=110.0), 2, 0, True, lambda: True, lambda _id: passed_front) == 0.0
        elif event == "illegal":
            t.update(queued(s=104.0), 2, 1, False, lambda: True, lambda _id: 110.0)     # solid line crossed
            assert t.update(queued(s=110.0), 3, 0, False, lambda: True, lambda _id: passed_front) == 0.0
        elif event == "timeout":
            late = 1 + ARM_TIMEOUT_STEPS + 1
            assert t.update(queued(s=110.0), late, 0, False, lambda: True, lambda _id: passed_front) == 0.0
        else:
            assert t.update(queued(s=110.0), 2, 0, False, lambda: True, lambda _id: None) == 0.0
        assert t.passes == 0 and t.armed is None, event


def test_tracker_off_by_default_and_reset_per_episode():
    t = OvertakeTracker(0.0)
    assert t.update(queued(), 1, 0, False, lambda: True, lambda _id: 50.0) == 0.0 and t.armed is None
    t = OvertakeTracker(5.0)
    t.update(queued(), 1, 0, False, lambda: True, lambda _id: 110.0)
    t.reset()
    assert t.armed is None and not t.paid and t.passes == 0


if __name__ == "__main__":
    import traceback

    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
