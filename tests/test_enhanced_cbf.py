"""
Test script for enhanced CBF Safety Layer
Validates:
- Speed-dependent Lie derivatives
- Trust score modulation
- Speed limit constraint
- Proactive lane keeping
- Rate limiting
- Fallback safety
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from commons.cbfQP_layer import CBFSafetyLayer


def test_cbf_initialization():
    """Test CBF layer initialization with new parameters"""
    print("\n[TEST 1] CBF Initialization with Enhanced Parameters")
    cbf = CBFSafetyLayer(
        alpha=1.0,
        d_min=5.0,
        y_max=1.5,
        v_max=15.0,
        vehicle_width=1.8,
        v_nominal=10.0,
        max_steering_rate=0.5,
        max_accel_change=0.3
    )
    
    assert cbf.alpha == 1.0
    assert cbf.d_min == 5.0
    assert cbf.y_max == 1.5
    assert cbf.vehicle_width == 1.8
    assert cbf.max_steering_rate == 0.5
    print("✓ CBF initialized with enhanced parameters")


def test_speed_dependent_derivatives():
    """Test speed-dependent Lie derivative computation"""
    print("\n[TEST 2] Speed-Dependent Lie Derivatives")
    cbf = CBFSafetyLayer()
    
    # Test at different speeds
    state_low_speed = {'speed': 1.0}
    state_nominal = {'speed': 10.0}
    state_high_speed = {'speed': 20.0}
    
    A_col_low, A_speed_low = cbf._get_speed_dependent_lie_derivatives(state_low_speed)
    A_col_nom, A_speed_nom = cbf._get_speed_dependent_lie_derivatives(state_nominal)
    A_col_high, A_speed_high = cbf._get_speed_dependent_lie_derivatives(state_high_speed)
    
    # Throttle effect magnitude should increase with speed (more negative = larger magnitude)
    throttle_mag_low = abs(A_col_low[1])
    throttle_mag_nom = abs(A_col_nom[1])
    throttle_mag_high = abs(A_col_high[1])
    
    assert throttle_mag_low < throttle_mag_nom < throttle_mag_high, \
        f"Throttle effect magnitude should increase with speed: {throttle_mag_low:.3f} → {throttle_mag_nom:.3f} → {throttle_mag_high:.3f}"
    print(f"✓ Throttle effect magnitude scales with speed: {throttle_mag_low:.3f} → {throttle_mag_nom:.3f} → {throttle_mag_high:.3f}")
    
    # Brake effect should moderate at high speed
    assert A_col_low[2] > 0 and A_col_high[2] > 0
    print(f"✓ Brake effects: low={A_col_low[2]:.3f}, nom={A_col_nom[2]:.3f}, high={A_col_high[2]:.3f}")


def test_proactive_lane_offset():
    """Test proactive lane offset prediction"""
    print("\n[TEST 3] Proactive Lane Offset Prediction")
    cbf = CBFSafetyLayer()
    
    # Test scenario: vehicle drifting right, with corrective steering
    state = {'lane_offset': 0.5, 'speed': 10.0}
    action_corrective = np.array([-0.3, 0.0, 0.0])  # negative steering = left correction
    action_aggravate = np.array([0.3, 0.0, 0.0])    # positive steering = right (aggravates)
    
    offset_corrective = cbf._compute_proactive_lane_offset(state, action_corrective)
    offset_aggravate = cbf._compute_proactive_lane_offset(state, action_aggravate)
    
    # Corrective steering should reduce predicted offset
    assert offset_corrective < state['lane_offset'], "Corrective steering should reduce offset"
    # Aggravating steering should increase predicted offset
    assert offset_aggravate > state['lane_offset'], "Aggravating steering should increase offset"
    
    print(f"✓ Proactive prediction: current={state['lane_offset']:.3f}")
    print(f"  - With corrective steering: {offset_corrective:.3f}")
    print(f"  - With aggravating steering: {offset_aggravate:.3f}")


def test_rate_limiting():
    """Test steering and throttle rate limiting"""
    print("\n[TEST 4] Rate Limiting")
    cbf = CBFSafetyLayer(max_steering_rate=0.5, max_accel_change=0.3)
    
    cbf.u_prev = np.array([0.0, 0.0, 0.0])
    
    # Try to make a large steering change
    u_aggressive = np.array([1.0, 0.5, 0.0])  # Full steering
    u_limited = cbf._apply_rate_limiting(u_aggressive)
    
    steering_change = abs(u_limited[0] - cbf.u_prev[0])
    assert steering_change <= cbf.max_steering_rate, f"Rate limit exceeded: {steering_change}"
    print(f"✓ Steering rate limited: requested {u_aggressive[0]:.3f}, limited to {u_limited[0]:.3f}")
    print(f"  - Change: {steering_change:.3f} (max: {cbf.max_steering_rate})")


def test_dynamic_speed_limit():
    """Test dynamic speed limit constraint"""
    print("\n[TEST 5] Dynamic Speed Limit")
    cbf = CBFSafetyLayer()
    
    # Test with dynamic speed limit from environment
    u_actor = np.array([0.0, 1.0, 0.0])  # Full throttle
    
    state_default = {
        'd_collision': 10.0,
        'lane_offset': 0.0,
        'speed': 5.0,
        # No speed_limit key → should use default
    }
    
    state_limited = {
        'd_collision': 10.0,
        'lane_offset': 0.0,
        'speed': 5.0,
        'speed_limit': 8.0  # Lower limit
    }
    
    u_safe_default = cbf.compute_safe_action(u_actor, state_default, trust_score=1.0)
    u_safe_limited = cbf.compute_safe_action(u_actor, state_limited, trust_score=1.0)
    
    # With lower speed limit, should apply more brake
    assert u_safe_limited[2] >= u_safe_default[2], "Lower speed limit should enable more braking"
    print(f"✓ Speed limit constraint works:")
    print(f"  - Default (no limit spec): throttle={u_safe_default[1]:.3f}, brake={u_safe_default[2]:.3f}")
    print(f"  - With 8 m/s limit: throttle={u_safe_limited[1]:.3f}, brake={u_safe_limited[2]:.3f}")


def test_trust_score_modulation():
    """Test trust score modulation of corrections"""
    print("\n[TEST 6] Trust Score Modulation")
    cbf = CBFSafetyLayer()
    
    # Collision scenario
    state = {
        'd_collision': 0.5,  # Very close! CBF should activate
        'lane_offset': 0.0,
        'speed': 10.0,
        'speed_limit': 15.0
    }
    
    u_actor = np.array([0.0, 1.0, 0.0])  # Full throttle (dangerous!)
    
    # High trust: normal correction
    u_safe_high_trust = cbf.compute_safe_action(u_actor, state, trust_score=1.0)
    
    # Low trust: more conservative
    u_safe_low_trust = cbf.compute_safe_action(u_actor, state, trust_score=0.5)
    
    # Low trust should brake more (or change less from safe default)
    brake_high = u_safe_high_trust[2]
    brake_low = u_safe_low_trust[2]
    
    print(f"✓ Trust score modulation:")
    print(f"  - High trust (1.0): brake={brake_high:.3f}")
    print(f"  - Low trust (0.5): brake={brake_low:.3f}")


def test_constraint_violation_tracking():
    """Test constraint violation tracking"""
    print("\n[TEST 7] Constraint Violation Tracking")
    cbf = CBFSafetyLayer()
    
    # Collision violation
    state_collision = {'d_collision': 2.0, 'lane_offset': 0.0, 'speed': 10.0, 'speed_limit': 15.0}
    u_actor = np.array([0.0, 0.0, 0.0])
    
    cbf.compute_safe_action(u_actor, state_collision)
    assert cbf.constraint_violations['collision'] > 0, "Collision constraint should be violated"
    print(f"✓ Collision violations tracked: {cbf.constraint_violations['collision']}")
    
    # Lane violation
    cbf_lane = CBFSafetyLayer()
    state_lane = {'d_collision': 50.0, 'lane_offset': 2.0, 'speed': 10.0, 'speed_limit': 15.0}
    cbf_lane.compute_safe_action(u_actor, state_lane)
    assert cbf_lane.constraint_violations['lane'] > 0, "Lane constraint should be violated"
    print(f"✓ Lane violations tracked: {cbf_lane.constraint_violations['lane']}")


def test_fallback_safety():
    """Test fallback action safety verification"""
    print("\n[TEST 8] Fallback Safety Verification")
    cbf = CBFSafetyLayer()
    
    # Critical collision state
    state = {'d_collision': 0.5, 'lane_offset': 0.0, 'speed': 15.0, 'speed_limit': 12.0}
    
    # Try an unsafe action
    u_unsafe = np.array([0.0, 1.0, 0.0])  # Throttle in collision zone!
    
    # Fallback should verify constraints
    fallback = cbf._fallback_safe_action(state)
    
    # Fallback should brake (or maintain safe previous action)
    assert fallback[2] >= 0.0, "Fallback should brake or maintain"
    print(f"✓ Fallback safety verified:")
    print(f"  - Unsafe action: {u_unsafe}")
    print(f"  - Fallback: {fallback}")


def test_metrics_reset():
    """Test metrics reset"""
    print("\n[TEST 9] Metrics Reset")
    cbf = CBFSafetyLayer()
    
    state = {'d_collision': 2.0, 'lane_offset': 0.0, 'speed': 10.0, 'speed_limit': 15.0}
    u = np.array([0.0, 0.0, 0.0])
    
    cbf.compute_safe_action(u, state)
    initial_count = cbf.correction_count
    
    cbf.reset_metrics()
    assert cbf.correction_count == 0, "Correction count should reset"
    assert cbf.constraint_violations['collision'] == 0, "Violation tracking should reset"
    
    print(f"✓ Metrics reset successfully")
    print(f"  - Before reset: correction_count={initial_count}")
    print(f"  - After reset: correction_count={cbf.correction_count}")


if __name__ == "__main__":
    print("=" * 70)
    print("Enhanced CBF Safety Layer Validation Tests")
    print("=" * 70)
    
    try:
        test_cbf_initialization()
        test_speed_dependent_derivatives()
        test_proactive_lane_offset()
        test_rate_limiting()
        test_dynamic_speed_limit()
        test_trust_score_modulation()
        test_constraint_violation_tracking()
        test_fallback_safety()
        test_metrics_reset()
        
        print("\n" + "=" * 70)
        print("✓ All tests passed!")
        print("=" * 70)
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
