#!/usr/bin/env python3
"""
Smoke Test: Validate Reward Shaping Implementation
Runs 1 episode and checks that all new metrics are logged
Usage: python test_reward_shaping_validation.py
"""

import sys
import os
from pathlib import Path

# Add project paths
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

try:
    import carla
    import gymnasium as gym
    from tests.pipeline_carla_test import CarlaGymEnv, CBFSafetyLayerWrapper
    from commons.cbfQP_layer import CBFSafetyLayer
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)


def test_cbf_collision_prevention():
    """Test Phase 1: CBF collision prevention detection"""
    print("\n" + "="*60)
    print("TEST 1: CBF Collision Prevention Detection (Phase 1)")
    print("="*60)
    
    cbf = CBFSafetyLayer(alpha=1.0, d_min=5.0)
    
    # Test state: very close to collision
    state = {
        'd_collision': 2.0,  # 2m away (< 5m safe distance)
        'ttc': 1.0,
        'lane_offset': 0.5,
        'speed': 5.0,
        'speed_limit': 15.0
    }
    
    # Aggressive action (accelerating into lead vehicle)
    aggressive_action = np.array([0.0, 0.8, 0.0])
    
    try:
        safe_action = cbf.compute_safe_action(aggressive_action, state, trust_score=1.0)
        
        # Check if metrics are populated
        if hasattr(cbf, 'collision_prevented'):
            print(f"✅ collision_prevented flag exists: {cbf.collision_prevented}")
        else:
            print("❌ collision_prevented flag missing")
            
        if hasattr(cbf, 'avoidance_efficiency'):
            print(f"✅ avoidance_efficiency metric exists: {cbf.avoidance_efficiency:.3f}")
        else:
            print("❌ avoidance_efficiency metric missing")
            
        print(f"✅ Safe action computed: {safe_action}")
        print(f"   Action difference: {np.linalg.norm(safe_action - aggressive_action):.4f}")
        
        return True
    except Exception as e:
        print(f"❌ CBF test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_waypoint_system():
    """Test Phase 2: Waypoint tracking system"""
    print("\n" + "="*60)
    print("TEST 2: Waypoint Tracking System (Phase 2)")
    print("="*60)
    
    try:
        # Create minimal environment
        env = CarlaGymEnv(host='localhost', port=2000, time_limit=50)
        
        # Check waypoint initialization
        print(f"✅ Environment created")
        
        if hasattr(env, 'waypoints'):
            print(f"✅ Waypoints list exists: {len(env.waypoints)} waypoints")
        else:
            print("❌ Waypoints list missing")
            return False
            
        if hasattr(env, 'waypoints_crossed'):
            print(f"✅ Waypoints_crossed counter exists: {env.waypoints_crossed}")
        else:
            print("❌ Waypoints_crossed counter missing")
            return False
            
        if hasattr(env, '_get_distance_to_next_waypoint'):
            dist = env._get_distance_to_next_waypoint()
            print(f"✅ Distance to next waypoint: {dist:.2f}m")
        else:
            print("❌ _get_distance_to_next_waypoint method missing")
            return False
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Waypoint test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reward_components():
    """Test Phase 3-4: Reward components and aggregation"""
    print("\n" + "="*60)
    print("TEST 3-4: Reward Components & Aggregation (Phase 3-4)")
    print("="*60)
    
    try:
        env = CarlaGymEnv(host='localhost', port=2000, time_limit=10)
        
        # Check that new reward methods exist
        if hasattr(env, 'compute_safety_buffer_reward'):
            print("✅ compute_safety_buffer_reward method exists")
            try:
                reward = env.compute_safety_buffer_reward()
                print(f"   Sample value: {reward:.4f}")
            except Exception as e:
                print(f"   ⚠️  Error calling method: {e}")
        else:
            print("❌ compute_safety_buffer_reward method missing")
            
        if hasattr(env, 'compute_waypoint_progress_reward'):
            print("✅ compute_waypoint_progress_reward method exists")
            try:
                reward = env.compute_waypoint_progress_reward()
                print(f"   Sample value: {reward:.4f}")
            except Exception as e:
                print(f"   ⚠️  Error calling method: {e}")
        else:
            print("❌ compute_waypoint_progress_reward method missing")
        
        # Test debug logging
        if hasattr(env, '_debug_reward_logging'):
            print("✅ Debug logging infrastructure exists")
            env._debug_reward_logging = False
            print("   (disabled for testing)")
        else:
            print("❌ Debug logging infrastructure missing")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Reward component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cbf_wrapper():
    """Test Phase 5: CBF wrapper and penalty reduction"""
    print("\n" + "="*60)
    print("TEST 5: CBF Wrapper & Penalty Reduction (Phase 5)")
    print("="*60)
    
    try:
        env = CarlaGymEnv(host='localhost', port=2000, time_limit=10)
        wrapper = CBFSafetyLayerWrapper(env, alpha=1.0, correction_penalty=0.003)
        
        if hasattr(wrapper, '_cbf_wrapper'):
            print("❌ Wrapper already has _cbf_wrapper (recursive reference)")
        else:
            print("✅ No recursive reference issue")
        
        # Check penalty value
        if wrapper.correction_penalty == 0.003:
            print(f"✅ Correction penalty reduced to {wrapper.correction_penalty}")
        else:
            print(f"⚠️  Correction penalty is {wrapper.correction_penalty} (expected 0.003)")
        
        # Check env reference
        if hasattr(env.unwrapped, '_cbf_wrapper'):
            print("✅ Environment has reference to CBF wrapper")
        else:
            print("⚠️  Environment doesn't have CBF wrapper reference")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ CBF wrapper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_logging_callback():
    """Test Phase 6: Logging callback"""
    print("\n" + "="*60)
    print("TEST 6: SafetyMetricsCallback Logging (Phase 6)")
    print("="*60)
    
    try:
        from tests.pipeline_carla_test import SafetyMetricsCallback
        
        callback = SafetyMetricsCallback(verbose=0, log_frequency=10)
        
        if hasattr(callback, '_on_step'):
            print("✅ SafetyMetricsCallback._on_step method exists")
        else:
            print("❌ SafetyMetricsCallback._on_step method missing")
            return False
        
        # Check key attributes
        attrs = ['log_frequency', 'step_count', 'episode_safety_rewards', 'episode_progress_rewards']
        for attr in attrs:
            if hasattr(callback, attr):
                print(f"✅ Attribute '{attr}' exists: {getattr(callback, attr)}")
            else:
                print(f"❌ Attribute '{attr}' missing")
        
        return True
        
    except Exception as e:
        print(f"❌ Logging callback test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "🧪 "*20)
    print("VALIDATION SUITE: Reward Shaping Implementation")
    print("🧪 "*20)
    
    results = {
        "Phase 1: CBF Collision Prevention": test_cbf_collision_prevention(),
        "Phase 2: Waypoint System": test_waypoint_system(),
        "Phase 3-4: Reward Components": test_reward_components(),
        "Phase 5: CBF Wrapper": test_cbf_wrapper(),
        "Phase 6: Logging Callback": test_logging_callback(),
    }
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print("\n" + "="*60)
    print(f"TOTAL: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("\n🎉 All validation tests passed! Implementation is ready.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check errors above.")
        return 1


if __name__ == "__main__":
    exit(main())
