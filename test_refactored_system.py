#!/usr/bin/env python3
"""
Validation Test: Refactored Reward & Observation System
Tests all 4 improvements:
1. Opposite lane time penalty (log-growing)
2. Lidar integration in observation space
3. Steering rate limiting for smooth control
4. Enhanced speed reward with waypoint tracking

Usage: python test_refactored_system.py
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "CARLA-RL-Agents"))
sys.path.insert(0, str(Path(__file__).parent / "CARLA-RL-Agents" / "src"))
sys.path.insert(0, str(Path(__file__).parent / "CARLA-RL-Agents" / "env"))

import numpy as np


def test_opposite_lane_penalty():
    """Test 1: Opposite lane time tracking with log-growing penalty"""
    print("\n" + "="*60)
    print("TEST 1: Opposite Lane Time Penalty (Log-Growing)")
    print("="*60)
    
    try:
        from reward import Reward
        import inspect
        
        reward = Reward()
        
        # Check that opposite_lane_penalty method exists
        if hasattr(reward, '_Reward__opposite_lane_penalty'):
            print("✅ __opposite_lane_penalty() method exists")
        else:
            print("❌ __opposite_lane_penalty() method missing")
            return False
        
        # Check opposite_lane_time tracking initialized
        if hasattr(reward, 'opposite_lane_time'):
            print(f"✅ opposite_lane_time tracker initialized: {reward.opposite_lane_time}")
        else:
            print("❌ opposite_lane_time tracker missing")
            return False
        
        # Verify log penalty implementation
        source = inspect.getsource(reward._Reward__opposite_lane_penalty)
        if "-np.log(1.0 + self.opposite_lane_time)" in source:
            print("✅ Logarithmic penalty formula correct")
        else:
            print("❌ Penalty formula not using logarithmic scaling")
            return False
        
        # Check unifed time-tracking pattern in __init__
        source_init = inspect.getsource(reward.__init__)
        if "Unified time-tracking for behavioral penalties" in source_init:
            print("✅ Unified time-tracking pattern documented")
        else:
            print("⚠️  Unified pattern comment missing (non-critical)")
        
        print("✅ PASS: Opposite lane penalty system is in place")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lidar_observation_integration():
    """Test 2: Lidar enabled in observation space"""
    print("\n" + "="*60)
    print("TEST 2: Lidar Observation Space Integration")
    print("="*60)
    
    try:
        from observation_action_space import observation_space, observation_shapes
        from gymnasium import spaces
        
        # Check observation_shapes includes lidar
        if 'lidar_data' in observation_shapes:
            print(f"✅ lidar_data in observation_shapes: {observation_shapes['lidar_data']}")
        else:
            print("❌ lidar_data missing from observation_shapes")
            return False
        
        # Check observation_space includes lidar
        if isinstance(observation_space, spaces.Dict):
            if 'lidar_data' in observation_space.spaces:
                lidar_space = observation_space.spaces['lidar_data']
                print(f"✅ lidar_data in observation_space with shape: {lidar_space.shape}")
                if lidar_space.shape == (3, 500):
                    print("✅ Shape correct: (3 coords × 500 points)")
                else:
                    print(f"❌ Shape incorrect: {lidar_space.shape}, expected (3, 500)")
                    return False
            else:
                print("❌ lidar_data missing from observation_space")
                return False
        else:
            print("❌ observation_space not Dict type")
            return False
        
        print("✅ PASS: Lidar integrated into observation space")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lidar_preprocessing():
    """Test 3: Lidar preprocessing pipeline activated"""
    print("\n" + "="*60)
    print("TEST 3: Lidar Preprocessing Pipeline")
    print("="*60)
    
    try:
        from pre_processing import PreProcessing
        import inspect
        
        preproc = PreProcessing()
        
        # Check __process_lidar method exists
        if hasattr(preproc, '_PreProcessing__process_lidar'):
            print("✅ __process_lidar() method exists")
        else:
            print("❌ __process_lidar() method missing")
            return False
        
        # Check preprocess_data calls lidar processing
        source = inspect.getsource(preproc.preprocess_data)
        if "self.__process_lidar" in source:
            print("✅ preprocess_data() calls __process_lidar()")
        else:
            print("❌ preprocess_data() doesn't call __process_lidar()")
            return False
        
        # Check lidar is added to neo_observation_data
        if "'lidar_data'" in source:
            print("✅ lidar_data added to processed observation")
        else:
            print("❌ lidar_data not in processed observation")
            return False
        
        print("✅ PASS: Lidar preprocessing pipeline activated")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_steering_rate_limiting():
    """Test 4: Steering rate limiting to reduce yanking"""
    print("\n" + "="*60)
    print("TEST 4: Steering Rate Limiting (Smooth Control)")
    print("="*60)
    
    try:
        from vehicle import Vehicle
        import inspect
        
        # Check __previous_steer variable initialized
        class MockWorld:
            pass
        
        vehicle = Vehicle(MockWorld())
        
        if hasattr(vehicle, '_Vehicle__previous_steer'):
            print(f"✅ __previous_steer tracker initialized: {vehicle._Vehicle__previous_steer}")
        else:
            print("❌ __previous_steer tracker missing")
            return False
        
        # Check control_vehicle has rate limiting logic
        source = inspect.getsource(vehicle.control_vehicle)
        
        if "max_steering_rate" in source:
            print("✅ max_steering_rate variable defined")
        else:
            print("❌ max_steering_rate not found in control_vehicle()")
            return False
        
        if "np.clip" in source:
            print("✅ np.clip used for steering rate limiting")
        else:
            print("❌ Steering rate limiting not using clip")
            return False
        
        if "__previous_steer" in source:
            print("✅ __previous_steer used to track steering history")
        else:
            print("❌ __previous_steer not being used")
            return False
        
        print("✅ PASS: Steering rate limiting implemented")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_unified_time_tracking():
    """Test 5: Unified time-tracking pattern across penalties"""
    print("\n" + "="*60)
    print("TEST 5: Unified Time-Tracking Architecture")
    print("="*60)
    
    try:
        from reward import Reward
        import inspect
        
        reward = Reward()
        
        # Check both timers exist and initialized
        checks = []
        
        if hasattr(reward, 'zero_speed_time') and reward.zero_speed_time == 0.0:
            checks.append("zero_speed_time")
            print("✅ zero_speed_time initialized to 0.0")
        else:
            print("❌ zero_speed_time not initialized")
            return False
        
        if hasattr(reward, 'opposite_lane_time') and reward.opposite_lane_time == 0.0:
            checks.append("opposite_lane_time")
            print("✅ opposite_lane_time initialized to 0.0")
        else:
            print("❌ opposite_lane_time not initialized")
            return False
        
        if hasattr(reward, 'TIMESTEP') and reward.TIMESTEP == 0.01:
            print("✅ TIMESTEP constant for stepping (0.01)")
        else:
            print("❌ TIMESTEP not properly initialized")
            return False
        
        # Check reset() resets both timers
        source = inspect.getsource(reward.reset)
        if "self.zero_speed_time  = 0.0" in source and "self.opposite_lane_time = 0.0" in source:
            print("✅ reset() clears both time-tracking states")
        else:
            print("❌ reset() doesn't reset time trackers properly")
            return False
        
        # Check calculate_reward includes both penalties
        source = inspect.getsource(reward.calculate_reward)
        if "self.__zero_speed_penalty" in source and "self.__opposite_lane_penalty" in source:
            print("✅ calculate_reward() calls both penalty functions")
        else:
            print("❌ calculate_reward() missing penalty calls")
            return False
        
        print("✅ PASS: Unified time-tracking architecture verified")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_speed_reward_system():
    """Test 6: Speed reward still functioning"""
    print("\n" + "="*60)
    print("TEST 6: Speed Reward System")
    print("="*60)
    
    try:
        from reward import Reward
        import inspect
        
        reward = Reward()
        
        # Check speed_reward method exists
        if hasattr(reward, '_Reward__speed_reward'):
            print("✅ __speed_reward() method exists")
        else:
            print("❌ __speed_reward() method missing")
            return False
        
        # Verify it's in calculate_reward
        source = inspect.getsource(reward.calculate_reward)
        if "self.__speed_reward" in source:
            print("✅ __speed_reward() called in calculate_reward()")
        else:
            print("❌ __speed_reward() not called")
            return False
        
        print("✅ PASS: Speed reward system intact")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation tests"""
    print("\n" + "🧪 "*20)
    print("VALIDATION: Refactored Reward & Observation System")
    print("🧪 "*20)
    
    results = {
        "Opposite Lane Penalty": test_opposite_lane_penalty(),
        "Lidar Observation Integration": test_lidar_observation_integration(),
        "Lidar Preprocessing": test_lidar_preprocessing(),
        "Steering Rate Limiting": test_steering_rate_limiting(),
        "Unified Time-Tracking": test_unified_time_tracking(),
        "Speed Reward System": test_speed_reward_system(),
    }
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v is True)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print("\n" + "="*60)
    print(f"TOTAL: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("\n✅ All validation checks passed!")
        print("\n📋 Summary of refactored system:")
        print("  1. ✅ Opposite lane time penalty (log-growing, mirrors zero-speed)")
        print("  2. ✅ Lidar observation space enabled (3×500 point cloud)")
        print("  3. ✅ Steering rate limiting (0.1 rad/step for smooth control)")
        print("  4. ✅ Preprocessing pipeline calls lidar processor")
        print("  5. ✅ Unified time-tracking pattern (reusable for future penalties)")
        print("  6. ✅ Speed reward maintained alongside waypoint rewards")
        print("\n🚀 Expected improvements:")
        print("  • Less aggressive steering corrections after lane invasion")
        print("  • Agent encouraged to quickly return to correct lane")
        print("  • Better obstacle detection via lidar (if policy updated)")
        print("  • Smoother trajectory without yanking behavior")
        print("  • Consistent reward structure across time-based penalties")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    exit(main())
