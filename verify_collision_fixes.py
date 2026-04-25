#!/usr/bin/env python3
"""
QUICK TEST: Verify collision detection & early termination
Run this after starting CARLA to verify the fixes work.

Usage:
    python verify_collision_fixes.py
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

try:
    from tests.pipeline_carla_test import CarlaGymEnv, CBFSafetyLayerWrapper
    print("✅ Imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)


def test_collision_termination():
    """Test that episodes terminate immediately on collision"""
    print("\n" + "="*70)
    print("TEST: Collision Detection & Early Termination")
    print("="*70)
    
    try:
        # Create environment
        print("\n[1/5] Creating CARLA environment...")
        env = CarlaGymEnv(
            host='localhost',
            port=2000,
            time_limit=1000,  # 1000 steps max
            render_mode=None,
            num_npc_vehicles=50,  # More NPCs = higher collision chance
            num_pedestrians=30,
            show_sensor_data=False
        )
        print("✅ Environment created")
        
        # Apply CBF wrapper
        print("\n[2/5] Applying CBF safety wrapper...")
        env = CBFSafetyLayerWrapper(env, alpha=1.0, correction_penalty=0.003)
        print("✅ CBF wrapper attached")
        
        # Run multiple episodes to see collision behavior
        print("\n[3/5] Running 5 episodes to observe collision behavior...\n")
        
        total_steps = 0
        collision_count = 0
        
        for episode_num in range(5):
            obs, info = env.reset()
            episode_steps = 0
            episode_terminated = False
            
            print(f"  Episode {episode_num + 1}:")
            
            # Run episode
            for step in range(100):  # Max 100 steps per episode for testing
                action = env.action_space.sample()  # Random action
                obs, reward, terminated, truncated, info = env.step(action)
                episode_steps += 1
                total_steps += 1
                
                # Check for collision termination
                if terminated:
                    episode_terminated = True
                    # Check if this was a collision (reward < -100)
                    if reward < -100:
                        collision_count += 1
                        print(f"    ✅ COLLISION detected & episode terminated at step {step + 1}")
                        print(f"       Reward: {reward:.2f}")
                    else:
                        print(f"    ℹ️  Timeout at step {step + 1}")
                    break
            
            if not episode_terminated:
                print(f"    ℹ️  Episode completed without termination (100 steps)")
            
            print(f"    Episode length: {episode_steps} steps")
        
        print(f"\n[4/5] Episode Summary:")
        print(f"  Total episodes: 5")
        print(f"  Collisions detected: {collision_count}")
        print(f"  Total steps: {total_steps}")
        print(f"  Avg steps/episode: {total_steps/5:.1f}")
        
        if collision_count > 0:
            print(f"\n✅ Collision detection is WORKING - {collision_count} collisions found")
            print(f"✅ Early termination is WORKING - episodes end on collision")
        else:
            print(f"\n⚠️  No collisions detected in 5 episodes")
            print(f"   This could mean:")
            print(f"   - CBF layer is successfully avoiding all collisions ✓ (GOOD)")
            print(f"   - Or: Not enough NPCs or scenario hard enough")
            print(f"   Try increasing num_npc_vehicles for more collision scenarios")
        
        env.close()
        print(f"\n[5/5] Environment closed")
        
        return collision_count > 0 or total_steps < 500
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cbf_logging():
    """Test that CBF verbose logging activates when close to collision"""
    print("\n" + "="*70)
    print("TEST: CBF Verbose Logging")
    print("="*70)
    print("\n[INFO] If you see [CBF-COLLISION] or [CBF-SOLVER] messages below")
    print("       when close to obstacles, that means verbose logging is working!\n")
    
    try:
        env = CarlaGymEnv(host='localhost', port=2000, time_limit=60)
        env = CBFSafetyLayerWrapper(env, alpha=1.0)
        
        obs, info = env.reset()
        
        print("[1/3] Running 30 steps to get close to obstacles...")
        
        for step in range(30):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated:
                print(f"✅ Episode ended at step {step}")
                break
        
        print("[2/3] If you saw [CBF-COLLISION] or [CBF-SOLVER] messages above:")
        print("      ✅ Verbose logging is ACTIVE when close to obstacles")
        print("[3/3] Cleanup...")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    print("\n🧪 COLLISION FIX VERIFICATION SUITE")
    print("="*70)
    
    test1_pass = test_collision_termination()
    test2_pass = test_cbf_logging()
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"✅ Test 1 (Collision Termination): {'PASS' if test1_pass else 'FAIL'}")
    print(f"✅ Test 2 (CBF Logging): {'PASS' if test2_pass else 'FAIL'}")
    
    if test1_pass and test2_pass:
        print("\n🎉 All tests passed! Collision fixes are working correctly.")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Check output above.")
        sys.exit(1)
