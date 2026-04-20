"""
Demo Script: View and Test Trained SAC Models
==============================================

Load pretrained SAC models from checkpoints and run inference in CARLA.

Supported Model Variants:
  - 5cnn.zip    : 5-layer CNN feature extractor
  - 2cnn.zip    : 2-layer CNN feature extractor
  - 5stt.zip    : 5-block spatiotemporal transformer
  - 2stt.zip    : 2-block spatiotemporal transformer

Usage:
    python demo/run_model_inference.py --model 5cnn --episodes 3 --render
    python demo/run_model_inference.py --model 2stt --episodes 5 --render --no-cbf
    python demo/run_model_inference.py --list-models
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Optional
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import carla
except ImportError:
    print("ERROR: CARLA module not found. Install via CARLA PythonAPI")
    sys.exit(1)

from stable_baselines3 import SAC
import gymnasium as gym
from tests.pipeline_carla_test import (
    CarlaGymEnv, 
    PipelineObservationWrapper, 
    CBFSafetyLayerWrapper,
)


class ModelInferenceEngine:
    """Load and run inference on trained SAC models"""
    
    # Model checkpoint directory
    CHECKPOINT_DIR = "demos"
    
    # Model name mapping
    MODEL_VARIANTS = {
        "5cnn": {"name": "SAC-5CNN", "desc": "5-layer CNN feature extractor"},
        "2cnn": {"name": "SAC-2CNN", "desc": "2-layer CNN feature extractor"},
        "5stt": {"name": "SAC-5STT", "desc": "5-block Spatiotemporal Transformer"},
        "2stt": {"name": "SAC-2STT", "desc": "2-block Spatiotemporal Transformer"},
    }
    
    def __init__(
        self,
        model_name: str,
        checkpoint_dir: str = None,
        use_cuda: bool = True,
        use_cbf: bool = True,
    ):
        """Initialize inference engine"""
        self.model_name = model_name
        
        # Fix checkpoint path - handle both relative and absolute paths
        if checkpoint_dir is None:
            checkpoint_dir = self.CHECKPOINT_DIR
        
        # If relative path, resolve from parent directory (not demo/)
        if not os.path.isabs(checkpoint_dir) and not checkpoint_dir.startswith('..'):
            # We're in demo folder, go up one level
            checkpoint_dir = os.path.join('..', checkpoint_dir)
        
        self.checkpoint_dir = os.path.abspath(checkpoint_dir)
        self.use_cuda = use_cuda
        self.use_cbf = use_cbf
        self.model = None
        self.env = None
        self.device = "cuda" if use_cuda else "cpu"
        
        # Stats tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_collisions = []
        self.episode_waypoints = []
    
    def list_available_models(self) -> dict:
        """List all available checkpoint models"""
        print("\n" + "="*70)
        print("AVAILABLE MODELS")
        print("="*70)
        
        for model_id, info in self.MODEL_VARIANTS.items():
            checkpoint_path = os.path.join(self.checkpoint_dir, f"{model_id}.zip")
            exists = os.path.exists(checkpoint_path)
            status = "✓ Available" if exists else "✗ Not found"
            print(f"  {model_id:8s} | {info['name']:30s} | {status}")
            print(f"           | {info['desc']:30s} |")
            if exists:
                size_mb = os.path.getsize(checkpoint_path) / (1024*1024)
                print(f"           | Size: {size_mb:.2f} MB                      |")
            print()
        
        print("="*70 + "\n")
        return self.MODEL_VARIANTS
    
    def load_model(self) -> bool:
        """Load model checkpoint"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.model_name}.zip")
        
        print(f"[DEBUG] Checkpoint directory: {self.checkpoint_dir}")
        print(f"[DEBUG] Looking for: {checkpoint_path}")
        
        # Check if file exists
        if not os.path.exists(checkpoint_path):
            print(f"[ERROR] Model not found: {checkpoint_path}")
            print(f"[ERROR] Directory exists: {os.path.isdir(self.checkpoint_dir)}")
            
            # List available files in directory
            if os.path.isdir(self.checkpoint_dir):
                files = os.listdir(self.checkpoint_dir)
                print(f"[ERROR] Files in {self.checkpoint_dir}: {files}")
            return False
        
        # Check file is readable
        if not os.access(checkpoint_path, os.R_OK):
            print(f"[ERROR] Model file not readable: {checkpoint_path}")
            return False
        
        print(f"[INFO] Loading model from: {checkpoint_path}")
        print(f"[INFO] File size: {os.path.getsize(checkpoint_path) / (1024*1024):.2f} MB")
        
        try:
            # Create environment for model loading
            print("[SETUP] Creating CARLA environment...")
            self.env = self._create_env()
            
            # Load SAC model
            model_path = checkpoint_path.replace(".zip", "")
            print(f"[INFO] Loading SAC model from: {model_path}")
            self.model = SAC.load(
                model_path,
                env=self.env,
                device=self.device,
            )
            
            print(f"[OK] Model loaded successfully")
            print(f"     Policy: {self.model.policy}")
            print(f"     Device: {self.device}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_env(self) -> gym.Env:
        """Create CARLA environment with pipeline"""
        print("[SETUP] Creating CARLA environment...")
        
        env = CarlaGymEnv(
            time_limit=300,
            render_mode="human",
            num_npc_vehicles=50,
            num_pedestrians=15,
            show_sensor_data=False
        )
        
        # Add pipeline wrapper
        env = PipelineObservationWrapper(env, embed_dim=512, num_frames=8)
        
        # Optionally add CBF safety layer
        if self.use_cbf:
            print("[SETUP] Adding CBF Safety Layer...")
            env = CBFSafetyLayerWrapper(env, alpha=1.0, use_trust_score=False)
        
        return env
    
    def run_inference_episode(self, episode_num: int = 1, verbose: bool = True) -> dict:
        """Run single inference episode"""
        if self.model is None or self.env is None:
            print("[ERROR] Model not loaded. Call load_model() first.")
            return {}
        
        print(f"\n{'='*70}")
        print(f"EPISODE {episode_num} - Inference")
        print(f"{'='*70}")
        print(f"Model: {self.MODEL_VARIANTS[self.model_name]['name']}")
        print(f"CBF Safety: {'Enabled' if self.use_cbf else 'Disabled'}")
        print(f"{'='*70}\n")
        
        # Reset environment
        obs, info = self.env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0
        step = 0
        
        # Episode loop
        while not (terminated or truncated):
            try:
                # Get action from model (deterministic mode)
                action, _ = self.model.predict(obs, deterministic=True)
                
                # Step environment
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                total_reward += reward
                step += 1
                
                # Log info every 50 steps
                if verbose and step % 50 == 0:
                    speed = obs[0] if hasattr(obs, '__len__') else 0  # Approximate
                    print(f"[Step {step:3d}] Reward: {reward:8.4f} | Total: {total_reward:10.4f}")
                    
                    # Print environment info if available
                    if hasattr(self.env.unwrapped, 'waypoints_crossed'):
                        wp = self.env.unwrapped.waypoints_crossed
                        total_wp = self.env.unwrapped.total_waypoints
                        dist = self.env.unwrapped._compute_collision_distance()
                        print(f"           | Waypoints: {wp}/{total_wp} | Collision Distance: {dist:.2f}m")
                
            except KeyboardInterrupt:
                print("\n[INTERRUPTED] Episode stopped by user")
                terminated = True
                break
            except Exception as e:
                print(f"[ERROR] Step {step}: {e}")
                break
        
        # Collect episode stats
        stats = {
            'episode_num': episode_num,
            'model': self.model_name,
            'total_reward': total_reward,
            'episode_length': step,
            'cbf_enabled': self.use_cbf,
        }
        
        # Get environment stats if available
        if hasattr(self.env.unwrapped, '_collision_count'):
            stats['collisions'] = self.env.unwrapped._collision_count
        if hasattr(self.env.unwrapped, 'waypoints_crossed'):
            stats['waypoints_crossed'] = self.env.unwrapped.waypoints_crossed
        if hasattr(self.env.unwrapped, '_cbf_correction_count_episode'):
            stats['cbf_corrections'] = self.env.unwrapped._cbf_correction_count_episode
        
        # Print episode summary
        print(f"\n{'='*70}")
        print("EPISODE SUMMARY")
        print(f"{'='*70}")
        print(f"Total Reward:      {total_reward:10.4f}")
        print(f"Episode Length:    {step:10d} steps")
        if 'collisions' in stats:
            print(f"Collisions:        {stats['collisions']:10d}")
        if 'waypoints_crossed' in stats:
            print(f"Waypoints Crossed: {stats['waypoints_crossed']:10d}")
        if 'cbf_corrections' in stats:
            print(f"CBF Corrections:   {stats['cbf_corrections']:10d}")
        print(f"{'='*70}\n")
        
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(step)
        if 'collisions' in stats:
            self.episode_collisions.append(stats['collisions'])
        if 'waypoints_crossed' in stats:
            self.episode_waypoints.append(stats['waypoints_crossed'])
        
        return stats
    
    def run_inference_episodes(self, num_episodes: int = 3, verbose: bool = True) -> list:
        """Run multiple inference episodes"""
        results = []
        
        for episode_idx in range(num_episodes):
            try:
                stats = self.run_inference_episode(episode_num=episode_idx + 1, verbose=verbose)
                results.append(stats)
            except Exception as e:
                print(f"[ERROR] Episode {episode_idx + 1} failed: {e}")
                continue
        
        # Print overall statistics
        self._print_summary(results)
        return results
    
    def _print_summary(self, results: list):
        """Print summary statistics across all episodes"""
        if not results:
            print("[ERROR] No episodes completed")
            return
        
        rewards = [r['total_reward'] for r in results]
        lengths = [r['episode_length'] for r in results]
        
        print(f"\n{'='*70}")
        print("INFERENCE SUMMARY")
        print(f"{'='*70}")
        print(f"Model:              {self.MODEL_VARIANTS[self.model_name]['name']}")
        print(f"Episodes Completed: {len(results)}/{len(results)}")
        print(f"\nReward Statistics:")
        print(f"  Mean:             {np.mean(rewards):10.4f}")
        print(f"  Std:              {np.std(rewards):10.4f}")
        print(f"  Min:              {np.min(rewards):10.4f}")
        print(f"  Max:              {np.max(rewards):10.4f}")
        print(f"\nLength Statistics:")
        print(f"  Mean:             {np.mean(lengths):10.2f} steps")
        print(f"  Std:              {np.std(lengths):10.2f} steps")
        print(f"  Min:              {np.min(lengths):10d} steps")
        print(f"  Max:              {np.max(lengths):10d} steps")
        
        if self.episode_collisions:
            print(f"\nSafety Statistics:")
            print(f"  Total Collisions: {sum(self.episode_collisions)}")
            print(f"  Mean per Ep:      {np.mean(self.episode_collisions):10.2f}")
        
        if self.episode_waypoints:
            print(f"\nNavigation Statistics:")
            print(f"  Mean Waypoints:   {np.mean(self.episode_waypoints):10.2f}")
            print(f"  Total Waypoints:  {sum(self.episode_waypoints)}")
        
        print(f"{'='*70}\n")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.env is not None:
            try:
                self.env.close()
            except:
                pass
        print("[OK] Cleanup complete")


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="View and test trained SAC models in CARLA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  
  View available models:
    python demo/run_model_inference.py --list-models
  
  Run inference with 5CNN model (3 episodes):
    python demo/run_model_inference.py --model 5cnn --episodes 3
  
  Test 2STT model with CBF disabled:
    python demo/run_model_inference.py --model 2stt --episodes 1 --no-cbf
  
  Test all models (1 episode each):
    python demo/run_model_inference.py --model 5cnn --episodes 1
    python demo/run_model_inference.py --model 2cnn --episodes 1
    python demo/run_model_inference.py --model 5stt --episodes 1
    python demo/run_model_inference.py --model 2stt --episodes 1
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        choices=["5cnn", "2cnn", "5stt", "2stt"],
        default="5cnn",
        help="Model variant to test (default: 5cnn)"
    )
    
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of inference episodes to run (default: 3)"
    )
    
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="demos",
        help="Directory containing model checkpoints (default: demos/checkpoints)"
    )
    
    parser.add_argument(
        "--no-cbf",
        action="store_true",
        help="Disable CBF safety layer during inference"
    )
    
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU instead of GPU"
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available model variants and exit"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    # Initialize inference engine
    engine = ModelInferenceEngine(
        model_name=args.model,
        checkpoint_dir=args.checkpoint_dir,
        use_cuda=not args.cpu,
        use_cbf=not args.no_cbf,
    )
    
    # List models if requested
    if args.list_models:
        engine.list_available_models()
        return
    
    # Print banner
    print("\n" + "="*70)
    print("SAC MODEL INFERENCE - CARLA")
    print("="*70)
    print(f"Model:           {engine.MODEL_VARIANTS[args.model]['name']}")
    print(f"Description:     {engine.MODEL_VARIANTS[args.model]['desc']}")
    print(f"Episodes:        {args.episodes}")
    print(f"CBF Safety:      {'Enabled' if not args.no_cbf else 'Disabled'}")
    print(f"Device:          {'GPU' if not args.cpu else 'CPU'}")
    print("="*70 + "\n")
    
    # Load model
    if not engine.load_model():
        print("[ERROR] Failed to load model. Exiting.")
        return
    
    # Run inference
    try:
        results = engine.run_inference_episodes(
            num_episodes=args.episodes,
            verbose=not args.quiet
        )
        
        # Save results
        results_file = f"demo/results_{args.model}_{args.episodes}eps.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"[OK] Results saved to: {results_file}\n")
        
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Inference stopped by user")
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        engine.cleanup()


if __name__ == "__main__":
    main()
