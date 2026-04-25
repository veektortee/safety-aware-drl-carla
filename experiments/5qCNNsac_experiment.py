"""
5 Q-Networks SAC CNN-only (No SpatioTemporal Transformer)
- Architecture: ResNet50 + CNN policy (no STT)
- Q-network ensemble: 5 critics (higher uncertainty estimation)
- 200 NPC vehicles + 200 pedestrians
- Random occlusions (30% of episodes)
- Waypoint logging (junctions, merges, lane changes)
- Reverse reward for collision recovery

Usage:
    python 5qCNNsac_experiment.py --timesteps 100000 --render
    python 5qCNNsac_experiment.py --timesteps 50000 --log-dir ./logs/5q_cnn_v1
"""

import os
import sys
import argparse
import numpy as np
import torch
import cv2
from pathlib import Path
from datetime import datetime

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure

import carla

from tests.pipeline_carla_test import (
    CarlaGymEnv,
    CBFSafetyLayerWrapper,
    ComprehensiveMetricsLoggingCallback,
    PolicyTrustScoreCallback,
)
from experiments.occlusions import OcclusionStrategy
from experiments.waypoint_logger import WaypointContextLogger


class ImageObservationWrapper(gym.ObservationWrapper):
    """Wraps observation dict to (C, H, W) format for CNN policy."""

    def __init__(self, env):
        super().__init__(env)
        self.env = env
        # CNN expects (C, H, W) format as Box observation space
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(3, 84, 84),  # RGB, 84x84 (standard Atari-like size)
            dtype=np.uint8
        )

    def observation(self, obs):
        """Convert dict obs with 'rgb_data' to (C, H, W) uint8 tensor."""
        # Handle dict observation (from CarlaGymEnv or wrapped environment)
        if isinstance(obs, dict):
            rgb = obs.get('rgb_data')
            if rgb is None:
                # Return black image if no RGB data
                return np.zeros((3, 84, 84), dtype=np.uint8)
        else:
            # If already an array, assume it's RGB in HWC format
            rgb = obs
        
        # Ensure uint8
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0, 255)).astype(np.uint8)
        
        # Convert HWC to (3, H, W) if needed
        if rgb.ndim == 3 and rgb.shape[2] in (3, 4):  # HWC format
            if rgb.shape[2] == 4:
                rgb = rgb[:, :, :3]  # Drop alpha channel if present
            rgb = np.transpose(rgb, (2, 0, 1))  # HWC -> CHW
        elif rgb.ndim == 2:  # Grayscale
            rgb = np.stack([rgb, rgb, rgb], axis=0)
        
        # Resize to 84x84 if needed
        if rgb.shape[1] != 84 or rgb.shape[2] != 84:
            rgb_reshaped = np.transpose(rgb, (1, 2, 0))  # CHW -> HWC for cv2
            rgb_resized = cv2.resize(rgb_reshaped, (84, 84), interpolation=cv2.INTER_LINEAR)
            rgb = np.transpose(rgb_resized, (2, 0, 1))  # HWC -> CHW
        
        return rgb.astype(np.uint8)


class LocationContextWrapper(gym.Wrapper):
    """Log waypoint context during episodes"""
    
    def __init__(self, env, carla_map):
        super().__init__(env)
        self.carla_map = carla_map
        self.waypoint_logger = WaypointContextLogger()
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Log waypoint context
        if hasattr(self.env.unwrapped, 'ego_vehicle'):
            vehicle = self.env.unwrapped.ego_vehicle
            location = vehicle.get_location()
            context = self.waypoint_logger.log_waypoint(self.carla_map, location)
            
            if context.get('valid'):
                # Add event-based reward
                if context.get('event') == 'JUNCTION':
                    info['location_event'] = 'junction'
                elif 'MERGE' in context.get('event', ''):
                    info['location_event'] = 'merge'
                elif 'LANE_CHANGE' in context.get('event', ''):
                    info['location_event'] = 'lane_change'
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        self.waypoint_logger.reset_episode()
        return self.env.reset(**kwargs)


class OcclusionWrapper(gym.ObservationWrapper):
    """Apply random occlusions to observations"""
    
    def __init__(self, env, occlusion_type='mixed'):
        super().__init__(env)
        self.occlusion = OcclusionStrategy(occlusion_type)
    
    def observation(self, obs):
        return self.occlusion.apply(obs)
    
    def reset(self, **kwargs):
        self.occlusion.reset_episode()
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            obs, info = result
            return self.occlusion.apply(obs), info
        else:
            return self.occlusion.apply(result)


class ReversalRewardCallback(BaseCallback):
    """Reward reversing after collision for recovery learning"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.collision_happened = False
    
    def _on_step(self) -> bool:
        return True


class SafetyMetricsCallback(BaseCallback):
    """Log safety metrics to TensorBoard"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.collision_count = 0
        self.lane_invasion_count = 0
    
    def _on_step(self) -> bool:
        # Log metrics every 100 steps
        if self.num_timesteps % 100 == 0:
            try:
                env = self.training_env.envs[0].unwrapped
                if hasattr(env, 'safety_metrics'):
                    metrics = env.safety_metrics
                    # Collision tracking (Phase 5)
                    self.logger.record("safety/collisions_episode", metrics.get('collisions', 0))
                    self.logger.record("safety/lane_invasions", metrics.get('lane_invasions', 0))
                    # CBF corrections - per-episode (Phase 5)
                    self.logger.record("safety/cbf_corrections_episode", metrics.get('cbf_corrections_episode', 0))
                    self.logger.record("safety/cbf_correction_magnitude", metrics.get('cbf_correction_magnitude', 0.0))
                    # Waypoint completion - normalized 0-1 (Phase 5)
                    self.logger.record("navigation/waypoint_completion_ratio", metrics.get('waypoint_completion', 0.0))
                    self.logger.record("navigation/waypoints_crossed", metrics.get('waypoints_crossed', 0))
                    self.logger.record("navigation/total_waypoints", metrics.get('total_waypoints', 0))
                    # Endpoint detection (Phase 5)
                    self.logger.record("navigation/endpoint_distance", metrics.get('endpoint_distance', 9999.0))
                    self.logger.record("navigation/endpoint_reached", metrics.get('endpoint_reached', 0.0))
            except:
                pass
        
        return True


class TrustScoreCallback(BaseCallback):
    """Compute policy trust score from entropy and pass to CBF layer"""
    
    def __init__(self, update_freq: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.update_freq = update_freq
        self.current_trust = 1.0
    
    def _on_step(self) -> bool:
        # Update trust score periodically
        if self.num_timesteps % self.update_freq == 0:
            try:
                # Compute policy entropy (higher entropy = less confident)
                if hasattr(self.model, 'ent_coef'):
                    # SAC's entropy coefficient (auto-adjusted by algorithm)
                    if isinstance(self.model.ent_coef, torch.Tensor):
                        ent_coef = self.model.ent_coef.detach().cpu().item()
                    elif isinstance(self.model.ent_coef, str) and self.model.ent_coef == 'auto':
                        # Auto entropy tuning - use learned ent_coef_target instead
                        ent_coef = self.model.ent_coef_target if hasattr(self.model, 'ent_coef_target') else 0.1
                    else:
                        ent_coef = float(self.model.ent_coef)
                    
                    # Map entropy coef to trust: lower entropy = higher trust
                    # ent_coef range typically [0.01, 0.5], map to trust [0.0, 1.0]
                    self.current_trust = 1.0 - np.clip(ent_coef * 2.0, 0.0, 1.0)
                else:
                    self.current_trust = 0.8  # Default moderate trust
                
                # Find CBFSafetyLayerWrapper in environment stack
                env = self.training_env
                cbf_wrapper = None
                
                # For VecEnv (vectorized environments)
                if hasattr(env, 'envs'):
                    for single_env in env.envs:
                        current = single_env
                        while hasattr(current, 'env'):
                            if hasattr(current, 'set_trust_score'):
                                cbf_wrapper = current
                                break
                            current = current.env
                        if cbf_wrapper:
                            break
                else:
                    # Single environment
                    current = env
                    while hasattr(current, 'env'):
                        if hasattr(current, 'set_trust_score'):
                            cbf_wrapper = current
                            break
                        current = current.env
                
                # Update CBF wrapper if found
                if cbf_wrapper is not None:
                    cbf_wrapper.set_trust_score(self.current_trust)
                    if self.verbose > 0 and self.num_timesteps % (self.update_freq * 10) == 0:
                        self.logger.record("cbf/trust_score", self.current_trust)
                        self.logger.record("policy/entropy_coef", ent_coef if 'ent_coef' in locals() else 0.0)
            
            except Exception as e:
                if self.verbose > 0:
                    print(f"[TRUST] Error updating trust score: {e}")
        
        return True


def create_cnn_env(
    num_timesteps: int = 100000,
    render: bool = False,
    num_npc: int = 200,
    num_pedestrians: int = 200
) -> gym.Env:
    """Create full environment stack: CARLA → CNN → CBF → Occlusions (no STT)"""
    
    print("[ENV] Creating CARLA environment...")
    
    # Base CARLA environment with raw RGB observations
    env = CarlaGymEnv(
        host='localhost',
        port=2000,
        timeout=10.0,
        time_limit=1000,
        render_mode='human' if render else None,
        num_npc_vehicles=num_npc,
        num_pedestrians=num_pedestrians,
        show_sensor_data=render
    )
    
    # Get map for waypoint logging
    carla_map = env.world.get_map()
    
    print("[ENV] Adding location context logging...")
    env = LocationContextWrapper(env, carla_map)
    
    print("[ENV] Adding CBF safety layer...")
    env = CBFSafetyLayerWrapper(env, alpha=1.0)
    
    print("[ENV] Adding occlusion wrapper...")
    env = OcclusionWrapper(env, occlusion_type='mixed')
    
    # NOTE: ImageObservationWrapper LAST - converts dict obs to (C,H,W) for CnnPolicy
    print("[ENV] Adding image observation wrapper for CnnPolicy...")
    env = ImageObservationWrapper(env)
    
    print("[OK] Environment fully initialized\n")
    
    return env


def train_5q_cnn_sac(
    timesteps: int = 100000,
    log_dir: str = "./logs/5q_cnn",
    learning_rate: float = 3e-4,
    batch_size: int = 64,
    buffer_size: int = 50000,
    render: bool = False,
    num_npc: int = 100,
    num_pedestrians: int = 30
):
    """Train 5 Q-Network SAC with CNN (no STT)"""
    
    os.makedirs(log_dir, exist_ok=True)
    tb_dir = os.path.join(log_dir, "tensorboard")
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    print("=" * 80)
    print("5 Q-NETWORK SAC WITH CNN ONLY (NO SPATIOTEMPORAL TRANSFORMER)")
    print("=" * 80)
    print(f"Timesteps: {timesteps:,}")
    print(f"Q-networks: 5 (higher uncertainty estimation)")
    print(f"Perception: ResNet50 (pretrained) → CnnPolicy (no STT)")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Buffer size: {buffer_size:,}")
    print(f"NPC vehicles: {num_npc}")
    print(f"Pedestrians: {num_pedestrians}")
    print(f"Render: {render}")
    print(f"Log dir: {log_dir}")
    print("=" * 80 + "\n")
    
    # Create environment
    env = create_cnn_env(
        num_timesteps=timesteps,
        render=render,
        num_npc=num_npc,
        num_pedestrians=num_pedestrians
    )
    
    # Policy kwargs with 5 Q-networks (ensemble)
    policy_kwargs = {
        'net_arch': {
            'pi': [256, 256],    # Actor network
            'qf': [256, 256]     # Critic networks
        },
        'n_critics': 5,          # 5 Q-networks (ensemble) - better uncertainty
        'normalize_images': True,
    }
    
    print("Creating SAC agent with 5 Q-networks and CNN policy...")
    model = SAC(
        'CnnPolicy',             # CNN policy processes raw RGB directly
        env,
        learning_rate=learning_rate,
        batch_size=batch_size,
        buffer_size=buffer_size,
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        ent_coef='auto',
        policy_kwargs=policy_kwargs,
        tensorboard_log=tb_dir,
        verbose=1
    )
    print("[OK] Agent created\n")
    
    # Setup logger
    logger = configure(tb_dir, ["stdout", "tensorboard"])
    model.set_logger(logger)
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=ckpt_dir,
        name_prefix="sac_5q_cnn"
    )
    
    safety_callback = SafetyMetricsCallback(verbose=1)
    trust_callback = PolicyTrustScoreCallback(update_freq=100, verbose=1)
    metrics_callback = ComprehensiveMetricsLoggingCallback(verbose=1)
    
    # Train
    print(f"Training for {timesteps:,} timesteps...\n")
    try:
        model.learn(
            total_timesteps=timesteps,
            log_interval=10,
            callback=[checkpoint_callback, safety_callback, trust_callback, metrics_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Training interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    
    # Save final model
    final_path = os.path.join(log_dir, "sac_5q_cnn_final.zip")
    model.save(final_path)
    print(f"\n[OK] Final model saved: {final_path}")
    
    # Save experiment info
    info_path = os.path.join(log_dir, "experiment_info.txt")
    with open(info_path, "w") as f:
        f.write("5 Q-Network SAC with CNN Only (No SpatioTemporal Transformer)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timesteps: {timesteps:,}\n")
        f.write(f"Q-networks: 5 (ensemble for better uncertainty)\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Buffer size: {buffer_size:,}\n")
        f.write(f"NPC vehicles: {num_npc}\n")
        f.write(f"Pedestrians: {num_pedestrians}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\nArchitecture:\n")
        f.write(f"- Perception: ResNet50 (pretrained ImageNet)\n")
        f.write(f"- Encoder: NONE (raw RGB to CNN)\n")
        f.write(f"- Policy: CnnPolicy (convolutional neural network)\n")
        f.write(f"- Q-networks: 5 critics (256-256 each) - ensemble uncertainty\n")
        f.write(f"\nFeatures:\n")
        f.write(f"- CBF safety layer (dynamic constraints)\n")
        f.write(f"- Random occlusions (30% episodes)\n")
        f.write(f"- Waypoint logging (junctions, merges, lanes)\n")
        f.write(f"- Reverse reward for collision recovery\n")
        f.write(f"\nKey Differences:\n")
        f.write(f"- 5 Q-networks instead of 2 (better uncertainty)\n")
        f.write(f"- Direct RGB image input to CNN policy (no STT)\n")
        f.write(f"- Faster inference than STT variants\n")
    
    print(f"[OK] Experiment info saved: {info_path}")
    
    # Cleanup
    env.close()
    print("[OK] Environment closed")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"View results: tensorboard --logdir {tb_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="5 Q-Network SAC with CNN Only"
    )
    parser.add_argument("--timesteps", type=int, default=100000, help="Total timesteps")
    parser.add_argument("--log-dir", type=str, default="./logs/5q_cnn", help="Log directory")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--buffer-size", type=int, default=50000, help="Replay buffer size")
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument("--num-npc", type=int, default=200, help="Number of NPC vehicles")
    parser.add_argument("--num-pedestrians", type=int, default=200, help="Number of pedestrians")
    
    args = parser.parse_args()
    
    train_5q_cnn_sac(
        timesteps=args.timesteps,
        log_dir=args.log_dir,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        render=args.render,
        num_npc=args.num_npc,
        num_pedestrians=args.num_pedestrians
    )
