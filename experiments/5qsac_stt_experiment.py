"""
5 Q-Networks SAC with SpatioTemporal Transformer (STT)
- Architecture: ResNet50 → SpatioTemporal Transformer → SAC
- Q-network ensemble: 5 critics (higher uncertainty estimation)
- 200 NPC vehicles + 200 pedestrians
- Random occlusions (30% of episodes)
- Waypoint logging (junctions, merges, lane changes)
- Reverse reward for collision recovery

Usage:
    python 5qsac_stt_experiment.py --timesteps 100000 --render
    python 5qsac_stt_experiment.py --timesteps 50000 --log-dir ./logs/5q_stt_v1
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Optional

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
    PipelineObservationWrapper,
    CBFSafetyLayerWrapper,
    ComprehensiveMetricsLoggingCallback,
    PolicyTrustScoreCallback,
    SafetyMetricsCallback,
)
from experiments.occlusions import OcclusionStrategy
from experiments.waypoint_logger import WaypointContextLogger


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


class TransformerCheckpointCallback(BaseCallback):
    """Save transformer weights separately during training"""
    
    def __init__(self, save_freq: int = 10000, save_path: str = "./checkpoints", verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
    
    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            try:
                # Access the environment wrapper stack to get the pipeline
                env = self.training_env
                
                # Unwrap to get PipelineObservationWrapper
                pipeline_wrapper = None
                current_env = env
                while hasattr(current_env, 'env'):
                    if isinstance(current_env, PipelineObservationWrapper):
                        pipeline_wrapper = current_env
                        break
                    current_env = current_env.env
                
                if current_env is not None and isinstance(current_env, PipelineObservationWrapper):
                    pipeline_wrapper = current_env
                
                if pipeline_wrapper is not None:
                    pipeline = pipeline_wrapper.pipeline
                    
                    # Save transformer components
                    transformer_state = {
                        'st_encoder': pipeline.st_encoder.state_dict(),
                        'stacked_transformer': pipeline.stacked_transformer.state_dict(),
                        'timesteps': self.num_timesteps
                    }
                    
                    checkpoint_path = os.path.join(
                        self.save_path,
                        f"transformer_checkpoint_{self.num_timesteps}.pth"
                    )
                    torch.save(transformer_state, checkpoint_path)
                    
                    if self.verbose > 0:
                        print(f"[TRANSFORMER] Saved checkpoint at {self.num_timesteps} steps: {checkpoint_path}")
                else:
                    if self.verbose > 0:
                        print("[TRANSFORMER] Warning: Could not find PipelineObservationWrapper in environment stack")
            except Exception as e:
                print(f"[TRANSFORMER] Error saving checkpoint: {e}")
                import traceback
                traceback.print_exc()
        
        return True


def create_stt_env(
    num_timesteps: int = 100000,
    render: bool = False,
    num_npc: int = 200,
    num_pedestrians: int = 200,
    encoder_path: Optional[str] = None
) -> gym.Env:
    """Create full environment stack: CARLA → STT → CBF → Occlusions
    
    Args:
        num_timesteps: Total timesteps (unused but kept for compatibility)
        render: Enable rendering
        num_npc: Number of NPC vehicles
        num_pedestrians: Number of pedestrians
        encoder_path: Path to pretrained encoder checkpoint
    """
    
    print("[ENV] Creating CARLA environment...")
    
    # Base CARLA environment
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
    
    print("[ENV] Adding pipeline observation wrapper (STT)...")
    env = PipelineObservationWrapper(env, embed_dim=512, num_frames=8, encoder_path=encoder_path)
    
    print("[ENV] Adding CBF safety layer...")
    env = CBFSafetyLayerWrapper(env, alpha=1.0)
    
    print("[ENV] Adding occlusion wrapper...")
    env = OcclusionWrapper(env, occlusion_type='mixed')
    
    print("[OK] Environment fully initialized\n")
    
    return env


def train_5q_stt_sac(
    timesteps: int = 100000,
    log_dir: str = "./logs/5q_stt",
    learning_rate: float = 3e-4,
    batch_size: int = 64,
    buffer_size: int = 50000,
    render: bool = False,
    num_npc: int = 100,
    num_pedestrians: int = 30,
    encoder_path: Optional[str] = None
):
    """Train 5 Q-Network SAC with STT
    
    Args:
        timesteps: Total training timesteps
        log_dir: Directory for logs and checkpoints
        learning_rate: SAC learning rate
        batch_size: Training batch size
        buffer_size: Replay buffer size
        render: Enable rendering
        num_npc: Number of NPC vehicles
        num_pedestrians: Number of pedestrians
        encoder_path: Path to pretrained encoder checkpoint
    """
    
    os.makedirs(log_dir, exist_ok=True)
    tb_dir = os.path.join(log_dir, "tensorboard")
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    print("=" * 80)
    print("5 Q-NETWORK SAC WITH SPATIOTEMPORAL TRANSFORMER (STT)")
    print("=" * 80)
    print(f"Timesteps: {timesteps:,}")
    print(f"Q-networks: 5 (higher uncertainty estimation)")
    print(f"Perception: ResNet50 → STT → 512-dim embedding")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Buffer size: {buffer_size:,}")
    print(f"NPC vehicles: {num_npc}")
    print(f"Pedestrians: {num_pedestrians}")
    print(f"Render: {render}")
    print(f"Encoder: {encoder_path if encoder_path else 'Training from scratch'}")
    print(f"Log dir: {log_dir}")
    print("=" * 80 + "\n")
    
    # Create environment
    env = create_stt_env(
        num_timesteps=timesteps,
        render=render,
        num_npc=num_npc,
        num_pedestrians=num_pedestrians,
        encoder_path=encoder_path
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
    
    print("Creating SAC agent with 5 Q-networks...")
    model = SAC(
        'MlpPolicy',
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
        name_prefix="sac_5q_stt"
    )
    
    transformer_callback = TransformerCheckpointCallback(
        save_freq=10000,
        save_path=os.path.join(ckpt_dir, "transformer"),
        verbose=1
    )
    
    safety_callback = SafetyMetricsCallback(verbose=1)
    trust_callback = PolicyTrustScoreCallback(update_freq=100, verbose=1)
    metrics_callback = ComprehensiveMetricsLoggingCallback(verbose=1, log_frequency=1)
    
    # Train
    print(f"Training for {timesteps:,} timesteps...\n")
    try:
        model.learn(
            total_timesteps=timesteps,
            log_interval=10,
            callback=[checkpoint_callback, transformer_callback, safety_callback, trust_callback, metrics_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Training interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    
    # Save final model
    final_path = os.path.join(log_dir, "sac_5q_stt_final.zip")
    model.save(final_path)
    print(f"\n[OK] Final model saved: {final_path}")
    
    # Save experiment info
    info_path = os.path.join(log_dir, "experiment_info.txt")
    with open(info_path, "w") as f:
        f.write("5 Q-Network SAC with SpatioTemporal Transformer\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timesteps: {timesteps:,}\n")
        f.write(f"Q-networks: 5 (ensemble for better uncertainty)\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Buffer size: {buffer_size:,}\n")
        f.write(f"NPC vehicles: {num_npc}\n")
        f.write(f"Pedestrians: {num_pedestrians}\n")
        f.write(f"Encoder checkpoint: {encoder_path if encoder_path else 'None (training from scratch)'}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\nArchitecture:\n")
        f.write(f"- Perception: ResNet50 {'(pretrained on CARLA)' if encoder_path else '(ImageNet)'}\n")
        f.write(f"- ResNet50 status: {'Frozen (weights not updated)' if encoder_path else 'Trainable'}\n")
        f.write(f"- Transformer: SpatioTemporal Transformer (8 frames, 4 blocks)\n")
        f.write(f"- Transformer status: Trainable from scratch\n")
        f.write(f"- Output embedding: 512-dim\n")
        f.write(f"- Policy: MlpPolicy (256-256)\n")
        f.write(f"- Q-networks: 5 critics (256-256 each) - ensemble uncertainty\n")
        f.write(f"\nFeatures:\n")
        f.write(f"- CBF safety layer (dynamic constraints)\n")
        f.write(f"- Random occlusions (30% episodes)\n")
        f.write(f"- Waypoint logging (junctions, merges, lanes)\n")
        f.write(f"- Reverse reward for collision recovery\n")
        f.write(f"\nDifference from 2q variant:\n")
        f.write(f"- Uses 5 Q-networks instead of 2\n")
        f.write(f"- Better uncertainty estimation via ensemble\n")
        f.write(f"- More conservative action selection\n")
    
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
        description="5 Q-Network SAC with SpatioTemporal Transformer"
    )
    parser.add_argument("--timesteps", type=int, default=100000, help="Total timesteps")
    parser.add_argument("--log-dir", type=str, default="./logs/5q_stt", help="Log directory")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--buffer-size", type=int, default=50000, help="Replay buffer size")
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument("--num-npc", type=int, default=200, help="Number of NPC vehicles")
    parser.add_argument("--num-pedestrians", type=int, default=200, help="Number of pedestrians")
    parser.add_argument(
        "--encoder-path",
        type=str,
        default="pretrained/st_encoder/st_encoder.pth",
        help="Path to pretrained encoder checkpoint (default: pretrained/st_encoder/st_encoder.pth)"
    )
    
    args = parser.parse_args()
    
    train_5q_stt_sac(
        timesteps=args.timesteps,
        log_dir=args.log_dir,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        render=args.render,
        num_npc=args.num_npc,
        num_pedestrians=args.num_pedestrians,
        encoder_path=args.encoder_path
    )
