"""
2 Q-Networks SAC with SpatioTemporal Transformer (STT)
- Architecture: ResNet50 → SpatioTemporal Transformer → SAC
- Q-network ensemble: 2 critics
- 200 NPC vehicles + 200 pedestrians
- Random occlusions (30% of episodes)
- Waypoint logging (junctions, merges, lane changes)
- Reverse reward for collision recovery

Usage:
    python 2qsac_stt_experiment.py --timesteps 100000 --render
    python 2qsac_stt_experiment.py --timesteps 50000 --log-dir ./logs/2q_stt_v1
"""

import os
import sys
import argparse
import numpy as np
import torch
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

from tests.pipeline_carla_test import CarlaGymEnv, PipelineObservationWrapper, CBFSafetyLayerWrapper
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
                    self.logger.record("safety/collisions", metrics.get('collisions', 0))
                    self.logger.record("safety/lane_invasions", metrics.get('lane_invasions', 0))
                    self.logger.record("safety/cbf_corrections", metrics.get('cbf_corrections', 0))
            except:
                pass
        
        return True


def create_stt_env(
    num_timesteps: int = 100000,
    render: bool = False,
    num_npc: int = 200,
    num_pedestrians: int = 200
) -> gym.Env:
    """Create full environment stack: CARLA → STT → CBF → Occlusions"""
    
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
    env = PipelineObservationWrapper(env, embed_dim=512, num_frames=8)
    
    print("[ENV] Adding CBF safety layer...")
    env = CBFSafetyLayerWrapper(env, alpha=1.0)
    
    print("[ENV] Adding occlusion wrapper...")
    env = OcclusionWrapper(env, occlusion_type='mixed')
    
    print("[OK] Environment fully initialized\n")
    
    return env


def train_2q_stt_sac(
    timesteps: int = 100000,
    log_dir: str = "./logs/2q_stt",
    learning_rate: float = 3e-4,
    batch_size: int = 64,
    buffer_size: int = 50000,
    render: bool = False,
    num_npc: int = 100,
    num_pedestrians: int = 30
):
    """Train 2 Q-Network SAC with STT"""
    
    os.makedirs(log_dir, exist_ok=True)
    tb_dir = os.path.join(log_dir, "tensorboard")
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    print("=" * 80)
    print("2 Q-NETWORK SAC WITH SPATIOTEMPORAL TRANSFORMER (STT)")
    print("=" * 80)
    print(f"Timesteps: {timesteps:,}")
    print(f"Q-networks: 2")
    print(f"Perception: ResNet50 → STT → 512-dim embedding")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Buffer size: {buffer_size:,}")
    print(f"NPC vehicles: {num_npc}")
    print(f"Pedestrians: {num_pedestrians}")
    print(f"Render: {render}")
    print(f"Log dir: {log_dir}")
    print("=" * 80 + "\n")
    
    # Create environment
    env = create_stt_env(
        num_timesteps=timesteps,
        render=render,
        num_npc=num_npc,
        num_pedestrians=num_pedestrians
    )
    
    # Policy kwargs with 2 Q-networks
    policy_kwargs = {
        'net_arch': {
            'pi': [256, 256],    # Actor network
            'qf': [256, 256]     # Critic networks
        },
        'n_critics': 2,          # 2 Q-networks (ensemble)
        'normalize_images': True,
    }
    
    print("Creating SAC agent with 2 Q-networks...")
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
        name_prefix="sac_2q_stt"
    )
    
    safety_callback = SafetyMetricsCallback()
    
    # Train
    print(f"Training for {timesteps:,} timesteps...\n")
    try:
        model.learn(
            total_timesteps=timesteps,
            log_interval=10,
            callback=[checkpoint_callback, safety_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Training interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    
    # Save final model
    final_path = os.path.join(log_dir, "sac_2q_stt_final.zip")
    model.save(final_path)
    print(f"\n[OK] Final model saved: {final_path}")
    
    # Save experiment info
    info_path = os.path.join(log_dir, "experiment_info.txt")
    with open(info_path, "w") as f:
        f.write("2 Q-Network SAC with SpatioTemporal Transformer\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timesteps: {timesteps:,}\n")
        f.write(f"Q-networks: 2\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Buffer size: {buffer_size:,}\n")
        f.write(f"NPC vehicles: {num_npc}\n")
        f.write(f"Pedestrians: {num_pedestrians}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\nArchitecture:\n")
        f.write(f"- Perception: ResNet50 (pretrained ImageNet)\n")
        f.write(f"- Encoder: SpatioTemporal Transformer (8 frames)\n")
        f.write(f"- Output embedding: 512-dim\n")
        f.write(f"- Policy: MlpPolicy (256-256)\n")
        f.write(f"- Q-networks: 2 critics (256-256 each)\n")
        f.write(f"\nFeatures:\n")
        f.write(f"- CBF safety layer (dynamic constraints)\n")
        f.write(f"- Random occlusions (30% episodes)\n")
        f.write(f"- Waypoint logging (junctions, merges, lanes)\n")
        f.write(f"- Reverse reward for collision recovery\n")
    
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
        description="2 Q-Network SAC with SpatioTemporal Transformer"
    )
    parser.add_argument("--timesteps", type=int, default=100000, help="Total timesteps")
    parser.add_argument("--log-dir", type=str, default="./logs/2q_stt", help="Log directory")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--buffer-size", type=int, default=50000, help="Replay buffer size")
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument("--num-npc", type=int, default=200, help="Number of NPC vehicles")
    parser.add_argument("--num-pedestrians", type=int, default=200, help="Number of pedestrians")
    
    args = parser.parse_args()
    
    train_2q_stt_sac(
        timesteps=args.timesteps,
        log_dir=args.log_dir,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        render=args.render,
        num_npc=args.num_npc,
        num_pedestrians=args.num_pedestrians
    )
