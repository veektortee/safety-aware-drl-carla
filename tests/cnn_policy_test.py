"""
CNN Policy Test Script for CARLA SAC Agent with Ensemble Critics
Tests the CnnPolicy with ensemble critics and trust scoring on the CARLA environment.
Includes safety constraints (CBF layer, lane invasion, speed limits).
"""

import os
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
from gymnasium import spaces
import torch
import carla
import cv2
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecCheckNan

# Import custom modules
from models.pipeline import Pipeline
from commons.cbfQP_layer import CBFSafetyLayer


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


class CNNSafetyMetricsCallback(BaseCallback):
    """Custom callback to log safety metrics to TensorBoard for CNN training."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.lane_invasion_count = 0
        self.speed_limit_violation_count = 0
        self.collision_count = 0

    def _on_step(self) -> bool:
        """Called after each step."""
        # Log custom metrics if they exist in the environment
        if hasattr(self.training_env, 'envs') and len(self.training_env.envs) > 0:
            env = self.training_env.envs[0].unwrapped
            if hasattr(env, 'safety_metrics'):
                metrics = env.safety_metrics
                self.logger.record("safety/lane_invasions", metrics.get('lane_invasions', 0))
                self.logger.record("safety/speed_violations", metrics.get('speed_violations', 0))
                self.logger.record("safety/collisions", metrics.get('collisions', 0))
                self.logger.record("safety/cbf_corrections", metrics.get('cbf_corrections', 0))

        return True


def create_cnn_env(client=None, show_sensor_data=False):
    """
    Create CARLA environment with CNN observation wrapper.
    
    Args:
        client: CARLA client (unused, for compatibility)
        show_sensor_data: Whether to display RGB sensor data
    
    Returns:
        environment wrapped with image observation wrapper and CBF safety layer
    """
    from tests.pipeline_carla_test import CarlaGymEnv, CBFSafetyLayerWrapper

    # Create base CARLA environment with proper host/port
    base_env = CarlaGymEnv(
        host='localhost',
        port=2000,
        timeout=10.0,
        time_limit=1000,
        show_sensor_data=show_sensor_data
    )

    # Wrap with CBF safety layer FIRST
    safety_env = CBFSafetyLayerWrapper(base_env)

    # Wrap with image observation wrapper LAST (this converts dict to (C,H,W))
    image_env = ImageObservationWrapper(safety_env)

    return image_env


def train_cnn_sac_agent(
    timesteps=100000,
    log_dir="./logs/cnn",
    learning_rate=3e-4,
    batch_size=64,
    buffer_size=50000,
    show_sensor_data=True,
):
    """
    Train SAC agent with CNN policy on CARLA environment.
    
    Args:
        timesteps: Total training timesteps
        log_dir: Directory for logs and checkpoints
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
        buffer_size: Replay buffer size
        show_sensor_data: Whether to display sensor data during training
    """
    
    # Setup logging directories
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    tensorboard_path = log_path / "tensorboard"
    checkpoint_path = log_path / "checkpoints"
    tensorboard_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    print(f"Creating CARLA environment...")
    env = create_cnn_env(show_sensor_data=show_sensor_data)

    # Define policy kwargs with ensemble critics and trust scoring
    policy_kwargs = {
        "net_arch": {"pi": [256, 256], "qf": [256, 256]},
        "n_critics": 5,  # Ensemble of 5 critics
        "trust_lambda": 0.01,  # Trust scoring hyperparameter
        "normalize_images": True,
    }

    # Create SAC agent with CNN policy
    print(f"Creating SAC agent with CNN policy...")
    agent = SAC(
        "CnnPolicy",
        env,
        learning_rate=learning_rate,
        batch_size=batch_size,
        buffer_size=buffer_size,
        gamma=0.99,
        tau=0.005,
        ent_coef="auto",
        target_update_interval=1,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(tensorboard_path),
        verbose=1,
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=str(checkpoint_path),
        name_prefix="cnn_sac"
    )
    
    safety_callback = CNNSafetyMetricsCallback(verbose=0)

    # Train
    print(f"Training CNN SAC agent for {timesteps} timesteps...")
    print(f"TensorBoard logs: {tensorboard_path}")
    print(f"Checkpoints: {checkpoint_path}")
    
    agent.learn(
        total_timesteps=timesteps,
        callback=[checkpoint_callback, safety_callback],
        progress_bar=True,
    )

    # Save final model
    model_path = log_path / "cnn_sac_final"
    agent.save(str(model_path))
    print(f"Saved final model to {model_path}")

    # Save model info
    model_info_path = log_path / "model_info.txt"
    with open(model_info_path, "w") as f:
        f.write("CNN SAC Agent Information\n")
        f.write("=" * 50 + "\n")
        f.write(f"Policy: CnnPolicy\n")
        f.write(f"Number of Critics: 5\n")
        f.write(f"Trust Lambda: 0.01\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Buffer Size: {buffer_size}\n")
        f.write(f"Total Timesteps: {timesteps}\n")
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\nObservation Space: (C=3, H=84, W=84) - CNN input\n")
        f.write(f"Action Space: [-1, 1] x [-1, 1] (steering, throttle/brake)\n")
        f.write(f"\nSafety Features:\n")
        f.write(f"- CBF Control Barrier Function layer\n")
        f.write(f"- Lane invasion detection\n")
        f.write(f"- Speed limit enforcement\n")
        f.write(f"- Collision distance monitoring\n")

    print(f"\nTraining complete!")
    print(f"Model info saved to {model_info_path}")

    env.close()


def evaluate_cnn_agent(
    model_path,
    num_episodes=5,
    show_sensor_data=True,
):
    """
    Evaluate trained CNN SAC agent.
    
    Args:
        model_path: Path to saved model
        num_episodes: Number of evaluation episodes
        show_sensor_data: Whether to display sensor data
    """
    
    print(f"Loading trained CNN SAC model from {model_path}...")
    
    env = create_cnn_env(show_sensor_data=show_sensor_data)
    agent = SAC.load(model_path, env=env)

    print(f"\nEvaluating agent for {num_episodes} episodes...")
    
    total_reward = 0.0
    episode_rewards = []
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        
        print(f"\nEpisode {ep + 1}/{num_episodes}")
        
        while not done:
            action, _states = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        episode_rewards.append(episode_reward)
        total_reward += episode_reward
        print(f"Episode {ep + 1} reward: {episode_reward:.2f}")
    
    avg_reward = total_reward / num_episodes
    print(f"\n{'=' * 50}")
    print(f"Evaluation Results:")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Std reward: {np.std(episode_rewards):.2f}")
    print(f"Min reward: {min(episode_rewards):.2f}")
    print(f"Max reward: {max(episode_rewards):.2f}")
    print(f"{'=' * 50}")
    
    env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train CNN SAC agent on CARLA")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--log-dir", type=str, default="./logs/cnn", help="Logging directory")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--buffer-size", type=int, default=50000, help="Replay buffer size")
    parser.add_argument("--render", action="store_true", help="Show sensor data during training")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate (no training)")
    parser.add_argument("--eval-model", type=str, default=None, help="Model path for evaluation")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Number of evaluation episodes")
    
    args = parser.parse_args()
    
    if args.eval_only and args.eval_model:
        evaluate_cnn_agent(
            model_path=args.eval_model,
            num_episodes=args.eval_episodes,
            show_sensor_data=args.render,
        )
    else:
        train_cnn_sac_agent(
            timesteps=args.timesteps,
            log_dir=args.log_dir,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            show_sensor_data=args.render,
        )
