"""
Enhanced SAC Agent Training with CBF Safety Layer & Transformer Perception
Features:
- Pretrained ResNet50 + SpatioTemporal Transformer perception
- CBF safety layer with dynamic constraints
- Dual training modes: visualization (--render) and headless (default)
- TensorBoard logging with CBF metrics
- Ensemble critics with trust scoring
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure

from tests.pipeline_carla_test import CarlaGymEnv, CBFSafetyLayerWrapper, PipelineObservationWrapper
from models.pipeline import Pipeline
from commons.cbfQP_layer import CBFSafetyLayer


# ============================================================================
# Specialized TensorBoard Callbacks
# ============================================================================

class EnhancedSafetyMetricsCallback(BaseCallback):
    """
    Log safety metrics, CBF activations, and perception confidence to TensorBoard.
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_cbf_corrections = 0
        self.episode_collisions = 0
        self.episode_lane_invasions = 0
        self.episode_speed_violations = 0
        
    def _on_step(self) -> bool:
        """Called after each environment step."""
        # Access environment
        if not hasattr(self.training_env, 'envs') or len(self.training_env.envs) == 0:
            return True
        
        env = self.training_env.envs[0].unwrapped
        
        # Log safety metrics if available
        if hasattr(env, 'safety_metrics'):
            metrics = env.safety_metrics
            
            # Safety metrics
            self.logger.record("safety/lane_invasions", metrics.get('lane_invasions', 0))
            self.logger.record("safety/speed_violations", metrics.get('speed_violations', 0))
            self.logger.record("safety/collisions", metrics.get('collisions', 0))
            self.logger.record("safety/cbf_corrections", metrics.get('cbf_corrections', 0))
            self.logger.record("safety/cbf_correction_magnitude", metrics.get('cbf_correction_magnitude', 0.0))
        
        # Get CBF layer stats if available
        cbf_wrapper = None
        current_env = self.training_env.envs[0]
        while hasattr(current_env, 'env'):
            if hasattr(current_env, 'cbf_layer'):
                cbf_wrapper = current_env
                break
            current_env = current_env.env
        
        if cbf_wrapper and hasattr(cbf_wrapper, 'correction_count'):
            self.logger.record("cbf/total_corrections", cbf_wrapper.correction_count)
            self.logger.record("cbf/episode_corrections", cbf_wrapper.episode_corrections)
            self.logger.record("cbf/avg_correction_mag", cbf_wrapper.episode_correction_mag)
            
            # Constraint violation counts
            if hasattr(cbf_wrapper.cbf_layer, 'constraint_violations'):
                violations = cbf_wrapper.cbf_layer.constraint_violations
                self.logger.record("cbf/collision_violations", violations.get('collision', 0))
                self.logger.record("cbf/lane_violations", violations.get('lane', 0))
                self.logger.record("cbf/speed_violations", violations.get('speed', 0))
        
        return True
    
    def _on_training_end(self) -> None:
        """Called at end of training."""
        pass


class PerceptionMonitorCallback(BaseCallback):
    """
    Monitor perception pipeline performance (feature extractor confidence).
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.feature_norms = []
        
    def _on_step(self) -> bool:
        """Monitor feature extraction."""
        # This would integrate with actual perception pipeline
        # Currently placeholder for future enhancement
        return True


# ============================================================================
# Visualization Wrapper (for --render mode)
# ============================================================================

class CarlaVisualizationWrapper(gym.Wrapper):
    """
    Wrapper that provides pygame visualization of:
    - Top-down ego vehicle view
    - RGB camera
    - Depth map
    - LiDAR point cloud
    """
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.env = env
        self.render_enabled = False
        self.pygame_initialized = False
        
        try:
            import pygame
            pygame.init()
            self.pygame_initialized = True
            self.pygame = pygame
            
            # Create display window
            self.display = pygame.display.set_mode((1600, 900))
            pygame.display.set_caption("CARLA Safety-Aware DRL Training")
            self.render_enabled = True
            self.clock = pygame.time.Clock()
        except Exception as e:
            print(f"[WARNING] Pygame initialization failed: {e}")
            print("[INFO] Training will continue without visualization")
            self.render_enabled = False
    
    def step(self, action):
        """Step environment and render if enabled."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if self.render_enabled:
            self._render_visualization(obs, info, reward)
        
        return obs, reward, terminated, truncated, info
    
    def _render_visualization(self, obs, info, reward):
        """Render multi-panel visualization."""
        if not self.pygame_initialized:
            return
        
        try:
            # Get raw environment observation for visualization
            base_env = self.env.unwrapped
            
            if hasattr(base_env, 'rgb_data') and base_env.rgb_data is not None:
                rgb = base_env.rgb_data
            else:
                rgb = np.zeros((360, 640, 3), dtype=np.uint8)
            
            if hasattr(base_env, 'depth_data') and base_env.depth_data is not None:
                depth = base_env.depth_data
                # Normalize depth for visualization
                depth_norm = np.clip(depth / 100.0, 0, 1)
                depth_vis = (depth_norm * 255).astype(np.uint8)
                depth_rgb = np.stack([depth_vis, depth_vis, depth_vis], axis=-1)
            else:
                depth_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Create layout: RGB (top-left), Depth (top-right), Info (bottom)
            surface_rgb = self.pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
            surface_depth = self.pygame.surfarray.make_surface(np.transpose(depth_rgb, (1, 0, 2)))
            
            # Draw surfaces
            self.display.blit(self.pygame.transform.scale(surface_rgb, (800, 450)), (0, 0))
            self.display.blit(self.pygame.transform.scale(surface_depth, (800, 450)), (800, 0))
            
            # Draw info text
            font = self.pygame.font.Font(None, 24)
            texts = [
                f"Step: {base_env._episode_step}",
                f"Collision Distance: {info.get('collision_distance', 0):.2f}m",
                f"Reward: {reward:.3f}",
                f"Lane Invaded: {info.get('lane_invaded', False)}",
                f"CBF Corrections: {getattr(base_env, '_cbf_correction_count', 0)}"
            ]
            
            for i, text in enumerate(texts):
                surf = font.render(text, True, (255, 255, 255))
                self.display.blit(surf, (10, 460 + i * 30))
            
            self.pygame.display.flip()
            self.clock.tick(20)  # 20 FPS for display
        
        except Exception as e:
            print(f"[WARNING] Rendering failed: {e}")


# ============================================================================
# Training Functions
# ============================================================================

def create_training_env(
    headless: bool = True,
    time_limit: int = 1000,
    num_npc: int = 100,
    perception_enabled: bool = True
):
    """
    Create training environment with optional visualization.
    
    Args:
        headless: If False, enables pygame visualization
        time_limit: Episode time limit in steps
        num_npc: Number of NPC vehicles
        perception_enabled: Use transformer perception pipeline
    
    Returns:
        Wrapped environment ready for training
    """
    print(f"Creating CARLA training environment...")
    print(f"  - Mode: {'Headless (Fast)' if headless else 'Visualized (Slow)'}")
    print(f"  - Perception: {'Enabled (Transformer)' if perception_enabled else 'Disabled'}")
    print(f"  - Time limit: {time_limit} steps")
    print(f"  - NPC vehicles: {num_npc}")
    
    # Create base environment
    base_env = CarlaGymEnv(
        host='localhost',
        port=2000,
        timeout=10.0,
        time_limit=time_limit,
        render_mode=None,
        num_npc_vehicles=num_npc,
        show_sensor_data=False
    )
    
    # Add perception pipeline if enabled
    if perception_enabled:
        print("  - Loading transformer perception pipeline...")
        try:
            base_env = PipelineObservationWrapper(base_env, embed_dim=512, num_frames=8)
        except Exception as e:
            print(f"[WARNING] Perception pipeline load failed: {e}")
            print("[INFO] Continuing without perception enhancement")
    
    # Add CBF safety layer
    print("  - Adding CBF safety layer...")
    safety_env = CBFSafetyLayerWrapper(
        base_env,
        alpha=1.0,
        use_trust_score=True,
        correction_penalty=0.01  # Reward penalty for corrections
    )
    
    # Add visualization if not headless
    if not headless:
        print("  - Initializing pygame visualization...")
        safety_env = CarlaVisualizationWrapper(safety_env)
    
    print("[OK] Environment created successfully")
    return safety_env


def train_enhanced_sac(
    timesteps: int = 100000,
    log_dir: str = "./logs/enhanced_training",
    headless: bool = True,
    learning_rate: float = 3e-4,
    batch_size: int = 64,
    buffer_size: int = 50000,
    num_npc: int = 20,
    perception_enabled: bool = True,
    checkpoint_freq: int = 10000,
):
    """
    Train SAC agent with enhanced CBF safety layer.
    
    Args:
        timesteps: Total training timesteps
        log_dir: Logging directory
        headless: Disable pygame visualization
        learning_rate: Actor learning rate
        batch_size: Training batch size
        buffer_size: Replay buffer size
        num_npc: Number of NPC vehicles
        perception_enabled: Use transformer perception
        checkpoint_freq: Checkpoint save frequency
    """
    
    # Setup logging
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    tensorboard_path = log_path / "tensorboard"
    checkpoint_path = log_path / "checkpoints"
    tensorboard_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("ENHANCED SAC TRAINING WITH CBF SAFETY LAYER")
    print("="*80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Log directory: {log_path.absolute()}")
    print(f"TensorBoard: {tensorboard_path.absolute()}")
    print("="*80 + "\n")
    
    # Create environment
    env = create_training_env(
        headless=headless,
        time_limit=1000,
        num_npc=num_npc,
        perception_enabled=perception_enabled
    )
    
    # Policy configuration
    policy_kwargs = {
        "net_arch": {"pi": [256, 256], "qf": [256, 256]},
        "n_critics": 5,            # Ensemble of 5 critics
        "trust_lambda": 0.01,      # Trust scoring sensitivity
        "normalize_images": False,
    }
    
    # Create SAC agent
    print("\nInitializing SAC agent...")
    agent = SAC(
        "MlpPolicy",
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
    
    print("[OK] SAC agent created")
    print(f"  - Policy: MlpPolicy")
    print(f"  - Ensemble critics: 5")
    print(f"  - Trust lambda: 0.01")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Batch size: {batch_size}")
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(checkpoint_path),
        name_prefix="sac_cbf"
    )
    
    safety_callback = EnhancedSafetyMetricsCallback(verbose=0)
    perception_callback = PerceptionMonitorCallback(verbose=0)
    
    callbacks = [checkpoint_callback, safety_callback, perception_callback]
    
    # Train
    print(f"\nStarting training for {timesteps} timesteps...")
    print(f"Callbacks: Checkpointing, Safety Metrics, Perception Monitoring")
    print("-"*80 + "\n")
    
    try:
        agent.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            progress_bar=True,
            log_interval=1,
        )
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Training stopped by user")
    
    # Save final model
    model_path = log_path / "sac_cbf_final"
    agent.save(str(model_path))
    print(f"\n[OK] Final model saved to {model_path}")
    
    # Save training info
    info_path = log_path / "training_info.txt"
    with open(info_path, "w") as f:
        f.write("Enhanced SAC Training with CBF Safety Layer\n")
        f.write("="*70 + "\n\n")
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Timesteps: {timesteps}\n")
        f.write(f"Headless Mode: {headless}\n")
        f.write(f"NPC Vehicles: {num_npc}\n")
        f.write(f"Perception Enabled: {perception_enabled}\n\n")
        f.write("SAC Configuration:\n")
        f.write(f"  - Policy: MlpPolicy\n")
        f.write(f"  - Learning Rate: {learning_rate}\n")
        f.write(f"  - Batch Size: {batch_size}\n")
        f.write(f"  - Buffer Size: {buffer_size}\n")
        f.write(f"  - Ensemble Critics: 5\n")
        f.write(f"  - Trust Lambda: 0.01\n\n")
        f.write("CBF Safety Layer:\n")
        f.write(f"  - Collision Distance: 5.0m\n")
        f.write(f"  - Lane Deviation: 1.5m\n")
        f.write(f"  - Speed Limit: Dynamic\n")
        f.write(f"  - Rate Limiting: Enabled\n")
        f.write(f"  - Correction Penalty: 0.01\n\n")
        f.write("TensorBoard Metrics:\n")
        f.write("  - safety/lane_invasions\n")
        f.write("  - safety/speed_violations\n")
        f.write("  - safety/collisions\n")
        f.write("  - safety/cbf_corrections\n")
        f.write("  - cbf/collision_violations\n")
        f.write("  - cbf/lane_violations\n")
        f.write("  - cbf/speed_violations\n")
    
    print(f"[OK] Training info saved to {info_path}")
    
    env.close()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"View TensorBoard: tensorboard --logdir {tensorboard_path}")
    print("="*80 + "\n")


def evaluate_trained_agent(
    model_path: str,
    num_episodes: int = 5,
    headless: bool = True,
    num_npc: int = 100,
):
    """
    Evaluate trained SAC agent.
    
    Args:
        model_path: Path to saved model
        num_episodes: Number of evaluation episodes
        headless: Disable visualization
        num_npc: Number of NPC vehicles
    """
    
    print(f"\nLoading model from {model_path}...")
    env = create_training_env(
        headless=headless,
        time_limit=1000,
        num_npc=num_npc,
        perception_enabled=True
    )
    
    agent = SAC.load(model_path, env=env)
    print("[OK] Model loaded successfully")
    
    print(f"\nEvaluating for {num_episodes} episodes...")
    
    total_reward = 0.0
    episode_rewards = []
    total_collisions = 0
    total_lane_invasions = 0
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        
        print(f"\nEpisode {ep + 1}/{num_episodes}: ", end="")
        
        while not done:
            action, _states = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        episode_rewards.append(episode_reward)
        total_reward += episode_reward
        
        base_env = env.unwrapped
        colls = getattr(base_env, '_collision_count', 0)
        lanes = getattr(base_env, '_lane_invasion_count', 0)
        total_collisions += colls
        total_lane_invasions += lanes
        
        print(f"Reward: {episode_reward:.2f} | Collisions: {colls} | Lane Invasions: {lanes}")
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"Episodes: {num_episodes}")
    print(f"Average Reward: {total_reward/num_episodes:.2f}")
    print(f"Reward Std Dev: {np.std(episode_rewards):.2f}")
    print(f"Total Collisions: {total_collisions}")
    print(f"Total Lane Invasions: {total_lane_invasions}")
    print("="*80 + "\n")
    
    env.close()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enhanced SAC Training with CBF Safety Layer"
    )
    
    # Training parameters
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs/enhanced_training",
        help="Logging directory"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size"
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=50000,
        help="Replay buffer size"
    )
    parser.add_argument(
        "--num-npc",
        type=int,
        default=20,
        help="Number of NPC vehicles"
    )
    
    # Mode selection
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable pygame visualization (slower)"
    )
    parser.add_argument(
        "--no-perception",
        action="store_true",
        help="Disable transformer perception pipeline"
    )
    
    # Evaluation mode
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only evaluate (no training)"
    )
    parser.add_argument(
        "--eval-model",
        type=str,
        default=None,
        help="Model path for evaluation"
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes"
    )
    
    args = parser.parse_args()
    
    # Execute based on arguments
    if args.eval_only and args.eval_model:
        evaluate_trained_agent(
            model_path=args.eval_model,
            num_episodes=args.eval_episodes,
            headless=not args.render,
            num_npc=args.num_npc,
        )
    else:
        train_enhanced_sac(
            timesteps=args.timesteps,
            log_dir=args.log_dir,
            headless=not args.render,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            num_npc=args.num_npc,
            perception_enabled=not args.no_perception,
            checkpoint_freq=10000,
        )
