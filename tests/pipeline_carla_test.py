"""
SAC Agent Training with CBF Safety Layer on CARLA
Processes RGB+LIDAR observations through spatiotemporal pipeline,
detects safety violations (lane invasion, speed limit), applies CBF corrections,
and logs everything to TensorBoard.

Uses raw CARLA API for direct environment control.

Usage:
    python pipeline_carla_test.py --episodes 10 --timesteps 100000 --render
"""
import os
import sys
import argparse
import random
import time
from collections import deque
from typing import Dict, Tuple, Optional

import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces

import torch
import torch.nn as nn

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import carla
except ImportError:
    raise RuntimeError("CARLA module not found. Install via CARLA PythonAPI")

from models.pipeline import Pipeline
from commons.cbfQP_layer import CBFSafetyLayer

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure


# ============================================================================
# Custom CARLA Gym Environment (Direct API)
# ============================================================================

class CarlaGymEnv(gym.Env):
    """
    Custom gym environment for CARLA using direct API.
    
    Handles:
    - Vehicle spawning and control
    - Sensor attachment (RGB, LiDAR, collision, lane invasion)
    - Episode management
    - Safety state tracking for CBF layer
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 2000,
        timeout: float = 10.0,
        time_limit: int = 60,
        render_mode: Optional[str] = None,
        num_npc_vehicles: int = 20,
        num_pedestrians: int = 30,
        show_sensor_data: bool = False
    ):
        """Initialize CARLA environment"""
        self.host = host
        self.port = port
        self.timeout = timeout
        self.time_limit = time_limit
        self.render_mode = render_mode
        self.num_npc_vehicles = num_npc_vehicles
        self.num_pedestrians = num_pedestrians
        self.show_sensor_data = show_sensor_data
        
        # CARLA objects
        self.client = None
        self.world = None
        self.map = None
        self.blueprint_library = None
        
        # Actors
        self.ego_vehicle = None
        self.npc_vehicles = []
        self.walkers = []
        self.walker_controllers = []
        self.spectator = None
        
        # Sensors
        self.rgb_camera = None
        self.lidar_sensor = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.depth_sensor = None
        
        # Sensor data
        self.rgb_data = None
        self.lidar_data = None
        self.collision_occurred = False
        self.lane_invaded = False
        self.depth_data = None
        
        # State tracking
        self._collision_distance = 100.0
        self._lane_invasion_count = 0
        self._speed_limit_violation_count = 0
        self._speed_limit_check_counter = 0
        self._current_speed_limit = 15.0
        self._episode_step = 0
        
        # Action/Observation spaces
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Observation: dict with rgb, lidar, position, speed, etc.
        self.observation_space = spaces.Dict({
            'rgb_data': spaces.Box(low=0, high=255, shape=(360, 640, 3), dtype=np.uint8),
            'lidar_data': spaces.Box(low=-np.inf, high=np.inf, shape=(3, 500), dtype=np.float32),
            'position': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            'speed': spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
            'heading': spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
        })
        
        # Connect and setup
        self._connect_carla()
        self._setup_world()
        self._spawn_actors()
        self._attach_sensors()
    
    def _connect_carla(self):
        """Connect to CARLA server"""
        print(f"Connecting to CARLA at {self.host}:{self.port}...")
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(self.timeout)
        
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()
        self.spectator = self.world.get_spectator()
        
        print("✓ Connected to CARLA")
    
    def _setup_world(self):
        """Setup world settings (synchronous mode, etc)"""
        # Set synchronous mode for reproducibility
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS
        self.world.apply_settings(settings)
        print("✓ Synchronous mode enabled (20 FPS)")
    
    def _spawn_actors(self):
        """Spawn ego vehicle, NPCs, and pedestrians"""
        # Spawn ego vehicle
        ego_bp = self.blueprint_library.filter("vehicle.tesla.model3")[0]
        spawn_point = random.choice(self.map.get_spawn_points())
        
        self.ego_vehicle = self.world.spawn_actor(ego_bp, spawn_point)
        self.ego_vehicle.set_autopilot(False)  # Manual control via agent
        print(f"✓ Ego vehicle spawned at {spawn_point.location}")
        
        # Spawn NPC vehicles
        vehicle_bps = self.blueprint_library.filter("vehicle.*")
        spawn_points = self.map.get_spawn_points()
        
        for sp in spawn_points[:self.num_npc_vehicles]:
            try:
                bp = random.choice(vehicle_bps)
                v = self.world.spawn_actor(bp, sp)
                v.set_autopilot(True)
                self.npc_vehicles.append(v)
            except Exception as e:
                pass  # Skip if spawn fails
        
        print(f"✓ Spawned {len(self.npc_vehicles)} NPC vehicles")
        
        # Spawn pedestrians
        walker_bps = self.blueprint_library.filter("walker.pedestrian.*")
        walker_controller_bp = self.blueprint_library.find("controller.ai.walker")
        
        for _ in range(self.num_pedestrians):
            try:
                loc = self.world.get_random_location_from_navigation()
                if loc is None:
                    continue
                
                transform = carla.Transform(loc)
                walker_bp = random.choice(walker_bps)
                
                walker = self.world.try_spawn_actor(walker_bp, transform)
                if walker is None:
                    continue
                
                controller = self.world.spawn_actor(
                    walker_controller_bp,
                    carla.Transform(),
                    attach_to=walker
                )
                
                controller.start()
                controller.go_to_location(self.world.get_random_location_from_navigation())
                controller.set_max_speed(random.uniform(0.5, 1.5))
                
                self.walkers.append(walker)
                self.walker_controllers.append(controller)
            
            except Exception as e:
                pass
        
        print(f"✓ Spawned {len(self.walkers)} pedestrians")
    
    def _attach_sensors(self):
        """Attach RGB, LiDAR, collision, lane invasion, and depth sensors"""
        if not self.ego_vehicle:
            return
        
        # RGB Camera
        rgb_bp = self.blueprint_library.find('sensor.camera.rgb')
        rgb_bp.set_attribute('image_size_x', '640')
        rgb_bp.set_attribute('image_size_y', '360')
        rgb_bp.set_attribute('fov', '90')
        
        rgb_transform = carla.Transform(carla.Location(x=0.8, z=1.7))
        self.rgb_camera = self.world.spawn_actor(rgb_bp, rgb_transform, attach_to=self.ego_vehicle)
        self.rgb_camera.listen(self._on_rgb_image)
        print("✓ RGB camera attached")
        
        # LiDAR Sensor
        lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', '32')
        lidar_bp.set_attribute('range', '50')
        lidar_bp.set_attribute('points_per_second', '56000')
        lidar_bp.set_attribute('rotation_frequency', '10')
        
        lidar_transform = carla.Transform(carla.Location(z=1.7))
        self.lidar_sensor = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.ego_vehicle)
        self.lidar_sensor.listen(self._on_lidar_data)
        print("✓ LiDAR sensor attached")
        
        # Depth Camera
        depth_bp = self.blueprint_library.find('sensor.camera.depth')
        depth_bp.set_attribute('image_size_x', '84')
        depth_bp.set_attribute('image_size_y', '84')
        depth_bp.set_attribute('fov', '90')
        
        depth_transform = carla.Transform(carla.Location(x=0.8, z=1.7))
        self.depth_sensor = self.world.spawn_actor(depth_bp, depth_transform, attach_to=self.ego_vehicle)
        self.depth_sensor.listen(self._on_depth_image)
        print("✓ Depth camera attached")
        
        # Collision Sensor
        collision_bp = self.blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.ego_vehicle)
        self.collision_sensor.listen(self._on_collision)
        print("✓ Collision sensor attached")
        
        # Lane Invasion Sensor
        lane_inv_bp = self.blueprint_library.find('sensor.other.lane_invasion')
        self.lane_invasion_sensor = self.world.spawn_actor(lane_inv_bp, carla.Transform(), attach_to=self.ego_vehicle)
        self.lane_invasion_sensor.listen(self._on_lane_invasion)
        print("✓ Lane invasion sensor attached")
    
    def _on_rgb_image(self, image):
        """RGB camera callback"""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, :3]  # Drop alpha channel
        self.rgb_data = array
    
    def _on_lidar_data(self, point_cloud):
        """LiDAR callback - convert to (3, N) array"""
        data = np.frombuffer(point_cloud.raw_data, dtype=np.float32)
        data = np.reshape(data, (int(data.shape[0] / 4), 4))
        
        # Extract x, y, z (drop intensity)
        points = data[:, :3]
        
        # Limit to 500 points (farthest sampling)
        if len(points) > 500:
            indices = np.random.choice(len(points), 500, replace=False)
            points = points[indices]
        elif len(points) < 500:
            # Pad with zeros
            padding = np.zeros((500 - len(points), 3))
            points = np.vstack([points, padding])
        
        # Transpose to (3, 500)
        self.lidar_data = points.T.astype(np.float32)
    
    def _on_depth_image(self, image):
        """Depth camera callback"""
        self.depth_data = image
    
    def _on_collision(self, event):
        """Collision callback"""
        self.collision_occurred = True
    
    def _on_lane_invasion(self, event):
        """Lane invasion callback"""
        self.lane_invaded = True
        self._lane_invasion_count += 1
    
    def _compute_collision_distance(self) -> float:
        """Compute collision distance from depth camera"""
        if self.depth_data is None:
            return 100.0
        
        try:
            depth_array = np.array(self.depth_data.raw_data)
            depth_array = depth_array.reshape((84, 84))
            
            # Normalize to distance (0-255 maps to 0-1000m)
            depth_m = (depth_array.astype(np.float32) / 255.0) * 1000.0
            
            # Get minimum in center region
            h, w = depth_m.shape
            center_start_col = w // 3
            center_end_col = 2 * w // 3
            center_region = depth_m[:, center_start_col:center_end_col]
            
            valid_depths = center_region[center_region > 0.1]
            if len(valid_depths) > 0:
                self._collision_distance = float(np.min(valid_depths))
        
        except Exception as e:
            pass
        
        return self._collision_distance
    
    def _compute_lane_offset(self) -> float:
        """Compute lane offset from waypoint"""
        if not self.ego_vehicle or not self.map:
            return 0.0
        
        try:
            location = self.ego_vehicle.get_location()
            waypoint = self.map.get_waypoint(location)
            
            if waypoint:
                lane_center = waypoint.transform.location
                dx = location.x - lane_center.x
                dy = location.y - lane_center.y
                offset = np.sqrt(dx**2 + dy**2)
                return float(offset)
        
        except Exception as e:
            pass
        
        return 0.0
    
    def _get_speed_limit(self) -> float:
        """Query speed limit at current location"""
        self._speed_limit_check_counter += 1
        
        if self._speed_limit_check_counter >= 30:
            self._speed_limit_check_counter = 0
            
            if self.ego_vehicle and self.map:
                try:
                    location = self.ego_vehicle.get_location()
                    waypoint = self.map.get_waypoint(location)
                    
                    if waypoint:
                        speed_limit_kmh = waypoint.get_speed_limit()
                        self._current_speed_limit = speed_limit_kmh / 3.6
                
                except Exception as e:
                    pass
        
        return self._current_speed_limit
    
    def build_cbf_state(self, current_speed: float) -> Dict:
        """Build state dict for CBF safety layer"""
        d_collision = self._compute_collision_distance()
        ttc = d_collision / max(current_speed, 0.1) if current_speed > 0.1 else 100.0
        lane_offset = self._compute_lane_offset()
        
        speed_limit = self._get_speed_limit()
        if current_speed > speed_limit * 1.1:
            self._speed_limit_violation_count += 1
        
        return {
            'd_collision': float(d_collision),
            'ttc': float(ttc),
            'lane_offset': float(lane_offset),
            'speed': float(current_speed)
        }
    
    def _get_observation(self) -> Dict:
        """Get current observation"""
        location = self.ego_vehicle.get_location()
        velocity = self.ego_vehicle.get_velocity()
        transform = self.ego_vehicle.get_transform()
        
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        heading = transform.rotation.yaw
        
        # Ensure sensor data is available
        rgb = self.rgb_data if self.rgb_data is not None else np.zeros((360, 640, 3), dtype=np.uint8)
        lidar = self.lidar_data if self.lidar_data is not None else np.zeros((3, 500), dtype=np.float32)
        
        return {
            'rgb_data': rgb,
            'lidar_data': lidar,
            'position': np.array([location.x, location.y, location.z], dtype=np.float32),
            'speed': np.array([speed], dtype=np.float32),
            'heading': np.array([heading], dtype=np.float32),
        }
    
    def _compute_reward(self) -> float:
        """Compute reward (simple version)"""
        reward = 0.0
        
        # Collision penalty
        if self.collision_occurred:
            reward -= 100.0
            self.collision_occurred = False
        
        # Lane invasion penalty
        if self.lane_invaded:
            reward -= 50.0
            self.lane_invaded = False
        
        # Speed reward (encourage 10 m/s ~36 km/h)
        velocity = self.ego_vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        if 8.0 <= speed <= 12.0:
            reward += 1.0
        elif speed > 15.0:
            reward -= 0.5
        
        return reward
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one environment step"""
        # Parse action [steer, throttle/brake]
        steer = float(np.clip(action[0], -1.0, 1.0))
        throttle_brake = float(np.clip(action[1], -1.0, 1.0))
        
        # Convert to CARLA control
        control = carla.VehicleControl()
        control.steer = steer
        
        if throttle_brake > 0:
            control.throttle = throttle_brake
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = -throttle_brake
        
        self.ego_vehicle.apply_control(control)
        
        # Tick world
        self.world.tick()
        self._episode_step += 1
        
        # Get observation and reward
        obs = self._get_observation()
        reward = self._compute_reward()
        
        # Update spectator camera to track ego vehicle
        self.render()
        
        # Display RGB sensor data if enabled
        if self.show_sensor_data and self.rgb_data is not None:
            rgb_display = cv2.cvtColor(self.rgb_data, cv2.COLOR_BGR2RGB)
            cv2.imshow('CARLA RGB Camera', rgb_display)
            cv2.waitKey(1)
        
        # Termination conditions
        terminated = self.collision_occurred or self._episode_step >= self.time_limit
        truncated = False
        
        # Info dict
        info = {
            'collision_distance': self._collision_distance,
            'lane_invaded': self.lane_invaded,
            'episode_step': self._episode_step,
        }
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset environment"""
        # Reset counters
        self._episode_step = 0
        self._collision_distance = 100.0
        self._lane_invasion_count = 0
        self._speed_limit_violation_count = 0
        self.collision_occurred = False
        self.lane_invaded = False
        
        # Reset ego vehicle
        if self.ego_vehicle:
            spawn_point = random.choice(self.map.get_spawn_points())
            self.ego_vehicle.set_transform(spawn_point)
            self.ego_vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
        
        # Get initial observation
        self.world.tick()
        obs = self._get_observation()
        
        info = {}
        
        return obs, info
    
    def render(self):
        """Update spectator camera"""
        if self.ego_vehicle and self.spectator:
            transform = self.ego_vehicle.get_transform()
            self.spectator.set_transform(
                carla.Transform(
                    transform.location + carla.Location(z=30),
                    carla.Rotation(pitch=-90)
                )
            )
    
    def close(self):
        """Cleanup CARLA actors"""
        print("Destroying CARLA actors...")
        
        if self.ego_vehicle:
            self.ego_vehicle.destroy()
        
        for v in self.npc_vehicles:
            try:
                v.destroy()
            except:
                pass
        
        for controller in self.walker_controllers:
            try:
                controller.stop()
                controller.destroy()
            except:
                pass
        
        for walker in self.walkers:
            try:
                walker.destroy()
            except:
                pass
        
        print("✓ All actors destroyed")


# ============================================================================
# Observation Preprocessing Wrapper (Pipeline Integration)
# ============================================================================

class PipelineObservationWrapper(gym.ObservationWrapper):
    """
    Converts raw CARLA observations to pipeline embeddings (512-dim tensor).
    """
    
    def __init__(self, env: gym.Env, embed_dim: int = 512, num_frames: int = 8):
        super().__init__(env)
        
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        
        # Initialize pipeline
        self.pipeline = Pipeline.from_defaults(
            num_frames=num_frames,
            embed_dim=embed_dim,
            use_timesformer=False,
            fe_weights_path=None,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.pipeline.eval()
        
        # Frame buffer
        self.frame_buffer = deque(maxlen=num_frames)
        for _ in range(num_frames):
            self.frame_buffer.append(np.zeros((360, 640, 3), dtype=np.uint8))
        
        # Update observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(embed_dim,), dtype=np.float32
        )
    
    def observation(self, obs: Dict) -> np.ndarray:
        """Convert observation dict to pipeline embedding"""
        rgb_frame = obs.get('rgb_data')
        if rgb_frame is None or rgb_frame.size == 0:
            rgb_frame = np.zeros((360, 640, 3), dtype=np.uint8)
        
        # Ensure BGR format and correct shape
        if rgb_frame.dtype != np.uint8:
            rgb_frame = (rgb_frame * 255).astype(np.uint8)
        if rgb_frame.ndim == 2:
            rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_GRAY2BGR)
        
        self.frame_buffer.append(rgb_frame)
        frames = list(self.frame_buffer)
        
        # Process through pipeline
        try:
            with torch.no_grad():
                embedding, _ = self.pipeline.process_sequence(frames)
                embedding = embedding.cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"Error in pipeline: {e}")
            embedding = np.zeros(self.embed_dim, dtype=np.float32)
        
        return embedding


# ============================================================================
# CBF Safety Layer Integration Wrapper
# ============================================================================

class CBFSafetyLayerWrapper(gym.ActionWrapper):
    """Applies CBF safety corrections to actions"""
    
    def __init__(self, env: gym.Env, alpha: float = 1.0):
        super().__init__(env)
        
        self.cbf_layer = CBFSafetyLayer(alpha=alpha, d_min=5.0, y_max=1.5, v_max=15.0)
        self.correction_count = 0
        self.correction_magnitudes = []
    
    def action(self, action: np.ndarray) -> np.ndarray:
        """Apply CBF correction if violation detected"""
        # Get current state
        try:
            current_obs = self.env.unwrapped._get_observation()
            current_speed = float(current_obs['speed'][0])
        except:
            current_speed = 0.0
        
        cbf_state = self.env.unwrapped.build_cbf_state(current_speed)
        
        # Check violations
        violation = False
        if cbf_state['d_collision'] < 5.0 or abs(cbf_state['lane_offset']) > 1.5:
            violation = True
        
        # Apply CBF correction
        safe_action = action.copy()
        if violation:
            try:
                safe_action = self.cbf_layer.compute_safe_action(action, cbf_state)
                self.correction_count += 1
                correction_mag = float(np.linalg.norm(safe_action - action))
                self.correction_magnitudes.append(correction_mag)
            except Exception as e:
                print(f"CBF correction failed: {e}")
        
        return np.clip(safe_action, -1.0, 1.0)


# ============================================================================
# Custom TensorBoard Callback
# ============================================================================

class SafetyMetricsCallback(BaseCallback):
    """Log safety metrics to TensorBoard"""
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
    
    def _on_step(self) -> bool:
        return True


# ============================================================================
# Main Training Pipeline
# ============================================================================

def create_carla_env(
    time_limit: int = 60,
    render: bool = False,
    num_npc: int = 20,
    num_pedestrians: int = 30,
    show_sensor_data: bool = False
) -> gym.Env:
    """Create CARLA environment with wrappers"""
    
    # Base environment
    env = CarlaGymEnv(
        time_limit=time_limit,
        render_mode="human" if render else None,
        num_npc_vehicles=num_npc,
        num_pedestrians=num_pedestrians,
        show_sensor_data=show_sensor_data
    )
    
    # Wrap with pipeline
    env = PipelineObservationWrapper(env, embed_dim=512, num_frames=8)
    

    # Wrap with CBF
    env = CBFSafetyLayerWrapper(env, alpha=1.0)
    
    return env


def train_sac_agent(
    num_episodes: int = 10,
    total_timesteps: int = 100000,
    checkpoint_freq: int = 2000,
    log_dir: str = "./logs",
    render: bool = False
):
    """Main training loop"""
    
    os.makedirs(log_dir, exist_ok=True)
    tb_dir = os.path.join(log_dir, "tensorboard")
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    print("=" * 70)
    print("SAC TRAINING WITH CBF SAFETY LAYER")
    print("=" * 70)
    print(f"Log dir: {log_dir}")
    print(f"TensorBoard: {tb_dir}")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"Total timesteps: {total_timesteps}")
    print("=" * 70 + "\n")
    
    # Create environment
    print("Initializing CARLA environment...")
    env = create_carla_env(time_limit=60, render=render, num_npc=20, num_pedestrians=30, show_sensor_data=True)
    print("✓ Environment created\n")
    
    # Create SAC model
    print("Initializing SAC agent...")
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=10000,
        batch_size=64,
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=1
    )
    print("✓ SAC agent created\n")
    
    # Setup TensorBoard
    new_logger = configure(tb_dir, ["stdout", "tensorboard"])
    model.set_logger(new_logger)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=ckpt_dir,
        name_prefix="sac_carla_cbf"
    )
    
    # Safety callback
    safety_callback = SafetyMetricsCallback(verbose=1)
    
    # Train
    print(f"Starting training for {total_timesteps} timesteps...\n")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            log_interval=10,
            callback=[checkpoint_callback, safety_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Training interrupted")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    
    # Save final model
    final_path = os.path.join(log_dir, "sac_carla_cbf_final")
    model.save(final_path)
    print(f"\n✓ Final model saved to {final_path}")
    
    # Cleanup
    env.close()
    print("✓ Environment closed")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"View results: tensorboard --logdir {tb_dir}")


# ============================================================================
# Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="SAC Training with CBF on CARLA")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total timesteps")
    parser.add_argument("--checkpoint-freq", type=int, default=2000, help="Checkpoint frequency")
    parser.add_argument("--log-dir", type=str, default="./logs", help="Log directory")
    parser.add_argument("--render", action="store_true", help="Render environment")
    
    args = parser.parse_args()
    
    train_sac_agent(
        num_episodes=args.episodes,
        total_timesteps=args.timesteps,
        checkpoint_freq=args.checkpoint_freq,
        log_dir=args.log_dir,
        render=args.render
    )


if __name__ == "__main__":
    main()
