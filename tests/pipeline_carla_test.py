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
        num_npc_vehicles: int = 100,
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
        self._collision_count = 0
        self._cbf_correction_count = 0
        self._avg_correction_mag = 0.0
        
        # Advanced reward shaping state (NEW)
        self._prev_steering = 0.0
        self._prev_speed = 0.0
        self._prev_acceleration = 0.0
        self._lane_change_count = 0
        self._last_lane_offset = 0.0
        self._use_advanced_rewards = True  # Toggle for backward compatibility
        
        # Waypoint tracking (NEW - Phase 2)
        self.waypoints = []  # List of target waypoints
        self.current_waypoint_idx = 0
        self.waypoint_cross_distance = 2.0  # Distance to trigger waypoint crossing
        self.waypoints_crossed = 0
        self.total_waypoints = 40  # Default; updated when route is generated
        self._generate_waypoints()  # Generate initial waypoints
        
        # Debug logging (Phase 3)
        self._debug_reward_logging = False  # Set to True to enable
        self._debug_log_frequency = 100  # Log every N steps
        self._reward_log_counter = 0
        
        # Action/Observation spaces
        # 3D action space: [steering, throttle, brake] for CBF compatibility
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
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
        
        print("[OK] Connected to CARLA")
    
    def _setup_world(self):
        """Setup world settings (synchronous mode, etc)"""
        # Set synchronous mode for reproducibility
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS
        self.world.apply_settings(settings)
        print("[OK] Synchronous mode enabled (20 FPS)")
    
    def _spawn_actors(self):
        """Spawn ego vehicle, NPCs, and pedestrians"""
        # STEP 1: Clear all existing actors from the map
        print("[S] Clearing existing actors from map...")
        actors = self.world.get_actors()
        for actor in actors:
            if actor.type_id.startswith('vehicle') or actor.type_id.startswith('walker') or actor.type_id.startswith('sensor'):
                try:
                    actor.destroy()
                except:
                    pass
        print("[OK] Map cleared\n")
        
        # STEP 2: Spawn ego vehicle first
        print("[S] Spawning ego vehicle...")
        spawn_points = self.map.get_spawn_points()
        
        # Try different vehicle blueprints
        blueprint_options = [
            "vehicle.tesla.model3",
            "vehicle.audi.a2",
            "vehicle.carlamotors.carlacola",
            "vehicle.dodge.charger",
            "vehicle.bmw.grandtourer"
        ]
        
        ego_bp = None
        for bp_filter in blueprint_options:
            filtered = self.blueprint_library.filter(bp_filter)
            if len(filtered) > 0:
                ego_bp = filtered[0]
                break
        
        if ego_bp is None:
            # Fallback: use any vehicle
            ego_bp = self.blueprint_library.filter("vehicle.*")[0]
        
        self.ego_vehicle = None
        for spawn_point in spawn_points:
            try:
                self.ego_vehicle = self.world.spawn_actor(ego_bp, spawn_point)
                self.ego_vehicle.set_autopilot(False)  # Manual control via agent
                print(f"[OK] Ego vehicle spawned at location ({spawn_point.location.x:.1f}, {spawn_point.location.y:.1f}, {spawn_point.location.z:.1f})\n")
                break
            except RuntimeError:
                continue
        
        if self.ego_vehicle is None:
            print("[ERROR] Could not spawn ego vehicle - no spawn points available\n")
            return
        
        # STEP 3: Spawn NPC vehicles (after ego vehicle)
        print(f"[S] Spawning {self.num_npc_vehicles} NPC vehicles...")
        vehicle_bps = self.blueprint_library.filter("vehicle.*")
        spawn_points = self.map.get_spawn_points()
        
        # Skip the first spawn point (used by ego)
        available_spawn_points = spawn_points[1:]
        
        npc_count = 0
        max_npcs = min(self.num_npc_vehicles, len(available_spawn_points))
        
        for sp in available_spawn_points[:max_npcs]:
            try:
                bp = random.choice(vehicle_bps)
                v = self.world.spawn_actor(bp, sp)
                v.set_autopilot(True)
                self.npc_vehicles.append(v)
                npc_count += 1
            except Exception:
                pass  # Skip if spawn fails
        
        print(f"[OK] Spawned {npc_count} NPC vehicles\n")
        
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
        
        print(f"[OK] Spawned {len(self.walkers)} pedestrians")
    
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
        print("[OK] RGB camera attached")
        
        # LiDAR Sensor
        lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', '32')
        lidar_bp.set_attribute('range', '50')
        lidar_bp.set_attribute('points_per_second', '56000')
        lidar_bp.set_attribute('rotation_frequency', '10')
        
        lidar_transform = carla.Transform(carla.Location(z=1.7))
        self.lidar_sensor = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.ego_vehicle)
        self.lidar_sensor.listen(self._on_lidar_data)
        print("[OK] LiDAR sensor attached")
        
        # Depth Camera
        depth_bp = self.blueprint_library.find('sensor.camera.depth')
        depth_bp.set_attribute('image_size_x', '84')
        depth_bp.set_attribute('image_size_y', '84')
        depth_bp.set_attribute('fov', '90')
        
        depth_transform = carla.Transform(carla.Location(x=0.8, z=1.7))
        self.depth_sensor = self.world.spawn_actor(depth_bp, depth_transform, attach_to=self.ego_vehicle)
        self.depth_sensor.listen(self._on_depth_image)
        print("[OK] Depth camera attached")
        
        # Collision Sensor
        collision_bp = self.blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.ego_vehicle)
        self.collision_sensor.listen(self._on_collision)
        print("[OK] Collision sensor attached")
        
        # Lane Invasion Sensor
        lane_inv_bp = self.blueprint_library.find('sensor.other.lane_invasion')
        self.lane_invasion_sensor = self.world.spawn_actor(lane_inv_bp, carla.Transform(), attach_to=self.ego_vehicle)
        self.lane_invasion_sensor.listen(self._on_lane_invasion)
        print("[OK] Lane invasion sensor attached")
    
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
        self._collision_count += 1
    
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
    
    def _generate_waypoints(self):
        """Generate waypoint route starting from ego vehicle location"""
        if not self.ego_vehicle or not self.map:
            return
        
        try:
            location = self.ego_vehicle.get_location()
            waypoint = self.map.get_waypoint(location, project_to_road=True)
            
            if not waypoint:
                return
            
            # Create a route of 40 waypoints by following the road
            self.waypoints = []
            current_wp = waypoint
            
            # Generate waypoints at 5m intervals
            for i in range(40):
                self.waypoints.append(current_wp.transform.location)
                
                # Move to next waypoint (5m ahead)
                next_wps = current_wp.next(5.0)
                if next_wps and len(next_wps) > 0:
                    current_wp = next_wps[0]
                else:
                    # Dead end; repeat the last waypoint
                    self.waypoints.extend([current_wp.transform.location] * (40 - len(self.waypoints)))
                    break
            
            self.total_waypoints = len(self.waypoints)
            self.current_waypoint_idx = 0
            self.waypoints_crossed = 0
        
        except Exception as e:
            # Fallback: create dummy waypoints
            if self.ego_vehicle:
                loc = self.ego_vehicle.get_location()
                self.waypoints = [carla.Location(loc.x + i * 5, loc.y, loc.z) for i in range(40)]
                self.total_waypoints = 40
    
    def _update_waypoint_progress(self):
        """Update which waypoint the vehicle is closest to"""
        if not self.ego_vehicle or len(self.waypoints) == 0:
            return
        
        try:
            ego_loc = self.ego_vehicle.get_location()
            ego_pos = np.array([ego_loc.x, ego_loc.y, ego_loc.z])
            
            # Find closest waypoint
            min_dist = float('inf')
            closest_idx = self.current_waypoint_idx
            
            # Search from current index onwards (prevents backtracking)
            for idx in range(self.current_waypoint_idx, len(self.waypoints)):
                wp = self.waypoints[idx]
                wp_pos = np.array([wp.x, wp.y, wp.z])
                dist = np.linalg.norm(ego_pos - wp_pos)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = idx
            
            # Check if we've crossed waypoint
            if min_dist < self.waypoint_cross_distance and closest_idx > self.current_waypoint_idx:
                self.waypoints_crossed += 1
                self.current_waypoint_idx = closest_idx
        
        except Exception as e:
            pass
    
    def _get_distance_to_next_waypoint(self) -> float:
        """Get distance to next waypoint"""
        if not self.ego_vehicle or len(self.waypoints) == 0:
            return 100.0
        
        try:
            ego_loc = self.ego_vehicle.get_location()
            ego_pos = np.array([ego_loc.x, ego_loc.y, ego_loc.z])
            
            # Get next waypoint (or current if at end)
            wp_idx = min(self.current_waypoint_idx, len(self.waypoints) - 1)
            wp = self.waypoints[wp_idx]
            wp_pos = np.array([wp.x, wp.y, wp.z])
            
            dist = np.linalg.norm(ego_pos - wp_pos)
            return float(dist)
        
        except Exception:
            return 100.0
    
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
            'speed': float(current_speed),
            'speed_limit': float(speed_limit)  # Add dynamic speed limit for CBF
        }
    
    def _get_empty_observation(self) -> Dict:
        """Return empty observation (for error cases)"""
        return {
            'rgb_data': np.zeros((360, 640, 3), dtype=np.uint8),
            'lidar_data': np.zeros((3, 500), dtype=np.float32),
            'position': np.zeros((3,), dtype=np.float32),
            'speed': np.zeros((1,), dtype=np.float32),
            'heading': np.zeros((1,), dtype=np.float32),
        }
    
    def _get_observation(self) -> Dict:
        """Get current observation"""
        if self.ego_vehicle is None:
            return self._get_empty_observation()  # or however your env returns a blank obs
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
    
    # ========================================================================
    # ADVANCED REWARD SHAPING MECHANISMS (NEW - 5 Reward Functions)
    # ========================================================================
    
    def compute_lane_centering_reward(self) -> float:
        """
        Reward for staying centered in lane; penalize edge-hugging.
        Bonus for smooth steering without jerky corrections.
        Returns: [-1, 1] reward value
        """
        lane_offset = self._compute_lane_offset()
        
        # Primary: lane centering (max 1.0 at center, -1.0 if >0.5m off)
        if abs(lane_offset) <= 0.5:
            lane_reward = 1.0 - (2.0 * abs(lane_offset))
        else:
            lane_reward = -1.0
        
        # Secondary: steering smoothness (penalize jerky corrections)
        steering_smoothness = -0.3 * abs(self._prev_steering)
        
        return lane_reward + steering_smoothness
    
    def compute_forward_motion_reward(self) -> float:
        """
        Reward for moving in intended heading direction.
        Penalize lateral (sideways) motion and crabbing.
        Returns: [-1, 1] reward value
        """
        if self.ego_vehicle is None:
            return 0.0
        
        velocity = self.ego_vehicle.get_velocity()
        transform = self.ego_vehicle.get_transform()
        
        # Vehicle heading vector
        heading_rad = np.radians(transform.rotation.yaw)
        heading_x = np.cos(heading_rad)
        heading_y = np.sin(heading_rad)
        heading = np.array([heading_x, heading_y])
        
        # Velocity vector
        vel_vector = np.array([velocity.x, velocity.y])
        vel_magnitude = np.linalg.norm(vel_vector)
        
        if vel_magnitude < 0.1:
            return 0.0
        
        vel_normalized = vel_vector / vel_magnitude
        
        # Forward component (dot product)
        forward_component = np.dot(vel_normalized, heading)
        
        # Lateral component (perpendicular magnitude)
        lateral_magnitude = np.sqrt(1.0 - forward_component**2) if abs(forward_component) <= 1.0 else 0.0
        
        # Rewards
        forward_reward = forward_component
        lateral_penalty = -0.4 * lateral_magnitude
        
        return forward_reward + lateral_penalty
    
    def compute_safe_following_reward(self) -> float:
        """
        Reward for maintaining safe headway from vehicle ahead.
        Uses LiDAR to detect leading vehicle.
        Bonus for 2-3s headway (optimal), penalty for tailgating.
        Returns: [-3, 1] reward value
        """
        if self.lidar_data is None or self.ego_vehicle is None:
            return 0.0
        
        velocity = self.ego_vehicle.get_velocity()
        ego_speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        # Extract forward-looking LiDAR points
        lidar_points = self.lidar_data  # Shape: (3, N)
        
        # Filter: points ahead (X > 0) and within lane width (Y: -2 to 2m)
        forward_points = lidar_points[0, :] > 0
        lateral_points = np.abs(lidar_points[1, :]) < 2.0
        mask = forward_points & lateral_points
        
        if not np.any(mask):
            return 0.0  # No vehicle ahead
        
        # Find closest point (likely leading vehicle)
        forward_distances = lidar_points[0, mask]
        min_distance = float(np.min(forward_distances))
        
        # TTC: Time-To-Collision (conservative: assume leader stationary)
        ttc = min_distance / max(ego_speed, 0.1)
        
        # Reward based on TTC zones
        if 2.0 <= ttc <= 3.5:
            headway_reward = 1.0  # Optimal spacing
        elif 1.5 <= ttc < 2.0:
            headway_reward = 0.5 - 0.5 * (2.0 - ttc)  # Approaching
        elif 1.0 <= ttc < 1.5:
            headway_reward = -1.0  # Too close
        elif ttc < 1.0:
            headway_reward = -3.0  # Extreme danger
        else:  # ttc > 3.5
            headway_reward = 0.0  # Too far
        
        return headway_reward
    
    def compute_traffic_flow_reward(self) -> float:
        """
        Penalize driving too slowly and blocking traffic.
        Reward smooth acceleration; penalize jerk and hard braking.
        Returns: [-1, 0] reward value
        """
        if self.ego_vehicle is None:
            return 0.0
        
        velocity = self.ego_vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        speed_limit = self._get_speed_limit()
        
        # Calculate acceleration
        acceleration = (speed - self._prev_speed) / 0.05  # delta_t = 0.05s
        
        # Calculate jerk (change in acceleration)
        jerk = abs(acceleration - self._prev_acceleration)
        self._prev_acceleration = acceleration
        
        # SLOW DRIVING PENALTY
        slow_threshold = speed_limit - 2.0
        slow_penalty = 0.0
        if speed < slow_threshold:
            slow_penalty = -0.5 * (slow_threshold - speed) ** 2
        
        # SMOOTHNESS BONUS (penalize jerky motion)
        smoothness_bonus = -0.1 * jerk
        
        # HARD BRAKING PENALTY
        brake_penalty = 0.0
        if acceleration < -2.0:
            brake_penalty = -0.3 * abs(acceleration)
        
        return slow_penalty + smoothness_bonus + brake_penalty
    
    def compute_yield_and_maneuver_reward(self) -> float:
        """
        Reward for yielding to faster vehicles approaching from behind.
        Penalize blocking traffic and excessive lane changes.
        Returns: [-1, 0.5] reward value
        """
        if self.lidar_data is None or self.ego_vehicle is None:
            return 0.0
        
        velocity = self.ego_vehicle.get_velocity()
        ego_speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        lidar_points = self.lidar_data
        
        # Extract rear-looking points (behind vehicle)
        behind_points = lidar_points[0, :] < -2.0
        lateral_points = np.abs(lidar_points[1, :]) < 3.0
        mask = behind_points & lateral_points
        
        yield_reward = 0.0
        
        if np.any(mask):
            # Vehicle detected behind
            rear_distances = -lidar_points[0, mask]
            min_rear_distance = float(np.min(rear_distances))
            
            # Estimate rear vehicle speed (assume 20% faster)
            estimated_rear_speed = ego_speed * 1.2
            
            # TTC from behind
            ttc_behind = min_rear_distance / max(estimated_rear_speed - ego_speed, 0.1)
            
            lane_offset = self._compute_lane_offset()
            
            # BLOCKING PENALTY or yield bonus
            if ttc_behind < 3.0 and estimated_rear_speed > ego_speed + 1.5:
                if abs(lane_offset) < 0.4:  # Centered (blocking)
                    yield_reward = -1.0
                elif abs(lane_offset) > 0.5:  # Moved to side (yielded)
                    yield_reward = 0.5
        
        # LANE CHANGE STABILITY: penalize excessive lane changes
        current_lane_offset = self._compute_lane_offset()
        
        if abs(current_lane_offset - self._last_lane_offset) > 0.3:
            self._lane_change_count += 1
            if self._lane_change_count > 2 and yield_reward == 0.0:
                yield_reward = -0.2  # Penalize fidgeting
        
        self._last_lane_offset = current_lane_offset
        
        return yield_reward
    
    def _log_reward_breakdown(self, components: Dict[str, float]):
        """Debug log reward component breakdown (Phase 3)"""
        if not self._debug_reward_logging:
            return
        
        self._reward_log_counter += 1
        if self._reward_log_counter % self._debug_log_frequency != 0:
            return
        
        print(f"\n[REWARD DEBUG] Step {self._episode_step}:")
        total = sum(components.values())
        for name, value in components.items():
            pct = 100.0 * value / max(abs(total), 0.01)
            print(f"  {name:30s}: {value:8.4f} ({pct:6.1f}%)")
        print(f"  {'TOTAL':30s}: {total:8.4f}")
    
    def compute_safety_buffer_reward(self) -> float:
        """
        Reward for maintaining safety margins and collision avoidance.
        Phase 1 Enhancement: Integrates CBF collision prevention metrics.
        Returns: [-0.5, 1.0] reward value
        """
        if self.ego_vehicle is None:
            return 0.0
        
        # Base: maintain collision distance margin
        d_collision = self._compute_collision_distance()
        
        # Normalize by d_min (5m is safe minimum)
        d_min = 5.0
        margin_ratio = min(d_collision / d_min, 2.0)  # Saturate at 2x
        
        # Reward for maintaining good margin (0 to +0.5)
        margin_reward = 0.25 * margin_ratio
        
        # Penalty for being too close
        if d_collision < d_min:
            margin_reward -= 0.5 * (1.0 - d_collision / d_min)
        
        # BONUS: Check if CBF layer is available and prevented a collision
        # This integrates Phase 1 collision prevention metrics
        collision_prevention_bonus = 0.0
        if hasattr(self, '_cbf_wrapper'):
            # If CBF wrapper is attached, check prevention metrics
            if hasattr(self._cbf_wrapper, 'cbf_layer'):
                cbf = self._cbf_wrapper.cbf_layer
                if hasattr(cbf, 'collision_prevented') and cbf.collision_prevented:
                    collision_prevention_bonus = 0.5  # Reward for preventing crash
        
        return margin_reward + collision_prevention_bonus
    
    def compute_waypoint_progress_reward(self) -> float:
        """
        Reward for making progress toward goal via waypoints.
        Dual-tier: milestone rewards + distance-based gradient.
        Phase 2 Enhancement: Tracks waypoint crossing and distance to next goal.
        Returns: [-0.1, 1.5] reward value
        """
        if len(self.waypoints) == 0:
            return 0.0
        
        # TIER 1: Milestone reward (sparse, high value)
        # Did we just cross a waypoint? Check if waypoints_crossed changed this step
        milestone_reward = 0.0
        
        # Track previous state to detect new crossing
        if not hasattr(self, '_prev_waypoints_crossed'):
            self._prev_waypoints_crossed = 0
        
        if self.waypoints_crossed > self._prev_waypoints_crossed:
            # Just crossed a waypoint!
            milestone_reward = 5.0  # High reward for crossing a waypoint
        self._prev_waypoints_crossed = self.waypoints_crossed
        
        # Scale milestone by progress percentage (encourage continuous progress)
        progress_pct = self.waypoints_crossed / max(1, self.total_waypoints)
        milestone_reward *= progress_pct  # 0 at start, 1.0 at 100%
        
        # TIER 2: Distance-based gradient (dense, low value)
        # Provide continuous signal even between waypoints
        dist_to_next = self._get_distance_to_next_waypoint()
        
        # Distance reward: inverse with 10m scaling
        # At 0m: +0.5, at 10m: +0.05, at 50m: +0.01
        distance_reward = 0.5 / (1.0 + dist_to_next / 10.0)
        
        # Combine tiers
        total_progress_reward = 0.25 * milestone_reward + 0.25 * distance_reward
        
        return total_progress_reward
    
    def _compute_reward(self) -> float:
        """Compute reward with advanced reward shaping mechanisms"""
        reward = 0.0
        reward_components = {}  # Track components for debug logging
        
        # ====== CORE SAFETY (BASE) ======
        # Collision penalty (critical) - BUG FIX: DON'T reset flag here!
        # Flag will be reset in step() after termination is decided
        if self.collision_occurred:
            reward -= 150.0  # Increased penalty from -100 to -150
            reward_components['collision'] = -150.0
            # DON'T reset flag here - causes step() to never terminate
            return reward  # Episode terminates; skip other rewards
        
        # Lane invasion penalty
        if self.lane_invaded:
            reward -= 50.0
            self.lane_invaded = False
            reward_components['lane_invasion'] = -50.0
        
        # Speed reward (encourage 8-12 m/s ~29-43 km/h)
        velocity = self.ego_vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        speed_reward = 0.0
        if 8.0 <= speed <= 12.0:
            speed_reward = 2.0  # Optimal speed zone
        elif speed > 15.0:
            speed_reward = -0.5  # Excessive speed penalty
        reward += speed_reward
        reward_components['speed'] = speed_reward
        
        # Track speed for flow reward calculation
        self._prev_speed = speed
        
        # ====== ADVANCED BEHAVIORAL REWARDS (7 COMPONENTS WITH BALANCED WEIGHTING) ======
        if self._use_advanced_rewards:
            # Weight each component equally: 1/7 ≈ 0.143 each
            component_weight = 1.0 / 7.0  # ~0.143
            
            # 1. Lane Centering (precision lane-keeping, smooth steering)
            lane_center_reward = self.compute_lane_centering_reward()
            weighted_lc = component_weight * lane_center_reward
            reward += weighted_lc
            reward_components['lane_centering'] = weighted_lc
            
            # 2. Forward Motion (drive in heading direction, no crabbing)
            forward_reward = self.compute_forward_motion_reward()
            weighted_fm = component_weight * forward_reward
            reward += weighted_fm
            reward_components['forward_motion'] = weighted_fm
            
            # 3. Safe Following (maintain headway from vehicle ahead)
            headway_reward = self.compute_safe_following_reward()
            weighted_hw = component_weight * headway_reward
            reward += weighted_hw
            reward_components['safe_following'] = weighted_hw
            
            # 4. Traffic Flow Efficiency (don't block, smooth acceleration)
            flow_reward = self.compute_traffic_flow_reward()
            weighted_tf = component_weight * flow_reward
            reward += weighted_tf
            reward_components['traffic_flow'] = weighted_tf
            
            # 5. Yield & Maneuver (move out of way for faster vehicles)
            yield_reward = self.compute_yield_and_maneuver_reward()
            weighted_ym = component_weight * yield_reward
            reward += weighted_ym
            reward_components['yield_maneuver'] = weighted_ym
            
            # 6. Safety Buffer (Phase 1) - margin maintenance + collision prevention
            safety_buffer_reward = self.compute_safety_buffer_reward()
            weighted_sb = component_weight * safety_buffer_reward
            reward += weighted_sb
            reward_components['safety_buffer'] = weighted_sb
            
            # 7. Waypoint Progress (Phase 2) - milestone + distance rewards
            waypoint_progress_reward = self.compute_waypoint_progress_reward()
            weighted_wp = component_weight * waypoint_progress_reward
            reward += weighted_wp
            reward_components['waypoint_progress'] = weighted_wp
            
            # Debug logging (Phase 3)
            self._log_reward_breakdown(reward_components)
        
        return reward
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one environment step"""
        # Parse 3D action: [steering, throttle, brake]
        # Each is in range [-1, 1]
        steer = float(np.clip(action[0], -1.0, 1.0))
        throttle = float(np.clip(action[1], -1.0, 1.0))
        brake = float(np.clip(action[2], -1.0, 1.0))
        
        # Clamp to [0, 1] for CARLA (can't apply both throttle and brake)
        throttle = max(0.0, throttle)  # Throttle is [0, 1]
        brake = max(0.0, brake)        # Brake is [0, 1]
        
        # Normalize if both are non-zero (shouldn't happen, but safety)
        total = throttle + brake
        if total > 1.0:
            throttle /= total
            brake /= total
        
        # Convert to CARLA control
        control = carla.VehicleControl()
        control.steer = steer
        control.throttle = throttle
        control.brake = brake
        
        self.ego_vehicle.apply_control(control)
        
        # Tick world
        self.world.tick()
        self._episode_step += 1
        
        # Get observation and reward
        obs = self._get_observation()
        reward = self._compute_reward()
        
        # Update waypoint progress (Phase 2)
        self._update_waypoint_progress()
        
        # Update spectator camera to track ego vehicle
        self.render()
        
        # Display RGB sensor data if enabled
        if self.show_sensor_data and self.rgb_data is not None:
            rgb_display = cv2.cvtColor(self.rgb_data, cv2.COLOR_BGR2RGB)
            cv2.imshow('CARLA RGB Camera', rgb_display)
            cv2.waitKey(1)
        
        # ===== IMMEDIATE TERMINATION ON COLLISION (BUG FIX) =====
        # CRITICAL: Check collision BEFORE returning and LOG it
        terminated = False
        if self.collision_occurred:
            print(f"\n{'='*60}")
            print(f"[COLLISION DETECTED] Episode ended at step {self._episode_step}")
            print(f"  Collision distance: {self._collision_distance:.2f}m")
            print(f"  Reward penalty applied: -150")
            print(f"  Total reward this step: {reward:.2f}")
            print(f"{'='*60}\n")
            terminated = True
            # ONLY reset flag after termination is decided and logged
            self.collision_occurred = False
        
        # Timeout termination (secondary)
        elif self._episode_step >= self.time_limit:
            terminated = True
        
        truncated = False
        
        # Info dict (Phase 2 - add waypoint tracking)
        info = {
            'collision_distance': self._collision_distance,
            'lane_invaded': self.lane_invaded,
            'episode_step': self._episode_step,
            'waypoints_crossed': self.waypoints_crossed,
            'waypoints_remaining': max(0, self.total_waypoints - self.waypoints_crossed),
            'distance_to_next_wp': self._get_distance_to_next_waypoint(),
            'progress_pct': 100.0 * self.waypoints_crossed / max(1, self.total_waypoints),
        }
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset environment"""
        # Reset counters
        self._episode_step = 0
        self._collision_distance = 100.0
        self._lane_invasion_count = 0
        self._speed_limit_violation_count = 0
        self._collision_count = 0
        self._cbf_correction_count = 0
        self._avg_correction_mag = 0.0
        self.collision_occurred = False
        self.lane_invaded = False
        
        # Reset advanced reward shaping state (NEW)
        self._prev_steering = 0.0
        self._prev_speed = 0.0
        self._prev_acceleration = 0.0
        self._lane_change_count = 0
        self._last_lane_offset = 0.0
        
        # Reset waypoint tracking (Phase 2)
        self.waypoints_crossed = 0
        self.current_waypoint_idx = 0
        
        # Reset ego vehicle
        if self.ego_vehicle:
            spawn_point = random.choice(self.map.get_spawn_points())
            self.ego_vehicle.set_transform(spawn_point)
            self.ego_vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
        
        # Generate new waypoint route (Phase 2)
        self._generate_waypoints()
        
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
        
        print("[OK] All actors destroyed")
    
    @property
    def safety_metrics(self) -> Dict:
        """Expose safety metrics for callback logging"""
        return {
            'lane_invasions': self._lane_invasion_count,
            'speed_violations': self._speed_limit_violation_count,
            'collisions': self._collision_count,
            'cbf_corrections': self._cbf_correction_count,
            'cbf_correction_magnitude': self._avg_correction_mag,
        }


# ============================================================================
# Observation Preprocessing Wrapper (Pipeline Integration)
# ============================================================================

class PipelineObservationWrapper(gym.ObservationWrapper):
    """
    Converts raw CARLA observations to pipeline embeddings (512-dim tensor).
    Supports loading pretrained encoder while keeping transformer trainable.
    """
    
    def __init__(
        self, 
        env: gym.Env, 
        embed_dim: int = 512, 
        num_frames: int = 8,
        encoder_path: Optional[str] = None
    ):
        super().__init__(env)
        
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.encoder_path = encoder_path
        
        # Initialize pipeline with ImageNet-pretrained ResNet50
        # (not loading feature extractor weights from encoder_path)
        print(f"[PIPELINE] Initializing with ImageNet-pretrained ResNet50")
        self.pipeline = Pipeline.from_defaults(
            num_frames=num_frames,
            embed_dim=embed_dim,
            use_timesformer=False,
            fe_weights_path=None,  # Use default ImageNet weights
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.pipeline.eval()
        
        # Load pretrained SpatioTemporalEncoder weights if provided
        if encoder_path is not None:
            print(f"[PIPELINE] Loading pretrained SpatioTemporalEncoder from: {encoder_path}")
            try:
                state_dict = torch.load(encoder_path, map_location=self.pipeline.device)
                # Handle both direct state_dict and checkpoint format
                if isinstance(state_dict, dict) and 'st_encoder_state' in state_dict:
                    self.pipeline.st_encoder.load_state_dict(state_dict['st_encoder_state'])
                else:
                    # Assume it's direct state_dict
                    self.pipeline.st_encoder.load_state_dict(state_dict)
                print("[PIPELINE] ✓ Loaded SpatioTemporalEncoder weights")
            except Exception as e:
                print(f"[PIPELINE] Error loading checkpoint: {e}")
        else:
            print("[PIPELINE] Training SpatioTemporalEncoder from scratch")
        
        # Freeze feature extractor (ResNet50)
        print("[PIPELINE] Freezing ResNet50 feature extractor weights")
        self.pipeline.feature_extractor.requires_grad = False
        for param in self.pipeline.feature_extractor.parameters():
            param.requires_grad = False
        
        # Ensure transformer modules are trainable
        print("[PIPELINE] Enabling training for SpatioTemporalEncoder + StackedHierarchicalTransformer")
        for param in self.pipeline.st_encoder.parameters():
            param.requires_grad = True
        for param in self.pipeline.stacked_transformer.parameters():
            param.requires_grad = True
        
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
    """
    Applies CBF safety corrections to actions
    Tracks corrections and integrates with trust scoring
    """
    
    def __init__(self, env: gym.Env, alpha: float = 1.0, use_trust_score: bool = True, correction_penalty: float = 0.003):
        super().__init__(env)
        
        self.cbf_layer = CBFSafetyLayer(alpha=alpha, d_min=5.0, y_max=1.5, v_max=15.0)
        self.use_trust_score = use_trust_score
        self.correction_penalty = correction_penalty  # Reduced from 0.01 to 0.003 (Phase 1)
        self.trust_score = 1.0  # Default full trust
        
        # Metrics for tracking
        self.correction_count = 0
        self.correction_magnitudes = []
        self.episode_corrections = 0
        self.episode_correction_mag = 0.0
        self.last_correction_mag = 0.0
        
        # Link to environment for reward function access (Phase 1)
        if hasattr(env.unwrapped, '__dict__'):
            env.unwrapped._cbf_wrapper = self
    
    def set_trust_score(self, score: float):
        """Update trust score from external source (e.g., training callback)"""
        self.trust_score = np.clip(float(score), 0.0, 1.0)
    
    def action(self, action: np.ndarray, trust_score: float = 1.0) -> np.ndarray:
        """
        Apply CBF correction if violation detected
        
        Args:
            action: np.ndarray shape (3,) - [steer, throttle, brake]
            trust_score: float in [0, 1] - ensemble confidence (optional)
        
        Returns:
            safe_action: np.ndarray shape (3,) - corrected action
        """
        # Get current state
        try:
            current_obs = self.env.unwrapped._get_observation()
            current_speed = float(current_obs['speed'][0])
        except:
            current_speed = 0.0
        
        cbf_state = self.env.unwrapped.build_cbf_state(current_speed)
        
        # Enable verbose logging if collision distance is small
        verbose_cbf = cbf_state['d_collision'] < 10.0  # Log when <10m away
        
        # Apply CBF correction with trust score modulation
        try:
            safe_action = self.cbf_layer.compute_safe_action(
                action, 
                cbf_state,
                trust_score=trust_score if self.use_trust_score else 1.0,
                verbose=verbose_cbf  # Pass verbose flag
            )
            
            # Track correction metrics
            correction_mag = float(np.linalg.norm(safe_action - action))
            self.last_correction_mag = correction_mag
            
            if correction_mag > 0.01:  # Only count non-negligible corrections
                self.correction_count += 1
                self.episode_corrections += 1
                self.correction_magnitudes.append(correction_mag)
                self.episode_correction_mag = np.mean(self.correction_magnitudes[-100:])  # Rolling avg
            
            # Update environment's safety metrics
            if hasattr(self.env.unwrapped, '_cbf_correction_count'):
                self.env.unwrapped._cbf_correction_count = self.correction_count
            if hasattr(self.env.unwrapped, '_avg_correction_mag'):
                self.env.unwrapped._avg_correction_mag = self.episode_correction_mag
        
        except Exception as e:
            print(f"[CBF] Correction failed: {e}")
            safe_action = action.copy()
            self.last_correction_mag = 0.0
        
        return np.clip(safe_action, -1.0, 1.0)
    
    def step(self, action: np.ndarray):
        """Override step to apply CBF corrections and add reward penalty"""
        # Apply CBF safety layer to action with current trust score
        safe_action = self.action(action, trust_score=self.trust_score)
        
        # Call parent step with safe action
        obs, reward, terminated, truncated, info = self.env.step(safe_action)
        
        # Apply penalty for CBF corrections to encourage safety learning
        if self.last_correction_mag > 0.01:
            correction_penalty = self.correction_penalty * self.last_correction_mag
            reward -= correction_penalty
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        """Reset episode metrics"""
        self.episode_corrections = 0
        self.episode_correction_mag = 0.0
        self.last_correction_mag = 0.0
        self.cbf_layer.reset_metrics()
        return self.env.reset(**kwargs)


# ============================================================================
# Custom TensorBoard Callback
# ============================================================================

class SafetyMetricsCallback(BaseCallback):
    """Log safety and progress metrics to TensorBoard"""
    
    def __init__(self, verbose: int = 0, log_frequency: int = 100):
        super().__init__(verbose)
        self.log_frequency = log_frequency
        self.step_count = 0
        
        # Accumulators for per-episode metrics
        self.episode_safety_rewards = 0.0
        self.episode_progress_rewards = 0.0
        self.episode_cbf_corrections = 0
        self.episode_collisions_prevented = 0
    
    def _on_step(self) -> bool:
        """Called at each training step"""
        self.step_count += 1
        
        # Extract environment from model
        env = self.model.get_env()
        if env is None:
            return True
        
        # Access the underlying CARLA environment
        carla_env = env.unwrapped if hasattr(env, 'unwrapped') else env
        
        # Log every log_frequency steps
        if self.step_count % self.log_frequency == 0:
            # ===== PHASE 1: Safety Metrics =====
            if hasattr(carla_env, '_cbf_wrapper') and hasattr(carla_env._cbf_wrapper, 'cbf_layer'):
                cbf = carla_env._cbf_wrapper.cbf_layer
                
                # Log CBF correction magnitude
                self.logger.record(
                    'safety/cbf_correction_mag',
                    float(carla_env._cbf_wrapper.last_correction_mag)
                )
                
                # Log collision prevention (binary: 0 or 1)
                if hasattr(cbf, 'collision_prevented'):
                    self.logger.record(
                        'safety/collision_prevented',
                        float(cbf.collision_prevented)
                    )
                
                # Log avoidance efficiency
                if hasattr(cbf, 'avoidance_efficiency'):
                    self.logger.record(
                        'safety/avoidance_efficiency',
                        float(cbf.avoidance_efficiency)
                    )
                
                # Log correction count
                self.logger.record(
                    'safety/total_cbf_corrections',
                    float(cbf.correction_count)
                )
                
                # Log constraint violations
                if hasattr(cbf, 'constraint_violations'):
                    self.logger.record(
                        'safety/collision_violations',
                        int(cbf.constraint_violations.get('collision', 0))
                    )
                    self.logger.record(
                        'safety/lane_violations',
                        int(cbf.constraint_violations.get('lane', 0))
                    )
                    self.logger.record(
                        'safety/speed_violations',
                        int(cbf.constraint_violations.get('speed', 0))
                    )
            
            # ===== PHASE 2: Progress Metrics =====
            if hasattr(carla_env, 'waypoints_crossed'):
                self.logger.record(
                    'progress/waypoints_crossed',
                    int(carla_env.waypoints_crossed)
                )
                self.logger.record(
                    'progress/waypoints_remaining',
                    max(0, carla_env.total_waypoints - carla_env.waypoints_crossed)
                )
                self.logger.record(
                    'progress/progress_pct',
                    100.0 * carla_env.waypoints_crossed / max(1, carla_env.total_waypoints)
                )
                self.logger.record(
                    'progress/distance_to_next_wp',
                    carla_env._get_distance_to_next_waypoint()
                )
            
            # ===== Collision & Lane Metrics =====
            self.logger.record(
                'safety/collision_distance',
                carla_env._compute_collision_distance()
            )
            self.logger.record(
                'safety/lane_offset',
                carla_env._compute_lane_offset()
            )
        
        # ===== Episode Termination Logging =====
        # Check if episode just finished
        if hasattr(self.model, 'env') and hasattr(self.model.env, 'buf_dones'):
            # Get done flags for each environment
            if self.model.env.buf_dones is not None and len(self.model.env.buf_dones) > 0:
                for env_idx, is_done in enumerate(self.model.env.buf_dones):
                    if is_done:
                        # Episode ended; log outcome
                        if hasattr(carla_env, 'collision_occurred') and carla_env.collision_occurred:
                            self.logger.record('episode/outcome_collision', 1)
                            self.logger.record('episode/outcome_timeout', 0)
                        else:
                            self.logger.record('episode/outcome_collision', 0)
                            self.logger.record('episode/outcome_timeout', 1)
                        
                        # Log final episode metrics
                        if hasattr(carla_env, 'waypoints_crossed'):
                            self.logger.record(
                                'episode/final_waypoint_progress',
                                100.0 * carla_env.waypoints_crossed / max(1, carla_env.total_waypoints)
                            )
        
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
    print("[OK] Environment created\n")
    
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
    print("[OK] SAC agent created\n")
    
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
    print(f"\n[OK] Final model saved to {final_path}")
    
    # Cleanup
    env.close()
    print("[OK] Environment closed")
    
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
