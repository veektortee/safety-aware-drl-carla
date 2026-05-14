"""
SAC Agent Training with CBF Safety Layer on CARLA — v2.1
Processes RGB+LIDAR observations through spatiotemporal pipeline,
detects safety violations (lane invasion, speed limit), applies CBF corrections,
and logs everything to TensorBoard.

Uses raw CARLA API for direct environment control.

Fixes applied in v2.1 (surgical — no mechanics changed):
  [3]  Action space bounds corrected: throttle/brake ∈ [0,1] not [-1,1].
       SAC init near 0 now maps to 0.5 (light pedal) instead of 0 (stall).
       step() parsing simplified to three direct clips — no conditional logic.
       CBF clip also updated to respect per-control physical bounds.
  [4]  reset(): world.tick() called BEFORE _generate_waypoints() so the
       post-teleport spawn transform is committed before get_location() is read.
  [5]  Traffic Manager set to synchronous mode before world sync is enabled.
       NPCs now tick on the same clock as the ego vehicle — eliminates phantom
       collisions from async TM. set_autopilot() passes TM port explicitly.
  [6]  lane_invaded flag no longer reset inside _compute_reward() (side-effect
       in a compute function). It is now reset in step() after reward is
       computed. obstacle_detected was already cleared at top of step().
  [10] train_sac_agent(): show_sensor_data=False — was True which called
       cv2.imshow()+waitKey(1) every step, breaking headless/SSH runs.
  [11] render() gated behind render_mode == "human" check — was firing
       unconditionally every step even in headless training mode.
  [12] _generate_waypoints() removed from __init__ — ego_vehicle is None at
       that point so the call silently no-ops. Waypoints are generated in
       reset() after the post-teleport tick.
  [13] Two warmup world.tick() calls added after _attach_sensors() so sensor
       callbacks fire before the first reset(), preventing all-zero initial obs.

Already correct in uploaded version (no change):
  [1]  _update_waypoint_progress() runs before _compute_reward() in step().
  [2]  _prev_waypoints_crossed reset to 0 in reset().
  [7]  obstacle_distance reset to 100.0 in reset().
  [8]  Dead code removed from _compute_collision_distance().
  [9]  Collision log prints actual reward value.
  [14] SafetyMetricsCallback uses locals['rewards'], not buf_rewards.

Experiment integration (2qCNNsac_experiment.py):
  No changes required. Wrapper stack, env unwrapping via _find_carla_and_cbf(),
  info dict keys, and callback imports all align correctly with this file.

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
        self.traffic_manager = None  # FIX [5]: explicit TM handle for sync mode
        
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
        self.obstacle_sensor = None  # Obstacle Detector (RSS) sensor (Phase 5.3)
        
        # Sensor data
        self.rgb_data = None
        self.lidar_data = None
        self.collision_occurred = False
        self.lane_invaded = False
        self.depth_data = None
        self.obstacle_detected = False  # Obstacle Detector state (Phase 5.3)
        self.obstacle_distance = 100.0  # Distance to nearest obstacle (Phase 5.3)
        self.obstacle_actor = None  # Actor causing obstacle (Phase 5.3)
        
        # State tracking
        self._collision_distance = 100.0
        self._lane_invasion_count = 0
        self._speed_limit_violation_count = 0
        self._speed_limit_check_counter = 0
        self._current_speed_limit = 15.0
        self._episode_step = 0
        self._collision_count = 0
        self._cbf_correction_count = 0
        self._obstacle_detection_count = 0  # Count of obstacle detections (Phase 5.3)
        self._avg_correction_mag = 0.0
        self._ensemble_uncertainty = 0.0
        self._trust_score = 1.0
        
        # Advanced reward shaping state (NEW)
        self._prev_steering = 0.0
        self._prev_speed = 0.0
        self._prev_acceleration = 0.0
        self._lane_change_count = 0
        self._last_lane_offset = 0.0
        self._prev_distance_to_next_wp = None
        self._stalled_steps = 0
        self._use_advanced_rewards = True  # Toggle for backward compatibility
        self._last_reward_components = {}
        self._last_total_reward = 0.0
        
        # Waypoint tracking (NEW - Phase 2)
        self.waypoints = []  # List of target waypoints
        self.current_waypoint_idx = 0
        self.waypoint_cross_distance = 2.0  # Distance to trigger waypoint crossing
        self.waypoints_crossed = 0
        self.total_waypoints = 40  # Default; updated when route is generated
        # FIX [12]: _generate_waypoints() removed from __init__ — ego_vehicle
        # is None at this point so the call silently no-ops and waypoints stay
        # empty until reset(). Waypoints are now generated in reset() after the
        # post-teleport world.tick() so get_location() is always correct.
        
        # Endpoint tracking & reward (Phase 5)
        self.endpoint_location = None
        self.endpoint_reached = False
        self.endpoint_distance = 9999.0
        self.waypoint_completion_ratio = 0.0
        
        # Per-episode CBF counter (Phase 5)
        self._cbf_correction_count_episode = 0  # Resets each episode
        
        # Debug logging (Phase 3)
        self._debug_reward_logging = False  # Set to True to enable
        self._debug_log_frequency = 100  # Log every N steps
        self._reward_log_counter = 0
        
        # Action/Observation spaces
        # 3D action space: [steering, throttle, brake] for CBF compatibility
        # FIX [3]: throttle and brake corrected to [0, 1] — physically these are
        # one-directional pedals (gas, brake) with no negative meaning.
        # With [-1,1]: SAC init near 0 → throttle=0, brake=0 → car stalls → flat
        # reward, no gradient to escape.
        # With [0,1]:  SAC init near 0.5 → light throttle + light brake → car
        # crawls forward → reward gradient exists from step 1.
        # The CBF layer and step() parsing receive the same (steer,throttle,brake)
        # tuple; only the declared range contract changes.
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
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

        # FIX [13]: Two warmup ticks so all sensor callbacks fire at least once
        # before the first reset() is called. Without this, the initial
        # observation may contain all-zero rgb_data and lidar_data.
        self.world.tick()
        time.sleep(0.1)
        self.world.tick()
    
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
        # FIX [5]: Traffic Manager must be put in synchronous mode BEFORE the
        # world is set to sync mode. If TM runs async while the world is sync,
        # NPC vehicles tick on a different clock — they teleport/phase through
        # objects causing phantom collisions the agent can't learn from.
        self.traffic_manager = self.client.get_trafficmanager(8000)
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        self.traffic_manager.set_hybrid_physics_mode(True)

        # Set synchronous mode for reproducibility
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS
        self.world.apply_settings(settings)
        print("[OK] Synchronous mode enabled (20 FPS), TM in sync mode")
    
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
                v.set_autopilot(True, self.traffic_manager.get_port())
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
        
        # Obstacle Detector (RSS) Sensor (Phase 5.3)
        try:
            obstacle_bp = self.blueprint_library.find('sensor.other.obstacle')
            obstacle_bp.set_attribute('hit_radius', '1')
            obstacle_bp.set_attribute('distance', '20')  # Detection range: 50 meters
            obstacle_transform = carla.Transform(carla.Location(x=0.8, z=1.7))
            self.obstacle_sensor = self.world.spawn_actor(obstacle_bp, obstacle_transform, attach_to=self.ego_vehicle)
            self.obstacle_sensor.listen(self._on_obstacle_detected)
            print("[OK] Obstacle Detector (RSS) sensor attached")
        except Exception as e:
            print(f"[WARN] Could not attach Obstacle Detector sensor: {e}")
            self.obstacle_sensor = None
    
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
        self._collision_distance = 0.0  # Physical contact detected
    
    def _on_lane_invasion(self, event):
        """Lane invasion callback"""
        self.lane_invaded = True
        self._lane_invasion_count += 1
    
    def _on_obstacle_detected(self, event):
        """Obstacle Detector (RSS) sensor callback (Phase 5.3)"""
        self.obstacle_detected = True
        self._obstacle_detection_count += 1
        
        # Extract obstacle distance and actor information
        if hasattr(event, 'distance') and event.distance is not None:
            # Validate distance is in reasonable range [0, 100]
            if 0.0 <= event.distance <= 100.0:
                self.obstacle_distance = event.distance
            else:
                # Invalid distance, use default safe value
                self.obstacle_distance = 100.0
        else:
            self.obstacle_distance = 100.0
        
        if hasattr(event, 'other_actor'):
            self.obstacle_actor = event.other_actor
    
    def _compute_collision_distance(self) -> float:
        """Return collision distance based on sensor event"""
        # Prefer physical contact when available, otherwise use the obstacle sensor.
        if self.collision_occurred:
            return 0.0
        
        if getattr(self, "obstacle_detected", False):
            obstacle_distance = float(getattr(self, "obstacle_distance", 100.0))
            return float(np.clip(obstacle_distance, 0.0, 100.0))

        # 3. Default: Path is clear
        return 100.0

    
    
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

    def _distance_to_nearest_vehicle(self, location) -> float:
        """Return distance in meters to the nearest live non-ego vehicle."""
        min_distance = float("inf")
        try:
            for actor in self.world.get_actors().filter("vehicle.*"):
                if self.ego_vehicle is not None and actor.id == self.ego_vehicle.id:
                    continue
                actor_location = actor.get_location()
                distance = location.distance(actor_location)
                if distance < min_distance:
                    min_distance = distance
        except Exception:
            return float("inf")
        return min_distance

    def _select_safe_spawn_point(self, min_clearance: float = 8.0):
        """
        Pick a spawn point that is not currently occupied by nearby traffic.

        Falls back to the furthest available spawn point if none meet the clearance threshold.
        """
        spawn_points = list(self.map.get_spawn_points()) if self.map else []
        if not spawn_points:
            return None

        random.shuffle(spawn_points)
        best_spawn = spawn_points[0]
        best_clearance = -1.0

        for spawn_point in spawn_points:
            clearance = self._distance_to_nearest_vehicle(spawn_point.location)
            if clearance >= min_clearance:
                return spawn_point
            if clearance > best_clearance:
                best_clearance = clearance
                best_spawn = spawn_point

        return best_spawn
    
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
        """Get distance to the NEXT uncrossed waypoint ahead of current position.

        FIX A: The original code used current_waypoint_idx (the waypoint the car
        has already reached / is sitting on).  Moving away from spawn therefore
        increased the distance to that same waypoint, making distance_delta
        negative and generating a negative progress signal for correct forward
        motion.  We now always target current_waypoint_idx + 1 so the gradient
        correctly rewards closing the gap to the next milestone.
        """
        if not self.ego_vehicle or len(self.waypoints) == 0:
            return 100.0

        try:
            ego_loc = self.ego_vehicle.get_location()
            ego_pos = np.array([ego_loc.x, ego_loc.y, ego_loc.z])

            # Target the NEXT uncrossed waypoint, clamped to the route end
            wp_idx = min(self.current_waypoint_idx + 1, len(self.waypoints) - 1)
            wp = self.waypoints[wp_idx]
            wp_pos = np.array([wp.x, wp.y, wp.z])

            dist = np.linalg.norm(ego_pos - wp_pos)
            return float(dist)

        except Exception:
            return 100.0
    
    def build_cbf_state(self, current_speed: float) -> Dict:
        """Build state dict for CBF safety layer"""
        d_collision = self._compute_collision_distance()
        # Clamp collision distance to valid range [0, 100]
        d_collision = np.clip(float(d_collision), 0.0, 100.0)
        
        ttc = d_collision / max(current_speed, 0.1) if current_speed > 0.1 else 100.0
        lane_offset = self._compute_lane_offset()
        lane_offset = np.clip(float(lane_offset), -3.0, 3.0)  # Clamp to reasonable lane values
        
        speed_limit = self._get_speed_limit()
        speed_limit = np.clip(float(speed_limit), 5.0, 30.0)  # Clamp speed limit to [5, 30] m/s
        
        if current_speed > speed_limit * 1.1:
            self._speed_limit_violation_count += 1
        
        cbf_state = {
            'd_collision': d_collision,
            'ttc': float(ttc),
            'lane_offset': lane_offset,
            'speed': float(current_speed),
            'speed_limit': speed_limit  # Add dynamic speed limit for CBF
        }
        
        # DEBUG: Log state values on first few steps and whenever constraints are tight
        if self._episode_step < 5 or d_collision < 10.0 or abs(lane_offset) > 1.0:
            print(f"[CBF-STATE] Step {self._episode_step}: d_col={d_collision:.2f}m, v={current_speed:.2f}m/s, v_lim={speed_limit:.2f}m/s, lane_off={lane_offset:.2f}m")
        
        return cbf_state
    
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
        Soft advisory reward for staying centered while actually moving.
        """
        if self.ego_vehicle is None:
            return 0.0

        offset = abs(self._compute_lane_offset())
        velocity = self.ego_vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

        if speed < 1.0:
            if offset <= 1.75:
                return 0.0
            return float(np.interp(min(offset, 2.75), [1.75, 2.75], [-0.4, -2.0]))

        if offset <= 0.25:
            lane_reward = 0.8
        elif offset <= 0.9:
            lane_reward = float(np.interp(offset, [0.25, 0.9], [0.8, 0.0]))
        elif offset <= 1.75:
            lane_reward = float(np.interp(offset, [0.9, 1.75], [0.0, -1.4]))
        else:
            lane_reward = -2.0 - 2.5 * min(offset - 1.75, 1.0)

        steering_penalty = -0.5 * abs(self._prev_steering) * min(speed / 5.0, 1.0)
        return lane_reward + steering_penalty
    
    def compute_forward_motion_reward(self) -> float:
        """
        Primary dense reward: move forward in the lane with meaningful speed.

        FIX B: The original code returned -2.5 whenever vel_magnitude < 0.25 m/s.
        SAC's initial random policy is zero-centered, so throttle ≈ 0 for many
        early steps — the car sits still and immediately accumulates -2.5/step on
        top of the traffic-flow penalty.  Changing idle to 0.0 removes this
        dominant negative attractor so the agent can escape the "stand still"
        local optimum.  The backward-motion penalty is also softened from
        (-3.0 - 0.8·|v| - 0.6·lateral) to (-1.0 - 0.3·|v| - 0.2·lateral) so
        early random reversals don't collapse the value estimate.
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
        
        #Velocity magnitude 
        move=0.0
        # FIX B-1: idle → 0.0 instead of -2.5
        if vel_magnitude < 0.25:
            move=0.0

        if vel_magnitude > 0.1:
    # A small "existence" bonus for moving forward
            move+= 0.5 + (vel_magnitude * 0.2) 
        else:
            move= 0.0 # Neutral, not negative

        # Forward and lateral velocity components in the vehicle frame
        forward_speed = float(np.dot(vel_vector, heading))
        lateral_vector = vel_vector - forward_speed * heading
        lateral_speed = float(np.linalg.norm(lateral_vector))
        speed_limit = max(self._get_speed_limit(), 5.0)

        # FIX B-2: soften backward-motion penalty
        if forward_speed <= 0.0:
            return float(-1.0 - 0.3 * abs(forward_speed) - 0.2 * lateral_speed)

        speed_ratio = np.clip(forward_speed / speed_limit, 0.0, 1.2)
        movement_reward = 15.0 * speed_ratio
        alignment_bonus = 1.5 * np.clip(forward_speed / max(vel_magnitude, 1e-6), 0.0, 1.0)
        lateral_penalty = 0.9 * min(lateral_speed, 4.0)

        return float(movement_reward + alignment_bonus - lateral_penalty+move)
    
    def compute_safe_following_reward(self) -> float:
        """
        Soft headway shaping that only matters when following at speed.
        """
        if self.lidar_data is None or self.ego_vehicle is None:
            return 0.0
        
        velocity = self.ego_vehicle.get_velocity()
        ego_speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        if ego_speed < 1.0:
            return 0.0

        lidar_points = self.lidar_data

        forward_points = lidar_points[0, :] > 0
        lateral_points = np.abs(lidar_points[1, :]) < 2.0
        mask = forward_points & lateral_points

        if not np.any(mask):
            return 0.0

        forward_distances = lidar_points[0, mask]
        min_distance = float(np.min(forward_distances))

        ttc = min_distance / max(ego_speed, 0.1)

        if ttc < 0.7:
            return -8.0
        if ttc < 1.5:
            return float(np.interp(ttc, [0.7, 1.5], [-8.0, -0.5]))
        if ttc <= 3.0:
            return float(np.interp(ttc, [1.5, 3.0], [0.2, 0.7]))
        if ttc <= 4.5:
            return float(np.interp(ttc, [3.0, 4.5], [0.7, 0.0]))
        return 0.0
    
    def compute_traffic_flow_reward(self) -> float:
        """
        Strong speed-band shaping: move quickly when the road is clear.

        FIX C: The original low-speed zone returned -3.5 for speed_ratio ≤ 0.15,
        then added another -2.5 × (0.35 - ratio) clear-road nudge on top.
        Combined with the (now-removed) -2.5 idle penalty from forward_motion,
        the agent received ≈ -9 per step in the first seconds — making "stand
        still with brake" the locally optimal policy.  We reduce the floor to
        -0.5 and the clear-road nudge to -0.8 so the gradient still pushes
        toward speed without creating an unescapable negative trap.
        """
        if self.ego_vehicle is None:
            return 0.0

        velocity = self.ego_vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        speed_limit = self._get_speed_limit()

        acceleration = (speed - self._prev_speed) / 0.05
        jerk = abs(acceleration - self._prev_acceleration)
        speed_ratio = speed / max(speed_limit, 5.0)
        road_clear = self._compute_collision_distance() > max(12.0, speed * 1.5)

        # FIX C-1: floor raised from -3.5 → 5
        if speed_ratio <= 0.15:
            flow_reward = -0.5
        elif speed_ratio <= 0.7:
            flow_reward = float(np.interp(speed_ratio, [0.15, 0.7], [-0.5, 2.5]))
        elif speed_ratio <= 0.9:
            flow_reward = float(np.interp(speed_ratio, [0.7, 0.9], [2.5, 3.8]))
        elif speed_ratio <= 1.0:
            flow_reward = float(np.interp(speed_ratio, [0.9, 1.0], [3.8, 2.5]))
        elif speed_ratio <= 1.1:
            flow_reward = float(np.interp(speed_ratio, [1.0, 1.1], [2.5, 0.0]))
        else:
            flow_reward = 5.0 - 8.0 * min(speed_ratio - 1.1, 0.4)

        # FIX C-2: clear-road nudge reduced from -2.5 → 8
        if road_clear and speed_ratio < 0.35:
            flow_reward += 8 * (0.35 - speed_ratio) / 0.35
        elif not road_clear and speed_ratio < 0.35:
            flow_reward = max(flow_reward, -0.5)

        flow_reward -= 0.01 * min(jerk, 20.0)
        if road_clear and acceleration < -4.0:
            flow_reward -= min(0.8, 0.12 * abs(acceleration + 4.0))

        return float(flow_reward)
    
    def compute_yield_and_maneuver_reward(self) -> float:
        """
        Small etiquette term for not trapping faster traffic behind us.
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
        
        speed_limit = self._get_speed_limit()
        yield_reward = 0.0

        if np.any(mask):
            rear_distances = -lidar_points[0, mask]
            min_rear_distance = float(np.min(rear_distances))

            closing_speed = max(0.5, speed_limit * 0.25)
            ttc_behind = min_rear_distance / closing_speed

            lane_offset = self._compute_lane_offset()

            if ttc_behind < 2.5 and ego_speed > 1.5 and ego_speed < 0.55 * speed_limit:
                if abs(lane_offset) < 0.35:
                    yield_reward = -0.7
                else:
                    yield_reward = 0.2

        current_lane_offset = self._compute_lane_offset()

        if ego_speed > 3.0 and abs(current_lane_offset - self._last_lane_offset) > 0.3:
            self._lane_change_count += 1
            if self._lane_change_count > 3 and yield_reward == 0.0:
                yield_reward = -0.15

        self._last_lane_offset = current_lane_offset

        return yield_reward
    
    def compute_opposite_lane_penalty(self) -> float:
        """
        Penalize leaving the drivable corridor and pushing beyond lane bounds.
        """
        lane_offset = abs(self._compute_lane_offset())
        if lane_offset <= 1.75:
            return 0.0
        if lane_offset <= 2.5:
            return float(np.interp(lane_offset, [1.75, 2.5], [-0.8, -3.0]))
        return float(-20.0 - 4.0 * min(lane_offset - 2.5, 1.0))
    
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
        Near-zero when spacing is fine; increasingly negative as margins collapse.
        """
        if self.ego_vehicle is None:
            return 0.0

        velocity = self.ego_vehicle.get_velocity()
        ego_speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        d_collision = min(self._compute_collision_distance(), float(self.obstacle_distance))

        if d_collision >= 20.0:
            margin_reward = 0.0
        elif d_collision >= 12.0:
            margin_reward = float(np.interp(d_collision, [12.0, 20.0], [-0.15, 0.0]))
        elif d_collision >= 6.0:
            margin_reward = float(np.interp(d_collision, [16.0, 22.0], [-1.5, -0.15]))
        else:
            margin_reward = -4.0 - 0.8 * (6.0 - d_collision)

        if ego_speed < 1.0 and d_collision >= 10.0:
            margin_reward = 0.0

        collision_prevention_bonus = 0.0
        if hasattr(self, '_cbf_wrapper'):
            if hasattr(self._cbf_wrapper, 'cbf_layer'):
                cbf = self._cbf_wrapper.cbf_layer
                if hasattr(cbf, 'collision_prevented') and cbf.collision_prevented:
                    collision_prevention_bonus = 0.15

        return margin_reward + collision_prevention_bonus
    
    def compute_waypoint_progress_reward(self) -> float:
        """
        Reward forward route progress, waypoint milestones, and completion.
        """
        if len(self.waypoints) == 0:
            return 0.0
        
        dist_to_next = self._get_distance_to_next_waypoint()

        if self._prev_distance_to_next_wp is None:
            self._prev_distance_to_next_wp = dist_to_next

        distance_delta = self._prev_distance_to_next_wp - dist_to_next
        dense_progress_reward = 3.5 * np.clip(distance_delta, -2.0, 2.0)

        milestone_reward = 0.0
        if not hasattr(self, '_prev_waypoints_crossed'):
            self._prev_waypoints_crossed = 0
        if self.waypoints_crossed > self._prev_waypoints_crossed:
            milestone_reward = 12.0

        route_completion_reward = 4.0 * self.waypoint_completion_ratio

        self._prev_waypoints_crossed = self.waypoints_crossed
        self._prev_distance_to_next_wp = dist_to_next

        return float(dense_progress_reward + milestone_reward + route_completion_reward)
    
    def compute_obstacle_avoidance_reward(self) -> float:
        """
        Local proximity penalty around obstacles; no free reward in open space.
        """
        if not self.obstacle_sensor or self.obstacle_distance >= 100.0:
            return 0.0

        if self.obstacle_distance > 15.0:
            return 0.0
        if self.obstacle_distance > 8.0:
            return float(5*(np.interp(self.obstacle_distance, [8.0, 15.0], [-0.8, 0.0])))
        if self.obstacle_distance > 4.0:
            return float(7*(np.interp(self.obstacle_distance, [4.0, 8.0], [-2.5, -0.8])))
        return float(-5.0 - 0.5 * (4.0 - self.obstacle_distance))
    
    def _compute_reward(self) -> float:
        """Robust reward shaping for progress, completion, containment, and traffic safety."""
        if self.collision_occurred:
            self._last_reward_components = {"collision": -600.0}
            self._last_total_reward = -600.0
            return -600.0

        if self.ego_vehicle is None:
            self._last_reward_components = {}
            self._last_total_reward = 0.0
            return 0.0

        velocity = self.ego_vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        speed_limit = max(self._get_speed_limit(), 5.0)
        clear_road = self._compute_collision_distance() > max(12.0, speed * 1.5)

        # FIX E: Delay stalling detection to step 60 (3 seconds at 20 Hz).
        # The original threshold of step 5 meant the agent was penalised for
        # being slow before SAC's random exploration had any chance to discover
        # a positive reward signal.  60 steps is enough time for a first
        # gradient update to bias the policy toward throttle.
        if speed < 1.0 and clear_road and self._episode_step > 60:
            self._stalled_steps += 1
        else:
            self._stalled_steps = 0

        components = {
            # FIX D: alive baseline raised from 0.0 → +0.5 so surviving each
            # step always provides a small positive signal.  This prevents
            # "terminate immediately" from ever being the optimal strategy and
            # gives the critic a stable, non-zero baseline to anchor around.
            "alive": 0.5,
            "movement": 8 * self.compute_forward_motion_reward(),
            "lane_centering": 3.5 * self.compute_lane_centering_reward(),
            "waypoint_progress": 4* self.compute_waypoint_progress_reward(),
            "traffic_flow": self.compute_traffic_flow_reward(),
            "safe_following": 4.5 * self.compute_safe_following_reward(),
            "yielding": 0.4 * self.compute_yield_and_maneuver_reward(),
            "safety_buffer": 2.5 * self.compute_safety_buffer_reward(),
            "containment": 2.5 * self.compute_opposite_lane_penalty(),
            "obstacle_avoidance": 3 * self.compute_obstacle_avoidance_reward(),
        }

        if self.lane_invaded:
            components["lane_invasion_event"] = -15.0

        if self.obstacle_detected and self.obstacle_distance < 8.0:
            components["close_obstacle_event"] = -6.0

        if speed > 1.08 * speed_limit:
            components["speeding"] = -8.0 * (speed / speed_limit - 1.08)

        # FIX E (continued): Stalling penalty capped at -3.0 (was -12.0) with a
        # slow growth rate of 0.05/step (was 0.8/step) and the 60-step grace
        # window above.  This keeps the incentive to move without making
        # extended early idling catastrophically loss-generating.
        if self._stalled_steps > 0:
            components["stalling"] = -min(3.0, 0.3 + 0.05 * self._stalled_steps)

        if self.endpoint_reached:
            time_efficiency = np.clip(
                1.0 - (self._episode_step / max(float(self.time_limit), 1.0)),
                0.0,
                1.0
            )
            components["episode_completion"] = 240.0 + 120.0 * time_efficiency

        total_reward = float(sum(components.values()))
        self._last_reward_components = components
        self._last_total_reward = total_reward
        self._log_reward_breakdown(components)

        # FIX [6]: lane_invaded and obstacle_detected flags are reset in step()
        # after _compute_reward() returns, not here. Resetting inside a compute
        # function is a side-effect that silently drops events if the function
        # is ever called more than once per step (e.g. during debugging or via
        # wrapper chains). The flags are still READ correctly above.

        return total_reward
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one environment step"""
        # FIX [3]: Parse 3D action [steer, throttle, brake] with correct bounds.
        # Each pedal is clipped to its declared range — no conditional logic,
        # no sign-based branching. The old if/else chain double-braked at SAC
        # initialisation because both action[1] and action[2] near 0 could
        # simultaneously contribute brake, stalling the car immediately.
        steer    = float(np.clip(action[0], -1.0, 1.0))
        throttle = float(np.clip(action[1],  0.0, 1.0))
        brake    = float(np.clip(action[2],  0.0, 1.0))
        

        # Clamp to [0, 1] for CARLA (can't apply both throttle and brake)
        throttle = max(0.0, throttle)  # Throttle is [0, 1]
        brake = max(0.0, brake)        # Brake is [0, 1]
        
        # Normalize if both are non-zero (shouldn't happen, but safety)
#        total = throttle + brake
#       if total > 1.0:
#          throttle /= total
#         brake /= total


        # FORCE MUTUAL EXCLUSION
        if throttle > brake:
            brake = 0.0
            # Add a small deadzone to help SAC escape the 0.5/0.5 initialization
            if throttle < 0.1: throttle = 0.0 
        else:
            throttle = 0.0

        # Per-step sensor events should reflect the upcoming world tick only.
        self.obstacle_detected = False
        self.obstacle_distance = 100.0
        self.obstacle_actor = None
        
        # Convert to CARLA control
        control = carla.VehicleControl()
        control.steer = steer
        control.throttle = throttle
        control.brake = brake

        
        
        # DEBUG: Log action on first 10 steps
        if self._episode_step < 10:
            print(f"[ACTION] Step {self._episode_step}: raw=[{action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}] -> ctrl=[steer={steer:.2f}, throttle={throttle:.2f}, brake={brake:.2f}]")
        
        self.ego_vehicle.apply_control(control)
        
        # Tick world
        self.world.tick()
        self._episode_step += 1
        
        # Update progress-related state before reward computation
        obs = self._get_observation()
        self._update_waypoint_progress()
        
        # Calculate endpoint distance and detect endpoint reached (Phase 5)
        if self.ego_vehicle and self.waypoints and not self.endpoint_reached:
            try:
                ego_loc = self.ego_vehicle.get_location()
                ego_pos = np.array([ego_loc.x, ego_loc.y, ego_loc.z])
                endpoint = self.waypoints[-1]
                endpoint_pos = np.array([endpoint.x, endpoint.y, endpoint.z])
                self.endpoint_distance = float(np.linalg.norm(ego_pos - endpoint_pos))
                
                # Check if reached endpoint (< 5m threshold)
                if self.endpoint_distance < 5.0 and not self.endpoint_reached:
                    self.endpoint_reached = True
                    print(f"\n{'='*60}")
                    print(f"[ENDPOINT REACHED] Episode completed at step {self._episode_step}")
                    print(f"  Endpoint distance: {self.endpoint_distance:.2f}m")
                    print(f"  Endpoint reward bonus: reward shaping bonus applied")
                    print(f"{'='*60}\n")
            except Exception as e:
                pass
        
        # Update waypoint completion ratio (Phase 5)
        self.waypoint_completion_ratio = self.waypoints_crossed / max(self.total_waypoints, 1)

        # Reward now reflects the updated route and endpoint state
        lane_invaded_event = self.lane_invaded
        obstacle_detected_event = self.obstacle_detected
        collision_event = self.collision_occurred
        reward = self._compute_reward()

        # FIX [6]: Reset sensor flags HERE after reward has read them.
        # obstacle_detected is already cleared at the top of step() each tick.
        # lane_invaded must be explicitly cleared here so the next tick starts clean.
        self.lane_invaded = False
        
        # FIX [11]: Only update spectator camera when rendering is requested.
        # The original unconditional call added overhead every step even when
        # render_mode=None (i.e. during all headless training runs).
        #if self.render_mode == "human": we need to render the spectator to view ego vehicle from above
        self.render()
        
        # Display RGB sensor data if enabled
        if self.show_sensor_data and self.rgb_data is not None:
            rgb_display = cv2.cvtColor(self.rgb_data, cv2.COLOR_BGR2RGB)
            cv2.imshow('CARLA RGB Camera', rgb_display)
            cv2.waitKey(1)
        
        # ===== IMMEDIATE TERMINATION ON COLLISION (BUG FIX) =====
        # CRITICAL: Check collision BEFORE returning and LOG it
        terminated = False
        terminated_reason = None
        if self.collision_occurred:
            print(f"\n{'='*60}")
            print(f"[COLLISION DETECTED] Episode ended at step {self._episode_step}")
            print(f"  Collision distance: {self._collision_distance:.2f}m")
            print(f"  Reward penalty applied: {reward:.2f}")
            print(f"  Total reward this step: {reward:.2f}")
            print(f"{'='*60}\n")
            terminated = True
            terminated_reason = "collision"
        
        # Endpoint termination (NEW - Phase 5)
        elif self.endpoint_reached:
            terminated = True
            terminated_reason = "endpoint"
        
        # Timeout termination (secondary)
        elif self._episode_step >= self.time_limit:
            terminated = True
            terminated_reason = "timeout"
        
        truncated = False

        
        # Info dict (Phase 2 - add waypoint tracking)
        info = {
            'collision_distance': 0.0 if collision_event else self._compute_collision_distance(),
            'collision_event': 1.0 if collision_event else 0.0,
            'collision_count_episode': int(self._collision_count),
            'lane_invaded': lane_invaded_event,
            'lane_invasion_event': 1.0 if lane_invaded_event else 0.0,
            'lane_invasion_count_episode': int(self._lane_invasion_count),
            'obstacle_detected': obstacle_detected_event,
            'episode_step': self._episode_step,
            'waypoints_crossed': self.waypoints_crossed,
            'waypoints_remaining': max(0, self.total_waypoints - self.waypoints_crossed),
            'distance_to_next_wp': self._get_distance_to_next_waypoint(),
            'progress_pct': 100.0 * self.waypoints_crossed / max(1, self.total_waypoints),
            'reward_total': self._last_total_reward,
            'reward_components': dict(self._last_reward_components),
            'ensemble_uncertainty': self._ensemble_uncertainty,
            'trust_score': self._trust_score,
            'cbf_corrections_episode': int(self._cbf_correction_count_episode),
            'cbf_correction_magnitude_avg': float(self._avg_correction_mag),
            'terminated_reason': terminated_reason,
        }

        current_speed = float(obs['speed'][0]) if obs is not None else 0.0
        current_acceleration = (current_speed - self._prev_speed) / 0.05
        self._prev_speed = current_speed
        self._prev_acceleration = current_acceleration
        self._prev_steering = steer

        if collision_event:
            # Keep the event visible in this returned step info; reset() will clear it for the next episode.
            self.collision_occurred = False
        
        
        if current_speed < 1.0 and self._episode_step > 120:
            terminated = True
            terminated_reason = "stuck"
            reward -= 50


        return obs, reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset environment"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Reset counters
        self._episode_step = 0
        self._collision_distance = 100.0
        self._lane_invasion_count = 0
        self._speed_limit_violation_count = 0
        self._collision_count = 0
        self._cbf_correction_count = 0
        self._cbf_correction_count_episode = 0  # Reset per-episode counter (Phase 5)
        self._avg_correction_mag = 0.0
        self.collision_occurred = False
        self.lane_invaded = False
        self.obstacle_detected = False  # Reset obstacle detection (Phase 5.3)
        self._obstacle_detection_count = 0  # Reset obstacle detection count (Phase 5.3)
        self.obstacle_distance = 100.0
        self.obstacle_actor = None
        self._ensemble_uncertainty = 0.0
        self._trust_score = 1.0
        
        # Reset advanced reward shaping state (NEW)
        self._prev_steering = 0.0
        self._prev_speed = 0.0
        self._prev_acceleration = 0.0
        self._lane_change_count = 0
        self._last_lane_offset = 0.0
        self._prev_distance_to_next_wp = None
        self._stalled_steps = 0
        self._last_reward_components = {}
        self._last_total_reward = 0.0
        
        # Reset waypoint tracking (Phase 2)
        self.waypoints_crossed = 0
        self.current_waypoint_idx = 0
        self._prev_waypoints_crossed = 0
        
        # Reset endpoint tracking (Phase 5)
        self.endpoint_reached = False
        self.endpoint_location = self.waypoints[-1] if self.waypoints else None
        self.waypoint_completion_ratio = 0.0
        
        # Reset ego vehicle
        spawn_point = None
        if self.ego_vehicle:
            spawn_point = self._select_safe_spawn_point(min_clearance=10.0)
            if spawn_point is None:
                raise RuntimeError("No spawn points available during reset")

            try:
                self.ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
                self.ego_vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
                self.ego_vehicle.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
                self.ego_vehicle.set_transform(spawn_point)
            except Exception as e:
                print(f"[RESET] Failed to reposition ego vehicle cleanly: {e}")
                raise
        
        # Generate new waypoint route (Phase 2)
        # FIX [4]: world.tick() is called first (inside the settle loop below)
        # so that set_transform() is committed before get_location() is called
        # inside _generate_waypoints(). Without this, waypoints point from the
        # *previous* episode's spawn position, not the new one.

        # Let physics and sensor streams settle after teleporting the ego vehicle.
        # FIX [4]: _generate_waypoints() is called AFTER the first tick so the
        # new spawn transform is already applied when we query get_location().
        for _ in range(2):
            self.world.tick()
        self._generate_waypoints()
        obs = self._get_observation()
        self.endpoint_location = self.waypoints[-1] if self.waypoints else None

        info = {
            "reset_spawn": (
                spawn_point.location.x,
                spawn_point.location.y,
                spawn_point.location.z,
            ) if self.ego_vehicle else None
        }
        
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
        """Cleanup CARLA actors and sensors"""
        print("Destroying CARLA actors and sensors...")
        
        # Stop and destroy sensors
        sensors_to_destroy = [
            self.rgb_camera,
            self.lidar_sensor,
            self.collision_sensor,
            self.lane_invasion_sensor,
            self.depth_sensor,
            self.obstacle_sensor  # Obstacle Detector sensor (Phase 5.3)
        ]
        
        for sensor in sensors_to_destroy:
            try:
                if sensor:
                    sensor.stop()
                    sensor.destroy()
            except:
                pass
        
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
        
        print("[OK] All actors and sensors destroyed")
    
    @property
    def safety_metrics(self) -> Dict:
        """Expose safety metrics for callback logging"""
        # Calculate normalized waypoint completion ratio (0-1)
        waypoint_ratio = self.waypoints_crossed / max(self.total_waypoints, 1)
        
        return {
            'lane_invasions': self._lane_invasion_count,
            'speed_violations': self._speed_limit_violation_count,
            'collisions': self._collision_count,
            'cbf_corrections_episode': self._cbf_correction_count_episode,  # Per-episode (Phase 5)
            'cbf_correction_magnitude': self._avg_correction_mag,
            'waypoint_completion': waypoint_ratio,  # Normalized 0-1 (Phase 5)
            'waypoints_crossed': self.waypoints_crossed,
            'total_waypoints': self.total_waypoints,
            'endpoint_distance': self.endpoint_distance,
            'endpoint_reached': 1.0 if self.endpoint_reached else 0.0,  # Phase 5
            'ensemble_uncertainty': self._ensemble_uncertainty,
            'trust_score': self._trust_score,
            'reward_total': self._last_total_reward,
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
        
        self.cbf_layer = CBFSafetyLayer(alpha=alpha, d_min=1.2, y_max=1.9, v_max=25.0)
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
        
        # Suppress verbose CBF logging to avoid cluttering terminal
        verbose_cbf = False  # Always suppress for cleaner training logs
        
        # Apply CBF correction with trust score modulation
        try:
            safe_action = self.cbf_layer.compute_safe_action(
                action, 
                cbf_state,
                trust_score=trust_score if self.use_trust_score else 1.0,
                verbose=verbose_cbf  # Always False to suppress output
            )
            
            # Track correction metrics
            correction_mag = float(np.linalg.norm(safe_action - action))
            self.last_correction_mag = correction_mag
            
            if correction_mag > 0.01:  # Only count non-negligible corrections
                self.correction_count += 1
                self.episode_corrections += 1
                self.correction_magnitudes.append(correction_mag)
                self.episode_correction_mag = np.mean(self.correction_magnitudes[-100:])  # Rolling avg
                # DEBUG: Log significant corrections
                if self.episode_corrections <= 5:
                    print(f"[CBF-CORRECTION] Step {self.correction_count}: mag={correction_mag:.3f}, action=[{action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}] -> safe=[{safe_action[0]:.2f}, {safe_action[1]:.2f}, {safe_action[2]:.2f}]")
            
            # Update environment's safety metrics
            if hasattr(self.env.unwrapped, '_cbf_correction_count'):
                self.env.unwrapped._cbf_correction_count = self.correction_count
            if hasattr(self.env.unwrapped, '_cbf_correction_count_episode'):
                self.env.unwrapped._cbf_correction_count_episode = self.episode_corrections  # Per-episode (Phase 5)
            if hasattr(self.env.unwrapped, '_avg_correction_mag'):
                self.env.unwrapped._avg_correction_mag = self.episode_correction_mag
        
        except Exception as e:
            print(f"[CBF] Correction failed: {e}")
            safe_action = action.copy()
            self.last_correction_mag = 0.0
        
        # FIX [3]: Clip each control to its physical range independently.
        # steer ∈ [-1,1], throttle ∈ [0,1], brake ∈ [0,1]
        safe_action[0] = np.clip(safe_action[0], -1.0, 1.0)
        safe_action[1] = np.clip(safe_action[1],  0.0, 1.0)
        safe_action[2] = np.clip(safe_action[2],  0.0, 1.0)
        return safe_action
    
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

        info["cbf_correction_event"] = 1.0 if self.last_correction_mag > 0.01 else 0.0
        info["cbf_correction_magnitude"] = float(self.last_correction_mag)
        info["cbf_invocations_episode"] = int(self.episode_corrections)
        info["cbf_collision_prevented"] = 1.0 if getattr(self.cbf_layer, "collision_prevented", False) else 0.0
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        """Reset episode metrics"""
        self.episode_corrections = 0
        self.episode_correction_mag = 0.0
        self.last_correction_mag = 0.0
        self.cbf_layer.reset_metrics()
        return self.env.reset(**kwargs)


# ============================================================================
# Env Unwrapping Utilities
# ============================================================================
# FIX BUG 2 + BUG 4: SB3 auto-wraps in DummyVecEnv, so model.get_env().unwrapped
# returns the DummyVecEnv shell, NOT CarlaGymEnv. We must go through .envs[0].
# The original traversal loop also never matched CBFSafetyLayerWrapper because it
# checked for _cbf_wrapper on each layer, but that attr lives on CarlaGymEnv (the
# bottom), not on the wrapper layers above it. Fixed with isinstance() checks.

def _unwrap_vec_env(model):
    """
    Return the raw gym.Env (index-0 worker) from an SB3 VecEnv.

    SB3 always wraps user envs in DummyVecEnv / SubprocVecEnv.
    DummyVecEnv exposes its workers via .envs[].
    """
    vec_env = model.get_env()
    if vec_env is None:
        return None
    # DummyVecEnv (most common during single-env training)
    if hasattr(vec_env, "envs"):
        return vec_env.envs[0]
    return vec_env


def _find_carla_and_cbf(model):
    """
    Walk the full wrapper chain and return (CarlaGymEnv, CBFSafetyLayerWrapper).
    Either value is None if not found.

    Wrapper stack built in create_carla_env():
        DummyVecEnv
          └─ CBFSafetyLayerWrapper          ← ActionWrapper
               └─ PipelineObservationWrapper ← ObservationWrapper
                    └─ CarlaGymEnv            ← base env
    """
    base = _unwrap_vec_env(model)
    if base is None:
        return None, None

    carla_env = None
    cbf_wrapper = None
    current = base

    while current is not None:
        # Match CBFSafetyLayerWrapper by isinstance, with duck-type fallback
        if isinstance(current, CBFSafetyLayerWrapper):
            cbf_wrapper = current
        elif hasattr(current, "cbf_layer") and hasattr(current, "correction_count"):
            cbf_wrapper = current

        # Match CarlaGymEnv by isinstance, with duck-type fallback
        if isinstance(current, CarlaGymEnv):
            carla_env = current
        elif hasattr(current, "_collision_count") and hasattr(current, "ego_vehicle"):
            carla_env = current

        current = getattr(current, "env", None)

    return carla_env, cbf_wrapper


def _get_policy_metric_observation(callback) -> Optional[np.ndarray]:
    """
    Return the freshest observation available for live policy metric evaluation.

    For off-policy algorithms this is usually `new_obs` from the callback locals.
    """
    observation = callback.locals.get("new_obs")
    if observation is None:
        observation = getattr(callback.model, "_last_obs", None)
    return observation


def _get_step_info(callback, env_idx: int = 0) -> Dict:
    """Return the info dict for the current VecEnv step."""
    infos = callback.locals.get("infos")
    if infos is None or len(infos) <= env_idx:
        return {}
    info = infos[env_idx]
    return info if isinstance(info, dict) else {}


def _compute_live_policy_metrics(model, observation) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute ensemble uncertainty and trust directly from the current policy.

    This uses the critic ensemble on the actor's deterministic action, which keeps
    the logged trust signal aligned with the live policy used by the safety layer.
    """
    if observation is None:
        return None, None

    try:
        obs_tensor, _ = model.policy.obs_to_tensor(observation)
        with torch.no_grad():
            policy_action = model.actor(obs_tensor, deterministic=True)
            _, uncertainty, trust_score = model.critic.predict(obs_tensor, policy_action)

        return (
            float(uncertainty.mean().detach().cpu().item()),
            float(trust_score.mean().detach().cpu().item()),
        )
    except Exception:
        return None, None


# ============================================================================
# Custom TensorBoard Callback
# ============================================================================

class SafetyMetricsCallback(BaseCallback):
    """
    Log safety and progress metrics to TensorBoard with SB3-style episode summaries.

    Fixes applied vs original:
    - Uses _find_carla_and_cbf() for reliable env unwrapping  (BUG 2 + 4)
    - Uses self.locals['dones'] for episode detection          (BUG 3)
    - Calls self.logger.dump(self.num_timesteps) to flush      (BUG 1)
    - Tracks episode reward via self.locals['rewards']
    """

    def __init__(self, verbose: int = 0, log_frequency: int = 100):
        super().__init__(verbose)
        self.log_frequency = log_frequency

        # Episode accumulators
        self.episode_count = 0
        self.episode_reward = 0.0
        self.episode_length = 0
        self.episode_waypoints = 0
        self.episode_cbf_corrections = 0
        self.episode_collisions = 0

    def _on_step(self) -> bool:

        # FIX BUG 3: use self.locals['rewards'], not buf_rewards
        rewards = self.locals.get("rewards", [0.0])
        self.episode_reward += float(rewards[0]) if len(rewards) > 0 else 0.0
        self.episode_length += 1
        step_info = _get_step_info(self)

        # FIX BUG 2 + 4: resolve env references correctly
        carla_env, cbf_wrapper = _find_carla_and_cbf(self.model)

        # Sync episode-level accumulators from step info first, falling back to live env state.
        self.episode_collisions = max(
            self.episode_collisions,
            int(step_info.get("collision_count_episode", getattr(carla_env, "_collision_count", 0) if carla_env is not None else 0)),
        )
        self.episode_waypoints = max(
            self.episode_waypoints,
            int(step_info.get("waypoints_crossed", getattr(carla_env, "waypoints_crossed", 0) if carla_env is not None else 0)),
        )
        self.episode_cbf_corrections = max(
            self.episode_cbf_corrections,
            int(step_info.get("cbf_invocations_episode", getattr(cbf_wrapper, "episode_corrections", 0) if cbf_wrapper is not None else 0)),
        )

        # Periodic per-step metric logging
        should_log = (
            self.num_timesteps % self.log_frequency == 0
            or float(step_info.get("collision_event", 0.0)) > 0.0
            or float(step_info.get("lane_invasion_event", 0.0)) > 0.0
        )
        if should_log:
            self._record_step_metrics(carla_env, cbf_wrapper, step_info)
            # FIX BUG 1: flush buffered records to TensorBoard
            self.logger.dump(self.num_timesteps)

        # FIX BUG 3: episode-end detection via locals['dones']
        dones = self.locals.get("dones", [False])
        if any(dones):
            self._record_episode_summary()
            self.episode_count += 1
            self.episode_reward = 0.0
            self.episode_length = 0
            self.episode_waypoints = 0
            self.episode_cbf_corrections = 0
            self.episode_collisions = 0

        return True

    def _record_step_metrics(self, carla_env, cbf_wrapper, step_info: Optional[Dict] = None):
        """Write all per-step metrics into the logger buffer."""
        if step_info is None:
            step_info = {}

        # CBF metrics
        if cbf_wrapper is not None:
            cbf = getattr(cbf_wrapper, "cbf_layer", None)
            self.logger.record(
                "safety/cbf_correction_mag",
                float(getattr(cbf_wrapper, "last_correction_mag", 0.0)),
            )
            self.logger.record(
                "safety/total_cbf_corrections",
                float(getattr(cbf_wrapper, "correction_count", 0)),
            )
            self.logger.record(
                "safety/cbf_corrections_episode",
                float(getattr(cbf_wrapper, "episode_corrections", 0)),
            )
            if cbf is not None:
                self.logger.record(
                    "safety/collision_prevented",
                    float(getattr(cbf, "collision_prevented", False)),
                )
                self.logger.record(
                    "safety/avoidance_efficiency",
                    float(getattr(cbf, "avoidance_efficiency", 0.0)),
                )
                violations = getattr(cbf, "constraint_violations", {})
                self.logger.record("safety/collision_violations", int(violations.get("collision", 0)))
                self.logger.record("safety/lane_violations",      int(violations.get("lane", 0)))
                self.logger.record("safety/speed_violations",     int(violations.get("speed", 0)))

        # Carla-env metrics
        if carla_env is not None or step_info:
            collision_distance = step_info.get(
                "collision_distance",
                float(carla_env._compute_collision_distance()) if carla_env is not None else 100.0,
            )
            collision_count = step_info.get(
                "collision_count_episode",
                int(getattr(carla_env, "_collision_count", 0)) if carla_env is not None else 0,
            )
            self.logger.record("safety/collision_distance", float(collision_distance))
            self.logger.record("safety/collision_event", float(step_info.get("collision_event", 0.0)))
            self.logger.record("safety/collisions_episode", int(collision_count))
        if carla_env is not None:
            self.logger.record(
                "safety/lane_offset",
                float(carla_env._compute_lane_offset()),
            )

            # Waypoint / navigation
            wp_crossed = getattr(carla_env, "waypoints_crossed", 0)
            wp_total   = getattr(carla_env, "total_waypoints", 1)
            self.logger.record("progress/waypoints_crossed",    int(wp_crossed))
            self.logger.record("progress/waypoints_remaining",  max(0, wp_total - wp_crossed))
            self.logger.record("progress/progress_pct",         100.0 * wp_crossed / max(1, wp_total))
            self.logger.record("progress/distance_to_next_wp",  float(carla_env._get_distance_to_next_waypoint()))
            self.logger.record(
                "navigation/waypoint_completion_ratio",
                float(getattr(carla_env, "waypoint_completion_ratio", 0.0)),
            )
            self.logger.record(
                "navigation/endpoint_distance",
                float(getattr(carla_env, "endpoint_distance", 9999.0)),
            )
            self.logger.record(
                "navigation/endpoint_reached",
                1.0 if getattr(carla_env, "endpoint_reached", False) else 0.0,
            )

    def _record_episode_summary(self):
        """Log end-of-episode summary and print SB3-style console line."""
        ep_len = max(self.episode_length, 1)
        self.logger.record("episode/return",           self.episode_reward)
        self.logger.record("episode/length",           ep_len)
        self.logger.record("episode/waypoints_crossed",self.episode_waypoints)
        self.logger.record("episode/cbf_corrections",  self.episode_cbf_corrections)
        self.logger.record("episode/collisions",       self.episode_collisions)
        # FIX BUG 1: flush episode summary immediately
        self.logger.dump(self.num_timesteps)

        print(
            f"Episode {self.episode_count + 1:5d} | "
            f"Return: {self.episode_reward:9.2f} | "
            f"Length: {ep_len:4d} | "
            f"Waypoints: {self.episode_waypoints:2d} | "
            f"CBF: {self.episode_cbf_corrections:3d} | "
            f"Collisions: {self.episode_collisions:1d}"
        )


# ============================================================================
# Live Policy Trust Callback
# ============================================================================

class PolicyTrustScoreCallback(BaseCallback):
    """
    Periodically push live ensemble trust metrics from the actor/critic into the CBF wrapper.
    """

    def __init__(self, update_freq: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.update_freq = max(1, int(update_freq))
        self.current_trust = 1.0
        self.current_uncertainty = 0.0

    def _on_step(self) -> bool:
        if self.num_timesteps % self.update_freq != 0:
            return True

        observation = _get_policy_metric_observation(self)
        uncertainty, trust_score = _compute_live_policy_metrics(self.model, observation)
        if uncertainty is None or trust_score is None:
            return True

        self.current_uncertainty = uncertainty
        self.current_trust = trust_score

        carla_env, cbf_wrapper = _find_carla_and_cbf(self.model)
        if cbf_wrapper is not None:
            cbf_wrapper.set_trust_score(trust_score)
            cbf_wrapper.ensemble_uncertainty = uncertainty

        if carla_env is not None:
            carla_env._ensemble_uncertainty = uncertainty
            carla_env._trust_score = trust_score

        self.logger.record("cbf/trust_score", self.current_trust)
        self.logger.record("cbf/ensemble_uncertainty", self.current_uncertainty)

        return True


# ============================================================================
# Comprehensive Metrics Logging Callback
# ============================================================================

class ComprehensiveMetricsLoggingCallback(BaseCallback):
    """
    Logs all 6 requested metrics without touching any other functionality:

      1. CBF logging         — when a CBF correction is applied
      2. CBF invoke rate     — CBF invocations / episode length
      3. Lane invasion rate  — invasions / episode length
      4. Car collision rate  — collisions / episode length
      5. Ensemble uncertainty score — critic ensemble variance
      6. Trust score         — critic-derived trust of the live policy action

    Fixes applied vs original:
    - _find_carla_and_cbf() for correct env traversal  (BUG 2 + 4)
    - self.locals['dones'] for episode detection        (BUG 3)
    - self.logger.dump(self.num_timesteps) after every
      record block so values actually reach TensorBoard (BUG 1)
    """

    def __init__(self, verbose: int = 0, log_frequency: int = 1):
        super().__init__(verbose)
        self.log_frequency = max(1, int(log_frequency))

        # Per-episode accumulators
        self.episode_number       = 0
        self.episode_length       = 0
        self.episode_cbf_invocations  = 0
        self.episode_lane_invasions   = 0
        self.episode_collisions       = 0

        # "Last seen" counters — detect increments each step
        self._prev_cbf_count          = 0
        self._prev_collision_count    = 0
        self._prev_lane_invasion_count = 0

        # Derived scalars updated each step
        self.current_trust_score = 1.0
        self.current_ensemble_uncertainty = 0.0

    def _on_step(self) -> bool:
        self.episode_length += 1
        step_info = _get_step_info(self)

        # FIX BUG 2 + 4: resolve envs via helper
        carla_env, cbf_wrapper = _find_carla_and_cbf(self.model)

        # 1 & 2. Ensemble uncertainty + trust score
        self._update_trust_score(carla_env=carla_env, cbf_wrapper=cbf_wrapper)

        # 3. CBF metrics
        current_cbf = int(
            step_info.get(
                "cbf_invocations_episode",
                getattr(cbf_wrapper, "episode_corrections", 0) if cbf_wrapper is not None else 0,
            )
        )
        delta_cbf = max(0, current_cbf - self._prev_cbf_count)
        cbf_event = float(step_info.get("cbf_correction_event", 1.0 if delta_cbf > 0 else 0.0))

        if cbf_event > 0.0 or delta_cbf > 0:
            self.episode_cbf_invocations = max(self.episode_cbf_invocations, current_cbf)
            if self.episode_cbf_invocations == 0:
                self.episode_cbf_invocations = self._prev_cbf_count + 1
            self.logger.record("cbf/invoke_event", 1.0)
            self.logger.record(
                "cbf/correction_magnitude",
                float(step_info.get("cbf_correction_magnitude", getattr(cbf_wrapper, "last_correction_mag", 0.0) if cbf_wrapper is not None else 0.0)),
            )
            if self.verbose > 0:
                print(
                    f"[CBF] Step {self.num_timesteps}: correction "
                    f"mag={float(step_info.get('cbf_correction_magnitude', getattr(cbf_wrapper, 'last_correction_mag', 0.0) if cbf_wrapper is not None else 0.0)):.4f}"
                )
        else:
            self.logger.record("cbf/invoke_event", 0.0)

        self.logger.record(
            "cbf/total_episode_invocations",
            float(self.episode_cbf_invocations),
        )
        self._prev_cbf_count = current_cbf

        # 4. Collision metrics
        current_col = int(
            step_info.get(
                "collision_count_episode",
                getattr(carla_env, "_collision_count", 0) if carla_env is not None else 0,
            )
        )
        delta_col = max(0, current_col - self._prev_collision_count)
        collision_event = float(step_info.get("collision_event", 1.0 if delta_col > 0 else 0.0))
        if collision_event > 0.0 or delta_col > 0:
            self.episode_collisions = max(self.episode_collisions, current_col)
            if self.episode_collisions == 0:
                self.episode_collisions = self._prev_collision_count + 1
            self.logger.record("collisions/collision_event", 1.0)
        else:
            self.logger.record("collisions/collision_event", 0.0)
        self.logger.record(
            "collisions/collision_count_episode",
            float(self.episode_collisions),
        )
        self._prev_collision_count = current_col

        # 5. Lane invasion metrics
        current_li = int(
            step_info.get(
                "lane_invasion_count_episode",
                getattr(carla_env, "_lane_invasion_count", 0) if carla_env is not None else 0,
            )
        )
        delta_li = max(0, current_li - self._prev_lane_invasion_count)
        lane_event = float(step_info.get("lane_invasion_event", 1.0 if delta_li > 0 else 0.0))
        if lane_event > 0.0 or delta_li > 0:
            self.episode_lane_invasions = max(self.episode_lane_invasions, current_li)
            if self.episode_lane_invasions == 0:
                self.episode_lane_invasions = self._prev_lane_invasion_count + 1
            self.logger.record("lane/invasion_event", 1.0)
        else:
            self.logger.record("lane/invasion_event", 0.0)
        self.logger.record(
            "lane/invasion_count_episode",
            float(self.episode_lane_invasions),
        )
        self._prev_lane_invasion_count = current_li

        # 6. Per-step rates
        ep_len = max(self.episode_length, 1)
        if self.num_timesteps % self.log_frequency == 0:
            self.logger.record("rates/cbf_invoke_rate",    self.episode_cbf_invocations / ep_len)
            self.logger.record("rates/collision_rate",     self.episode_collisions       / ep_len)
            self.logger.record("rates/lane_invasion_rate", self.episode_lane_invasions   / ep_len)
            self.logger.record("metrics/ensemble_uncertainty_score", self.current_ensemble_uncertainty)
            self.logger.record("metrics/trust_score",                self.current_trust_score)
            self.logger.dump(self.num_timesteps)

        # FIX BUG 3: episode-end detection via locals['dones']
        dones = self.locals.get("dones", [False])
        if any(dones):
            self._on_episode_end()

        return True

    def _update_trust_score(self, carla_env=None, cbf_wrapper=None):
        """Read uncertainty and trust from the live actor/critic ensemble."""
        observation = _get_policy_metric_observation(self)
        uncertainty, trust_score = _compute_live_policy_metrics(self.model, observation)

        if uncertainty is None or trust_score is None:
            return

        self.current_ensemble_uncertainty = uncertainty
        self.current_trust_score = trust_score

        if cbf_wrapper is not None:
            cbf_wrapper.set_trust_score(trust_score)
            cbf_wrapper.ensemble_uncertainty = uncertainty

        if carla_env is not None:
            carla_env._ensemble_uncertainty = uncertainty
            carla_env._trust_score = trust_score

    def _on_episode_end(self):
        """Log episode-level summary and reset accumulators."""
        ep_len = max(self.episode_length, 1)
        self.episode_number += 1

        summary = {
            "cbf_invoke_rate":      self.episode_cbf_invocations / ep_len,
            "collision_rate":       self.episode_collisions       / ep_len,
            "lane_invasion_rate":   self.episode_lane_invasions   / ep_len,
            "cbf_invocations":      float(self.episode_cbf_invocations),
            "collisions":           float(self.episode_collisions),
            "lane_invasions":       float(self.episode_lane_invasions),
            "ensemble_uncertainty": self.current_ensemble_uncertainty,
            "trust_score":          self.current_trust_score,
        }
        for k, v in summary.items():
            self.logger.record(f"episode_summary/{k}", v)
        # FIX BUG 1: flush episode summary immediately
        self.logger.dump(self.num_timesteps)

        if self.verbose > 0:
            sep = "=" * 72
            print(f"\n{sep}")
            print(f"  EPISODE {self.episode_number} SUMMARY")
            print(f"{sep}")
            for k, v in summary.items():
                print(f"  {k:<30s}: {v:.4f}")
            print(f"{sep}\n")

        # Reset episode accumulators
        self.episode_length            = 0
        self.episode_cbf_invocations   = 0
        self.episode_lane_invasions    = 0
        self.episode_collisions        = 0
        self._prev_cbf_count           = 0
        self._prev_collision_count     = 0
        self._prev_lane_invasion_count = 0


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
    # FIX [10]: show_sensor_data=False during training — the original True
    # called cv2.imshow()+waitKey(1) on every step, adding latency and
    # breaking headless/SSH runs. Pass --render to see the RGB feed.
    env = create_carla_env(time_limit=60, render=render, num_npc=20, num_pedestrians=30, show_sensor_data=False)
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
    
    # Safety callback (fixed)
    safety_callback = SafetyMetricsCallback(verbose=1)

    # Comprehensive metrics callback (fixed)
    metrics_callback = ComprehensiveMetricsLoggingCallback(verbose=1)
    
    # Train
    print(f"Starting training for {total_timesteps} timesteps...\n")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            log_interval=1,
            callback=[checkpoint_callback, safety_callback, metrics_callback],
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
    parser.add_argument("--checkpoint-freq", type=int, default=20000, help="Checkpoint frequency")
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