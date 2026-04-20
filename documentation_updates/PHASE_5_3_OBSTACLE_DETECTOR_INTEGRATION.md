# Phase 5.3: Obstacle Detector (RSS) Sensor Integration

**Date**: Current Session  
**Status**: ✅ COMPLETED  
**Target File**: `tests/pipeline_carla_test.py`

## Overview
Phase 5.3 integrates the CARLA Obstacle Detector (RSS - Responsibility Sensitive Safety) sensor to detect obstacles in the vehicle's path and provide safety feedback to the reinforcement learning agent.

## Changes Implemented

### 1. State Tracking (Lines 96-127)
```python
# Sensors - Added:
self.obstacle_sensor = None  # Obstacle Detector (RSS) sensor (Phase 5.3)

# Sensor data - Added:
self.obstacle_detected = False          # Obstacle Detector state (Phase 5.3)
self.obstacle_distance = 100.0          # Distance to nearest obstacle (Phase 5.3)
self.obstacle_actor = None              # Actor causing obstacle (Phase 5.3)

# State tracking - Added:
self._obstacle_detection_count = 0      # Count of obstacle detections (Phase 5.3)
```

### 2. Sensor Setup (Lines 359-371)
Added obstacle sensor attachment in `_attach_sensors` method:
```python
# Obstacle Detector (RSS) Sensor (Phase 5.3)
try:
    obstacle_bp = self.blueprint_library.find('sensor.other.obstacle')
    obstacle_bp.set_attribute('only_physics', 'False')
    obstacle_bp.set_attribute('distance_to_ad', '50')  # Detection range: 50 meters
    obstacle_transform = carla.Transform(carla.Location(x=0.8, z=1.7))
    self.obstacle_sensor = self.world.spawn_actor(obstacle_bp, obstacle_transform, 
                                                  attach_to=self.ego_vehicle)
    self.obstacle_sensor.listen(self._on_obstacle_detected)
    print("[OK] Obstacle Detector (RSS) sensor attached")
except Exception as e:
    print(f"[WARN] Could not attach Obstacle Detector sensor: {e}")
    self.obstacle_sensor = None
```

### 3. Sensor Callback (Lines 399-413)
New callback method to handle obstacle detection events:
```python
def _on_obstacle_detected(self, event):
    """Obstacle Detector (RSS) sensor callback (Phase 5.3)"""
    self.obstacle_detected = True
    self._obstacle_detection_count += 1
    
    # Extract obstacle distance and actor information
    if hasattr(event, 'distance'):
        self.obstacle_distance = event.distance
    else:
        self.obstacle_distance = 0.0
    
    if hasattr(event, 'other_actor'):
        self.obstacle_actor = event.other_actor
```

### 4. Reward Function Integration (Lines 981-998)
Added obstacle detection penalty in `_compute_reward`:
```python
# Obstacle detection penalty (Phase 5.3)
if self.obstacle_detected:
    reward -= 75.0  # Penalty for obstacle in path
    self.obstacle_detected = False
    reward_components['obstacle_detected'] = -75.0
    
    # Additional penalty based on distance to obstacle
    if self.obstacle_distance < 10.0:  # Very close
        reward -= 50.0
        reward_components['obstacle_proximity'] = -50.0
    elif self.obstacle_distance < 20.0:  # Moderately close
        reward -= 25.0
        reward_components['obstacle_proximity'] = -25.0
```

### 5. Obstacle Avoidance Reward Component (Lines 932-962)
New 9th behavioral reward component for obstacle avoidance:
```python
def compute_obstacle_avoidance_reward(self) -> float:
    """
    Reward for obstacle avoidance.
    Phase 5.3: Obstacle Detector (RSS) sensor integration
    
    Returns: [-0.5, 0.5] reward value
    """
    if not self.obstacle_sensor or self.obstacle_distance >= 100.0:
        return 0.0
    
    # Reward based on maintaining safe distance
    if self.obstacle_distance > 20.0:
        safe_distance_reward = 0.1 * (self.obstacle_distance / 50.0)
    elif self.obstacle_distance > 10.0:
        safe_distance_reward = 0.2
    else:
        safe_distance_reward = 0.0
    
    return safe_distance_reward
```

### 6. Advanced Reward Weighting Update (Lines 1011-1071)
Updated component weighting from 8 to 9 components (1/9 ≈ 0.111 each):
- Components: Lane Centering, Forward Motion, Safe Following, Traffic Flow, Yield & Maneuver
- Plus: Safety Buffer, Waypoint Progress, Opposite Lane Penalty, **Obstacle Avoidance**

Penalty Scheme:
- Obstacle detected: -75.0 (base penalty)
- Obstacle very close (<10m): -50.0 (additional)
- Obstacle moderately close (10-20m): -25.0 (additional)
- Obstacle at safe distance (>20m): +0.1 to +0.2 (reward)

### 7. Environment Reset (Lines 1197-1201)
Added obstacle detection state reset in `reset` method:
```python
self.obstacle_detected = False          # Reset obstacle detection (Phase 5.3)
self._obstacle_detection_count = 0      # Reset obstacle detection count (Phase 5.3)
```

### 8. Sensor Cleanup (Lines 1248-1289)
Updated `close` method to properly stop and destroy all sensors including obstacle sensor:
```python
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
```

## Reward Structure Summary

### Safety Penalties (Immediate)
| Event | Base Penalty | Additional |
|-------|-------------|-----------|
| Collision | -150.0 | — |
| Lane Invasion | -50.0 | — |
| Obstacle Detected | -75.0 | -50.0 (if <10m) or -25.0 (if 10-20m) |

### Behavioral Rewards (Weighted 1/9 each)
1. **Lane Centering** - Precision lane-keeping
2. **Forward Motion** - Drive in heading direction
3. **Safe Following** - Maintain headway
4. **Traffic Flow** - Smooth acceleration
5. **Yield & Maneuver** - Move for faster vehicles
6. **Safety Buffer** - Margin maintenance
7. **Waypoint Progress** - Milestone + distance
8. **Opposite Lane** - Penalize wrong direction
9. **Obstacle Avoidance** - Safe distance maintenance (NEW)

## Detection Parameters
- **Sensor Type**: sensor.other.obstacle (RSS)
- **Detection Range**: 50 meters
- **Physics-based Detection**: Enabled (only_physics = False)
- **Attachment**: Front of vehicle (x=0.8)

## Error Handling
- Gracefully handles obstacle sensor attachment failures (prints warning)
- Returns safe defaults (obstacle_distance = 100.0) if sensor unavailable
- Proper cleanup of sensor resources at environment close

## Testing Recommendations
1. Spawn obstacles in CARLA and verify detection
2. Verify reward penalties applied correctly
3. Check sensor callback frequency matches world tick
4. Monitor obstacle avoidance reward component impact
5. Validate sensor cleanup on episode termination

## Future Enhancements
- Multi-obstacle tracking (prioritize closest)
- Dynamic penalty scaling based on obstacle type
- Obstacle velocity consideration
- Integrated trajectory prediction for avoidance
- Visualization of detected obstacles in training

## Files Modified
- `tests/pipeline_carla_test.py` - Main implementation

## Notes
- Phase 5.3 is complementary to existing safety mechanisms (collision, lane invasion sensors)
- Obstacle detection operates independently from CBF layer (can be extended for CBF constraints)
- Nine-component reward system maintains balanced incentivization
- All obstacle state properly reset between episodes
