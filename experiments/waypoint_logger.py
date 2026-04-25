"""
Waypoint logging for detailed location context analysis
- Junctions (intersections)
- Lane changes
- Highway merges
- Road and lane IDs
"""

import numpy as np
from typing import Dict, List
from collections import defaultdict


class WaypointContextLogger:
    """Log semantic location information from CARLA waypoints"""
    
    def __init__(self):
        self.junction_count = 0
        self.lane_change_count = 0
        self.merge_count = 0
        
        # Tracking per episode
        self.episode_junctions = 0
        self.episode_merges = 0
        self.episode_lane_changes = 0
        
        # History
        self.location_history = []
        self.prev_road_id = None
    
    def log_waypoint(self, carla_map, vehicle_location) -> Dict:
        """
        Log waypoint information at vehicle location
        
        Args:
            carla_map: carla.Map object
            vehicle_location: carla.Location of vehicle
        
        Returns:
            dict with location context
        """
        try:
            # Get waypoint with proper projection
            waypoint = carla_map.get_waypoint(
                vehicle_location,
                project_to_road=True,
                lane_type=None
            )
            
            if waypoint is None:
                return {'valid': False}
            
            context = {
                'valid': True,
                'road_id': waypoint.road_id,
                'lane_id': waypoint.lane_id,
                'section_id': waypoint.section_id,
                'is_junction': waypoint.is_junction,
                'lane_change': self._parse_lane_change(waypoint.lane_change),
                'lane_type': str(waypoint.lane_type),
                'road_marked': waypoint.road_marked,
            }
            
            # Detect junction entry
            if context['is_junction'] and (self.prev_road_id != context['road_id'] or self.prev_road_id is None):
                self.junction_count += 1
                self.episode_junctions += 1
                context['event'] = 'JUNCTION'
            
            # Detect lane change opportunity
            if context['lane_change'] != 'NONE':
                self.lane_change_count += 1
                self.episode_lane_changes += 1
                context['event'] = f'LANE_CHANGE_{context["lane_change"]}'
            
            # Detect merge by checking next waypoints
            if not context['is_junction']:
                next_waypoints = waypoint.next(2.0)
                if len(next_waypoints) > 1:
                    # Multiple paths = merge/fork
                    self.merge_count += 1
                    self.episode_merges += 1
                    context['event'] = 'MERGE'
            
            self.prev_road_id = context['road_id']
            self.location_history.append(context)
            
            return context
        
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def _parse_lane_change(self, lane_change_enum) -> str:
        """Convert carla.LaneChange enum to string"""
        try:
            # carla.LaneChange.Left, Right, Both, None
            if str(lane_change_enum) == 'Left':
                return 'LEFT'
            elif str(lane_change_enum) == 'Right':
                return 'RIGHT'
            elif str(lane_change_enum) == 'Both':
                return 'BOTH'
        except:
            pass
        return 'NONE'
    
    def reset_episode(self):
        """Reset episode counters"""
        self.episode_junctions = 0
        self.episode_merges = 0
        self.episode_lane_changes = 0
    
    def get_episode_summary(self) -> Dict:
        """Get summary of location events in current episode"""
        return {
            'junctions': self.episode_junctions,
            'merges': self.episode_merges,
            'lane_changes': self.episode_lane_changes,
            'total_events': self.episode_junctions + self.episode_merges + self.episode_lane_changes
        }
