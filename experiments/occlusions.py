"""
Occlusion strategies for perception robustness experiments
- Random black bars
- White noise areas  
- LiDAR point dropout
- Occurs randomly in 30% of episodes, stays consistent throughout episode
"""

import numpy as np
import cv2
from typing import Dict, Optional


class OcclusionStrategy:
    """Apply random occlusions to environment observations"""
    
    def __init__(self, occlusion_type: str = 'none'):
        """
        Args:
            occlusion_type: 'none', 'black_bars', 'white_noise', 'lidar_dropout', 'mixed'
        """
        self.occlusion_type = occlusion_type
        self.active = False
        self.episode_start = None
        
        # Occlusion params (set once per episode)
        self.bar_height = None
        self.bar_position = None
        self.noise_region = None
    
    def activate_for_episode(self):
        """Randomly activate occlusion for entire episode (30% chance)"""
        if np.random.rand() < 0.3:
            self.active = True
            # Randomize parameters
            self.bar_height = np.random.randint(20, 100)
            self.bar_position = np.random.choice(['top', 'bottom', 'both'])
            self.noise_region = np.random.choice(['left', 'right', 'center'])
        else:
            self.active = False
    
    def reset_episode(self):
        """Call at episode start"""
        self.activate_for_episode()
    
    def apply(self, observation: Dict) -> Dict:
        """Apply occlusion to observation"""
        if not self.active:
            return observation
        
        observation = observation.copy()
        
        if self.occlusion_type in ['none']:
            return observation
        
        # RGB camera occlusion
        if 'rgb_data' in observation and observation['rgb_data'] is not None:
            rgb = observation['rgb_data'].copy().astype(np.uint8)
            h, w = rgb.shape[:2]
            
            if self.occlusion_type in ['black_bars', 'mixed']:
                # Black horizontal bars
                if self.bar_position in ['top', 'both']:
                    rgb[0:self.bar_height, :] = 0
                if self.bar_position in ['bottom', 'both']:
                    rgb[-self.bar_height:, :] = 0
            
            if self.occlusion_type in ['white_noise', 'mixed']:
                # White Gaussian noise in region
                noise = np.random.normal(200, 30, (h, w, 3))
                noise = np.clip(noise, 0, 255).astype(np.uint8)
                
                region_width = w // 3
                if self.noise_region == 'left':
                    rgb[:, :region_width] = noise[:, :region_width]
                elif self.noise_region == 'right':
                    rgb[:, -region_width:] = noise[:, -region_width:]
                else:  # center
                    start = w // 3
                    end = 2 * w // 3
                    rgb[:, start:end] = noise[:, start:end]
            
            observation['rgb_data'] = rgb
        
        # LiDAR occlusion
        if 'lidar_data' in observation and observation['lidar_data'] is not None:
            if self.occlusion_type in ['lidar_dropout', 'mixed']:
                lidar = observation['lidar_data'].copy()
                # Drop 40% of LiDAR points randomly
                mask = np.random.rand(lidar.shape[1]) < 0.6  # Keep 60%
                lidar_occluded = lidar.copy()
                lidar_occluded[:, ~mask] = 0
                observation['lidar_data'] = lidar_occluded
        
        return observation
