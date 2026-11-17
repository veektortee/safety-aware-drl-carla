"""Observation wrappers that turn CarlaGymEnv's Dict(rgb_data, nav) observation into policy inputs.

Stacks used by the experiments (both with SAC 'MultiInputPolicy'):

    CarlaGymEnv -> CBFSafetyLayerWrapper -> CnnNavObservationWrapper  (this file)
    CarlaGymEnv -> CBFSafetyLayerWrapper -> SttNavObservationWrapper  (commons/stt.py)

Occlusions (experiments.occlusions.OcclusionStrategy) are applied to the camera image only, at full
resolution, and re-rolled at every reset (30% of episodes are occluded, consistently for the episode).
The nav vector is never occluded. The helpers below are shared by both wrappers.
"""

from typing import Optional

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from commons.nav_layout import NAV_DIM
from experiments.occlusions import OcclusionStrategy

# The nav features are normalised to roughly [-2, 2]; this only guards the replay buffer against
# NaN/inf or garbage values (one NaN in the buffer is enough to destroy the critic).
NAV_CLIP = 10.0


def _make_occlusion(occlusion: Optional[str]) -> Optional[OcclusionStrategy]:
    if occlusion is None or occlusion == "none":
        return None
    return OcclusionStrategy(occlusion)


def _nav_dim_of(env: gym.Env) -> int:
    space = getattr(env, "observation_space", None)
    if isinstance(space, spaces.Dict) and "nav" in space.spaces:
        return int(np.prod(space.spaces["nav"].shape))
    return NAV_DIM


def _rgb_from_obs(obs) -> Optional[np.ndarray]:
    """HxWx3 uint8 camera image from the env observation (None if missing)."""
    rgb = obs.get("rgb_data") if isinstance(obs, dict) else obs
    if rgb is None:
        return None
    rgb = np.asarray(rgb)
    if rgb.size == 0:
        return None
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=2)
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def _occlude(strategy: Optional[OcclusionStrategy], rgb: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Apply the episode's occlusion to the image only. Never mutates the env's array."""
    if strategy is None or rgb is None or not strategy.active:
        return rgb
    return strategy.apply({"rgb_data": rgb})["rgb_data"]


def _nav_from_obs(obs, nav_dim: int) -> np.ndarray:
    nav = obs.get("nav") if isinstance(obs, dict) else None
    if nav is None:
        return np.zeros(nav_dim, dtype=np.float32)
    nav = np.asarray(nav, dtype=np.float32).reshape(-1)
    if nav.shape[0] != nav_dim:
        raise ValueError(f"obs['nav'] has {nav.shape[0]} entries, expected {nav_dim}")
    nav = np.nan_to_num(nav, nan=0.0, posinf=NAV_CLIP, neginf=-NAV_CLIP)
    return np.clip(nav, -NAV_CLIP, NAV_CLIP).astype(np.float32)


class CnnNavObservationWrapper(gym.ObservationWrapper):
    """Dict(rgb_data HxWx3, nav) -> Dict(image (3,size,size) uint8, nav (NAV_DIM,) float32).

    The image key is channel-first uint8 in [0, 255], so SB3's CombinedExtractor routes it through
    NatureCNN (normalize_images=True) and flattens the nav vector next to it.
    """

    def __init__(self, env: gym.Env, size: int = 84, occlusion: Optional[str] = "mixed"):
        super().__init__(env)
        self.size = int(size)
        self.nav_dim = _nav_dim_of(env)
        self.occlusion = _make_occlusion(occlusion)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(3, self.size, self.size), dtype=np.uint8),
            "nav": spaces.Box(low=-np.inf, high=np.inf, shape=(self.nav_dim,), dtype=np.float32),
        })

    def reset(self, **kwargs):
        if self.occlusion is not None:
            self.occlusion.reset_episode()
        return super().reset(**kwargs)

    def observation(self, obs):
        rgb = _occlude(self.occlusion, _rgb_from_obs(obs))
        if rgb is None:
            image = np.zeros((3, self.size, self.size), dtype=np.uint8)
        else:
            if rgb.shape[0] != self.size or rgb.shape[1] != self.size:
                rgb = cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_AREA)
            image = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1)), dtype=np.uint8)
        return {"image": image, "nav": _nav_from_obs(obs, self.nav_dim)}
