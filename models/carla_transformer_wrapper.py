from collections import deque
from typing import Any, Dict, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TransformerObservationWrapper(gym.ObservationWrapper):
    """
    Wrap an env to convert CARLA RGB observations into a temporal stack
    of feature-maps for your SpatioTemporal encoder.

    - Expects env observations to contain an RGB frame under one of the keys
      'rgb_data', 'camera' or 'rgb' (or the env may return a raw frame).
    - `vision_extractor` can be any callable that accepts an RGB frame and
      returns a numpy feature-map shaped `(C, H, W)` (e.g. ResNet trunk output).

    The wrapper appends each extracted feature-map to an internal deque and
    exposes a numpy array of shape `(T, C, H, W)` under the observation key
    `'vision_features'`.
    """

    def __init__(
        self,
        env: gym.Env,
        vision_extractor: Any,
        temporal_window: int = 8,
        fmap_shape: tuple = (2048, 7, 7),
        dtype: np.dtype = np.float32,
    ):
        super().__init__(env)
        self.vision_extractor = vision_extractor
        self.temporal_window = int(temporal_window)
        self.fmap_shape = tuple(fmap_shape)
        self.dtype = dtype
        self.buffer = deque(maxlen=self.temporal_window)

        # Create a Box space for the stacked feature maps: (T, C, H, W)
        fmap_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.temporal_window,) + self.fmap_shape,
            dtype=self.dtype,
        )

        base_space = getattr(env, "observation_space", None)
        if isinstance(base_space, spaces.Dict):
            new_spaces = dict(base_space.spaces)
            new_spaces["vision_features"] = fmap_space
            self.observation_space = spaces.Dict(new_spaces)
        else:
            # If original env is not a Dict, return a Dict with the original under 'orig'
            self.observation_space = spaces.Dict({"orig": base_space, "vision_features": fmap_space})

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Clear buffer and prime with the first observation repeated
        self.buffer.clear()
        # Process initial observation
        processed = self.observation(obs)
        return processed, info

    def observation(self, obs: Any) -> Any:
        # Extract rgb frame
        rgb = None
        if isinstance(obs, dict):
            rgb = obs.get("rgb_data") or obs.get("camera") or obs.get("rgb")
        else:
            rgb = obs

        if rgb is None:
            raise KeyError(
                "TransformerObservationWrapper: no rgb frame found in observation dict."
                " Expected one of keys: 'rgb_data','camera','rgb' or a raw frame."
            )

        fmap = self._extract_fmap(rgb)
        # ensure dtype
        fmap = np.asarray(fmap, dtype=self.dtype)

        # push and pad
        self.buffer.append(fmap)
        while len(self.buffer) < self.temporal_window:
            self.buffer.appendleft(np.zeros(self.fmap_shape, dtype=self.dtype))

        stacked = np.stack(list(self.buffer), axis=0)  # (T, C, H, W)

        if isinstance(obs, dict):
            out = dict(obs)
            out["vision_features"] = stacked
            return out
        else:
            return {"vision_features": stacked}

    def _extract_fmap(self, rgb_frame: Any) -> np.ndarray:
        # Support vision_extractor callable or object with extract_feature_map
        if hasattr(self.vision_extractor, "extract_feature_map"):
            fmap = self.vision_extractor.extract_feature_map(rgb_frame)
        else:
            fmap = self.vision_extractor(rgb_frame)

        fmap = np.asarray(fmap)

        # If user-provided extractor returns (H, W, C), try to transpose
        if fmap.ndim == 3 and fmap.shape != self.fmap_shape:
            if fmap.shape == (self.fmap_shape[1], self.fmap_shape[2], self.fmap_shape[0]):
                fmap = np.transpose(fmap, (2, 0, 1))

        if fmap.shape != self.fmap_shape:
            raise ValueError(
                f"Extracted feature-map shape {fmap.shape} does not match expected "
                f"fmap_shape={self.fmap_shape}. Provide a vision_extractor that returns (C,H,W)."
            )

        return fmap


__all__ = ["TransformerObservationWrapper"]


# Quick local smoke test (no CARLA required)
if __name__ == "__main__":
    class DummyEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = spaces.Dict({"rgb_data": spaces.Box(0, 255, shape=(360, 640, 3), dtype=np.uint8)})

        def reset(self):
            return {"rgb_data": np.zeros((360, 640, 3), dtype=np.uint8)}, {}

        def step(self, action):
            return {"rgb_data": np.zeros((360, 640, 3), dtype=np.uint8)}, 0.0, False, False, {}

    # dummy extractor that downsamples and returns (C,H,W)
    def dummy_extractor(frame):
        # produce a small fake feature map (2048,7,7) but fill with zeros for speed
        return np.zeros((2048, 7, 7), dtype=np.float32)

    env = DummyEnv()
    wrap = TransformerObservationWrapper(env, dummy_extractor, temporal_window=4, fmap_shape=(2048, 7, 7))
    o, info = wrap.reset()
    print("vision_features shape after reset:", o["vision_features"].shape)
    o2, r = wrap.step(None)[:2]
    print("vision_features shape after step:", o2["vision_features"].shape)
