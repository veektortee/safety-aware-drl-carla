"""Observation wrappers that turn CarlaGymEnv's Dict(rgb_data, nav) observation into policy inputs.

Stack used by every experiment (see CONTRACT section 6):

    CarlaGymEnv -> CBFSafetyLayerWrapper -> CnnNavObservationWrapper  (SAC 'MultiInputPolicy')
    CarlaGymEnv -> CBFSafetyLayerWrapper -> SttNavObservationWrapper  (SAC 'MlpPolicy')

Occlusions (experiments.occlusions.OcclusionStrategy) are applied to the camera image only, at full
resolution, and re-rolled at every reset (30% of episodes are occluded, consistently for the episode).
The nav vector is never occluded.
"""

import os
import re
from collections import deque
from typing import Optional

import cv2
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback

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


class SttNavObservationWrapper(gym.ObservationWrapper):
    """Dict(rgb_data, nav) -> Box(embed_dim + NAV_DIM,) float32 = [STT embedding, nav vector].

    The embedding comes from tests.pipeline_carla_test.PipelineObservationWrapper (ResNet50 ->
    SpatioTemporal Transformer over the last `num_frames` camera frames). It is composed, not
    subclassed: this wrapper owns a PipelineObservationWrapper instance and calls its
    observation() with the (occluded) image, so that file is used unchanged.

    The frame history is cleared at every reset, so an episode never sees frames of the previous one.
    """

    def __init__(self, env: gym.Env, embed_dim: int = 512, num_frames: int = 8,
                 encoder_path: Optional[str] = None, occlusion: Optional[str] = "mixed"):
        super().__init__(env)
        # Imported here so that the CNN wrapper does not pull in the whole CARLA env module.
        from tests.pipeline_carla_test import PipelineObservationWrapper

        self.embed_dim = int(embed_dim)
        self.num_frames = int(num_frames)
        self.nav_dim = _nav_dim_of(env)
        self.occlusion = _make_occlusion(occlusion)
        # Not part of the step chain: only used as the embedding engine (it never steps `env`).
        self.embedder = PipelineObservationWrapper(env, embed_dim=self.embed_dim,
                                                   num_frames=self.num_frames, encoder_path=encoder_path)
        self.pipeline = self.embedder.pipeline
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.embed_dim + self.nav_dim,), dtype=np.float32)

    def _clear_frame_history(self):
        buf = getattr(self.embedder, "frame_buffer", None)
        if isinstance(buf, deque) and len(buf) > 0:
            blank = np.zeros_like(buf[-1])
            n = buf.maxlen or len(buf)
            buf.clear()
            buf.extend(blank for _ in range(n))

    def reset(self, **kwargs):
        if self.occlusion is not None:
            self.occlusion.reset_episode()
        self._clear_frame_history()
        return super().reset(**kwargs)

    def observation(self, obs):
        rgb = _occlude(self.occlusion, _rgb_from_obs(obs))
        embedding = np.asarray(self.embedder.observation({"rgb_data": rgb}), dtype=np.float32).reshape(-1)
        if embedding.shape[0] != self.embed_dim:
            raise ValueError(f"STT embedding has {embedding.shape[0]} entries, expected {self.embed_dim}")
        embedding = np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)
        return np.concatenate([embedding, _nav_from_obs(obs, self.nav_dim)]).astype(np.float32)

    # --- transformer weights -------------------------------------------------------------------
    # The ResNet50 -> STT pipeline is not part of the SAC .zip. It is never optimised during
    # training (it runs in eval mode under no_grad), so its randomly initialised weights must be
    # saved once per run and restored on resume / in the demo, otherwise the policy sees a
    # different embedding than the one it was trained on.
    def transformer_state(self) -> dict:
        return {
            "st_encoder": self.pipeline.st_encoder.state_dict(),
            "stacked_transformer": self.pipeline.stacked_transformer.state_dict(),
        }

    def save_transformer(self, path: str, timesteps: Optional[int] = None) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        state = self.transformer_state()
        state["timesteps"] = int(timesteps) if timesteps is not None else None
        tmp = path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, path)
        return path

    def load_transformer(self, path: str) -> bool:
        state = torch.load(path, map_location=self.pipeline.device)
        self.pipeline.st_encoder.load_state_dict(state["st_encoder"])
        self.pipeline.stacked_transformer.load_state_dict(state["stacked_transformer"])
        self.pipeline.eval()
        return True


TRANSFORMER_CKPT_RE = re.compile(r"transformer_checkpoint_(\d+)\.pth$")


def find_stt_wrapper(model_or_env) -> Optional[SttNavObservationWrapper]:
    """The SttNavObservationWrapper inside an SB3 model's VecEnv or inside a plain gym wrapper chain."""
    env = model_or_env.get_env() if hasattr(model_or_env, "get_env") else model_or_env
    if env is not None and hasattr(env, "envs"):
        env = env.envs[0]
    while env is not None:
        if isinstance(env, SttNavObservationWrapper):
            return env
        env = getattr(env, "env", None)
    return None


def find_transformer_checkpoint(model_path: str, transformer_dir: Optional[str] = None) -> Optional[str]:
    """Transformer weights that belong to a SAC checkpoint.

    Looks in `transformer_dir`, else in '<dir of model>/transformer' (checkpoints) and
    '<dir of model>/checkpoints/transformer' (final model in the log dir). Prefers the file with the
    same step count as the SAC checkpoint, else the newest one (all files of one run hold the same
    weights because the transformer is not trained).
    """
    model_dir = os.path.dirname(os.path.abspath(model_path))
    dirs = [transformer_dir] if transformer_dir else [
        os.path.join(model_dir, "transformer"),
        os.path.join(model_dir, "checkpoints", "transformer"),
    ]
    m = re.search(r"_(\d+)_steps\.zip$", os.path.basename(model_path))
    wanted = int(m.group(1)) if m else None
    for d in dirs:
        if not d or not os.path.isdir(d):
            continue
        found = {}
        for name in os.listdir(d):
            mm = TRANSFORMER_CKPT_RE.match(name)
            if mm:
                found[int(mm.group(1))] = os.path.join(d, name)
        if not found:
            continue
        if wanted is not None and wanted in found:
            return found[wanted]
        return found[max(found)]
    return None


class TransformerCheckpointCallback(BaseCallback):
    """Save the STT transformer weights next to the SAC checkpoints.

    Saves 'transformer_checkpoint_<num_timesteps>.pth' once at training start (so every run has at
    least one file, even if it crashes early) and then every `save_freq` callback calls, in step
    with CheckpointCallback(save_freq) so the step numbers in the file names match.
    """

    def __init__(self, save_freq: int = 25000, save_path: str = "./checkpoints/transformer", verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = int(save_freq)
        self.save_path = save_path

    def _save(self):
        wrapper = find_stt_wrapper(self.model)
        if wrapper is None:
            if self.verbose > 0:
                print("[TRANSFORMER] SttNavObservationWrapper not found in the env stack; nothing saved")
            return
        path = os.path.join(self.save_path, f"transformer_checkpoint_{self.num_timesteps}.pth")
        wrapper.save_transformer(path, timesteps=self.num_timesteps)
        if self.verbose > 0:
            print(f"[TRANSFORMER] saved {path}")

    def _on_training_start(self) -> None:
        path = os.path.join(self.save_path, f"transformer_checkpoint_{self.num_timesteps}.pth")
        if not os.path.isfile(path):
            self._save()

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self._save()
        return True
