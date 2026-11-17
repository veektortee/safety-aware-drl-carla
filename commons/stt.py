"""SpatioTemporal Transformer (STT) perception, trained end-to-end by SAC.

Env side - SttNavObservationWrapper (frozen, no gradients, runs once per camera frame):

    camera frame (BGR 360x640) -> resize to 224x128 (keeps the 16:9 layout)
      -> ImageNet ResNet50 trunk (frozen)          -> (2048, 4, 7) feature map
      -> frozen PCA projection, whitened, clipped  -> (K, 4, 7) float16

    Each frame is encoded ONCE and cached; the observation is the stack of the last T cached frames
    (repeat-padded with the episode's first frame after a reset, never black frames) plus the nav
    vector: Dict(frames (T, K, 4, 7) float16, nav (NAV_DIM,) float32). The compact per-frame code is
    what makes end-to-end training possible: the replay buffer holds these instead of raw 2048x7x7
    ResNet maps (~100x smaller).

Policy side - SttFeaturesExtractor (trained by the SAC actor / critic losses):

    frames -> SpatioTemporalEncoder (divided space-time attention over T x 28 tokens) -> st (D)
    newest frame, spatially averaged -> cnn_proj Linear(K, D)                         -> f  (D)
    StackedHierarchicalTransformer(f, st) -> LayerNorm -> concat nav                  -> D + NAV_DIM

Saving: every trainable weight (encoder, cnn_proj, transformer) is part of the SB3 policy, so it is
inside every checkpoint .zip. The frozen PCA projection is fitted once at the start of a fresh run
(fit_stt_projection) and stored on the model as `model.stt_projection`, which SB3 also writes into
the .zip. Resume and the demo restore it with `wrapper.set_projection(model.stt_projection)`.
"""

import os
from collections import deque
from typing import Optional

import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from commons.obs_wrappers import _make_occlusion, _nav_dim_of, _nav_from_obs, _occlude, _rgb_from_obs
from commons.spatiotemporal_transformer import SpatioTemporalEncoder, StackedHierarchicalTransformer

NUM_FRAMES = 8              # frames in the temporal window (0.4 s at 20 FPS)
CHANNELS = 32               # PCA components kept per spatial cell
INPUT_SIZE = (224, 128)     # (W, H) fed to ResNet50
GRID = (4, 7)               # (H, W) of the ResNet50 output for INPUT_SIZE (stride 32)
TRUNK_CHANNELS = 2048
FEATURE_CLIP = 8.0          # whitened PCA codes are clipped to +-FEATURE_CLIP (float16-safe)
EMBED_DIM = 128             # transformer width D

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESNET50_WEIGHTS = os.path.join(_REPO_ROOT, "pretrained", "feature_extractor", "pretrained_resnet50.pth")


# ------------------------------------------------------------------------------------------------
# Frozen ResNet50 trunk
# ------------------------------------------------------------------------------------------------

def _resnet50_state_dict(path: str = RESNET50_WEIGHTS) -> dict:
    """ImageNet ResNet50 weights from `path`; downloaded once (torchvision) and saved there if missing."""
    if os.path.isfile(path):
        return torch.load(path, map_location="cpu")
    from torchvision import models

    print(f"[STT] {path} not found; downloading torchvision's ImageNet ResNet50 weights once")
    state = models.ResNet50_Weights.IMAGENET1K_V2.get_state_dict(progress=False, check_hash=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    return state


class FrozenResNet50:
    """ImageNet ResNet50 without avgpool/fc. __call__(bgr frame) -> (GRID cells, 2048) float32 tensor."""

    def __init__(self, device: Optional[str] = None, weights_path: str = RESNET50_WEIGHTS):
        from torchvision import models

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        net = models.resnet50(weights=None)
        net.load_state_dict(_resnet50_state_dict(weights_path), strict=True)
        self.trunk = nn.Sequential(*list(net.children())[:-2]).to(self.device).eval()
        for p in self.trunk.parameters():
            p.requires_grad_(False)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def __call__(self, bgr: np.ndarray) -> torch.Tensor:
        img = cv2.resize(bgr, INPUT_SIZE, interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1]))          # BGR -> RGB
        x = x.to(self.device).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
        x = (x - self.mean) / self.std
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
            fmap = self.trunk(x)                                              # (1, 2048, 4, 7)
        if tuple(fmap.shape[2:]) != GRID:
            raise RuntimeError(f"ResNet50 grid {tuple(fmap.shape[2:])} != expected {GRID}")
        return fmap[0].float().flatten(1).T                                  # (28, 2048)


# ------------------------------------------------------------------------------------------------
# Frozen PCA projection (fitted once per run, stored on the SB3 model)
# ------------------------------------------------------------------------------------------------

class ProjectionFitter:
    """Streaming PCA: accumulate the token covariance, then keep the top-k whitened components."""

    def __init__(self, dim: int = TRUNK_CHANNELS, device=None):
        self.device = torch.device(device or "cpu")
        self.n = 0
        self.sum = torch.zeros(dim, dtype=torch.float64, device=self.device)
        self.outer = torch.zeros(dim, dim, dtype=torch.float64, device=self.device)

    def add(self, tokens: torch.Tensor):
        x = tokens.to(self.device, torch.float64)
        self.n += x.shape[0]
        self.sum += x.sum(0)
        self.outer += x.T @ x

    def projection(self, k: int = CHANNELS) -> dict:
        if self.n < 2 * k:
            raise ValueError(f"need at least {2 * k} tokens to fit {k} components, got {self.n}")
        mean = self.sum / self.n
        cov = (self.outer - self.n * torch.outer(mean, mean)) / (self.n - 1)
        evals, evecs = torch.linalg.eigh(cov)                                # ascending
        evals, evecs = evals.flip(0).clamp_min(0.0), evecs.flip(1)
        top = evals[:k]
        return {
            "mean": mean.float().cpu().numpy(),
            "components": evecs[:, :k].T.float().cpu().numpy(),             # (k, 2048)
            "scale": top.sqrt().clamp_min(1e-6).float().cpu().numpy(),      # whitening
            "explained_variance": float(top.sum() / evals.sum().clamp_min(1e-12)),
            "n_tokens": int(self.n),
            "grid": GRID,
            "input_size": INPUT_SIZE,
        }


def projection_summary(proj: dict) -> str:
    k = int(np.asarray(proj["components"]).shape[0])
    return (f"PCA {TRUNK_CHANNELS}->{k} per cell, {proj['explained_variance']:.0%} of the variance, "
            f"fitted on {proj['n_tokens']:,} tokens")


# ------------------------------------------------------------------------------------------------
# Observation wrapper (feature cache)
# ------------------------------------------------------------------------------------------------

class SttNavObservationWrapper(gym.ObservationWrapper):
    """Dict(rgb_data, nav) -> Dict(frames (T, K, 4, 7) float16, nav (NAV_DIM,) float32).

    Needs a projection before the first observation: fit_stt_projection(...) for a fresh run, or
    set_projection(model.stt_projection) when resuming / in the demo.
    """

    def __init__(self, env: gym.Env, num_frames: int = NUM_FRAMES, channels: int = CHANNELS,
                 occlusion: Optional[str] = "mixed", backbone=None):
        super().__init__(env)
        self.num_frames = int(num_frames)
        self.channels = int(channels)
        self.nav_dim = _nav_dim_of(env)
        self.occlusion = _make_occlusion(occlusion)
        self.backbone = backbone if backbone is not None else FrozenResNet50()
        self.device = getattr(self.backbone, "device", torch.device("cpu"))
        self.projection = None
        self._proj = None
        self.frames = deque(maxlen=self.num_frames)                          # the feature cache
        self.observation_space = spaces.Dict({
            "frames": spaces.Box(-FEATURE_CLIP, FEATURE_CLIP,
                                 (self.num_frames, self.channels, *GRID), dtype=np.float16),
            "nav": spaces.Box(-np.inf, np.inf, (self.nav_dim,), dtype=np.float32),
        })

    def set_projection(self, proj: dict):
        comps = np.asarray(proj["components"], dtype=np.float32)
        if comps.shape != (self.channels, TRUNK_CHANNELS):
            raise ValueError(f"projection has shape {comps.shape}, expected ({self.channels}, {TRUNK_CHANNELS})")
        to = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32), device=self.device)  # noqa: E731
        self._proj = (to(proj["mean"]), to(comps).T.contiguous(), to(proj["scale"]))
        self.projection = proj

    @torch.inference_mode()
    def encode(self, rgb: np.ndarray) -> np.ndarray:
        """One camera frame -> (K, 4, 7) float16 code."""
        mean, comps_t, scale = self._proj
        z = (self.backbone(rgb) - mean) @ comps_t / scale                    # (28, K)
        z = z.clamp_(-FEATURE_CLIP, FEATURE_CLIP).T.reshape(self.channels, *GRID)
        return z.half().cpu().numpy()

    def reset(self, **kwargs):
        if self.occlusion is not None:
            self.occlusion.reset_episode()
        self.frames.clear()
        return super().reset(**kwargs)

    def observation(self, obs):
        if self._proj is None:
            raise RuntimeError("SttNavObservationWrapper: no projection set (fit_stt_projection or "
                               "set_projection(model.stt_projection) first)")
        rgb = _occlude(self.occlusion, _rgb_from_obs(obs))
        if rgb is None:
            raise RuntimeError("SttNavObservationWrapper: observation has no camera image")
        code = self.encode(rgb)
        if not self.frames:                                                  # new episode: repeat-pad
            self.frames.extend(code for _ in range(self.num_frames))
        else:
            self.frames.append(code)
        return {"frames": np.stack(self.frames), "nav": _nav_from_obs(obs, self.nav_dim)}


def _find_env_with(env, attr: str):
    while env is not None:
        if hasattr(type(env), attr) or attr in getattr(env, "__dict__", {}):
            return env
        env = getattr(env, "env", None)
    return None


def fit_stt_projection(wrapper: SttNavObservationWrapper, n_steps: int = 1000, reset_every: int = 250,
                       stage: Optional[int] = None, verbose: int = 1) -> dict:
    """Fit the frozen PCA projection on frames from expert driving, then set it on `wrapper`.

    Drives the env under the wrapper (CBF wrapper -> CarlaGymEnv) with CarlaGymEnv.expert_action(),
    restarting every `reset_every` steps for varied scenes. `stage` (e.g. the final curriculum stage)
    makes the frames include traffic and pedestrians even if training starts on empty roads; the
    previous stage is restored afterwards (applied at the next reset). Nothing is stored anywhere.
    """
    inner = wrapper.env
    carla_env = _find_env_with(inner, "expert_action")
    if carla_env is None:
        raise RuntimeError("fit_stt_projection: no env with expert_action() under the wrapper")
    old_stage = getattr(carla_env, "curriculum_stage", None)
    if stage is not None and hasattr(carla_env, "set_curriculum_stage"):
        carla_env.set_curriculum_stage(stage)

    fitter = ProjectionFitter(device=wrapper.device)
    if verbose:
        print(f"[STT] fitting the frame projection on {n_steps} expert-driving frames...")
    obs, _ = inner.reset()
    t = 0
    for step in range(1, n_steps + 1):
        rgb = _rgb_from_obs(obs)
        if rgb is not None:
            fitter.add(wrapper.backbone(rgb))
        obs, _, terminated, truncated, _ = inner.step(np.asarray(carla_env.expert_action(), dtype=np.float32))
        t += 1
        if terminated or truncated or t >= reset_every:
            obs, _ = inner.reset()
            t = 0
        if verbose and step % 250 == 0:
            print(f"[STT]   {step}/{n_steps} frames")

    proj = fitter.projection(wrapper.channels)
    if stage is not None and old_stage is not None:
        carla_env.set_curriculum_stage(old_stage)
    wrapper.set_projection(proj)
    if verbose:
        print(f"[STT] {projection_summary(proj)}")
    return proj


# ------------------------------------------------------------------------------------------------
# Trainable policy features extractor
# ------------------------------------------------------------------------------------------------

class SttFeaturesExtractor(BaseFeaturesExtractor):
    """SB3 features extractor: Dict(frames, nav) -> [STT embedding (D), nav]. Trained by SAC."""

    def __init__(self, observation_space: spaces.Dict, embed_dim: int = EMBED_DIM, num_heads: int = 4,
                 num_layers: int = 2, num_blocks: int = 5):
        frames_shape = observation_space["frames"].shape
        nav_dim = int(np.prod(observation_space["nav"].shape))
        super().__init__(observation_space, features_dim=embed_dim + nav_dim)
        t, k, h, w = frames_shape
        self.encoder = SpatioTemporalEncoder(img_size=(w, h), in_channels=k, embed_dim=embed_dim,
                                             num_frames=t, num_heads=num_heads, num_layers=num_layers,
                                             dropout=0.0)
        # The encoder draws its positional embeddings from N(0, 1), which swamps the projected
        # features at the start of training; use the usual ViT scale instead.
        nn.init.trunc_normal_(self.encoder.pos_embed_spatial, std=0.02)
        nn.init.trunc_normal_(self.encoder.pos_embed_temporal, std=0.02)
        self.cnn_proj = nn.Linear(k, embed_dim)
        self.transformer = StackedHierarchicalTransformer(embed_dim=embed_dim, num_heads=num_heads,
                                                          num_blocks=num_blocks)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, observations: dict) -> torch.Tensor:
        frames = observations["frames"]                                      # (B, T, K, H, W)
        st = self.encoder(frames)                                            # (B, D)
        f = self.cnn_proj(frames[:, -1].mean(dim=(2, 3)))                    # (B, D) newest frame
        out, _ = self.transformer(f, st)
        return torch.cat([self.norm(out), observations["nav"].flatten(1)], dim=1)


def stt_policy_kwargs(n_critics: int, embed_dim: int = EMBED_DIM) -> dict:
    """policy_kwargs for SAC('MultiInputPolicy') with the trainable STT extractor."""
    return {
        "net_arch": {"pi": [256, 256], "qf": [256, 256]},
        "n_critics": n_critics,
        "features_extractor_class": SttFeaturesExtractor,
        "features_extractor_kwargs": {"embed_dim": embed_dim},
    }
