# ============================================================
# SAC Transformer Pipeline Test with Logging & Checkpoints
# ============================================================

import os
import sys
from collections import deque
import numpy as np
import gymnasium as gym

# ------------------------------------------------------------
# Path setup (allow importing from /commons)
# ------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
commons_path = os.path.join(project_root, "commons")
if commons_path not in sys.path:
    sys.path.append(commons_path)

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure

from spatiotemporal_transformer import SpatioTemporalEncoder
from transformer_feature_extractor import TransformerFeatureExtractor


# ============================================================
# Temporal Image Wrapper (FAKE IMAGE PIPELINE FOR TESTING)
# ============================================================
class TemporalImageWrapper(gym.Wrapper):
    """
    Converts low-dim obs into fake image frames and stacks them temporally.
    ONLY FOR PIPELINE TESTING.
    """

    def __init__(self, env, T=4, H=64, W=64):
        super().__init__(env)
        self.T = T
        self.H = H
        self.W = W
        self.frames = deque(maxlen=T)

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(T, 3, H, W),
            dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        frame = self._obs_to_image(obs)
        for _ in range(self.T):
            self.frames.append(frame)
        return np.stack(self.frames), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        frame = self._obs_to_image(obs)
        self.frames.append(frame)
        return np.stack(self.frames), reward, terminated, truncated, info

    def _obs_to_image(self, obs):
        img = np.zeros((3, self.H, self.W), dtype=np.uint8)
        val = int((obs[0] + 2) / 4 * 255)
        img[:] = val
        return img


# ============================================================
# MAIN
# ============================================================
def main():

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # --------------------------------------------------------
    # Logging directories
    # --------------------------------------------------------
    LOG_DIR = os.path.join(project_root, "logs")
    TB_DIR = os.path.join(LOG_DIR, "tensorboard")
    CKPT_DIR = os.path.join(LOG_DIR, "checkpoints")
    EVAL_DIR = os.path.join(LOG_DIR, "eval")

    os.makedirs(TB_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)

    # --------------------------------------------------------
    # Environments
    # --------------------------------------------------------
    base_env = gym.make("Pendulum-v1")
    env = TemporalImageWrapper(base_env, T=4)
    env = make_vec_env(lambda: env, n_envs=1)

    eval_env = TemporalImageWrapper(gym.make("Pendulum-v1"), T=4)
    eval_env = make_vec_env(lambda: eval_env, n_envs=1)

    # --------------------------------------------------------
    # Spatio-Temporal Transformer Encoder
    # --------------------------------------------------------
    st_encoder = SpatioTemporalEncoder(
        in_channels=2048,
        num_frames=4,
        embed_dim=512,
        num_layers=5,
        num_heads=8,
        img_size=(7, 7)
    ).to(DEVICE)

    # --------------------------------------------------------
    # Policy kwargs (CRITICAL)
    # --------------------------------------------------------
    policy_kwargs = dict(
        features_extractor_class=TransformerFeatureExtractor,
        features_extractor_kwargs=dict(
            st_encoder=st_encoder,
            embed_dim=512,
            device=DEVICE
        ),
        net_arch=dict(pi=[256, 256], qf=[256, 256])
    )

    # --------------------------------------------------------
    # SAC Model
    # --------------------------------------------------------
    model = SAC(
        policy="MlpPolicy",
        env=env,
        buffer_size=50_000,          # SAFE buffer size
        learning_starts=1_000,
        batch_size=64,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=DEVICE
    )

    # --------------------------------------------------------
    # Logger
    # --------------------------------------------------------
    new_logger = configure(TB_DIR, ["stdout", "tensorboard"])
    model.set_logger(new_logger)

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------
    checkpoint_callback = CheckpointCallback(
        save_freq=2_000,
        save_path=CKPT_DIR,
        name_prefix="sac_transformer"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=EVAL_DIR,
        log_path=EVAL_DIR,
        eval_freq=2_000,
        deterministic=True,
        render=False
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------
    model.learn(
        total_timesteps=10_000,
        callback=[checkpoint_callback, eval_callback]
    )

    # --------------------------------------------------------
    # Save & Reload Best Model
    # --------------------------------------------------------
    del model
    model = SAC.load(
        os.path.join(EVAL_DIR, "best_model"),
        env=env,
        device=DEVICE
    )

    # --------------------------------------------------------
    # Run trained model
    # --------------------------------------------------------
    obs = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs = env.reset()


# ============================================================
if __name__ == "__main__":
    main()
