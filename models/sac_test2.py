import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__)) 
project_root = os.path.abspath(os.path.join(current_dir, "..")) 
commons_path = os.path.join(project_root, "commons") 
if commons_path not in sys.path: 
    sys.path.append(commons_path)



import gymnasium as gym
import numpy as np
from collections import deque

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
        """
        Encode low-dim obs into a fake RGB image.
        """
        img = np.zeros((3, self.H, self.W), dtype=np.uint8)
        val = int((obs[0] + 2) / 4 * 255)
        img[:] = val
        return img




import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from spatiotemporal_transformer import SpatioTemporalEncoder
from transformer_feature_extractor import TransformerFeatureExtractor

# ------------------------------------
# Environment
# ------------------------------------


base_env = gym.make("Pendulum-v1")
env = TemporalImageWrapper(base_env, T=4)
env = make_vec_env(lambda: env, n_envs=1)

# ------------------------------------
# Spatio-temporal encoder
# ------------------------------------
st_encoder = SpatioTemporalEncoder(
    in_channels=2048,
    num_frames=2,
    embed_dim=512,
    num_layers=5,
    num_heads=8,
    img_size=(7, 7)
)

# ------------------------------------
# Policy kwargs (CRITICAL)
# ------------------------------------
policy_kwargs = dict(
    features_extractor_class=TransformerFeatureExtractor,
    features_extractor_kwargs=dict(
        st_encoder=st_encoder,
        embed_dim=512,
        device="cuda"
    ),
    net_arch=dict(pi=[256, 256], qf=[256, 256])
)

# ------------------------------------
# SAC model
# ------------------------------------
model = SAC(
    policy="MlpPolicy",     # still MLP policy (correct)
    env=env,
    buffer_size=50_000,   # ← was 1_000_000
    learning_starts=1_000,
    batch_size=64,
    policy_kwargs=policy_kwargs,
    verbose=1,
    device="cuda"
)

# ------------------------------------
# Train
# ------------------------------------
model.learn(total_timesteps=10_000)

# ------------------------------------
# Save / Load
# ------------------------------------
model.save("sac_transformer_test")
del model

model = SAC.load("sac_transformer_test", env=env)

# ------------------------------------
# Run
# ------------------------------------
obs = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, info = env.step(action)
