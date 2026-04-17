# examples/policy_kwargs_snippet.py
# Shows how to create the SpatioTemporalEncoder and pass it to SAC via policy_kwargs.

import torch
from commons.spatiotemporal_transformer import SpatioTemporalEncoder   # your transformer file
from commons.transformer_feature_extractor import TransformerFeatureExtractor

# Config
NUM_FRAMES = 8
IMG_SIZE = 7     # trunk output spatial size (C,7,7)
IN_CHANNELS = 2048
EMBED_DIM = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1) Create SpatioTemporalEncoder (same class as you already implemented)
st_encoder = SpatioTemporalEncoder(
    img_size=IMG_SIZE,
    in_channels=IN_CHANNELS,
    embed_dim=EMBED_DIM,
    num_frames=NUM_FRAMES,
    num_heads=8,
    num_layers=4
).to(DEVICE)

# 2) Prepare policy_kwargs for SAC
policy_kwargs = dict(
    features_extractor_class=TransformerFeatureExtractor,
    features_extractor_kwargs=dict(
        st_encoder=st_encoder,
        backbone=None,       # default ResNet trunk will be built inside extractor
        embed_dim=EMBED_DIM,
        device=DEVICE
    ),
    net_arch=[256, 256],     # typical actor/critic head sizes (you can tune)
)


import gymnasium as gym

from stable_baselines3 import SAC

env = gym.make("Pendulum-v1", render_mode="human")

model = SAC("MlpPolicy", env, policy_kwargs=policy_kwargs ,  verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
#model.save("sac_pendulum")

del model # remove to demonstrate saving and loading

model = SAC.load("sac_pendulum")

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()