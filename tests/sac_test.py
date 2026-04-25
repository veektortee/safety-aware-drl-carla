import os
import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

# --------------------------------------------------
# Paths for logging and checkpoints
# --------------------------------------------------
LOG_DIR = "./logs"
TB_DIR = os.path.join(LOG_DIR, "tensorboard")
CKPT_DIR = os.path.join(LOG_DIR, "checkpoints")

os.makedirs(TB_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# --------------------------------------------------
# Environment (render during evaluation, not training)
# --------------------------------------------------
env = gym.make("Pendulum-v1")

# --------------------------------------------------
# Model
# --------------------------------------------------
model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
)

# --------------------------------------------------
# TensorBoard logger
# --------------------------------------------------
new_logger = configure(TB_DIR, ["stdout", "tensorboard"])
model.set_logger(new_logger)

# --------------------------------------------------
# Checkpoint callback
# --------------------------------------------------
checkpoint_callback = CheckpointCallback(
    save_freq=2_000,
    save_path=CKPT_DIR,
    name_prefix="sac_pendulum"
)

# --------------------------------------------------
# Train (NO rendering here – keeps training fast)
# --------------------------------------------------
model.learn(
    total_timesteps=10_000,
    log_interval=4,
    callback=checkpoint_callback
)

# --------------------------------------------------
# Save final model
# --------------------------------------------------
model.save("sac_pendulum_final")
del model

# --------------------------------------------------
# Reload and RUN WITH RENDERING (real-time viewing)
# --------------------------------------------------
env = gym.make("Pendulum-v1", render_mode="human")
model = SAC.load("sac_pendulum_final", env=env)

obs, info = env.reset()

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
