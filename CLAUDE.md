# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Safety-aware deep RL for autonomous driving in the CARLA simulator. A SAC agent is trained on RGB+LiDAR observations; a **Control Barrier Function (CBF) safety layer** minimally corrects the agent's actions to enforce collision/lane/speed constraints, and a **critic ensemble** produces an uncertainty-derived *trust score* that modulates how aggressively the CBF intervenes. Optional spatiotemporal-transformer perception sits between the camera and the policy.

## Prerequisites: a running CARLA server

Nothing auto-launches CARLA. Before running anything, a CARLA server must be listening on `localhost:<port>` (synchronous mode, 20 FPS / `fixed_delta_seconds=0.05`). Scripts timeout after ~10 s if the server isn't up, and raise `RuntimeError: CARLA module not found` if the `carla` PythonAPI isn't importable.

```bash
./CarlaUE4.sh -fps=20 --carla-port=2000     # Linux
CarlaUE4.exe -fps=20 --carla-port=2000      # Windows
```

`stable_baselines3` is **vendored** as a top-level directory, not a pip install — it is a *modified* copy (see "Vendored SB3" below). Do not `pip install stable-baselines3` over it. Local ResNet50 ImageNet weights live at `pretrained/feature_extractor/pretrained_resnet50.pth` and are loaded by `commons/feature_extractor.py` to avoid a torchvision download (which fails behind SSL-inspecting proxies). Keep that file present.

## Running training

Four experiment scripts in `experiments/`, named `<N>q<Arch>sac_experiment.py` where `N` ∈ {2,5} is the critic-ensemble size and `Arch` ∈ {CNN, stt}:

- `2qCNNsac_experiment.py`, `5qCNNsac_experiment.py` — ResNet50 → `CnnPolicy` (no transformer)
- `2qsac_stt_experiment.py`, `5qsac_stt_experiment.py` — ResNet50 → SpatioTemporal Transformer → `MlpPolicy` on a 512-dim embedding

```bash
# From the repo root. NOTE: default --port differs per script (e.g. 2000 vs 2004) — match your server.
python experiments/5qCNNsac_experiment.py --port 2000 --timesteps 1000000 --render
```

Common flags: `--timesteps` (target TOTAL steps), `--port`, `--log-dir`, `--lr`, `--batch-size`, `--buffer-size`, `--render`, `--num-npc`, `--num-pedestrians`.

Outputs go to `<log-dir>/{tensorboard,checkpoints}` and a final `sac_<tag>_final.zip`. View curves with `tensorboard --logdir <log-dir>/tensorboard`.

### Resume from a checkpoint (5qCNN and 2q_stt scripts)

`--load-checkpoint <path/to/sac_..._steps.zip>` restores weights/optimizer/hyperparameters and continues (`reset_num_timesteps=False`, so the step counter and TB curves continue). Caveats that bite:

- **`--timesteps` is the target total**, not an increment — resuming a 440k checkpoint with `--timesteps 1000000` trains ~560k more.
- **The replay buffer is NOT in the `.zip`.** Old checkpoints resume with an empty buffer (warm-up refill; weights are intact). Newer runs save `*_replay_buffer.pkl` beside the checkpoint (`CheckpointCallback(save_replay_buffer=True)`), and the resume code auto-loads it if present.
- `--lr` / `--buffer-size` are read from the checkpoint on resume, not the CLI.
- **The checkpoint's reward function must match the code's current reward**, or the critic's Q-values are miscalibrated against the new rewards (instability, not a clean continuation).

## Tests and checks

The root `test_*.py` and `check_*.py` files are **standalone scripts**, not pytest: `python test_integration.py`, `python check_dependencies.py`, etc. `tests/pipeline_carla_test.py` is **not a test** despite the name and path — it is the core library module (the env, wrappers, and callbacks) imported by every experiment.

## Architecture (the parts that span files)

**The wrapper stack is order-sensitive** and assembled in each experiment's `create_*_env()`:

```
CarlaGymEnv                    (tests/pipeline_carla_test.py)  — raw CARLA API gym env
  → PipelineObservationWrapper (STT variants only)            — frames → 512-dim embedding
  → CBFSafetyLayerWrapper                                     — QP action correction (gym.ActionWrapper)
  → [OcclusionWrapper / ImageObservationWrapper]              — CNN variants reshape to (3,84,84)
```

SB3 then auto-wraps this in a `DummyVecEnv`, which breaks `.unwrapped`. Use the helpers `_unwrap_vec_env(model)` and `_find_carla_and_cbf(model)` (in `pipeline_carla_test.py`) to reach the real `CarlaGymEnv` / `CBFSafetyLayerWrapper` — they traverse `.envs[0]` then the `.env` chain. Callbacks rely on these.

**The trust-score control loop** is the non-obvious heart of the system and spans three files:
1. The vendored SAC critic ensemble (`stable_baselines3/sac/`) returns `(mean_q, uncertainty, trust_score)` from `critic.predict(...)` — upstream SB3 returns only `mean_q`. Uncertainty is the variance across the K critics; trust ≈ `exp(-λ·uncertainty)`.
2. `PolicyTrustScoreCallback` reads this live each ~100 steps and calls `cbf_wrapper.set_trust_score(...)`.
3. `CBFSafetyLayer.compute_safe_action(u_actor, state, trust_score)` (`commons/cbfQP_layer.py`) modulates constraint tightness by trust — lower trust → more aggressive correction. If the callback isn't registered, trust stays 1.0.

**CBF layer** solves a small OSQP quadratic program: minimize `‖u − u_actor‖²` subject to collision (`d ≥ d_min`), lane (`|y| ≤ y_max`), speed (`v ≤ v_max`, queried dynamically from the CARLA waypoint), and steer/throttle rate limits. The env feeds it state via `CarlaGymEnv.build_cbf_state()` → `{d_collision, ttc, lane_offset, speed, speed_limit}`.

**Perception pipeline** (`models/pipeline.py`, `Pipeline.from_defaults(...)`): ResNet50 trunk (frozen, `(2048,7,7)` maps) → `SpatioTemporalEncoder` → stacked transformer, producing one 512-dim embedding per 8-frame window. Only the encoder + transformer train; the ResNet50 is frozen.

## Reward function — read this before changing it

`CarlaGymEnv._compute_reward()` sums ~9 weighted dense components plus event terms (lane invasion, speeding, stalling, collision, completion). Two things to know:

- The summed components differ by orders of magnitude (e.g. `movement` is weighted ×8; `yielding` ×0.4), so a few terms dominate the gradient. Per-component means are logged under `reward_components/*_ep_mean` in TensorBoard — use those to see what actually drives the policy, and confirm they roughly sum to the per-step reward.
- Terminal magnitudes are large relative to dense reward (collision is a large negative terminal; completion a large positive one). When changing terminal scalars, keep them in a sane ratio to the dense reward or the critic's value estimate becomes dominated by rare events. **Collision detection/termination logic and waypoint-progress shaping are treated as load-bearing — change their *scalars* deliberately, not their logic.**

The action path in `step()` enforces throttle/brake mutual exclusion with a deadzone; near SAC's centered output this can resolve to coast/brake and starve the car of throttle — a known cause of "creep" / "won't move" behavior. The stall safeguards are: a `stalling` reward penalty after `_episode_step > 60` at `speed < 1.0` on a clear road, and a hard `"stuck"` termination at `_episode_step > 120` with `speed < 1.0`.

## Git / branches

History contains several revert/reapply cycles of a "Driving curriculum" change — the net state of a branch may *not* contain work that a commit message implies (e.g. the curriculum/lane-aligned-reward work lives in commit `73782fc`, but was reverted on `updates` HEAD). Verify with `git log -S '<symbol>'` before assuming a fix is present on a given branch. End commit messages with the Co-Authored-By trailer.
