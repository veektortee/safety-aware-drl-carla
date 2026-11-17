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

- `2qCNNsac_experiment.py`, `5qCNNsac_experiment.py` — 84×84 camera + 26-dim nav vector → `MultiInputPolicy`
- `2qsac_stt_experiment.py`, `5qsac_stt_experiment.py` — ResNet50 → SpatioTemporal Transformer embedding (512) + nav vector → `MlpPolicy`

```bash
# From the repo root. NOTE: default --port differs per script (e.g. 2000 vs 2004) — match your server.
python experiments/5qCNNsac_experiment.py --port 2000 --timesteps 1000000 --render
python train_all.py --only 2q_cnn            # runner with auto-resume; --render is on by default
```

Common flags: `--timesteps` (target TOTAL steps), `--port`, `--log-dir`, `--lr`, `--batch-size`, `--buffer-size`, `--render`, `--num-npc`, `--num-pedestrians`, `--warmup-steps` (expert autopilot transitions pre-filled into the replay buffer, default 20000), `--start-stage` / `--no-curriculum`.

Outputs go to `<log-dir>/{tensorboard,checkpoints}`, `episodes.jsonl`, `eval.jsonl`, `run_config.json`, `curriculum_state.json` and a final `sac_<tag>_final.zip`. The console prints one line per episode and a rolling-100 summary; metric definitions are in `METRICS.md`; `python make_report.py` builds an HTML report.

### Resume from a checkpoint

`--load-checkpoint <path/to/sac_..._steps.zip>` restores weights/optimizer/hyperparameters and continues (`reset_num_timesteps=False`, so the step counter and TB curves continue). Caveats that bite:

- **`--timesteps` is the target total**, not an increment — resuming a 440k checkpoint with `--timesteps 1000000` trains ~560k more.
- **The replay buffer is NOT in the `.zip`.** Runs save `*_replay_buffer_<N>_steps.pkl` beside the checkpoint and the resume code auto-loads it (both file-name patterns are tried).
- Checkpoints from before reward v3 (3-D action, image-only observation) are **incompatible** with the current env.
- `--lr` / `--buffer-size` are read from the checkpoint on resume, not the CLI.
- **The checkpoint's reward function must match the code's current reward**, or the critic's Q-values are miscalibrated against the new rewards (instability, not a clean continuation).

## Tests and checks

Offline unit tests (no CARLA needed): `python tests/unit/test_<name>.py` (each file has its own runner; pytest is not in the venv). The root `test_*.py` / `check_*.py` files are older **standalone scripts** written for the previous env API and may not run. `tests/pipeline_carla_test.py` is **not a test** despite the name and path — it is the core library module (the env, wrappers, and callbacks) imported by every experiment.

## Architecture (the parts that span files)

**The wrapper stack is order-sensitive** and assembled in each experiment's `create_*_env()`:

```
CarlaGymEnv              (tests/pipeline_carla_test.py)  — obs {rgb_data, nav}, action [steer, accel]
  → CBFSafetyLayerWrapper (commons/cbf_wrapper.py)       — discrete-time CBF on the action
  → CnnNavObservationWrapper / SttNavObservationWrapper (commons/obs_wrappers.py) — image/embedding + nav, occlusion
```

Supporting modules: `commons/route.py` + `commons/navigation/` (CARLA GlobalRoutePlanner routes, route projection), `commons/perception.py` (lead gap from the actor list + ground-filtered LiDAR along the curved route corridor), `commons/traffic_rules.py` (red lights, stop signs, lane-change legality), `commons/expert.py` (autopilot used for the replay-buffer warm-up), `commons/vehicle_model.py` (measured CARLA dynamics), `commons/curriculum_stages.py`, `commons/metrics_callbacks.py`.

SB3 then auto-wraps this in a `DummyVecEnv`, which breaks `.unwrapped`. Use the helpers `_unwrap_vec_env(model)` and `_find_carla_and_cbf(model)` (in `pipeline_carla_test.py`) to reach the real `CarlaGymEnv` / `CBFSafetyLayerWrapper` — they traverse `.envs[0]` then the `.env` chain. Callbacks rely on these.

**The trust-score control loop** is the non-obvious heart of the system and spans three files:
1. The vendored SAC critic ensemble (`stable_baselines3/sac/`) returns `(mean_q, uncertainty, trust_score)` from `critic.predict(...)` — upstream SB3 returns only `mean_q`. Uncertainty is the variance across the K critics; trust = `exp(-λ·std(Q)/(|mean(Q)|+1))` (scale-free, λ=2).
2. `PolicyTrustScoreCallback` reads this live each ~100 steps and calls `cbf_wrapper.set_trust_score(...)`.
3. `CBFSafetyLayer.compute_safe_action(u_actor, state, trust_score)` (`commons/cbfQP_layer.py`): lower trust only tightens the barrier parameters (larger standoff/reaction time, smaller γ); it never blends actions. If the callback isn't registered, trust stays 1.0.

**CBF layer** is a discrete-time CBF filter on `[steer, accel]` in physical units: a hard stopping-distance collision barrier (gap + lead speed; red-light stop lines count as obstacles when stoppable), a soft speed-limit row and soft signed lane rows (lookahead/HOCBF), with slack. It passes the action through untouched when no constraint is active and brakes fully when infeasible. The wrapper penalises the reward by `0.2·min(1,‖Δu‖²)` only on steps where it intervened; SB3 still stores the policy's proposed action. State comes from `CarlaGymEnv.build_cbf_state()` (the snapshot from the end of the previous step).

**Perception pipeline** (`models/pipeline.py`, `Pipeline.from_defaults(...)`): ResNet50 trunk (frozen, `(2048,7,7)` maps) → `SpatioTemporalEncoder` → stacked transformer, producing one 512-dim embedding per 8-frame window. Only the encoder + transformer train; the ResNet50 is frozen.

## Reward function — read this before changing it

Reward v3 (`compute_reward_v3` in `tests/pipeline_carla_test.py`, constants in `RewardConfigV3`): `+1·clip(Δs_route/(v_limit·dt), −1, 1)` progress along the planned route, `−0.2`/step time cost, small speeding/lateral/heading penalties (capped at −0.5/step), −1 per illegal lane crossing, −10 per stop-sign violation; terminal +50 at the route end, crash −C (C = 50/100/150/300 by curriculum stage, ×2 for pedestrians, scaled by speed), and −C for red-light runs, off-route (1 s) and wrong-way (2 s). Timeout and "blocked without a reason" (15 s stopped with no lead obstacle/red light) are **truncations** (bootstrapped, no penalty). Two invariants to keep when changing it:

- **No alive bonus.** Progress is a finite supply, so finishing sooner always scores higher; a positive per-step reward makes the agent avoid every termination including the goal, a negative one with a cheap exit makes it seek exits (both happened in earlier versions: 95% crashes, then 93–97% parking).
- **A crash must cost more than any non-crash outcome**, but not so much early on that parking beats learning to drive — hence the stage-scaled C.

Actions are `[steer, accel]`; `accel ≥ 0` is throttle, `< 0` brake (no deadzone). The speed limit comes from `vehicle.get_speed_limit()` (`carla.Waypoint` has no speed limit). Sensor events are frame-stamped and anything from before the current episode's start frame is ignored (this fixed the "respawns into its own crash" bug).

## Git / branches

History contains several revert/reapply cycles of a "Driving curriculum" change — the net state of a branch may *not* contain work that a commit message implies (e.g. the curriculum/lane-aligned-reward work lives in commit `73782fc`, but was reverted on `updates` HEAD). Verify with `git log -S '<symbol>'` before assuming a fix is present on a given branch. Do not add Claude / Claude Code as author or co-author on commits.
