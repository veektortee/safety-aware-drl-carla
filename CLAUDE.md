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

- `2qCNNsac_experiment.py`, `5qCNNsac_experiment.py` — 84×84 camera + 26-dim nav vector → `MultiInputPolicy`. The 5q script also takes `--cnn {nature,resnet50}` (default `nature` = SB3's NatureCNN) and `--resnet-train {head,layer4,all}` (default `layer4`): the complete ImageNet ResNet-50 (stem, layer1-4, avgpool, fc re-headed 2048→256) replaces NatureCNN with the same 256-d output, so the SAC decision layers are unchanged (`commons/resnet_extractor.py`). ResNet runs use their own log dir / checkpoint prefix (`5q_cnn_resnet_<mode>`; `train_all.py --only 5q_cnn_resnet_layer4`) and must start fresh
- `2qsac_stt_experiment.py`, `5qsac_stt_experiment.py` — cached per-frame ResNet50 codes (8 frames) + nav vector → `MultiInputPolicy` with the trainable SpatioTemporal Transformer as its features extractor (`commons/stt.py`)

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
- **Only the latest checkpoint is kept** (`LatestCheckpointCallback`, every 25k total steps: zip + replay buffer + `*_curriculum_<N>_steps.json`, written atomically, older ones deleted afterwards).
- **A resume erases everything logged after the checkpoint**: TensorBoard events (rewritten in place, works while TensorBoard is running), `episodes.jsonl` / `eval.jsonl` records, and the curriculum state is restored from the checkpoint's snapshot (`discard_logs_after`, `restore_curriculum_snapshot`, `mark_tensorboard_resume` in `commons/warmup.py`). TensorBoard continues the same run from the checkpoint step.
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
  → CnnNavObservationWrapper (commons/obs_wrappers.py) / SttNavObservationWrapper (commons/stt.py) — image or frame codes + nav, occlusion
```

Supporting modules: `commons/route.py` + `commons/navigation/` (CARLA GlobalRoutePlanner routes, route projection), `commons/perception.py` (lead gap from the actor list + ground-filtered LiDAR along the curved route corridor), `commons/traffic_rules.py` (red lights, stop signs, lane-change legality), `commons/expert.py` (autopilot used for the replay-buffer warm-up), `commons/vehicle_model.py` (measured CARLA dynamics), `commons/curriculum_stages.py`, `commons/metrics_callbacks.py`.

**Maps**: `CarlaGymEnv(maps="all")` (the default) rotates through every installed town, one map per town with the `_Opt` build preferred (`client.get_available_maps()`; here Town01-05 and Town10HD — `D:\WORK\Carla\HDMaps` only holds point clouds). Training switches to the next town every `map_rotate_every=10` episodes; each eval round spreads its 10 fixed episodes over all towns (reset options `map_slot`/`map_slots`). `_load_map` drops to asynchronous mode for the load and waits until the server reports the new town (in synchronous mode `load_world` can return the old world). Episode records carry `map`.

**Curriculum** (`commons/curriculum_stages.py`): 6 stages — S0 empty short routes, S1 empty full routes, S2 15 cars/15 pedestrians, S3 60/60, S4 150/150, S5 300/300 (cars capped by each map's spawn points minus a 10% reserve; `--num-npc` / `--num-pedestrians` are maxima, default 300). Promotion at >= 80% success over 50 episodes, and at the latest at total steps 15k/35k/65k/100k/140k, so the whole curriculum fits in 200k steps; a resumed run starts at least at the stage the schedule allows for its step count.

**Demo** (`demo/run_model_inference.py`): one episode per town, each run to its natural end (no step cap), with game-style checkpoint rings every 30 m (`demo/checkpoint_rings.py`): HUD ovals in the server window (`draw_hud_line`, not seen by camera sensors) and projected rings + HUD in the camera window via `CarlaGymEnv.display_overlay` (display only, the observation is untouched).

SB3 then auto-wraps this in a `DummyVecEnv`, which breaks `.unwrapped`. Use the helpers `_unwrap_vec_env(model)` and `_find_carla_and_cbf(model)` (in `pipeline_carla_test.py`) to reach the real `CarlaGymEnv` / `CBFSafetyLayerWrapper` — they traverse `.envs[0]` then the `.env` chain. Callbacks rely on these.

**The trust-score control loop** is the non-obvious heart of the system and spans three files:
1. The vendored SAC critic ensemble (`stable_baselines3/sac/`) returns `(mean_q, uncertainty, trust_score)` from `critic.predict(...)` — upstream SB3 returns only `mean_q`. Uncertainty is the variance across the K critics; trust = `exp(-λ·std(Q)/(|mean(Q)|+1))` (scale-free, λ=2).
2. `PolicyTrustScoreCallback` reads this live each ~100 steps and calls `cbf_wrapper.set_trust_score(...)`.
3. `CBFSafetyLayer.compute_safe_action(u_actor, state, trust_score)` (`commons/cbfQP_layer.py`): lower trust only tightens the barrier parameters (larger standoff/reaction time, smaller γ); it never blends actions. If the callback isn't registered, trust stays 1.0.

**CBF layer** is a discrete-time CBF filter on `[steer, accel]` in physical units: a hard stopping-distance collision barrier (gap + lead speed; red-light stop lines count as obstacles when stoppable), a soft speed-limit row and soft signed lane rows (lookahead/HOCBF), with slack. It passes the action through untouched when no constraint is active and brakes fully when infeasible. The wrapper penalises the reward by `0.2·min(1,‖Δu‖²)` only on steps where it intervened; SB3 still stores the policy's proposed action. State comes from `CarlaGymEnv.build_cbf_state()` (the snapshot from the end of the previous step).

**STT perception** (`commons/stt.py`). Env side, frozen: each camera frame goes once through ImageNet ResNet50 at 224×128 → `(2048,4,7)` → a PCA projection to `(32,4,7)` float16; the wrapper caches the last 8 codes (repeat-padded after reset) and returns `{frames (8,32,4,7), nav}`. Policy side, trained by the SAC losses: `SttFeaturesExtractor` = `SpatioTemporalEncoder` (D=128, 2 layers) + `cnn_proj` + `StackedHierarchicalTransformer` + LayerNorm, concatenated with nav. All trainable weights are in the SAC `.zip`; the PCA is fitted once per fresh run on 1000 expert-driving frames in the full-traffic stage and stored as `model.stt_projection` (also in the `.zip`) — resume/demo call `wrapper.set_projection(model.stt_projection)`. ImageNet ResNet50 weights: `pretrained/feature_extractor/pretrained_resnet50.pth` (gitignored; downloaded once from torchvision if missing). `models/pipeline.py` / `PipelineObservationWrapper` are the old, untrained pipeline and are no longer used by the experiments.

## Reward function — read this before changing it

Reward v3 (`compute_reward_v3` in `tests/pipeline_carla_test.py`, constants in `RewardConfigV3`): `+1·clip(Δs_route/(v_limit·dt), −1, 1)` progress along the planned route, `−0.2`/step time cost, small speeding/lateral/heading penalties (capped at −0.5/step), −1 per illegal lane crossing, −10 per stop-sign violation; terminal +50 at the route end, crash −C (C = 50/100/150/200/250/300 by curriculum stage, ×2 for pedestrians, scaled by speed), and −C for red-light runs, off-route (1 s) and wrong-way (2 s). Timeout and "blocked without a reason" (15 s stopped with no lead obstacle/red light) are **truncations** (bootstrapped, no penalty). Two invariants to keep when changing it:

- **No alive bonus.** Progress is a finite supply, so finishing sooner always scores higher; a positive per-step reward makes the agent avoid every termination including the goal, a negative one with a cheap exit makes it seek exits (both happened in earlier versions: 95% crashes, then 93–97% parking).
- **A crash must cost more than any non-crash outcome**, but not so much early on that parking beats learning to drive — hence the stage-scaled C.

Actions are `[steer, accel]`; `accel ≥ 0` is throttle, `< 0` brake (no deadzone). The speed limit comes from `vehicle.get_speed_limit()` (`carla.Waypoint` has no speed limit). Sensor events are frame-stamped and anything from before the current episode's start frame is ignored (this fixed the "respawns into its own crash" bug).

## Git / branches

History contains several revert/reapply cycles of a "Driving curriculum" change — the net state of a branch may *not* contain work that a commit message implies (e.g. the curriculum/lane-aligned-reward work lives in commit `73782fc`, but was reverted on `updates` HEAD). Verify with `git log -S '<symbol>'` before assuming a fix is present on a given branch. Do not add Claude / Claude Code as author or co-author on commits.
