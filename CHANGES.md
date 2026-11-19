# Rework change log and architecture (591047a..feat/maps-curriculum)

This document records every code change of the November 2026 rework, including the stability fixes
of 2026-11-19, and explains how each module works now. It is written for two readers:

- **A coding agent**: every claim carries a repo-relative `path:line` reference, checked against
  branch `feat/maps-curriculum` at commit `dce8ace` (worktree `D:\safety-aware-drl-carla\wt-maps`; the
  commit that adds this file changes only Markdown, so its code is the same).
  Start from the module index in section 4 or the file-to-section map below.
- **A human**: section 1 is a one-table summary, section 2 says what behaved differently before and
  after, section 3 shows how the parts connect.

Read these first; this file does not repeat them:

| File | What it covers |
|---|---|
| [CLAUDE.md](CLAUDE.md) | Short architecture summary and the rules for changing the reward |
| [RUNNING.md](RUNNING.md) | How to set up, start CARLA, train each experiment, use the queue, run the demo |
| [METRICS.md](METRICS.md) | Definition of every logged number, TensorBoard tag and log file field |

Contents

1. [Summary by area](#1-summary-by-area)
2. [Before and after](#2-before-and-after)
3. [How the parts connect](#3-how-the-parts-connect)
4. [Module reference](#4-module-reference)
5. [Runner and training queue](#5-runner-and-training-queue)
6. [Compatibility and gotchas](#6-compatibility-and-gotchas)
7. [Stability incident of 2026-11-18: diagnosis and fixes](#7-stability-incident-of-2026-11-18-diagnosis-and-fixes)
8. [Commit index](#8-commit-index)

File-to-section map (for agents):

| Path | Section |
|---|---|
| `tests/pipeline_carla_test.py` | 4.1 |
| `commons/vehicle_model.py` | 4.2 |
| `commons/nav_layout.py`, `commons/debug.py` | 4.3 |
| `commons/cbfQP_layer.py` | 4.4 |
| `commons/cbf_wrapper.py` | 4.5 |
| `stable_baselines3/sac/policies.py`, `stable_baselines3/sac/sac.py` | 4.6 |
| `commons/navigation/` | 4.7 |
| `commons/route.py` | 4.8 |
| `commons/perception.py` | 4.9 (corridor offset: 4.1.4) |
| `commons/overtake.py` | 4.1.4 |
| `commons/traffic_rules.py` | 4.10 |
| `commons/expert.py` | 4.11 |
| `commons/obs_wrappers.py` | 4.12 |
| `commons/stt.py`, `commons/spatiotemporal_transformer.py` | 4.13 |
| `commons/resnet_extractor.py` | 4.14 |
| `commons/curriculum_stages.py` | 4.15 |
| `commons/curriculum.py` | 4.16 |
| `commons/warmup.py` | 4.17 |
| `commons/metrics_callbacks.py` | 4.18 |
| `experiments/*_experiment.py` | 4.19 |
| `make_report.py` | 4.20 |
| `demo/run_model_inference.py` | 4.21 |
| `demo/checkpoint_rings.py` | 4.22 |
| `tests/unit/` | 4.23 |
| Removed or retired code | 4.24 |
| `train_all.py` | 5.1, memory guard and stall watchdog 5.2 |
| `training_queue\run_queue.ps1`, `queue_order.txt` (outside the repo) | 5.3 |
| Changes of 2026-11-19 (stability fixes, training curriculum, runner guards, 5k checkpoints) | 4.1.1–4.1.4, 4.19, 5.2, 7 |

### Scope and baseline

- Range: `git log 591047a..bc4855d` has 39 commits. 30 of them (2026-11-16 to 2026-11-18, from
  `26946b8` to `bc4855d`) are the rework. The other 9 (2026-04-25 to 2026-09-21) are older commits of
  the `update-v1` line. `update-v1` has its own root commit (`74b22ab`) and no common ancestor with
  `main`; the rework's first commit `26946b8` merged it into `main` (591047a) and resolved the
  conflicts in favour of `update-v1`.
- On 2026-11-19 four more commits followed on `feat/maps-curriculum`: three code commits `9b93e48`,
  `81fc91f`, `dce8ace` (stability fixes after the 2026-11-18 incident, section 7, and the opt-in
  training curriculum, section 4.1.4) and the commit that adds this file and updates RUNNING.md,
  CLAUDE.md and METRICS.md. `git diff --stat bc4855d..dce8ace`: 13 files, +1,046 / −38 lines.
- "Before" in this document means the tree at `26946b8` (main after the `update-v1` merge), which is
  the code the rework started from. "After" means `dce8ace`. The commit counts and diff stats in the
  first and last bullet of this list cover `591047a..bc4855d` only.
- Branch flow: `merge-update-v1` (26946b8, 9cc0e2f, a38ee61) → `fix/env`, `fix/cbf`, `fix/exp`,
  `fix/metrics`, `fix/nav` (in parallel) → merged into `fix/integration` → `fix/stt` →
  `feat/maps-curriculum`, which also merges `feat/5q-resnet`. `main` is still at `591047a`.
- `git diff --stat 591047a..bc4855d`: 146 files, +20,585 / −4,653 lines. Of that, 26946b8..bc4855d
  (the rework without the merged `update-v1` content) is 53 non-`.pyc` files, +13,494 / −5,203.

---

## 1. Summary by area

| Area | What changed | Branch: commits | Main files |
|---|---|---|---|
| Env and reward v3 | New `CarlaGymEnv`: pure reward function `compute_reward_v3`, action `[steer, accel]`, observation `{rgb_data, nav}`, frame-stamped sensors, ordered reset, one state snapshot per tick, terminations vs truncations, 60 s red-light allowance, Traffic Manager port per CARLA port, map rotation | fix/env: 3da9b1e; fix/integration: ba24d51, 63c1b23; feat/maps-curriculum: 0f2d2a2 | `tests/pipeline_carla_test.py` |
| Env stability (2026-11-19) | Town01 left out of `maps="all"`; a new Traffic Manager on a new port for every map load, the old one shut down first; NPC cars with broken physics destroyed after each tick and replaced at the next reset | feat/maps-curriculum: 9b93e48 | `tests/pipeline_carla_test.py` |
| Training curriculum (2026-11-19) | Opt-in env options, used by the 5q CNN, 2q STT and 5q STT scripts: 70% traffic fill, steering low-pass and steering-change penalty, safe-overtake bonus, perception corridor on the ego; reward 3.1 when a reward option is on. 2q CNN keeps the old behaviour | feat/maps-curriculum: 9b93e48, dce8ace | `tests/pipeline_carla_test.py`, `commons/overtake.py`, `commons/perception.py`, `commons/metrics_callbacks.py`, `experiments/*.py` |
| Shared interfaces | Measured vehicle model and pedal mapping, nav-vector layout, debug switch | merge-update-v1: a38ee61 | `commons/vehicle_model.py`, `commons/nav_layout.py`, `commons/debug.py` |
| CBF layer and wrapper | Discrete-time CBF on `[steer, accel]` in physical units, closed-form solve, passthrough when no constraint is active; new wrapper module with intervention penalty and per-episode stats | fix/cbf: e383c5a, 82bc4be | `commons/cbfQP_layer.py`, `commons/cbf_wrapper.py` |
| Trust score | `exp(−2·std(Q)/(|mean(Q)|+1))`, detached; trust only tightens the CBF; actor loss skips the critic's feature graph | fix/cbf: e9c072e; fix/stt: 11dc732 | `stable_baselines3/sac/policies.py`, `stable_baselines3/sac/sac.py`, `PolicyTrustScoreCallback` in `tests/pipeline_carla_test.py` |
| Navigation, route, perception, traffic rules, expert | Bundled CARLA GlobalRoutePlanner; route geometry and lane queries; lead gap from actors and LiDAR along the curved route; red lights, stop signs, lane-marking legality; rule-based expert driver | fix/nav: c127036, 1dd0032; feat/maps-curriculum: 247e89c | `commons/navigation/`, `commons/route.py`, `commons/perception.py`, `commons/traffic_rules.py`, `commons/expert.py` |
| Observation wrappers | Image (3,84,84) + nav dict for the CNN experiments; occlusion on the image only | fix/exp: 0470e13; trimmed in fix/stt: c001d68 | `commons/obs_wrappers.py` |
| Curriculum and map rotation | 6 stages up to 300 cars / 300 pedestrians, success gate plus a step schedule that fits 200k steps; every installed town except Town01 in rotation (since `9b93e48`) | a38ee61, fix/exp: 0470e13, feat/maps-curriculum: 0f2d2a2 | `commons/curriculum_stages.py`, `commons/curriculum.py`, `tests/pipeline_carla_test.py` |
| STT perception | Frozen ResNet50 + PCA codes cached per frame in the env; trainable transformer as the SAC features extractor; projection saved in the `.zip` | fix/stt: c001d68, 11dc732 | `commons/stt.py`, `commons/spatiotemporal_transformer.py`, `experiments/*_stt_experiment.py` |
| 5q ResNet-50 option | `--cnn resnet50 --resnet-train {head,layer4,all}` for the 5q CNN script | feat/5q-resnet: 6fdf5d4; merged in fc1975c | `commons/resnet_extractor.py`, `experiments/5qCNNsac_experiment.py`, `train_all.py` |
| Experiments on the v3 stack | All four scripts rebuilt around the same flow: expert warm-up, curriculum, shared callbacks | fix/exp: 0470e13, 14f9a28 | `experiments/*.py`, `commons/warmup.py` |
| Metrics, callbacks, run config, report | Per-episode records, rolling summaries, deterministic evaluation, run config, HTML report | fix/metrics: 16ec944 | `commons/metrics_callbacks.py`, `make_report.py`, `METRICS.md` |
| Checkpointing and resume | `--timesteps` is a target total; keep-latest checkpoints with replay buffer and curriculum snapshot, every 5k steps since `dce8ace` (was 25k); logs after the checkpoint erased on resume | fix/exp: 0470e13, 14f9a28; feat/maps-curriculum: 0f2d2a2, dce8ace | `commons/warmup.py`, `experiments/*.py` |
| `train_all.py` runner | Renamed from `train_all (1).py`; resumes from the newest checkpoint, starts/restarts CARLA, retries, gives up after 5 attempts without progress | fix/metrics: 16ec944; 63c1b23; 6fdf5d4 | `train_all.py` |
| Runner memory guard and stall watchdog (2026-11-19) | Each experiment in a Windows Job Object with an OS cap on committed memory (`--mem-hard-gb`, 16) plus a watchdog that kills it at `--mem-limit-gb` (14), or after `--stall-minutes` (15) without output, for a restart from the latest checkpoint; the runner's children die with it; `train_all.log`; decoded exit codes and peak memory in `runner.log` | feat/maps-curriculum: 81fc91f | `train_all.py` |
| Training queue | Windows scheduled task that runs experiments one at a time to 200k steps and removes itself; since 2026-11-19 it passes the memory-guard flags and writes its log file before the console | outside the repo | `D:\safety-aware-drl-carla\training_queue\run_queue.ps1`, `queue_order.txt` |
| Demo | Same stack as training, model type detected, live trust, one episode per town, checkpoint rings | 14f9a28, c001d68, 0f2d2a2, 247e89c | `demo/run_model_inference.py`, `demo/checkpoint_rings.py` |
| Unit tests | 152 offline `test_*` functions in 12 files, no CARLA server needed | b9de273, 3da9b1e, 0470e13, 16ec944, 1dd0032, c001d68, 6fdf5d4, 0f2d2a2, bc4855d, 9b93e48 | `tests/unit/` |
| Docs and housekeeping | RUNNING.md, requirements_train.txt, CLAUDE.md updates, stale import removed, `documentation_updates/` untracked; this file | 9cc0e2f, 328fad1, 9970ebd, bc4855d, and the docs commit of 2026-11-19 | `RUNNING.md`, `requirements_train.txt`, `CLAUDE.md`, `CHANGES.md`, `__init__.py`, `.gitignore` |

---

## 2. Before and after

### 2.1 Reward function

| | Before (`CarlaGymEnv._compute_reward` at 26946b8) | After (`compute_reward_v3`, `tests/pipeline_carla_test.py:374`) |
|---|---|---|
| Dense terms | 10 weighted terms summed every step: `alive` +0.5, `movement` ×8, `lane_centering` ×3.5, `waypoint_progress` ×4, `traffic_flow` ×1, `safe_following` ×4.5, `yielding` ×0.4, `safety_buffer` ×2.5, `containment` ×2.5, `obstacle_avoidance` ×3 | Progress `+1·clip(Δs/(v_limit·dt), −1, 1)` along the planned route, time cost −0.2/step, speeding −2·max(0, v/L−1), lateral −0.2 (deadband 0.5 m), heading −0.1; the three shaping penalties together are capped at −0.5/step |
| Events | Lane invasion −15, obstacle closer than 8 m −6, speeding −8·(v/L−1.08), stalling up to −3 | −1 per illegal lane crossing (legal crossings are free), −10 per stop-sign violation |
| Terminal | Completion +240 + 120·time-efficiency; a collision step returned −600 on its own | Goal +50 when < 3 m of route remain; crash −C·(0.5 + 0.5·min(v/L, 1.5)), ×2 for pedestrians; −C for red light, off-route (1 s), wrong-way (2 s). C = 50/100/150/200/250/300 by curriculum stage |
| Episode ends | Collision, endpoint (< 5 m to the last waypoint) and timeout all `terminated=True`; `truncated` was always False. "Stuck" (planar speed < 1 m/s after step 200) terminated with −50 | `endpoint`, `collision`, `collision_pedestrian`, `red_light`, `off_route`, `wrong_way` terminate. `timeout` and `blocked` (15 s stopped with no lead, light or stop line) are truncations: no penalty, SB3 bootstraps |
| Speed limit | `waypoint.get_speed_limit()`, which `carla.Waypoint` does not have; the exception was swallowed, so the limit stayed at the initial 15 m/s | `ego.get_speed_limit()` km/h → m/s, clipped to [5, 25] m/s (`clip_speed_limit`, `tests/pipeline_carla_test.py:151`) |
| Where | Methods on the env, reading sensor flags | A pure function of two snapshot dicts, unit-tested without CARLA (`tests/unit/test_env_reward.py`) |

The design rules behind v3 (no alive bonus, crash must cost more than any non-crash outcome) are in
CLAUDE.md, "Reward function". The constants are class attributes of `RewardConfigV3`
(`tests/pipeline_carla_test.py:78`), which `CarlaGymEnv` inherits, so `run_config.json` records them.

**Reward 3.1 (opt-in, 2026-11-19).** With `steer_change_penalty` or `overtake_bonus` on, two shaping
terms are added to v3: `steer_change` = −w·(Δ steering command)² and `overtake` = +B per safe pass
(section 4.1.4). Everything else, including the terminal −C for a red-light run, is v3 unchanged. The
5q CNN, 2q STT and 5q STT scripts train with 3.1 (w = 0.5, B = 5); 2q CNN stays on 3.0.

### 2.2 Action space

| | Before | After |
|---|---|---|
| Space | `Box([-1, 0, 0], [1, 1, 1])` = `[steer, throttle, brake]` | `Box([-1, -1], [1, 1])` = `[steer, accel]` (`tests/pipeline_carla_test.py:660`) |
| Pedals | `step()` forced mutual exclusion: the larger pedal won, the other was zeroed, and the winner was zeroed below 0.1 (deadzone). Near SAC's initial output this gave coast/brake | `accel ≥ 0` → throttle = accel, `accel < 0` → brake = −accel (`accel_to_pedals`, `commons/vehicle_model.py:50`). No deadzone |
| Feedback | none | The executed (post-CBF) action is fed back in the nav vector (`prev_steer`, `prev_accel`) |

### 2.3 Observation

| | Before | After |
|---|---|---|
| Env | `Dict(rgb_data 360×640×3, lidar_data (3,500), position (3), speed (1), heading (1))` | `Dict(rgb_data 360×640×3 uint8 BGR, nav (26,) float32)` (`tests/pipeline_carla_test.py:665`). LiDAR is no longer in the observation; it feeds the obstacle gap |
| CNN experiments | `ImageObservationWrapper` → `Box(3,84,84)`, image only, SB3 `CnnPolicy` | `CnnNavObservationWrapper` → `Dict(image (3,84,84) uint8, nav (26,))`, SB3 `MultiInputPolicy` (NatureCNN + flattened nav, or ResNet-50 with `--cnn resnet50`) |
| STT experiments | `PipelineObservationWrapper` → `Box(512)` embedding from a pipeline run in eval mode under `no_grad`, so the transformer never trained; `MlpPolicy` | `SttNavObservationWrapper` → `Dict(frames (8,32,4,7) float16, nav (26,))`; the transformer is the policy's features extractor and is trained by the SAC losses |

The 26 nav features and their normalisation are listed in `commons/nav_layout.py:9`.

### 2.4 CBF formulation

| | Before (`commons/cbfQP_layer.py` at 26946b8) | After (`commons/cbfQP_layer.py`) |
|---|---|---|
| Action | 3-D `[steer, throttle, brake]` | 2-D `[steer, accel]`; accel converted to m/s² with the measured `AccelModel` |
| Constraints | Linearised "Lie derivative" rows for collision (`d ≥ d_min`), lane (`|offset| ≤ y_max`, unsigned offset) and speed, always passed to OSQP | Hard collision row from a stopping-distance barrier with an exact one-step bound; soft speed row; soft signed lane rows with lookahead (HOCBF), all with slack |
| When it acts | OSQP solved every step; result then rate-limited (steer change ≤ 0.5/step, pedal change ≤ 1.9/step) | Rows are "active" only if the actor's own command violates them. No active row → the action is returned bit-exact. No rate limiting, no blending |
| Infeasible | Solver failure → previous action, or `[0, 0.1, 0]` if that violated a row | Collision row infeasible → full brake (`accel = −1`); at standstill → "hold" |
| Solver | OSQP | Exact closed form of the separable QP (`_solve_1d`); `solver='osqp'` kept as a reference path |
| Reward penalty | `0.003·‖Δu‖` whenever `‖Δu‖ > 0.01` | `0.2·min(1, ‖Δu‖²)` only on steps where a row was active (clipping alone does not count) |
| Parameters | `alpha=1.0, d_min=1.2, y_max=1.9, v_max=25` | `DEFAULT_PARAMS` (`commons/cbfQP_layer.py:67`); legacy keyword arguments are accepted and ignored with a warning |

### 2.5 Trust score

| | Before | After |
|---|---|---|
| Formula | `trust = exp(−0.01·var(Q))` over the K critics; default `trust_lambda=0.01`. It saturated to about 0 or 1 depending on the Q magnitude | `trust = exp(−λ·std(Q)/(|mean(Q)|+1))`, λ = 2.0, computed on detached values (`stable_baselines3/sac/policies.py:117`) |
| Effect on the CBF | Output blended toward a fixed fallback `[0, 0.25, 0.2]` with weight `0.3·(1 − trust)`, so low trust changed the action even when nothing was unsafe | Trust only tightens the barrier parameters: γ_c and γ_2 grow with trust (0.1 to 0.3), standoff +1.0 m·(1−trust), reaction time +0.2 s·(1−trust), lane bounds −0.2 m·(1−trust). With no active row the action is untouched |
| Uncertainty | Variance across critics | Unchanged: variance across critics (`train/ensemble_uncertainty`) |

### 2.6 Checkpoint and resume

| | Before | After |
|---|---|---|
| `--timesteps` in the scripts | Increment: SB3 adds it to the restored counter (`reset_num_timesteps=not bool(load_checkpoint)`); the old runner computed `remaining` and passed that | Target total: scripts train `target − checkpoint steps` and exit 0 if already reached; the runner passes the target unchanged |
| Checkpoint frequency | `CheckpointCallback` every 100k (2q CNN) or 10k (other three); all kept | `LatestCheckpointCallback` every 5k total steps (`SAVE_FREQ`, 25k until `dce8ace`); zip, replay buffer and curriculum snapshot written atomically; older ones deleted afterwards (keep 1) |
| Replay buffer on resume | The 2q CNN script looked only for `<ckpt>_replay_buffer.pkl`, while `CheckpointCallback` writes `<prefix>_replay_buffer_<N>_steps.pkl`; the other three tried both names. A failing load was not caught | Both names tried in all four; unreadable or mismatched buffers skipped; a restored buffer skips the expert warm-up |
| Crash or Ctrl+C | The exception was printed and `<log-dir>/sac_<tag>_final.zip` was saved (weights only; not a name the runner resumes from) | A resumable step checkpoint (zip + buffer) is saved and the script returns 1 |
| Logs after the checkpoint | Left in place, so curves showed steps the restored model never saw | Erased: TensorBoard events, `episodes.jsonl`, `eval.jsonl`; curriculum state restored from the checkpoint's snapshot |
| Model-type checks | none | Space mismatch → clear error; 5q CNN refuses a checkpoint of another image branch; STT needs `stt_projection` in the `.zip` |

### 2.7 Routes, sensors and episode boundaries

| | Before | After |
|---|---|---|
| Route | 40 waypoints 5 m apart from the spawn, always taking the first branch (`_generate_waypoints`) | CARLA GlobalRoutePlanner route to a random spawn point, length window by stage (80–150 m or 200–400 m), glitch and length checks (`Route.plan`, `commons/route.py:233`) |
| Sensors | RGB camera, LiDAR at 10 Hz, depth camera, obstacle sensor, collision, lane invasion; callbacks set flags | RGB camera, LiDAR at 20 Hz (one full sweep per tick, ground and ego filtered), collision, lane invasion. Every message is queued with its simulator frame |
| Reset | `collision_occurred` was cleared before the teleport and the settle ticks, so a late collision event from the previous crash could end the next episode at step 0 | Brake and zero velocity, teleport, settle in first gear, then flush the event queues and set the episode start frame; only events with `start_frame < frame ≤ current frame` count |
| Maps | The server's current map | Every installed town except Town01 (`_Opt` build preferred), next town every 10 training episodes; a new Traffic Manager for every map load |
| NPC physics | Not checked (also not at `bc4855d`) | Cars the physics engine launched, sank under the road or sped past 40 m/s are destroyed after the tick that shows them and replaced at the next reset (section 4.1.3) |

---

## 3. How the parts connect

### 3.1 Wrapper stack and one training step

The stack is built in each experiment's `create_*_env()` (`experiments/2qCNNsac_experiment.py:65`,
`experiments/2qsac_stt_experiment.py:76`). SB3 then wraps it in `Monitor` and `DummyVecEnv`.

```
SAC policy (MultiInputPolicy; NatureCNN / ResNetNavExtractor / SttFeaturesExtractor)
  │ action a = [steer, accel]                           ▲ obs Dict(image|frames, nav), reward, done, info
  ▼                                                     │
DummyVecEnv → Monitor                                   │
  → CnnNavObservationWrapper | SttNavObservationWrapper │  (action passes through; obs transformed)
    → CBFSafetyLayerWrapper                             │  state = env.build_cbf_state()  (snapshot of the previous tick)
        safe = cbf_layer.compute_safe_action(a, state, trust)
        r -= 0.2·min(1, ‖safe − a‖²) if a row was active; info['cbf'] = {...}
      → CarlaGymEnv.step(safe)
          accel_to_pedals → apply_control → world.tick() → wait for the frame's camera image and LiDAR sweep
          drain collision / lane events of this episode → _build_snapshot() (route, lane, lead, light, stop sign)
          compute_reward_v3(snap, prev) → obs {rgb_data, nav}, reward, terminated, truncated, info
```

SB3 stores the action the policy proposed (`a`), not `safe`; the intervention penalty is what teaches
the policy to stay inside the safe set (`commons/cbf_wrapper.py:15-16`).

Because of the `DummyVecEnv`, `.unwrapped` on the SB3 env does not reach `CarlaGymEnv`. Callbacks use
`_find_carla_and_cbf(model)` (`tests/pipeline_carla_test.py:1718`), which walks `.envs[0]` and then
the `.env` chain.

### 3.2 Trust-score loop (three files)

1. `EnsembleCritic.ensemble_forward` (`stable_baselines3/sac/policies.py:81`) returns all K
   Q-values, their variance (uncertainty) and the trust score. `critic.predict`
   (`stable_baselines3/sac/policies.py:137`) returns `(mean_q, uncertainty, trust)`.
2. `PolicyTrustScoreCallback` (`tests/pipeline_carla_test.py:1805`) runs every 100 steps. It
   evaluates the critics on the actor's deterministic action for the newest observation
   (`_compute_live_policy_metrics`, `tests/pipeline_carla_test.py:1777`) and calls
   `cbf_wrapper.set_trust_score(trust)`.
3. `CBFSafetyLayerWrapper` passes the trust to `CBFSafetyLayer.compute_safe_action`
   (`commons/cbfQP_layer.py:233`), which uses it only to tighten the barrier parameters
   (`commons/cbfQP_layer.py:399-403`, `commons/cbfQP_layer.py:443-446`).

If the callback is not registered, trust stays 1.0. The demo computes trust every step instead
(`demo/run_model_inference.py:146`). The training-batch average `train/trust_score` that SAC logs
(`stable_baselines3/sac/sac.py:369`) is a diagnostic only and is filtered from TensorBoard.

### 3.3 Episode-end record path

1. On the last step `CarlaGymEnv.step` builds the episode dict (`_episode_summary`,
   `tests/pipeline_carla_test.py:1495`) and puts it in `info['episode_metrics']` (and the same object
   in `info['episode']`, which SB3's `Monitor` overwrites).
2. `CBFSafetyLayerWrapper.step` merges `cbf_interventions`, `cbf_intervention_frac`, `cbf_infeasible`,
   `cbf_penalty_sum`, `trust_mean`, `cbf_rows` into that dict (`commons/cbf_wrapper.py:152-157`).
3. Readers: `EpisodeMetricsCallback` (`episodes.jsonl`, TensorBoard, console), `CurriculumCallback`
   (promotion), `prefill_replay_buffer` (warm-up summary), `EvalCallback` (eval records),
   the demo. Field definitions: METRICS.md.

### 3.4 Lifecycle of one experiment process

The four scripts share one flow (`experiments/2qCNNsac_experiment.py:86`):

1. If `--load-checkpoint`: read its step count from the zip without building a model; exit 0 if the
   target is reached; (5q CNN) refuse a different image branch; restore the curriculum snapshot and
   erase logs after the checkpoint step.
2. Choose the start stage (`resolve_start_stage`, `commons/curriculum.py:58`).
3. Build the env stack.
4. New `SAC(...)` or `SAC.load(...)`; on resume restore the replay buffer (STT: restore
   `stt_projection`; fresh STT: fit it first).
5. Replace the logger, mark the TensorBoard resume point, write `run_config.json`.
6. Expert warm-up (20,000 transitions by default) unless a buffer was restored.
7. `model.learn(remaining, reset_num_timesteps=False)` with callbacks, in this order:
   `LatestCheckpointCallback`, `EvalCallback`, `PolicyTrustScoreCallback`,
   `EpisodeMetricsCallback`, `CurriculumCallback`, `LogEveryNTimesteps`.
8. On Ctrl+C or an exception: save a step checkpoint and return 1. On success: save
   `<log-dir>/sac_<tag>_final.zip` and return 0. The env is closed in `finally`.

---

## 4. Module reference

Each subsection: purpose, key symbols, one or two verbatim excerpts, and what they do.

### 4.1 `tests/pipeline_carla_test.py`: the environment (reward v3)

**Purpose.** Despite its name and folder this is the core library module: the CARLA gym env, the
pure reward and helper functions, the trust callback and the unwrap helpers. Every experiment and
the demo import from it. It was rewritten in `3da9b1e` and extended in `ba24d51`, `63c1b23` and
`0f2d2a2`. The stability fixes of `9b93e48` (town exclusion, Traffic Manager per map load, NPC physics
guard) are described in 4.1.1–4.1.3, the opt-in training-curriculum options of the same commit
(70% traffic, steering smoothing and penalty, overtake bonus, corridor on the ego) in 4.1.4.

| Symbol | Where | Role |
|---|---|---|
| `RewardConfigV3` | `tests/pipeline_carla_test.py:78` | All reward, termination, reset and sensor constants; `REWARD_VERSION` "3.0" (`tests/pipeline_carla_test.py:81`) |
| `REWARD_COMPONENT_KEYS` | `tests/pipeline_carla_test.py:135` | Every reward component; `steer_change` and `overtake` added in `9b93e48` (always present, 0.0 when off) |
| `episode_step_limit` | `tests/pipeline_carla_test.py:156` | Per-route limit `clip((route/(0.4·v_limit) + 60 s)/dt, 600, time_limit)` |
| `drain_frame_events` | `tests/pipeline_carla_test.py:163` | Keeps only sensor events of the current episode and tick |
| `wait_for_frame` | `tests/pipeline_carla_test.py:188` | Blocks until the camera / LiDAR message of a frame arrives (2 s timeout, then reuses the last one) |
| `select_spawn_index` | `tests/pipeline_carla_test.py:206` | Spawn ≥ 10 m from vehicles, ≥ 5 m from walkers, ≥ 50 m from the last crash |
| `pick_town_maps` | `tests/pipeline_carla_test.py:242` | One map per installed town, `_Opt` preferred |
| `UNSTABLE_TOWNS` | `tests/pipeline_carla_test.py:264` | `("Town01",)`: towns left out of `maps="all"` (4.1.1) |
| `town_of` | `tests/pipeline_carla_test.py:267` | `'Carla/Maps/Town01_Opt'` → `'Town01'` |
| `eval_map` | `tests/pipeline_carla_test.py:280` | Town of evaluation slot i (contiguous blocks over all towns) |
| `vehicle_cap` | `tests/pipeline_carla_test.py:287` | Cars a map can hold: spawn points minus max(10, 10%) |
| `TRAINING_TRAFFIC_FILL` | `tests/pipeline_carla_test.py:299` | 0.7: the traffic fill the training curriculum passes (env default 1.0) (4.1.4) |
| `traffic_caps` | `tests/pipeline_carla_test.py:302` | (cars, walkers) a town may hold at a fill: `fill·vehicle_cap(spawn points)`, `fill·pedestrian maximum` |
| `TRAINING_STEER_SMOOTHING` … `TRAINING_OVERTAKE_BONUS` | `tests/pipeline_carla_test.py:311-313` | 0.6, 0.5, 5.0: the training curriculum's driving-quality defaults |
| `add_training_curriculum_args` | `tests/pipeline_carla_test.py:316` | Adds `--traffic-fill`, `--steer-smoothing`, `--steer-change-penalty`, `--overtake-bonus`, `--route-corridor` to a script's parser |
| `training_curriculum_kwargs` | `tests/pipeline_carla_test.py:335` | Those options as `CarlaGymEnv` keyword arguments |
| `NPC_MAX_SPEED_MPS` … `NPC_Z_ABOVE_ROADS_M` | `tests/pipeline_carla_test.py:347-350` | NPC physics guard thresholds: 40 m/s, 8 m/s vertical, 6 m below / 8 m above the road band |
| `npc_physics_broken` | `tests/pipeline_carla_test.py:353` | True for a launched, sunk, non-finite or absurdly fast NPC car (4.1.3) |
| `condition_flags` | `tests/pipeline_carla_test.py:362` | Off-route, wrong-way and blocked conditions of one snapshot |
| `compute_reward_v3` | `tests/pipeline_carla_test.py:374` | The reward, termination and truncation for one step |
| `cbf_gap` | `tests/pipeline_carla_test.py:450` | Gap for the CBF: lead obstacle, or a red/yellow stop line if nearer and still stoppable |
| `build_nav_vector` | `tests/pipeline_carla_test.py:474` | The 26 nav features in `NAV_FEATURES` order |
| `CarlaGymEnv` | `tests/pipeline_carla_test.py:523` | The env (inherits `RewardConfigV3`); constructor options `traffic_fill`, `steer_smoothing`, `steer_change_penalty`, `overtake_bonus`, `lead_corridor_follows_ego` default to the old behaviour (`tests/pipeline_carla_test.py:541`) |
| `TM_PORT_OFFSET` | `tests/pipeline_carla_test.py:529` | First Traffic Manager port = CARLA port + 6000 |
| `TM_PORT_STRIDE`, `TM_GENERATIONS` | `tests/pipeline_carla_test.py:535-536` | Each map load moves the Traffic Manager 100 ports up, wrapping after 50 (4.1.2) |
| `_resolve_maps` | `tests/pipeline_carla_test.py:685` | `maps="all"` → installed towns minus `UNSTABLE_TOWNS`; an explicit list is used as given |
| `_init_world_objects` | `tests/pipeline_carla_test.py:700` | Ego, sensors, route planner, rule trackers; records the map's road height band for the NPC guard |
| `_load_map` | `tests/pipeline_carla_test.py:737` | Map switch in asynchronous mode, retires the Traffic Manager, waits until the server reports the new town |
| `_target_map` | `tests/pipeline_carla_test.py:787` | Which town a reset must switch to |
| `_retire_traffic_manager` | `tests/pipeline_carla_test.py:811` | `set_synchronous_mode(False)` + `shut_down()` on the old Traffic Manager |
| `_setup_world` | `tests/pipeline_carla_test.py:822` | New Traffic Manager on this generation's port, synchronous mode, 20 FPS |
| `_attach_sensors` | `tests/pipeline_carla_test.py:872` | RGB 640×360, LiDAR 32 ch / 50 m / 20 Hz, collision, lane invasion |
| `_tick` | `tests/pipeline_carla_test.py:926` | `world.tick()`, NPC physics guard, then frame-matched camera and LiDAR |
| `_adjacent_lane_clear` | `tests/pipeline_carla_test.py:944` | Overtake check: a legal same-direction Driving lane beside the ego, free of vehicles 12 m behind to 25 m ahead |
| `_overtake_step` | `tests/pipeline_carla_test.py:976` | Runs `OvertakeTracker.update` for the step; never raises |
| `_remove_broken_npcs` | `tests/pipeline_carla_test.py:997` | Destroys the NPC cars `npc_physics_broken` flags; counts them for replacement |
| `_population_target` | `tests/pipeline_carla_test.py:1042` | Stage counts capped by the maxima and by `traffic_caps(spawn points, num_pedestrians, traffic_fill)` |
| `_apply_stage_population` | `tests/pipeline_carla_test.py:1052` | Respawns traffic when the stage counts change; otherwise tops up cars the guard removed |
| `_teleport_ego` | `tests/pipeline_carla_test.py:1178` | Brake, zero velocity, teleport, settle in manual first gear |
| `reset` | `tests/pipeline_carla_test.py:1198` | Map switch, stage traffic, spawn, flush, route, first snapshot |
| `_build_snapshot` | `tests/pipeline_carla_test.py:1272` | The single per-tick state dict |
| `build_cbf_state` | `tests/pipeline_carla_test.py:1361` | CBF input: the snapshot of the previous tick |
| `expert_action` | `tests/pipeline_carla_test.py:1372` | Expert `[steer, accel]` for the current snapshot |
| `step` | `tests/pipeline_carla_test.py:1382` | Steering low-pass, apply action, tick, events, snapshot, overtake check, reward, info |
| `_episode_summary` | `tests/pipeline_carla_test.py:1495` | The `episode_metrics` dict |
| `set_curriculum_stage` | `tests/pipeline_carla_test.py:1529` | Takes effect at the next reset |
| `PipelineObservationWrapper` | `tests/pipeline_carla_test.py:1598` | Old untrained STT wrapper; kept, no longer used by the experiments |
| `_find_carla_and_cbf` | `tests/pipeline_carla_test.py:1718` | Reach `CarlaGymEnv` and the CBF wrapper through SB3's wrappers |
| `PolicyTrustScoreCallback` | `tests/pipeline_carla_test.py:1805` | Pushes live trust into the CBF wrapper every 100 steps |

The core of one step, from `tests/pipeline_carla_test.py:1385-1403`:

```python
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        cmd_steer = float(np.clip(a[0], -1.0, 1.0))
        accel = float(np.clip(a[1], -1.0, 1.0))
        k = self.steer_smoothing          # 0.0 = off (applied steer = command)
        steer = float(k * float(self._prev_action[0]) + (1.0 - k) * cmd_steer) if k > 0.0 else cmd_steer
        throttle, brake = accel_to_pedals(accel)
        self.ego_vehicle.apply_control(carla.VehicleControl(throttle=float(throttle), steer=steer,
                                                            brake=float(brake)))
        self._prev_action = np.array([steer, accel], dtype=np.float32)   # executed (post-CBF) action
        prev_cmd_steer, self._prev_cmd_steer = self._prev_cmd_steer, cmd_steer

        prev = self._snapshot
        frame = self._tick()
        self._episode_step += 1

        collisions = drain_frame_events(self._q_collision, self._episode_start_frame, frame)
        invasions = drain_frame_events(self._q_lane, self._episode_start_frame, frame)
        actors = self.world.get_actors()                       # the one get_actors() call of this tick
        snap = self._build_snapshot(actors)
```

The action that arrives here has already been through the CBF. With `steer_smoothing` > 0 (off by
default, 0.6 in the training curriculum, 4.1.4) the applied steer is low-passed; otherwise it is the
command. The accel is mapped to pedals, the control applied, and the world advances one 0.05 s tick.
`_tick` waits for the camera image and LiDAR sweep stamped with that frame (and runs the NPC physics
guard, 4.1.3). Collision and lane-invasion events are taken only if their frame lies after the
episode start frame, so events from the previous episode are dropped. One `get_actors()` call feeds
one snapshot, which is then used by the nav vector, the reward, the terminations, the next
`build_cbf_state()`, `expert_action()` and the overtake check.

The dense part of the reward, from `tests/pipeline_carla_test.py:390-401`:

```python
    comps["progress"] = cfg.W_PROGRESS * float(np.clip(ds / (L * DT), -1.0, 1.0))
    comps["time"] = cfg.TIME_PENALTY

    speeding = cfg.W_SPEEDING * max(0.0, v / L - 1.0)
    lateral = cfg.W_LATERAL * float(np.clip((abs(float(snap["e_lane"])) - cfg.LATERAL_DEADBAND_M)
                                            / cfg.LATERAL_RANGE_M, 0.0, 1.0))
    heading = cfg.W_HEADING * min(abs(float(snap["e_psi_lane"])) / (math.pi / 2.0), 1.0)
    dense = speeding + lateral + heading
    if dense < cfg.DENSE_PENALTY_CAP:
        scale = cfg.DENSE_PENALTY_CAP / dense      # keep the split, cap the total
        speeding, lateral, heading = speeding * scale, lateral * scale, heading * scale
    comps["speeding"], comps["lateral"], comps["heading"] = speeding, lateral, heading
```

Progress is the route arc length gained this tick, normalised so driving at the speed limit earns
+1 per step; reversing earns down to −1. Lateral and heading errors are measured against the lane
the car is in when that lane is legal (same direction, drivable), else against the route
(`tests/pipeline_carla_test.py:1304-1306`). The three penalties are scaled down together if their
sum would go below −0.5. The two reward-3.1 terms `steer_change` and `overtake` follow
(`tests/pipeline_carla_test.py:407-412`); they stay 0.0 unless the env options are on (4.1.4). The
terminal chain that follows (`tests/pipeline_carla_test.py:420-444`) checks pedestrian collision,
collision, red light, off-route, wrong-way and goal in that order, then the two truncations (timeout,
blocked). Running a red light has been a terminal −C since reward v3 (`red_light_run` →
`comps["red_light"] = -C`, `tests/pipeline_carla_test.py:426-428`); the 2026-11-19 changes leave it
as it is. Every component key is always present, and the components sum to the reward.

Map selection at reset, from `tests/pipeline_carla_test.py:787-799`:

```python
    def _target_map(self, options: Dict) -> Optional[str]:
        """The map this reset must switch to, or None to stay."""
        if options.get("map"):
            want = map_basename(options["map"])
        elif len(self._maps) < 2:
            return None
        elif options.get("map_slot") is not None:
            want = eval_map(self._maps, options["map_slot"], options.get("map_slots", 1))
        elif self._episodes_on_map >= self.map_rotate_every:
            want = next_map(self._maps, self.current_map_name)
        else:
            return None
        return None if want == self.current_map_name else want
```

Evaluation resets pass `map_slot`; training resets rotate after `map_rotate_every` (10) training
episodes. `reset` counts only training episodes toward the rotation
(`tests/pipeline_carla_test.py:1213-1214`), and `_load_map` resets the counter. A map switch destroys
and rebuilds the ego, sensors, traffic, the GlobalRoutePlanner, the traffic-light tracker and the
stop-sign monitor (`_init_world_objects`, `tests/pipeline_carla_test.py:700`), and since `9b93e48` also
the Traffic Manager (4.1.2).

#### 4.1.1 Town exclusion (`UNSTABLE_TOWNS`, `9b93e48`)

**Why.** Town01_Opt with S5 traffic failed in 7 of 8 reproductions on 2026-11-19, always inside
`world.tick()` (section 7). The fix leaves the town out; it does not repair it.

From `tests/pipeline_carla_test.py:258-270`:

```python
# Towns left out of maps="all" (training rotation, eval rounds, demo); an explicit maps=[...] still loads them.
# Town01_Opt with S5 traffic (230 cars, ~250 walkers) failed inside CARLA's native client code (Traffic Manager
# / walker navigation run in-process during world.tick()) in 7 of 8 reproductions on 2026-11-19, within 1-4
# minutes: unbounded memory growth (75 GB in training on 2026-11-18) or an access violation, with hybrid
# physics on or off, with or without walkers, at 140 or 230 cars, with or without a map switch (the 8th run
# lasted 6.7 minutes without failing). Town10HD at S5 ran for hours.
UNSTABLE_TOWNS = ("Town01",)


def town_of(map_name: str) -> str:
    """'Carla/Maps/Town01_Opt' -> 'Town01'."""
    name = map_basename(map_name)
    return name[:-4] if name.endswith("_Opt") else name
```

The filter is applied where `maps="all"` is resolved, from `tests/pipeline_carla_test.py:688-694`:

```python
        if isinstance(maps, str) and maps.lower() == "all":
            listing = getattr(self.client, "get_available_maps", None)
            found = pick_town_maps(listing()) if callable(listing) else []
            found = [m for m in found if town_of(m) not in UNSTABLE_TOWNS]
            return found or [self.current_map_name]
        names = [map_basename(m) for m in ([maps] if isinstance(maps, str) else maps)]
        return names or [self.current_map_name]
```

`maps="all"` is the default for training, the eval rounds and the demo, so on this machine the
rotation is Town02, Town03, Town04, Town05, Town10HD. An explicit list (`maps=["Town01_Opt"]`) still
loads Town01. The exclusion does not move a server that already has Town01 loaded: the env starts on
the server's current map and `_target_map` switches only after `map_rotate_every` training episodes
(10; the demo uses 1), for an eval slot or for `options['map']`. The next town is then the first of
the list, because `next_map` (`tests/pipeline_carla_test.py:273`) returns the first map when the
current one is not in the rotation. Test:
`test_all_maps_leaves_out_unstable_towns_but_explicit_lists_keep_them` in
`tests/unit/test_maps_curriculum.py`.

#### 4.1.2 A new Traffic Manager for every map load (`9b93e48`)

**Why.** In synchronous mode the Traffic Manager runs inside the experiment process and works during
`world.tick()`. After a Town10HD → Town01 switch, a Traffic Manager carried across `load_world()`
produced 6 launched or sunk cars in the first 7 s; one created on the already loaded map produced none.

From `tests/pipeline_carla_test.py:811-825`:

```python
    def _retire_traffic_manager(self):
        """Shut the current Traffic Manager down (its cars are already destroyed) so none survives a map load."""
        tm, self.traffic_manager = self.traffic_manager, None
        if tm is None:
            return
        try:
            tm.set_synchronous_mode(False)
            tm.shut_down()
        except Exception as exc:  # noqa: BLE001 - a new one on a new port is created after the load anyway
            print(f"[ENV] WARNING could not shut down the old Traffic Manager: {exc}")

    def _setup_world(self):
        # TM must be synchronous before the world is, or NPCs tick on a different clock.
        tm_port = self.port + self.TM_PORT_OFFSET + self.TM_PORT_STRIDE * self._tm_generation
        self.traffic_manager = self.client.get_trafficmanager(tm_port)
```

`_load_map` first destroys the old town's actors (`_release_world_actors`), then calls
`_retire_traffic_manager()` before `load_world()` (`tests/pipeline_carla_test.py:752`), and after the
load moves to the next generation and builds the new Traffic Manager on the loaded map
(`tests/pipeline_carla_test.py:768-769`). The first Traffic Manager of an env (generation 0) uses
CARLA port + 6000, each map load adds 100, and the generation wraps after 50
(`TM_PORT_STRIDE`, `TM_GENERATIONS`, `tests/pipeline_carla_test.py:535-536`): for CARLA on 2000 the
ports are 8000, 8100, …, 12900, then 8000 again. Two servers whose ports differ by a multiple of 100
would share Traffic Manager ports; with the runner's 10-port spacing that needs more than 10
parallel runs. `close()` does not shut the Traffic Manager down; it only switches it out of
synchronous mode, as before.

#### 4.1.3 NPC physics guard (`npc_physics_broken`, `_remove_broken_npcs`, `9b93e48`)

**Why.** In dense traffic PhysX sometimes launched cars into the air or dropped them through the road
(on Town01_Opt at S5: cars 40 m up, 60 m under the map, 56 m/s launches). Perception projects the
actor list onto the route (`find_lead`, `commons/perception.py:77`), so a car sunk under the road was a
phantom obstacle the CBF braked for.

The predicate, from `tests/pipeline_carla_test.py:347-359`:

```python
NPC_MAX_SPEED_MPS = 40.0        # no NPC legitimately drives this fast on these towns (highways are 90 km/h)
NPC_MAX_VERTICAL_MPS = 8.0      # launched or falling; driving over hills and kerbs stays well below
NPC_Z_BELOW_ROADS_M = 6.0       # this far below the map's lowest road = fell through
NPC_Z_ABOVE_ROADS_M = 8.0       # this far above the map's highest road = launched


def npc_physics_broken(vx: float, vy: float, vz: float, z: float, road_z_min: float, road_z_max: float) -> bool:
    """True for an NPC car the physics engine has launched, dropped through the road or sped up absurdly."""
    speed_sq = vx * vx + vy * vy + vz * vz
    if not math.isfinite(speed_sq) or not math.isfinite(z):
        return True
    return (speed_sq > NPC_MAX_SPEED_MPS ** 2 or abs(vz) > NPC_MAX_VERTICAL_MPS
            or z < road_z_min - NPC_Z_BELOW_ROADS_M or z > road_z_max + NPC_Z_ABOVE_ROADS_M)
```

The road band is the lowest and highest waypoint of the loaded map, sampled every 10 m when the town
is loaded (`tests/pipeline_carla_test.py:705-706`). The guard runs right after every `world.tick()`
(`_tick`, `tests/pipeline_carla_test.py:926`), from `tests/pipeline_carla_test.py:1001-1022`:

```python
        if not self.npc_vehicles:
            return 0
        snapshot = self.world.get_snapshot()
        z_min, z_max = self._road_z_range
        broken = []
        for vid in self.npc_vehicles:
            actor = snapshot.find(vid)
            if actor is None:
                continue
            v = actor.get_velocity()
            if npc_physics_broken(v.x, v.y, v.z, actor.get_transform().location.z, z_min, z_max):
                broken.append(vid)
        if not broken:
            return 0
        try:
            self.client.apply_batch_sync([carla.command.DestroyActor(i) for i in broken], False)
        except Exception as exc:  # noqa: BLE001 - never let the guard end an episode
            print(f"[ENV] WARNING could not destroy broken NPC cars {broken}: {exc}")
        gone = set(broken)
        self.npc_vehicles = [i for i in self.npc_vehicles if i not in gone]
        self._npc_removed_on_map += len(broken)
        self._npc_removed_pending += len(broken)   # replaced at the next reset
```

Velocities and positions come from one `world.get_snapshot()` per tick. Destroyed cars are counted in
`_npc_removed_pending` and topped up by `_apply_stage_population` at the next reset when the stage's
counts have not changed (`tests/pipeline_carla_test.py:1056-1061`); a full respawn or a map load resets
the counter (`tests/pipeline_carla_test.py:1153`, `tests/pipeline_carla_test.py:707-708`). The ego is
not checked. Tests: `test_npc_physics_guard_flags_launched_sunk_and_absurdly_fast_cars` and
`test_env_destroys_only_broken_npcs_after_a_tick` in `tests/unit/test_maps_curriculum.py`. The guard
did not stop the Town01 failure (section 7); it removes the phantom obstacles.

#### 4.1.4 Training curriculum (70%): opt-in env options (`9b93e48`, used by the scripts since `dce8ace`)

**Why.** For the experiments that had not started yet, the user asked for less traffic (at 100% the
S5 towns jammed solid and cars piled onto each other), smoother steering and a reward for safe
overtaking. The 2q_cnn run that was training at the time had to keep its exact behaviour, so every
change is a `CarlaGymEnv` constructor option whose default is the old behaviour. Only the 5q CNN,
2q STT and 5q STT scripts opt in (section 4.19); `experiments/2qCNNsac_experiment.py` does not, so
2q_cnn keeps the 100% curriculum and reward 3.0.

| `CarlaGymEnv` option | Script flag | Env default | Training default | Effect |
|---|---|---|---|---|
| `traffic_fill` | `--traffic-fill` | 1.0 | 0.7 | Cars ≤ `fill·vehicle_cap(spawn points)`, walkers ≤ `fill·num_pedestrians` (`_population_target`) |
| `steer_smoothing` | `--steer-smoothing` | 0.0 | 0.6 | Applied steer = k·(previous applied steer) + (1−k)·command; k clipped to [0, 0.95] |
| `steer_change_penalty` | `--steer-change-penalty` | 0.0 | 0.5 | Reward component `steer_change` = −w·(command_t − command_{t−1})² |
| `overtake_bonus` | `--overtake-bonus` | 0.0 | 5.0 | Reward component `overtake` = +B once per vehicle passed safely (rules below) |
| `lead_corridor_follows_ego` | `--route-corridor` turns it off | False | True | Perception corridor centred on the ego instead of the route centre line |

The constants and the reason for each value, from `tests/pipeline_carla_test.py:299-313`:

```python
TRAINING_TRAFFIC_FILL = 0.7


def traffic_caps(n_spawn_points: int, max_pedestrians: int, fill: float = 1.0) -> Tuple[int, int]:
    """(NPC cars, walkers) a town may hold at the given fill."""
    return int(fill * vehicle_cap(n_spawn_points)), int(fill * max(0, int(max_pedestrians)))


# The training curriculum used by the experiments started after 2026-11-19 (5q CNN, 2q STT, 5q STT): 70%
# traffic, smooth steering (low-pass 0.6: per-step jitter -75%, time constant ~0.1 s; penalty
# -0.5 * (command change)^2: a full left-right flip costs -2 on that step, steady driving ~0),
# +5 per safe overtake (well below any crash penalty, C = 50..300) and the perception corridor on the ego.
TRAINING_STEER_SMOOTHING = 0.6
TRAINING_STEER_CHANGE_PENALTY = 0.5
TRAINING_OVERTAKE_BONUS = 5.0
```

**Traffic at 70%.** With `--num-npc 300 --num-pedestrians 300` the S5 caps seen live on 2026-11-19
were Town10HD 98 cars, Town02 63, Town03 167, Town04 234, Town05 190, and up to 210 walkers (fewer
spawn where a town's navmesh is small). S2 (15/15) and S3 (60/60) are below every cap; S4's 150 cars
are capped on the smaller towns (98 on Town10HD) while its 150 walkers stay below 210
(`test_offline_env_traffic_fill_caps_the_curriculum_at_70_percent` in `tests/unit/test_env_reward.py`,
`test_traffic_caps_are_70_percent_of_each_town` in `tests/unit/test_maps_curriculum.py`).

**Steering.** The penalty acts on the command the env receives (the policy's output after the CBF),
the low-pass on what is applied. The observation's `prev_steer` is the applied value, so the state stays
Markov (constructor docstring, `tests/pipeline_carla_test.py:569-579`). A full left-to-right flip
costs −2 on that step; in the live smoke test the penalty summed to about −8 per 400 steps for the
lightly noisy expert and −55 to −117 for random steering.

**Reward version.** Either reward option changes the reward, so the instance reports "3.1", from
`tests/pipeline_carla_test.py:592-593`:

```python
        if self.steer_change_penalty > 0.0 or self.overtake_bonus > 0.0:
            self.REWARD_VERSION = "3.1"      # instance override: v3 + steering-change / overtaking terms
```

`write_run_config` records the instance value as `reward_version` and the five options in its `env`
section (`commons/metrics_callbacks.py:1117-1122`); the three scripts also store them as
`traffic_and_driving`. The class constant stays "3.0", which is what 2q_cnn records.

**Overtake rules** (`commons/overtake.py`, new in `9b93e48`; module docstring
`commons/overtake.py:1-19`):

| Symbol | Where | Role |
|---|---|---|
| `QUEUE_LEAD_GAP_M` … `EGO_HALF_LENGTH_M` | `commons/overtake.py:23-32` | 20 m queue gap, lead slower than max(2 m/s, 0.5·limit), 30 m light/stop margin, clear box 12 m behind / 25 m ahead + 0.3 m, pass margin 2 m, 400-step (20 s) timeout |
| `is_queued_behind` | `commons/overtake.py:35` | The ego waits behind a slow vehicle for no traffic-control reason |
| `lane_clear` | `commons/overtake.py:54` | No vehicle in the adjacent-lane box (vehicle length included) |
| `OvertakeTracker` | `commons/overtake.py:66` | Arms on the queued lead, pays once when it is passed |
| `_adjacent_lane_clear` | `tests/pipeline_carla_test.py:944` | CARLA side: a lane change is allowed by the marking (`wp.lane_change`), the neighbour is a same-direction Driving lane, not in a junction, and `lane_clear` holds |
| `_overtake_step` | `tests/pipeline_carla_test.py:976` | Calls the tracker; an exception warns once and pays 0 |

1. **Queued** (`is_queued_behind`): the lead is a vehicle, not a pedestrian, at most 20 m ahead and
   slower than max(2 m/s, half the speed limit); the ego is on a legal same-direction drivable lane
   outside a junction; and no red/yellow stop line or stop sign lies within the lead gap + 30 m
   (waiting there is correct, so it is never rewarded).
2. **Armed** on that lead when an adjacent lane is legal and clear (and the vehicle has not paid yet in
   this episode).
3. **Paid** (+B) when the lead's front bumper is more than 2 m behind the ego's rear while the ego is on
   a legal same-direction lane, within 400 steps of arming. A collision or the timeout disarms; an
   illegal lane crossing while armed voids the bonus. Each vehicle pays at most once per episode, so
   weaving earns nothing.

From `commons/overtake.py:95-111`:

```python
            front = lead_front_s(lead_id)
            if front is None:
                self.armed = None
                return 0.0
            passed = front < float(snap["s"]) - EGO_HALF_LENGTH_M - PASS_MARGIN_M
            if passed and snap.get("in_same_dir_lane") and snap.get("on_drivable"):
                self.armed = None
                if self.illegal or lead_id in self.paid:
                    return 0.0
                self.paid.add(lead_id)
                self.passes += 1
                return self.bonus
            return 0.0
        if is_queued_behind(snap) and int(snap["lead_id"]) not in self.paid and adjacent_clear():
            self.armed = (int(snap["lead_id"]), int(step))
            self.illegal = False
        return 0.0
```

+5 is small next to the crash penalty (C = 50–300), and the CBF and the collision terms are unchanged,
so the bonus never pays for an unsafe pass. In the live smoke test the tracker armed on a real queued
car through the CARLA lane API. Tests: `tests/unit/test_overtake.py` (5 tests).

**Perception corridor on the ego.** `find_lead` and `lidar_static_gap` take `lateral_offset`
(m, + = right, `commons/perception.py:77-78`, `commons/perception.py:164-165`). With
`lead_corridor_follows_ego` the env passes the ego's lateral error, so after a lane change the car in
the ego's new lane is the lead and the one in the route lane is not, from
`tests/pipeline_carla_test.py:1288-1292`:

```python
        offset = float(proj.e_y) if self.lead_corridor_follows_ego else 0.0   # corridor centre [m, + = right]
        lead_actor = find_lead(actors, ego, route, proj, lateral_offset=offset)
        if self.lidar_points is not None and self.lidar_transform is not None:
            static_gap = float(lidar_static_gap(self.lidar_points, self.lidar_transform, route, proj,
                                                lateral_offset=offset))
```

The lead gap from this corridor also feeds the CBF's collision row and the reward's `blocked` test, so
the option changes what the CBF brakes for while the ego is off the route centre line.

### 4.2 `commons/vehicle_model.py`: measured dynamics

**Purpose.** One place for the ego car's measured numbers (Tesla Model 3 in CARLA 0.9.15, dt 0.05 s)
and the mapping between the single longitudinal action and pedals/acceleration. Used by the env,
the CBF and the expert. Added in `a38ee61`.

| Symbol | Where | Role |
|---|---|---|
| `STEER_GAIN` | `commons/vehicle_model.py:26` | yaw rate ≈ 0.36·v·steer |
| `B_MAX` | `commons/vehicle_model.py:29` | 0.9 × measured full brake (5.13 m/s²) |
| `B_DESIGN` | `commons/vehicle_model.py:30` | 0.7 × `B_MAX`, what the CBF plans with |
| `BRAKE_DELAY_S` | `commons/vehicle_model.py:33` | 0.35 s brake response delay |
| `_ACCEL_TABLE` | `commons/vehicle_model.py:38` | Measured acceleration for 8 action values × 7 speeds |
| `accel_to_pedals` | `commons/vehicle_model.py:50` | `[−1, 1]` → (throttle, brake) |
| `AccelModel` | `commons/vehicle_model.py:61` | Bilinear interpolation `accel(u, v)` and its inverse |

From `commons/vehicle_model.py:50-58`:

```python
def accel_to_pedals(u: float) -> tuple[float, float]:
    """Map the single longitudinal action u in [-1, 1] to (throttle, brake)."""
    u = float(np.clip(u, -1.0, 1.0))
    return (u, 0.0) if u >= 0.0 else (0.0, -u)


def pedals_to_accel_cmd(throttle: float, brake: float) -> float:
    """Inverse of accel_to_pedals (brake wins when both are pressed)."""
    return -float(brake) if brake > 0.0 else float(throttle)
```

Positive values are throttle, negative values are brake, and both pedals are never pressed together.
`AccelModel.inverse` (`commons/vehicle_model.py:75`) turns a desired acceleration into the smallest
action that produces it; the CBF and the expert use it so that "hold speed" gets the throttle needed
against the measured ~3 m/s² coast-down.

### 4.3 `commons/nav_layout.py` and `commons/debug.py`: small shared modules

**Purpose.** `nav_layout.py` fixes the order of the 26 nav features, so the env, wrappers and policies
agree on `NAV_DIM` (`commons/nav_layout.py:38`). `debug.py` turns per-step prints on only when the
environment variable `CARLA_DEBUG=1` is set (`DEBUG`, `commons/debug.py:4`; `dprint`,
`commons/debug.py:7`). Both added in `a38ee61`.

From `commons/nav_layout.py:9-20`:

```python
NAV_FEATURES = [
    "speed_over_limit",      # v / v_limit, clipped to [0, 2]
    "speed_limit_norm",      # v_limit / 30 m/s
    "lateral_error_norm",    # e_y / 2 m, clipped to [-2, 2]; + = right of the route lane centre
    "heading_err_sin",       # sin(e_psi); + = vehicle yawed clockwise (right) of the route tangent
    "heading_err_cos",       # cos(e_psi)
    "route_pt5_x",           # route point 5 m ahead (arc length), ego frame, / 20 m
    "route_pt5_y",
    "route_pt10_x",          # route point 10 m ahead
    "route_pt10_y",
    "route_pt20_x",          # route point 20 m ahead
    "route_pt20_y",
```

The list continues with the next-manoeuvre one-hot and distance, lead gap / relative speed /
pedestrian flag, traffic-light state and stop-line gap, the previous executed action and the
remaining route fraction. `build_nav_vector` (`tests/pipeline_carla_test.py:474`) fills it from the
snapshot and replaces NaN/inf with 0; `_nav_from_obs` (`commons/obs_wrappers.py:65`) clips it to
±10 before it can reach the replay buffer.

### 4.4 `commons/cbfQP_layer.py`: discrete-time CBF safety filter

**Purpose.** Minimally corrects `[steer, accel]` so the next step keeps a stopping-distance barrier
to the obstacle ahead (hard), stays near the speed limit (soft) and inside the lane bounds (soft).
Rewritten in `e383c5a`. The module docstring (`commons/cbfQP_layer.py:1-53`) is the reference for
the math.

| Symbol | Where | Role |
|---|---|---|
| `DEFAULT_PARAMS` | `commons/cbfQP_layer.py:67` | All parameters (standoffs, γ ranges, trust margins, weights, solver) |
| `LEGACY_PARAMS` | `commons/cbfQP_layer.py:110` | Old keyword names, accepted and ignored with a warning |
| `_solve_1d` | `commons/cbfQP_layer.py:127` | Exact minimiser of a 1-D convex piecewise quadratic |
| `CBFSafetyLayer` | `commons/cbfQP_layer.py:166` | The filter |
| `compute_safe_action` | `commons/cbfQP_layer.py:233` | Public entry point; fills `last_info` |
| `_parse_state` | `commons/cbfQP_layer.py:356` | Tolerates missing / non-finite / old-style state keys |
| `_collision_row` | `commons/cbfQP_layer.py:391` | Barrier `h` and the exact bound `a ≤ U_col` |
| `_speed_bound` | `commons/cbfQP_layer.py:424` | `a ≤ max(γ_v (v_lim + 1 − v)/dt, −3)` |
| `_lane_rows` | `commons/cbfQP_layer.py:431` | Lookahead lane rows, normalised to steering units |
| `_solve_osqp` | `commons/cbfQP_layer.py:489` | Same QP with OSQP (tests cross-check) |

Which rows are active, from `commons/cbfQP_layer.py:258-268`:

```python
        if col is not None and not col["feasible"]:
            if v < self.v_stop:
                hold = True
            else:
                infeasible = True
            col_active = u_acc > -1.0
        else:
            col_active = col is not None and a_ref > col["U"]
        speed_active = U_v is not None and a_ref > U_v
        right_active = lane is not None and lane["k_s"] * s_ref > lane["B_R"]
        left_active = lane is not None and -lane["k_s"] * s_ref > lane["B_L"]
```

A row only acts if the actor's own command violates it. `a_ref` is the acceleration the actor's
`accel` would produce according to `AccelModel`. If no row is active the function returns a copy of
the input unchanged.

The collision barrier, from `commons/cbfQP_layer.py:412-422`:

```python
        h = d + vL * vL / (2.0 * self.b_L) - d0 - rho * v - v * v / (2.0 * b_e)
        q2 = dt * dt / (2.0 * b_e)
        q1 = dt * (0.5 * dt + rho + v / b_e)
        R = gamma * h - v * dt
        disc = q1 * q1 + 4.0 * q2 * R
        if disc < 0.0:
            U = -math.inf
        else:
            U = 2.0 * R / (q1 + math.sqrt(disc))
        feasible = U >= -self.b_max
        return dict(h=float(h), U=float(U), feasible=bool(feasible), d0=d0, gamma=gamma)
```

`h` is the distance left after both cars brake to a stop (lead braking at 8 m/s², ego planning with
`b_e`), minus the standstill distance `d0` and the reaction distance `ρ·v`. Requiring
`h(k+1) ≥ (1 − γ)·h(k)` is quadratic in the acceleration and gives the upper bound `U`. If no
acceleration above `−b_max` satisfies it, the row is infeasible and the output is full brake. The
`gap` comes from `cbf_gap` (`tests/pipeline_carla_test.py:450`): a red/yellow stop line counts as a
stationary obstacle when it is nearer than the lead and the car can still stop before it.

Applying the solution, from `commons/cbfQP_layer.py:293-301`:

```python
            if lon_active:
                if a_lon is None:                       # infeasible / standstill hold
                    flat[1] = -1.0
                    a_safe = self.accel_model.accel(-1.0, v)
                else:
                    a_safe = a_sol
                    flat[1] = min(self.accel_model.inverse(a_safe, v), u_acc)
            if lat_active:
                flat[0] = float(np.clip(s_sol, -1.0, 1.0))
```

The longitudinal result is converted back to an action and can only lower the actor's command
(`min(..., u_acc)`). Steering changes only if a lane row was active. Because the longitudinal and
lateral parts are separable, each is solved exactly in closed form.

### 4.5 `commons/cbf_wrapper.py`: CBF action wrapper

**Purpose.** A `gym.ActionWrapper` that runs the CBF before every env step, subtracts the
intervention penalty and reports what it did. Moved out of `pipeline_carla_test.py` into its own
module in `82bc4be` (the env module re-exports it, `tests/pipeline_carla_test.py:69`).

| Symbol | Where | Role |
|---|---|---|
| `CBFSafetyLayerWrapper` | `commons/cbf_wrapper.py:34` | The wrapper; default `intervention_penalty=0.2` |
| `set_trust_score` | `commons/cbf_wrapper.py:72` | Called by `PolicyTrustScoreCallback` and the demo; ignores non-finite values |
| `action` | `commons/cbf_wrapper.py:86` | State query + filter; on an exception passes the action through and counts it |
| `step` | `commons/cbf_wrapper.py:106` | Filter, env step, penalty, `info['cbf']`, episode stats |
| `episode_cbf_stats` | `commons/cbf_wrapper.py:169` | Per-episode CBF summary merged into `episode_metrics` |

From `commons/cbf_wrapper.py:106-119`:

```python
    def step(self, action):
        safe = self.action(action)
        layer_info = self.cbf_layer.last_info
        trust = self._effective_trust()

        obs, reward, terminated, truncated, info = self.env.step(safe)

        du = np.asarray(safe, dtype=np.float64).reshape(-1) - np.asarray(action, dtype=np.float64).reshape(-1)
        du2 = float(du @ du)
        active = bool(layer_info.get("active", False))
        infeasible = bool(layer_info.get("infeasible", False))
        penalty = self.intervention_penalty * min(1.0, du2) if active else 0.0
        if penalty:
            reward = reward - penalty
```

The CBF state is read from the env before the step (it is the snapshot built at the end of the
previous step). The penalty is charged only when a constraint row was active, and it is capped at
0.2 per step. On the last step of an episode the wrapper writes its counters into the env's
`episode_metrics` dict, which is how `cbf_penalty_sum` and `trust_mean` reach `episodes.jsonl`.

### 4.6 `stable_baselines3/sac/`: vendored SAC changes

**Purpose.** The repo vendors a modified SB3 with a K-critic ensemble that also returns uncertainty
and trust. The rework changed two things: the trust formula (`e9c072e`) and a cheaper actor loss
(`11dc732`).

| Symbol | Where | Role |
|---|---|---|
| `EnsembleCritic` | `stable_baselines3/sac/policies.py:27` | K Q-networks; `trust_lambda` default 2.0 |
| `ensemble_forward` | `stable_baselines3/sac/policies.py:81` | All Q-values, variance, trust; `detach_features` option |
| `predict` | `stable_baselines3/sac/policies.py:137` | `(mean_q, uncertainty, trust)` under `no_grad` |
| `n_critics_to_update` | `stable_baselines3/sac/sac.py:123` | M critics for the min-Q target (unchanged) |
| `bootstrap_masks` | `stable_baselines3/sac/sac.py:240` | Each critic trains on a Bernoulli(0.8) subset (unchanged) |

From `stable_baselines3/sac/policies.py:110-117`:

```python
        # Trust Score: exp(-lambda * std(Q) / (|mean(Q)| + 1)). The relative spread is meaningful at any
        # reward scale (exp(-lambda * var) saturated to ~0 or ~1 depending on the Q magnitude). Computed
        # on detached values: trust is a diagnostic / CBF input, never part of a loss (and sqrt has an
        # infinite gradient at zero variance).
        q_detached = q_values_ensemble.detach()
        std_q = th.sqrt(uncertainty.detach().clamp_min(0.0))
        mean_q = q_detached.mean(dim=1, keepdim=True)
        trust_score = th.exp(-self.trust_lambda * std_q / (mean_q.abs() + 1.0))
```

Trust is the critics' relative disagreement mapped to (0, 1]. Dividing by `|mean(Q)| + 1` makes it
independent of the reward scale. It never enters a loss.

From `stable_baselines3/sac/sac.py:338-343`:

```python
            q_values_ensemble_pi, uncertainty, trust_score = self.critic.ensemble_forward(
                replay_data.observations, actions_pi, detach_features=True
            )
            # Min over all K critic networks for the actor loss
            min_qf_pi, _ = th.min(q_values_ensemble_pi, dim=1, keepdim=True)# Standard SAC Actor Loss
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
```

The actor loss now reuses `actions_pi` / `log_prob` sampled at the top of the gradient step
(`stable_baselines3/sac/sac.py:254`) and computes the critic's observation features without a graph:
the actor only needs dQ/d(action). The commit message states the updates are unchanged, which
`tests/unit/test_stt.py` checks (`test_actor_loss_critic_features_detached`). This matters most for
image and transformer extractors.

### 4.7 `commons/navigation/`: bundled GlobalRoutePlanner

**Purpose.** A copy of CARLA 0.9.15's `GlobalRoutePlanner`, `RoadOption` and one helper, so the
project no longer needs CARLA's `agents` package on `sys.path`. Only the `RoadOption` import was
changed (`c127036`). The env builds one planner per map load (`tests/pipeline_carla_test.py:711`).

From `commons/navigation/__init__.py:1-14`:

```python
"""CARLA 0.9.15 global route planner bundled into the project (MIT licensed, see file headers).

Exports:
    GlobalRoutePlanner(wmap, sampling_resolution) -> .trace_route(origin, destination)
        returns list[(carla.Waypoint, RoadOption)]. Building it walks the whole map topology,
        so build it once per map and reuse it.
    RoadOption: IntEnum VOID=-1, LEFT=1, RIGHT=2, STRAIGHT=3, LANEFOLLOW=4,
        CHANGELANELEFT=5, CHANGELANERIGHT=6.
"""

from commons.navigation.road_option import RoadOption
from commons.navigation.global_route_planner import GlobalRoutePlanner

__all__ = ["GlobalRoutePlanner", "RoadOption"]
```

The commit message reports that `trace_route` gave identical results to CARLA's original on 180
random origin/destination pairs (checked offline against three towns' `.xodr`).

### 4.8 `commons/route.py`: route planning and geometry

**Purpose.** The episode's route as a polyline with per-waypoint `RoadOption`s: planning, projection
of the ego (arc length `s`, lateral error `e_y`, heading error `e_psi`, curvature), points ahead,
next manoeuvre, lane bounds and lane status. Added in `1dd0032`, hardened in `247e89c`.
Sign conventions follow CARLA (y to the right): `commons/route.py:1-13`.

| Symbol | Where | Role |
|---|---|---|
| `MAX_KINK_DEG` | `commons/route.py:40` | Routes with a sharper polyline kink (GRP lane-change glitch) are rejected |
| `RouteProjection` | `commons/route.py:75` | `s, e_y, e_psi, kappa, idx, lane_width, remaining_m` |
| `LaneStatus` | `commons/route.py:86` | Which lane the ego is in, relative to the route and travel direction |
| `Route.plan` | `commons/route.py:233` | GRP route of the stage's length; never empty or degenerate |
| `project` | `commons/route.py:308` | Windowed projection using the previous index as hint (no jumps back on overlapping routes) |
| `points_ahead_ego` | `commons/route.py:410` | Route points 5/10/20 m ahead in the ego frame (nav vector) |
| `next_maneuver` | `commons/route.py:420` | Next LEFT/RIGHT/STRAIGHT within 50 m and its distance |
| `project_points` | `commons/route.py:443` | Vectorised projection used by perception and traffic rules |
| `lane_bounds` | `commons/route.py:488` | Lateral free space left/right, wider where a same-direction lane may legally be used |
| `lane_status` | `commons/route.py:505` | On drivable / route lane / same-direction / opposite lane, lane offset |

From `commons/route.py:262-278` (inside `Route.plan`):

```python
            try:
                trace = grp.trace_route(start_loc, dest)
                if len(trace) < 2:
                    continue
                route = cls.from_trace(trace)
            except Exception:   # no path / start not localisable
                continue
            length = route.length_m
            clean = route.max_kink_deg <= MAX_KINK_DEG
            if clean and min_len_m <= length <= max_len_m:
                return route
            err = max(0.0, min_len_m - length, length - max_len_m) + (0.0 if clean else 1e4)
            if err < best_err:
                best, best_err = route, err
        if best is not None and best.length_m >= 0.5 * min_len_m:
            return best
        return cls._lane_follow_route(wmap, start_loc, min_len_m, max_len_m, rng)
```

Up to 30 random destinations (spawn points within reach of the length window) are traced. The first
glitch-free route in the length window wins. Otherwise the closest one is used if it is at least half
the minimum length; otherwise a plain lane-follow route is built (`commons/route.py:281`). The
half-minimum check was added in `247e89c` after a failed trace on Town04 produced a 2 m route, which
handed out the goal reward for free.

### 4.9 `commons/perception.py`: obstacles along the route

**Purpose.** The gap and speed of the nearest obstacle in the route corridor, measured along the
curved route from the front bumper. Feeds the nav vector, the reward's `blocked` test, the CBF and
the expert. Added in `1dd0032`. Since `9b93e48` both corridor functions take `lateral_offset` (m,
+ = right of the route centre line, default 0.0 = unchanged); the env passes the ego's lateral error
when `lead_corridor_follows_ego` is on (section 4.1.4).

| Symbol | Where | Role |
|---|---|---|
| `LeadInfo` | `commons/perception.py:27` | `gap_m`, `v_lead_mps`, `kind` (`vehicle`/`walker`/`static`/`none`), `actor_id` |
| `find_lead` | `commons/perception.py:77` | Actor bounding boxes projected into the route frame and intersected with the corridor; walkers swept 2 s along their lateral motion |
| `lidar_static_gap` | `commons/perception.py:164` | Nearest LiDAR return in the curved corridor, with ground, ceiling, ego-box and road-relative ground cuts |
| `combine` | `commons/perception.py:209` | Nearest of the two |
| `ttc` | `commons/perception.py:222` | gap / closing speed, inf if not closing |

From `commons/perception.py:209-219`:

```python
def combine(lead_actor: LeadInfo, static_gap: float) -> LeadInfo:
    """Nearest of the actor lead and the LiDAR gap.

    The LiDAR also sees the lead actor itself (usually at the same or a slightly larger gap than
    its bounding box), so the static return only wins when it is more than COMBINE_TOL_M nearer;
    this keeps the actor's velocity for car following.
    """
    static_gap = float(static_gap)
    if static_gap < NO_GAP and static_gap < lead_actor.gap_m - COMBINE_TOL_M:
        return LeadInfo(static_gap, 0.0, "static", None)
    return lead_actor
```

The actor list gives speed and type; the LiDAR catches static objects the actor list does not
contain. The LiDAR result replaces the actor only if it is more than 1 m nearer, so car following
keeps the lead car's speed. "Nothing ahead" is a gap of 100 m.

### 4.10 `commons/traffic_rules.py`: lights, stop signs, lane markings

**Purpose.** Rule checks along the route, modelled on the CARLA leaderboard rules. Added in `1dd0032`.

| Symbol | Where | Role |
|---|---|---|
| `TrafficLightTracker` | `commons/traffic_rules.py:56` | Lights whose stop line lies on the route lanes; cached per map load |
| `query` | `commons/traffic_rules.py:165` | Nearest relevant light ahead and its stop-line gap (negative while still crossing on red) |
| `RedLightMonitor` | `commons/traffic_rules.py:187` | Violation when the tail crosses the line on red after the front crossed on red |
| `StopSignMonitor` | `commons/traffic_rules.py:228` | Stop counts if speed < 0.1 m/s within 4 m of the line; violation if the centre crosses without stopping |
| `lane_invasion_is_illegal` | `commons/traffic_rules.py:358` | Solid/curb/grass, wrong side of a mixed marking, or into an oncoming lane |
| `DurationMonitor` | `commons/traffic_rules.py:390` | Generic N-consecutive-steps counter |

From `commons/traffic_rules.py:211-225`:

```python
        violated = False
        if cur is not None and self._prev is not None and self._prev[0] == cur[0]:
            lid, gap = cur[0], cur[1]
            _, pgap, ps = self._prev
            if -1.0 <= s - ps < 10.0:          # ignore relocalisation jumps
                if pgap > 0.0 >= gap:
                    self._front_state[lid] = tl.state
                ptail = pgap + EGO_HALF_LENGTH + TAIL_OFFSET_M
                tail = gap + EGO_HALF_LENGTH + TAIL_OFFSET_M
                if (ptail > 0.0 >= tail and tl.state == "red" and self._front_state.get(lid) == "red"
                        and lid not in self._flagged):
                    self._flagged.add(lid)
                    violated = True
        self._prev = cur
        return violated
```

The monitor remembers the light state when the front bumper crossed the stop line. A red-light run is
reported only when the tail also crosses on red and the front had crossed on red, so a legal yellow
entry is not punished. Each light is reported at most once per episode. A red-light run terminates
the episode with −C.

### 4.11 `commons/expert.py`: rule-based expert driver

**Purpose.** Drives well enough to pre-fill the replay buffer and to fit the STT projection. Pure
pursuit for steering; a target speed from the speed limit, curves, lead, lights and stop signs,
tracked by PI control through `AccelModel.inverse`. Added in `1dd0032`.

| Symbol | Where | Role |
|---|---|---|
| `ExpertDriver` | `commons/expert.py:29` | Cruise at 90% of the limit, lateral acceleration ≤ 2.5 m/s² in curves |
| `steer` | `commons/expert.py:54` | Pure pursuit from the rear axle |
| `target_speed` | `commons/expert.py:75` | Minimum of cruise, curve and stopping-ramp speeds, plus constraints to meet |
| `act` | `commons/expert.py:110` | `[steer, accel]` |

From `commons/expert.py:132-144`:

```python
        # Exact constant deceleration that meets a constraint, once the ramp is no longer enough
        # (a P controller alone lags a decelerating profile by target_decel / KP ~ 2 m/s).
        v_pos = max(0.0, v)
        for g, v_end in constraints:
            if v_pos > v_end + 0.1:
                g_eff = max(g - 0.5 * BRAKE_DELAY_S * v_pos, 0.05)
                a_need = (v_end * v_end - v_pos * v_pos) / (2.0 * g_eff)
                if a_need < -0.8 * self.target_decel:
                    a = min(a, a_need)

        a = float(np.clip(a, -FULL_BRAKE_DECEL, A_MAX))
        u = self.accel_model.inverse(a, v_pos)
        return np.array([steer, float(np.clip(u, -1.0, 1.0))], dtype=np.float32)
```

When the PI controller would brake too late for a stop point or a slow curve, the expert applies
the exact constant deceleration that reaches it. The acceleration is converted to an action with the
measured model. `CarlaGymEnv.expert_action()` (`tests/pipeline_carla_test.py:1372`) calls it with the
current snapshot.

### 4.12 `commons/obs_wrappers.py`: CNN observation wrapper

**Purpose.** Turns the env's `{rgb_data, nav}` into `{image (3,84,84) uint8, nav}` for SB3's
`MultiInputPolicy`, with occlusions applied to the image only. Added in `0470e13`; the first STT
wrapper and the transformer checkpoint helpers that also lived here were removed in `c001d68`.

| Symbol | Where | Role |
|---|---|---|
| `NAV_CLIP` | `commons/obs_wrappers.py:25` | Guard against NaN/inf/garbage in the replay buffer |
| `_nav_from_obs` | `commons/obs_wrappers.py:65` | Checks length, replaces non-finite values, clips |
| `CnnNavObservationWrapper` | `commons/obs_wrappers.py:76` | The wrapper; re-rolls the occlusion at every reset |

From `commons/obs_wrappers.py:98-106`:

```python
    def observation(self, obs):
        rgb = _occlude(self.occlusion, _rgb_from_obs(obs))
        if rgb is None:
            image = np.zeros((3, self.size, self.size), dtype=np.uint8)
        else:
            if rgb.shape[0] != self.size or rgb.shape[1] != self.size:
                rgb = cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_AREA)
            image = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1)), dtype=np.uint8)
        return {"image": image, "nav": _nav_from_obs(obs, self.nav_dim)}
```

The occlusion (`experiments/occlusions.py`, unchanged) is chosen per episode: 30% of episodes are
occluded, consistently for the whole episode. It is applied at full resolution before the resize.
The image stays BGR; SB3 normalises it to [0, 1].

### 4.13 `commons/stt.py` (and `commons/spatiotemporal_transformer.py`): trainable STT

**Purpose.** Makes the SpatioTemporal Transformer trainable by SAC. The env side encodes each camera
frame once with a frozen ImageNet ResNet50 and a frozen PCA projection and caches the last 8 codes.
The policy side is a features extractor (encoder + stacked transformer) trained by the actor and
critic losses. Added in `c001d68`; `11dc732` added a single-token attention fast path and
`need_weights=False` in `commons/spatiotemporal_transformer.py:25`.

| Symbol | Where | Role |
|---|---|---|
| `NUM_FRAMES` … `EMBED_DIM` | `commons/stt.py:42` | 8 frames, 32 PCA channels, 224×128 input, 4×7 grid, D = 128 |
| `_resnet50_state_dict` | `commons/stt.py:58` | Loads `pretrained/feature_extractor/pretrained_resnet50.pth`, downloads once if missing |
| `FrozenResNet50` | `commons/stt.py:71` | Frame → (28, 2048) tokens, fp16 autocast on CUDA |
| `ProjectionFitter` | `commons/stt.py:103` | Streaming PCA, whitened top-k components |
| `SttNavObservationWrapper` | `commons/stt.py:147` | Feature cache; `Dict(frames (8,32,4,7) float16, nav)` |
| `set_projection` | `commons/stt.py:172` | Installs a fitted or restored projection |
| `fit_stt_projection` | `commons/stt.py:217` | 1000 expert-driving frames in a given stage; restores the previous stage afterwards |
| `SttFeaturesExtractor` | `commons/stt.py:264` | Encoder + `cnn_proj` + transformer + LayerNorm, concatenated with nav |
| `stt_policy_kwargs` | `commons/stt.py:293` | `policy_kwargs` for `MultiInputPolicy` |

From `commons/stt.py:194-206`:

```python
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
```

Each new frame goes through ResNet50 and the PCA exactly once. After a reset the window is filled
with copies of the first frame, never black frames. Errors raise instead of returning silent zeros.
The compact codes are what fit in the replay buffer.

From `commons/stt.py:285-290`:

```python
    def forward(self, observations: dict) -> torch.Tensor:
        frames = observations["frames"]                                      # (B, T, K, H, W)
        st = self.encoder(frames)                                            # (B, D)
        f = self.cnn_proj(frames[:, -1].mean(dim=(2, 3)))                    # (B, D) newest frame
        out, _ = self.transformer(f, st)
        return torch.cat([self.norm(out), observations["nav"].flatten(1)], dim=1)
```

Because the transformer is the policy's features extractor, SB3 trains it and saves it in every
checkpoint `.zip`. With SB3's default `share_features_extractor=False` the actor and the critic each
have their own copy (the experiment records this in `run_config.json`,
`experiments/2qsac_stt_experiment.py:194`). The PCA projection is stored as `model.stt_projection`,
which SB3 writes into the `.zip`; resume and the demo restore it with `set_projection`.

### 4.14 `commons/resnet_extractor.py`: ResNet-50 image branch (5q CNN)

**Purpose.** Optional replacement for NatureCNN in the 5q CNN experiment: the complete ImageNet
ResNet-50 with its classifier replaced by `Linear(2048, 256) + ReLU`, so the image features have
NatureCNN's size and the SAC layers are unchanged. Added in `6fdf5d4`.

| Symbol | Where | Role |
|---|---|---|
| `TRAINABLE_CHOICES` | `commons/resnet_extractor.py:52` | `head` (fc only), `layer4` (default), `all` |
| `FrozenTrunk` | `commons/resnet_extractor.py:126` | Frozen stages, BatchNorm folded, fp16 channels-last on CUDA, reuses the output for a repeated batch |
| `frozen_trunk` | `commons/resnet_extractor.py:170` | One shared frozen trunk per (backbone, split, device), outside the state dict |
| `ResNetNavExtractor` | `commons/resnet_extractor.py:185` | SB3 features extractor: ResNet features then nav |
| `resnet_extractor_kwargs` | `commons/resnet_extractor.py:244` | `policy_kwargs` entries for the 5q script |
| `checkpoint_image_extractor` | `commons/resnet_extractor.py:254` | Reads a checkpoint's image branch from its JSON data |
| `extractor_mismatch` | `commons/resnet_extractor.py:272` | Message if a checkpoint's branch differs from the requested one |

From `commons/resnet_extractor.py:228-241`:

```python
    def forward(self, observations: dict) -> torch.Tensor:
        img = observations["image"]                                  # (B, 3, H, W) BGR in [0, 1]
        x = (img.flip(1) - self.mean) / self.std                     # RGB, ImageNet normalisation
        x = x.contiguous(memory_format=torch.channels_last)
        if self.split > 0:
            x = frozen_trunk(self.backbone, self.split, x.device)(x)
        if len(self.body) > 0:
            amp = _amp_dtype(x.device)
            if amp is None:
                x = x.float()                                        # frozen fp16 output -> fp32
            with torch.autocast(device_type=x.device.type, dtype=amp, enabled=amp is not None):
                x = self.body(x)
        image_features = self.fc(torch.flatten(x, 1).float())
        return torch.cat([image_features, observations["nav"].flatten(1)], dim=1)
```

The camera image is BGR, so it is flipped to RGB before the ImageNet normalisation. The frozen part
runs once per distinct batch and is shared by the actor, critic and target extractors; the trained
part (layer4 + fc by default) runs in bf16 autocast on CUDA with BatchNorm kept in eval mode. The
file's docstring reports one SAC gradient step at batch 64 as NatureCNN 22 ms, head 32 ms,
layer4 49 ms, all 107 ms on the training laptop (`commons/resnet_extractor.py:21-22`).

### 4.15 `commons/curriculum_stages.py`: stage table and schedule

**Purpose.** The curriculum, shared by the env (applies a stage at reset) and the curriculum callback
(promotes). Added with 4 stages in `a38ee61`; 6 stages and the step schedule since `0f2d2a2`.

| Stage | Cars / pedestrians (maxima) | Route length | Crash penalty C | Left at the latest at |
|---|---|---|---|---|
| S0 empty short | 0 / 0 | 80–150 m | 50 | 15,000 steps |
| S1 empty full | 0 / 0 | 200–400 m | 100 | 35,000 |
| S2 light | 15 / 15 | 200–400 m | 150 | 65,000 |
| S3 medium | 60 / 60 | 200–400 m | 200 | 100,000 |
| S4 heavy | 150 / 150 | 200–400 m | 250 | 140,000 |
| S5 dense | 300 / 300 | 200–400 m | 300 | final |

Source: `STAGES` (`commons/curriculum_stages.py:18`). Car counts are further capped per map by
`vehicle_cap` (`tests/pipeline_carla_test.py:287`). With `traffic_fill` below 1 (0.7 in the training
curriculum of the 5q CNN, 2q STT and 5q STT scripts, 4.1.4) `traffic_caps` lowers the car and walker
caps to that share; the stage table itself is unchanged.

From `commons/curriculum_stages.py:32-40`:

```python
CURRICULUM_TOTAL_STEPS = 200_000
# Total-step marks by which S0..S4 are left even if the success gate was not met.
PROMOTE_BY_STEP = [15_000, 35_000, 65_000, 100_000, 140_000]
assert len(PROMOTE_BY_STEP) == FINAL_STAGE and PROMOTE_BY_STEP == sorted(PROMOTE_BY_STEP)


def scheduled_stage(num_timesteps: int) -> int:
    """The lowest stage the step schedule allows at `num_timesteps` (number of marks passed)."""
    return sum(int(num_timesteps) >= mark for mark in PROMOTE_BY_STEP)
```

`scheduled_stage` is the lowest stage a run may be in at a given total step count. It forces
promotion when the success gate (80% of the last 50 episodes, at least 50 episodes in the stage) has
not been met, and it sets the minimum stage of a resumed run.

### 4.16 `commons/curriculum.py`: curriculum callback

**Purpose.** Promotes the env through the stages from training-episode outcomes, persists the state
to `<log-dir>/curriculum_state.json` after every episode and restores it on resume. Added in
`0470e13`; step schedule in `0f2d2a2`.

| Symbol | Where | Role |
|---|---|---|
| `resolve_start_stage` | `commons/curriculum.py:58` | Fresh: `--start-stage`. Resume: max of start stage, saved stage, scheduled stage |
| `CurriculumCallback` | `commons/curriculum.py:70` | The callback |
| `_on_training_start` | `commons/curriculum.py:94` | Restore saved state; jump to the scheduled stage if behind (logged as a `schedule` event) |
| `_on_step` | `commons/curriculum.py:134` | Counts only episodes that ran under the current stage |
| `_should_promote` | `commons/curriculum.py:167` | Gate met or schedule due |
| `_save` | `commons/curriculum.py:198` | Atomic write of the state file |

From `commons/curriculum.py:157-168`:

```python
    def _gate_met(self) -> bool:
        return (
            self.episodes_in_stage >= MIN_EPISODES_PER_STAGE
            and len(self.window) >= PROMOTE_WINDOW
            and self.success_rate() >= PROMOTE_SUCCESS_RATE
        )

    def _schedule_due(self) -> bool:
        return self.stage < FINAL_STAGE and self.num_timesteps >= PROMOTE_BY_STEP[self.stage]

    def _should_promote(self) -> bool:
        return self.stage < FINAL_STAGE and (self._gate_met() or self._schedule_due())
```

A success is an episode with `end_reason == "endpoint"`. The check runs when an episode ends, so a
schedule promotion happens at the first episode end after the mark. The stage never goes down. The
new stage is applied with `CarlaGymEnv.set_curriculum_stage` and takes effect at the next reset.

### 4.17 `commons/warmup.py`: expert warm-up, checkpoints, resume clean-up

**Purpose.** Everything the four scripts share around SAC: the expert replay-buffer warm-up, replay
buffer restore, atomic checkpoints, keep-latest checkpointing, log clean-up on resume and the env
description for `run_config.json`. Added in `0470e13`; keep-latest and clean resume in `0f2d2a2`.

| Symbol | Where | Role |
|---|---|---|
| `prefill_replay_buffer` | `commons/warmup.py:39` | N expert transitions through the full stack; does not advance `num_timesteps` |
| `replay_buffer_candidates` | `commons/warmup.py:131` | Both buffer file names for a checkpoint |
| `restore_replay_buffer` | `commons/warmup.py:145` | Loads the first usable one; rejects a different observation space |
| `checkpoint_num_timesteps` | `commons/warmup.py:176` | Step count from the zip's JSON, no model built |
| `save_checkpoint` | `commons/warmup.py:186` | Zip + buffer, each written to a temp name then renamed |
| `prune_checkpoints` | `commons/warmup.py:226` | Deletes older zips, buffers and curriculum snapshots |
| `LatestCheckpointCallback` | `commons/warmup.py:242` | Every multiple of `save_freq` total steps (the experiments pass `SAVE_FREQ` = 5k); keep 1 |
| `restore_curriculum_snapshot` | `commons/warmup.py:280` | Copies `*_curriculum_<N>_steps.json` back to `curriculum_state.json` |
| `truncate_tensorboard` | `commons/warmup.py:293` | Rewrites event files in place without events past a step |
| `discard_logs_after` | `commons/warmup.py:360` | TensorBoard + `episodes.jsonl` + `eval.jsonl` |
| `mark_tensorboard_resume` | `commons/warmup.py:374` | SessionLog.START at step+1, so a running TensorBoard drops later points |
| `describe_env_stack` | `commons/warmup.py:407` | Wrapper chain and env/CBF parameters |

From `commons/warmup.py:86-95`:

```python
    for step in range(1, n_steps + 1):
        expert = np.asarray(carla_env.expert_action(), dtype=np.float32).reshape(action_space.shape)
        expert = np.clip(expert, low, high)
        actions = expert[None, :]                          # (n_envs=1, 2) for the VecEnv
        buffer_actions = model.policy.scale_action(actions)  # what SB3 stores ([-1, 1] -> identity)

        new_obs, rewards, dones, infos = vec_env.step(actions)
        # Exactly SB3's storage path (terminal_observation, TimeLimit.truncated, deep copies,
        # updates model._last_obs / _last_original_obs).
        model._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)
```

The warm-up drives the expert through the same VecEnv, observation wrapper and CBF wrapper as
training, so stored observations, rewards and CBF penalties match what the policy sees later. It
uses SB3's own `_store_transition`. A zero-centred random policy would mostly coast to a stop, which
is why the expert replaces SB3's random warm-up.

From `commons/warmup.py:263-277`:

```python
    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            self.save()
            self._next = (int(self.num_timesteps) // self.save_freq + 1) * self.save_freq
        return True

    def save(self) -> str:
        path = save_checkpoint(self.model, self.save_path, self.name_prefix, verbose=self.verbose)
        if self.log_dir:
            state = os.path.join(self.log_dir, "curriculum_state.json")
            if os.path.isfile(state):
                shutil.copyfile(state, curriculum_snapshot_path(self.save_path, self.name_prefix,
                                                                self.model.num_timesteps))
        prune_checkpoints(self.save_path, self.name_prefix, keep=self.keep, verbose=self.verbose)
        return path
```

Saves happen on multiples of `save_freq` total steps (5k in all four experiments since `dce8ace`, 25k
before), so a run that ends exactly at the target (200k) also writes a `200000` checkpoint, which is what the runner and the queue look for. Old checkpoints are
deleted only after the new zip and buffer are complete.

### 4.18 `commons/metrics_callbacks.py`: metrics, evaluation, run config

**Purpose.** One record per training episode, rolling summaries, deterministic evaluation every 25k
steps, `run_config.json`. What each number means is in METRICS.md; this section covers the code.
Added in `16ec944`; map field and per-town eval in `0f2d2a2`.

| Symbol | Where | Role |
|---|---|---|
| `SB3_TB_DUPLICATES` | `commons/metrics_callbacks.py:132` | SB3 keys filtered from TensorBoard (console unaffected) |
| `_RUN_ID` | `commons/metrics_callbacks.py:148` | `YYYYmmdd-HHMMSS-<pid>`, stamped on every record |
| `drop_superseded` | `commons/metrics_callbacks.py:247` | Drops records an earlier process logged past the resume point |
| `build_episode_record` | `commons/metrics_callbacks.py:276` | Flattens `episode_metrics`, adds `return_total`, reward groups |
| `summarize` | `commons/metrics_callbacks.py:331` | Aggregates a window or an eval round |
| `_tb_writer` | `commons/metrics_callbacks.py:507` | One `SummaryWriter` per run in `tensorboard/metrics`, `purge_step` on resume |
| `_DedupTensorBoardFormat` | `commons/metrics_callbacks.py:544` | Wraps SB3's TensorBoard output and drops duplicates |
| `EpisodeMetricsCallback` | `commons/metrics_callbacks.py:592` | `episodes.jsonl`, TensorBoard, console lines and blocks, perf timing |
| `EvalCallback` | `commons/metrics_callbacks.py:761` | 10 deterministic episodes on fixed spawns and towns |
| `_restore_training_state` | `commons/metrics_callbacks.py:902` | Leaves SB3 on a fresh training episode after evaluation |
| `git_info` | `commons/metrics_callbacks.py:976` | Commit, branch, dirty (tracked files only) |
| `write_run_config` | `commons/metrics_callbacks.py:1036` | `run_config.json`; previous version appended to `run_config_history.jsonl`; since `9b93e48` also the training-curriculum options and the instance `reward_version` (`commons/metrics_callbacks.py:1117-1122`) |

From `commons/metrics_callbacks.py:836-845`:

```python
            for i in range(self.n_episodes):
                # map_slot: with map rotation, each round covers every town in the same fixed order
                venv.set_options({"spawn_index": i, "route_seed": i, "map_slot": i, "map_slots": self.n_episodes})
                obs = venv.reset()
                last_done, metrics, agent_return, n_steps = False, None, 0.0, 0
                while n_steps < self.max_episode_steps:
                    action, _ = model.predict(obs, deterministic=self.deterministic)
                    obs, rewards, dones, infos = venv.step(action)
                    n_steps += 1
                    agent_return += float(np.asarray(rewards).reshape(-1)[0])
```

Evaluation runs on the training env at the end of a rollout, steps the VecEnv directly and never
calls the other callbacks, so nothing reaches the replay buffer, `episodes.jsonl` or the curriculum.
Episode i uses spawn point i, route seed i and town slot i; NPC traffic is still random. Afterwards
`_restore_training_state` points `model._last_obs` at a fresh training episode.

From `commons/metrics_callbacks.py:247-263`:

```python
def drop_superseded(records: List[dict]) -> List[dict]:
    """After a resume, records an earlier process logged past the restored checkpoint are void.

    Every record carries run_id and run_start_step; when a new run starts at step S, records of
    earlier runs with step > S are dropped (their weights/replay buffer were never restored).
    """
    kept: List[dict] = []
    seen = set()
    for rec in records:
        run = rec.get("run_id")
        if run is not None and run not in seen:
            seen.add(run)
            start = rec.get("run_start_step")
            if start is not None:
                kept = [r for r in kept if _num(r.get("step"), 0) <= start]
        kept.append(rec)
    return kept
```

Readers (`EpisodeMetricsCallback` on resume, `make_report.py`) apply this filter. Since `0f2d2a2`
the resume also physically removes those records (`discard_logs_after`), so the filter is a second
line of defence.

### 4.19 `experiments/*_experiment.py`: the four training scripts

**Purpose.** Each script builds its env stack and SAC model and runs the flow in section 3.4. All four
were rewritten in `14f9a28`; STT scripts again in `c001d68`; keep-latest checkpoints and clean resume
in `0f2d2a2`; the 5q CNN script got the ResNet option in `6fdf5d4`. In `dce8ace` all four save every 5k
steps, and the 5q CNN, 2q STT and 5q STT scripts opt into the training curriculum (4.1.4): their
parsers call `add_training_curriculum_args`, the env factory takes `env_kwargs` and gets
`training_curriculum_kwargs(args)`, the settings are printed at start-up and stored in
`run_config.json` as `traffic_and_driving` (`experiments/5qCNNsac_experiment.py:145-148`,
`experiments/5qCNNsac_experiment.py:214`, `experiments/5qCNNsac_experiment.py:297`). The 2q CNN
script changed only `SAVE_FREQ`.

| Script | Critics | Observation / extractor | Default `--port` | Default `--log-dir` | Traffic / reward |
|---|---|---|---|---|---|
| `experiments/2qCNNsac_experiment.py` | 2 | image + nav, NatureCNN | 2000 | `./logs/2q_cnn` | 100%, reward 3.0 |
| `experiments/5qCNNsac_experiment.py` | 5 | image + nav, NatureCNN or ResNet-50 | 2000 | `./logs/5q_cnn` or `./logs/5q_cnn_resnet_<mode>` | 70% curriculum, reward 3.1 |
| `experiments/2qsac_stt_experiment.py` | 2 | frame codes + nav, `SttFeaturesExtractor` | 2004 | `./logs/2q_stt` | 70% curriculum, reward 3.1 |
| `experiments/5qsac_stt_experiment.py` | 5 | frame codes + nav, `SttFeaturesExtractor` | 2000 | `./logs/5q_stt` | 70% curriculum, reward 3.1 |

Shared settings: `MultiInputPolicy`, `net_arch` pi/qf `[256, 256]`, lr 3e-4, batch 64, buffer 50,000,
`learning_starts=1000`, `train_freq=1`, `gradient_steps=1`, `ent_coef="auto"`, time limit 3000,
occlusion `mixed`, checkpoint every 5k (`SAVE_FREQ`, 25k until `dce8ace`) and eval every 25k
(`experiments/2qCNNsac_experiment.py:52-62`,
`experiments/2qCNNsac_experiment.py:140-155`).

| Symbol | Where | Role |
|---|---|---|
| `create_cnn_env` | `experiments/2qCNNsac_experiment.py:65` | Env → CBF wrapper → CNN wrapper |
| `train` | `experiments/2qCNNsac_experiment.py:86` | The flow of section 3.4 |
| `parse_args` | `experiments/2qCNNsac_experiment.py:235` | CLI flags (table in RUNNING.md §3.2) |
| `run_names` | `experiments/5qCNNsac_experiment.py:97` | Per-branch experiment tag and checkpoint prefix |
| `extractor_mismatch` | `experiments/5qCNNsac_experiment.py:121` | Refuses a checkpoint of another image branch |
| `create_stt_env` | `experiments/2qsac_stt_experiment.py:76` | Env → CBF wrapper → STT wrapper |
| `fit_stt_projection` | `experiments/2qsac_stt_experiment.py:158` | Fresh STT run: fit the PCA in the final stage first |

The resume preamble, from `experiments/2qCNNsac_experiment.py:93-108`:

```python
    resume = bool(args.load_checkpoint)
    if resume:
        if not os.path.isfile(args.load_checkpoint):
            raise FileNotFoundError(f"--load-checkpoint not found: {args.load_checkpoint}")
        done = checkpoint_num_timesteps(args.load_checkpoint)
        if done is not None and done >= args.timesteps:
            print(f"[RESUME] checkpoint is at {done:,} steps; target {args.timesteps:,} already reached. Nothing to do.")
            return 0
        if done is not None:
            # The run continues from the checkpoint: anything an earlier process logged after it
            # (TensorBoard, episodes/eval jsonl, curriculum state) is erased or restored.
            restore_curriculum_snapshot(log_dir, args.load_checkpoint)
            discard_logs_after(log_dir, done)

    start_stage = FINAL_STAGE if args.no_curriculum else args.start_stage
    env_stage = resolve_start_stage(log_dir, start_stage, resume=resume, step=(done or 0) if resume else 0)
```

All of this runs before the env and the model are built. The clean-up therefore also happens when
the later `SAC.load` fails (see section 6). The env is constructed directly with the stage the
curriculum resolves, so the first episode already has the right traffic.

### 4.20 `make_report.py`: HTML report

**Purpose.** Reads `episodes.jsonl`, `eval.jsonl` and `run_config.json` of each run and writes
`report.html`, `summary.csv`, per-run CSVs and `plots/*.png`. Headline numbers come from evaluation.
Added in `16ec944`. Safe to run during training (read only).

| Symbol | Where | Role |
|---|---|---|
| `TARGET` | `make_report.py:46` | Default target for the status column (1,000,000; override with `--target`) |
| `find_runs` | `make_report.py:162` | Finds runs by the four standard names, or `--run name=folder` |
| `load_run` | `make_report.py:190` | Loads and de-duplicates (`drop_superseded`) one run |
| `main` | `make_report.py:507` | CLI: `--log-root`, `--run`, `--out`, `--target` |

From `make_report.py:162-177`:

```python
def find_runs(log_roots, explicit):
    runs = {}
    for spec in explicit or []:
        name, _, path = spec.partition("=")
        if not path:
            name, path = os.path.basename(os.path.normpath(spec)), spec
        runs[name] = os.path.abspath(path)
    roots = list(log_roots or []) + DEFAULT_LOG_ROOTS
    for name in EXPERIMENTS:
        if name in runs:
            continue
        for root in roots:
            folder = os.path.join(root, name)
            if os.path.isfile(os.path.join(folder, "episodes.jsonl")):
                runs[name] = os.path.abspath(folder)
                break
```

Automatic discovery only looks for `2q_cnn`, `5q_cnn`, `2q_stt`, `5q_stt` under the given roots plus
`<repo>/CARLA-RL-Agents/logs` and `<repo>/logs`. ResNet runs need `--run 5q_cnn_resnet_layer4=<folder>`,
and logs written elsewhere (the queue writes to the main checkout's `CARLA-RL-Agents\logs`) need
`--log-root`.

### 4.21 `demo/run_model_inference.py`: watch a trained model

**Purpose.** Loads any experiment's `.zip`, builds the same stack as training (model type detected
from the observation space), runs deterministic episodes and prints one line per episode. Rewritten
in `14f9a28`; STT support in `c001d68`; one town per episode and rings in `0f2d2a2`; dense traffic by
default in `247e89c`. Commands: RUNNING.md §7.

| Symbol | Where | Role |
|---|---|---|
| `build_env` | `demo/run_model_inference.py:86` | CNN or STT stack, `map_rotate_every=1`, stage 5 by default |
| `run_episode` | `demo/run_model_inference.py:133` | Runs one episode to its natural end; live trust |
| `main` | `demo/run_model_inference.py:191` | CLI; writes `demo/results_<model>_<N>eps.json` by default |

From `demo/run_model_inference.py:141-155`:

```python
    done = False
    ep_return, steps, trust_values = 0.0, 0, []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        if cbf_wrapper is not None and use_trust:
            _, trust = _compute_live_policy_metrics(model, obs)
            if trust is not None:
                cbf_wrapper.set_trust_score(trust)
                trust_values.append(trust)
        obs, reward, terminated, truncated, info = env.step(action)
        if rings is not None:
            rings.update()
        ep_return += float(reward)
        steps += 1
        done = bool(terminated or truncated)
```

There is no step cap in the demo; the env's own time limit ends an episode. Trust is computed every
step from the loaded critics, where training updates it every 100 steps.

### 4.22 `demo/checkpoint_rings.py`: display-only checkpoint rings

**Purpose.** Game-style gates every 30 m along the route for the demo: ovals in the server window
(`draw_hud_line`, not rendered into camera sensors) and projected rings plus a HUD in the camera
window through `CarlaGymEnv.display_overlay`. The observation is never touched. Added in `0f2d2a2`;
finish gate counted in `247e89c`.

| Symbol | Where | Role |
|---|---|---|
| `CheckpointRings` | `demo/checkpoint_rings.py:80` | Installs itself as `display_overlay` |
| `start_episode` | `demo/checkpoint_rings.py:94` | Gate positions from the new route |
| `update` | `demo/checkpoint_rings.py:105` | Collects gates by route progress, draws the overview |
| `finish_episode` | `demo/checkpoint_rings.py:119` | A finished route collects all remaining gates |
| `overlay` | `demo/checkpoint_rings.py:146` | Draws rings and HUD on a copy of the camera image |

From `demo/checkpoint_rings.py:105-117`:

```python
    def update(self):
        snap = getattr(self.env, "_snapshot", None)
        if not snap or not self.gates:
            return
        s = float(snap.get("s", 0.0))
        while self.collected < len(self.gates) and s + 1.0 >= self.gates[self.collected]["s"]:
            self._flash.append((self.gates[self.collected], FLASH_TICKS))
            self.collected += 1
            self._banner = FLASH_TICKS
        self._flash = [(g, n - 1) for g, n in self._flash if n > 1]
        self._banner = max(self._banner - 1, 0)
        if self.overview:
            self._draw_overview()
```

A gate counts as collected when the car's route progress `s` reaches it. The env ends an episode
with less than 3 m of route left, slightly before the finish gate, so `finish_episode` credits all
gates on `endpoint`.

### 4.23 `tests/unit/`: offline unit tests

**Purpose.** Tests that need no CARLA server (fakes for the world, map and actors, plus real SB3 SAC
on toy envs). Each file is its own runner: `python tests/unit/test_<name>.py` (RUNNING.md §1).
The root-level `test_*.py` / `check_*.py` scripts predate the rework and target the old env API.

| File | Tests | Covers | Added in |
|---|---|---|---|
| `tests/unit/test_env_reward.py` | 39 | Reward v3, terminations, nav vector, helpers, offline env against a fake world, traffic fill, steering smoothing and penalty, corridor on the ego | 3da9b1e, 21c9889, 0f2d2a2, 9b93e48 |
| `tests/unit/test_cbf.py` | 29 | Passthrough, closed-loop stopping with brake delay, rows, trust monotonicity, OSQP cross-check, wrapper, `trust_lambda` loading | b9de273 |
| `tests/unit/test_exp_glue.py` | 4 | CNN wrapper, occlusion on image only, warm-up and resume helpers, curriculum persistence | 0470e13, c001d68 |
| `tests/unit/test_metrics_callbacks.py` | 10 | Episode/eval callbacks, tag inventory, resume de-duplication, run config, train_all parser, make_report | 16ec944 |
| `tests/unit/test_route.py` | 11 | Route geometry and planning | 1dd0032, 247e89c |
| `tests/unit/test_perception.py` | 13 | Lead finding, LiDAR gap, corridor offset | 1dd0032, 9b93e48 |
| `tests/unit/test_overtake.py` | 5 | Queue detection, clear-lane box, overtake tracker (pays once; nothing after a crash, an illegal crossing or the timeout; off by default) | 9b93e48 |
| `tests/unit/test_traffic_rules.py` | 7 | Lights, stop signs, lane-marking legality | 1dd0032 |
| `tests/unit/test_expert.py` | 8 | Expert driver | 1dd0032 |
| `tests/unit/test_stt.py` | 7 | Cache and padding, PCA, gradients, SAC trains the transformer, save/load, fast paths | c001d68 |
| `tests/unit/test_resnet_extractor.py` | 8 | Output layout, frozen/trained split, trunk reuse, checkpoint round trip, CLI names | 6fdf5d4 |
| `tests/unit/test_maps_curriculum.py` | 11 | Map helpers, 6-stage schedule, keep-latest checkpoint, log discard, rings, Town01 exclusion, NPC physics guard, 70% traffic caps | 0f2d2a2, bc4855d, 9b93e48 |

`tests/unit/_nav_fakes.py` holds the fake map, waypoints and actors shared by the navigation tests.
Test counts are the number of `def test_` functions per file. On 2026-11-19 at `dce8ace` all 12 files
passed (152 tests) with the project venv and `PYTHONDONTWRITEBYTECODE=1`.

From `tests/unit/test_maps_curriculum.py:58-67`:

```python
def test_six_stages_up_to_300_and_200k_schedule():
    from commons import curriculum_stages as cs

    assert len(cs.STAGES) == 6 and cs.FINAL_STAGE == 5
    assert cs.STAGES[-1]["num_npc"] == 300 and cs.STAGES[-1]["num_walkers"] == 300
    npcs = [s["num_npc"] for s in cs.STAGES]
    assert npcs == sorted(npcs), "traffic only grows"
    assert [s["crash_penalty"] for s in cs.STAGES[:3]] == [50.0, 100.0, 150.0], "earlier stages unchanged"
    assert cs.PROMOTE_BY_STEP[-1] < cs.CURRICULUM_TOTAL_STEPS == 200_000
    assert [cs.scheduled_stage(s) for s in (0, 14_999, 15_000, 75_000, 125_000, 140_000, 200_000)] == [0, 0, 1, 3, 4, 5, 5]
```

This test pins the stage table and the schedule; changing either should fail it first.

### 4.24 Removed or retired code

| What | Status | Commit |
|---|---|---|
| v1 reward methods, `SafetyMetricsCallback`, `ComprehensiveMetricsLoggingCallback`, `train_sac_agent`, `main` in `tests/pipeline_carla_test.py` | Removed | 3da9b1e |
| Obstacle sensor and depth camera | Removed | 3da9b1e |
| Per-script callback classes (`TrustScoreCallback`, `ReversalRewardCallback`, ...) and wrappers (`ImageObservationWrapper`, `LocationContextWrapper`, `OcclusionWrapper`) in the experiments | Removed | 14f9a28 |
| `TransformerCheckpointCallback`, transformer side files, `--encoder-path` | Removed | c001d68 |
| `PipelineObservationWrapper` (`tests/pipeline_carla_test.py:1598`), `create_carla_env` (`tests/pipeline_carla_test.py:1847`) | Kept, unused by experiments and demo | — |
| `patch_experiments.py` | Stub that prints "nothing to patch" | 14f9a28 |
| `examples/` and `import examples` in the root `__init__.py` | Package deleted by the update-v1 merge; import removed | 26946b8, 328fad1 |
| `train_all (1).py` | Renamed to `train_all.py` | 16ec944 |

---

## 5. Runner and training queue

### 5.1 `train_all.py`: supervisor with auto-resume

**Purpose.** Runs one or more experiments to a target total, always resuming from the newest
checkpoint, restarting CARLA and the script after a crash, and giving up after repeated attempts with
no progress. Run it from `CARLA-RL-Agents` (it finds `../experiments` from there). Flags: RUNNING.md
§3.1. Reworked in `16ec944`, `--headless` in `63c1b23`, ResNet entries in `6fdf5d4`, memory guard and
stall watchdog in `81fc91f` (section 5.2).

| Symbol | Where | Role |
|---|---|---|
| `EXPERIMENTS` | `train_all.py:71` | Script, log dir and checkpoint prefix per name; `5q_cnn_resnet_*` are opt-in (run only with `--only`) |
| `parse_step` | `train_all.py:98` | Reads the step from the per-episode line, else from SB3's table |
| `latest_checkpoint` | `train_all.py:137` | Newest `<prefix>_<N>_steps.zip` in `<log-dir>/checkpoints` |
| `prune_checkpoints` | `train_all.py:152` | Keeps the newest N (default 3) with both buffer names |
| `kill_process_tree` | `train_all.py:179` | `taskkill /F /T` on Windows |
| `carla_command` | `train_all.py:464` | Visible `-windowed`, or `-RenderOffScreen` with `--headless`; `-quality-level=Low` |
| `log` | `train_all.py:114` | Runner messages to the console and, since `81fc91f`, to `<log-root>/train_all.log` |
| `stream_run` | `train_all.py:502` | Starts the script suspended inside its memory job, mirrors output to the console and `runner.log`, tracks steps, errors and the time of the last output line, writes the exit summary |
| `build_command` | `train_all.py:597` | Passes `--timesteps` (target), `--port`, `--log-dir`, `--buffer-size`, `--render`, traffic maxima, `--load-checkpoint` |
| `run_experiment` | `train_all.py:619` | The attempt loop; names a stall kill, a memory-guard kill or the decoded exit code as the reason |
| `main` | `train_all.py:795` | Selects experiments, ports, `skip_experiments.txt`, serial or parallel; sets up `train_all.log` and the runner's kill-on-close job |

How one experiment is driven (`run_experiment`):

1. Find the newest checkpoint. If its step count meets the target: **done**.
2. If `failures ≥ --max-retries` (default 5): **give up** and return False.
3. With `--carla-exe`: kill the CARLA process the runner started last time (if any) and start a new
   one, then wait up to `--carla-wait` (180 s) for the port. Without it: wait for a server on the
   port; if none answers, count a failure.
4. Run the script with `--load-checkpoint <newest>` (or fresh) under the memory guard (section 5.2)
   and stream its output.
5. After the script exits (or the memory guard killed it): if a newer checkpoint exists, reset
   `failures` to 0 and prune; otherwise `failures += 1`. If the highest step printed reached the
   target and no `[ERROR]` / traceback line was seen: **done**. Otherwise the log line names the reason:
   `stalled: no output for N min`, `memory guard at N GB`, `error in the script`, or the exit code
   decoded by `describe_exit`.
6. Wait `--restart-delay` (20 s) and loop. The CARLA process is killed in `finally`.

From `train_all.py:643-653`:

```python
        while not STOP.is_set():
            checkpoint, done = latest_checkpoint(log_dir, cfg["prefix"])
            remaining = args.timesteps - done

            if remaining <= 0:
                log(f"[{name}] DONE - checkpoint at {done:,} steps meets the {args.timesteps:,} target")
                return True

            if failures >= args.max_retries:
                log(f"[{name}] GIVING UP after {failures} attempts with no progress. See {runner_log}")
                return False
```

From `train_all.py:696-705`:

```python
            _, done_after = latest_checkpoint(log_dir, cfg["prefix"])
            if done_after > done:
                failures = 0
                prune_checkpoints(log_dir, cfg["prefix"], args.keep_checkpoints, name)
            else:
                failures += 1

            if highest_step >= args.timesteps and not saw_error:
                log(f"[{name}] DONE - reached {highest_step:,} steps")
                return True
```

"Progress" means a new checkpoint file. Checkpoints are written every 5k total steps since `dce8ace`
(25k before; plus one on a caught exception or Ctrl+C), so an attempt that dies hard before the next
5k mark counts as a failure. With 25k an attempt could train for hours and still count as one: on
2026-11-18, 2q_cnn stayed at its 150k checkpoint all day (section 7).

Other behaviour: `--render` is on by default; `--parallel K` runs K experiments on ports
`--port-base + i·--port-step` (step 10 when parallel); without `--only`, names in
`../skip_experiments.txt` are skipped; the child env gets `PYTHONPATH`, unbuffered UTF-8 output and
`WANDB_MODE=offline`.

### 5.2 `train_all.py` memory guard and stall watchdog (`81fc91f`)

**Purpose.** On 2026-11-18 an experiment process reached 75.5 GB of virtual memory and the PC went
down (section 7). The runner now bounds every experiment's memory in two layers and makes sure nothing
it started outlives it. On 2026-11-19 a native hang inside `world.tick()` preceded a crash, so the same
thread also restarts an experiment that stops printing. The module docstring summarises the memory
part (`train_all.py:50-57`). The job objects and the memory checks are Windows-only; the stall check
runs on every OS.

| Symbol | Where | Role |
|---|---|---|
| `CREATE_SUSPENDED` | `train_all.py:214` | Creation flag: the experiment starts suspended |
| `Job` | `train_all.py:272` | Windows Job Object: kill-on-close, no crash dialog, optional per-process commit cap |
| `Job.peak_gb` | `train_all.py:301` | Highest committed memory of any process in the job (`PeakProcessMemoryUsed`) |
| `Job.max_private_gb` | `train_all.py:320` | Largest private memory of the processes now in the job |
| `Job.terminate` | `train_all.py:332` | `TerminateJobObject`: kills the launcher and the interpreter together |
| `resume_process` | `train_all.py:351` | `NtResumeProcess` on the suspended child |
| `experiment_private_gb` | `train_all.py:359` | Private memory of the experiment (the job's largest process), None once it exited |
| `MemoryGuard` | `train_all.py:369` | Watchdog thread: polls every 0.25 s, logs `[MEM]`, kills above the soft limit or after `stall_s` without output |
| `_EXIT_CODES`, `describe_exit` | `train_all.py:422-437` | Exit code with its NTSTATUS name, e.g. `3221225477 (0xC0000005, access violation (native crash))` |
| `--mem-limit-gb`, `--mem-hard-gb`, `--stall-minutes`, `--mem-log-every` | `train_all.py:781-791` | Soft limit 14 GB, hard cap 16 GB, stall limit 15 min, a reading every 300 s; 0 disables a limit |

**Layer 1, the OS cap.** The experiment is created with `CREATE_SUSPENDED`, put into its own job and
only then resumed. `venv\Scripts\python.exe` is a launcher whose child is the real interpreter, so the
interpreter is born inside the job and inherits the cap; there is no window in which it runs
unguarded. From `train_all.py:528-543`:

```python
            creationflags=CREATE_SUSPENDED if os.name == "nt" else 0,
        )
        job = None
        try:
            job = Job.for_popen(proc, mem_hard_gb)
            if mem_hard_gb > 0 and os.name == "nt":
                write(f"memory guard: hard cap {mem_hard_gb:g} GB (OS job limit), "
                      f"watchdog kill at {mem_limit_gb:g} GB")
        except OSError as exc:   # never run unguarded silently
            job = None
            write(f"WARNING could not attach the memory job object: {exc}")
            log(f"[{name}] WARNING could not attach the memory job object: {exc}")
        finally:
            resume_process(proc)   # only now may the launcher start the interpreter (inside the job)
        guard = MemoryGuard(proc, job, name, mem_limit_gb, mem_log_every, write, stall_s=60.0 * stall_minutes)
        guard.start()
```

The job's limits, from `train_all.py:283-288`:

```python
        info = _EXTENDED_LIMITS()
        flags = _JOB_LIMIT_KILL_ON_JOB_CLOSE | _JOB_LIMIT_DIE_ON_UNHANDLED_EXCEPTION
        if hard_limit_gb > 0:
            flags |= _JOB_LIMIT_PROCESS_MEMORY
            info.ProcessMemoryLimit = int(hard_limit_gb * GB)
        info.BasicLimitInformation.LimitFlags = flags
```

`ProcessMemoryLimit` caps each process's committed memory: an allocation beyond it fails inside the
experiment (in the offline hog test, with `MemoryError`) instead of exhausting the page file.
`DIE_ON_UNHANDLED_EXCEPTION` stops Windows from holding a crashed process open behind a "has stopped
working" dialog. If the job cannot be created, the runner says so in both logs and runs the
experiment without it.

**Layer 2, the watchdog.** From `train_all.py:392-416`:

```python
    def run(self):
        last_report = time.time()
        while not self._halt.wait(0.25):   # a runaway reached ~2 GB/s; poll fast
            if self.proc.poll() is not None:
                return
            silent = time.time() - self.last_output
            if self.stall_s > 0 and silent > self.stall_s:
                self.stalled_s = silent
                self._kill(f"[STALL] {self.name}: no output for {silent / 60:.1f} min (limit "
                           f"{self.stall_s / 60:g} min) - killing it; it restarts from the latest checkpoint")
                return
            if os.name != "nt":
                continue
            gb = experiment_private_gb(self.proc, self.job)
            if gb is None:
                return
            self.peak_gb = max(self.peak_gb, gb)
            if self.limit_gb > 0 and gb > self.limit_gb:
                self.tripped_gb = gb
                self._kill(f"[MEMGUARD] {self.name}: experiment uses {gb:.1f} GB private memory (limit "
                           f"{self.limit_gb:g} GB) - killing it; it restarts from the latest checkpoint")
                return
            if self.log_every_s > 0 and time.time() - last_report >= self.log_every_s:
                last_report = time.time()
                self.write(f"[MEM] {self.name}: private {gb:.2f} GB (peak {self.peak_gb:.2f} GB)")
```

The memory watchdog fires below the OS cap, so a runaway normally ends as a clean, logged kill and the
attempt loop restarts the experiment from the latest checkpoint. If `--mem-hard-gb` is not above
`--mem-limit-gb`, `main` lowers the soft limit to 90% of the hard cap (`train_all.py:814-815`).

**Stall watchdog.** `stream_run` stamps `guard.last_output` for every line the experiment prints
(`train_all.py:545-546`). A native hang inside `world.tick()` neither exits nor grows, so after
`--stall-minutes` (default 15) without a line the same thread kills the job (`[STALL]` in both logs)
and the attempt loop restarts from the latest checkpoint. The training scripts print one line per
episode plus SB3's table every 1,000 steps. A phase that legitimately prints nothing for longer
would also be killed; raise the limit or pass 0 to disable it.

**Exit summary.** When the experiment ends, `stream_run` writes one line to `runner.log` and
`train_all.log`, from `train_all.py:578-586`:

```python
            summary = f"experiment exited: code {describe_exit(returncode)}, peak memory {peak:.2f} GB"
            if guard.stalled_s is not None:
                summary += f" - killed by the stall watchdog after {guard.stalled_s / 60:.1f} min without output"
            elif guard.tripped_gb is not None:
                summary += f" - killed by the memory guard at {guard.tripped_gb:.1f} GB"
            elif mem_hard_gb > 0 and peak >= 0.98 * mem_hard_gb:
                summary += f" - hit the {mem_hard_gb:g} GB hard cap"
            write(summary)
            log(f"[{name}] {summary}")
```

**The runner's own job and log.** `main` creates `<log-root>/train_all.log` (default
`CARLA-RL-Agents/logs/train_all.log` when run from `CARLA-RL-Agents`) and puts the runner itself into
a kill-on-close job without a memory cap (`train_all.py:810-821`). CARLA, the experiments and
`taskkill` inherit it, so when the runner dies (console closed, Ctrl+C, crash) the CARLA server and
the experiment it started die with it, instead of being left for the queue's clean-slate check. Every
`log()` line also goes to `train_all.log` with a full date (`train_all.py:114-122`). The start-up
banner reports the guard settings (`train_all.py:855-859`).

Offline checks on 2026-11-19 (no CARLA): a memory-hog test confirmed that the watchdog kills at the
soft limit, and that the OS cap alone stops a 1 GB/s leak with `MemoryError`; a silent child was
killed by the stall watchdog. The 2q_cnn session that resumed at 08:09 under the memory guard reached
200k without a crash, peak memory 6.88 GB (section 7).

### 5.3 Training queue (outside the repo)

Files (read only for this document; not in git):
`D:\safety-aware-drl-carla\training_queue\run_queue.ps1` and
`D:\safety-aware-drl-carla\training_queue\queue_order.txt`.

**Scheduled task "CARLA-Training-Queue"** (as registered on 2026-11-19): logon trigger; action
`powershell.exe -NoProfile -ExecutionPolicy Bypass -NoExit -File "...\training_queue\run_queue.ps1"`;
no execution time limit; a second start while one runs is ignored (`MultipleInstances = IgnoreNew`);
runs on battery.

**Configuration** (`run_queue.ps1:12-22`): Python from
`D:\safety-aware-drl-carla\safety-aware-drl-carla-update-v1\venv`, code from `wt-maps`, logs to
`D:\safety-aware-drl-carla\safety-aware-drl-carla-main\CARLA-RL-Agents\logs`, CARLA
`D:\WORK\Carla\CarlaUE4.exe`, target 200,000 steps, default order `5q_cnn, 2q_cnn, 2q_stt, 5q_stt`
if `queue_order.txt` is missing or empty.

**What it does, in order:**

1. Keeps the PC awake (`SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)`,
   `run_queue.ps1:72-73`), sets `PYTHONDONTWRITEBYTECODE=1`, logs the current step of every listed
   experiment, and starts TensorBoard on 127.0.0.1:6006 if nothing listens there (`run_queue.ps1:81-86`).
2. Loop (`run_queue.ps1:88-108`): re-read `queue_order.txt` (lines starting with `#` ignored); pick the
   first experiment that has not been tried in this session and whose newest checkpoint is below
   200k; wait for a clean slate; run `train_all.py --only <name> --timesteps 200000 --carla-exe ...
   --headless --port-base 2000 --keep-checkpoints 1 --num-npc 300 --num-pedestrians 300 --log-root <logs>
   --mem-limit-gb 14 --mem-hard-gb 16`
   from `wt-maps\CARLA-RL-Agents`; this blocks until the runner returns; wait for a clean slate again.
3. Each experiment is tried **once per queue session**. If the runner gives up, the queue moves on.
4. At the end, if any listed experiment is below 200k: log "not finished", `exit 1`; the task stays
   registered and runs again at the next logon. If all reached 200k: unregister the task, `cd` to
   `%TEMP%`, delete the `training_queue` folder. `training_queue.log` lives in the logs folder and
   is kept.

From `run_queue.ps1:88-108`:

```powershell
$attempted = @{}
while ($true) {
    # Re-read the order before every experiment; each one is tried once per queue session.
    $name = Get-Order | Where-Object { -not $attempted.ContainsKey($_) -and (Get-LatestStep $_) -lt $Target } |
        Select-Object -First 1
    if (-not $name) { break }
    $attempted[$name] = $true
    $done = Get-LatestStep $name
    Wait-CleanSlate "before $name"
    Log "${name}: starting at $done steps (target $Target)"
    Set-Location (Join-Path $Repo 'CARLA-RL-Agents')
    # Blocks until this experiment has ended (target reached, or the runner gave up after repeated failures).
    # Memory guard (train_all.py): watchdog restart above 14 GB, OS hard cap 16 GB per experiment;
    # CARLA and the experiment die with the runner. Runner log: <logs>\train_all.log
    & $Py ..\train_all.py --only $name --timesteps $Target --carla-exe $CarlaExe --headless --port-base 2000 `
        --keep-checkpoints 1 --num-npc 300 --num-pedestrians 300 --log-root $Logs `
        --mem-limit-gb 14 --mem-hard-gb 16
    $rc = $LASTEXITCODE
    Wait-CleanSlate "after $name"
    Log "${name}: runner exited (code $rc); latest checkpoint $(Get-LatestStep $name) steps"
}
```

**Changed on 2026-11-19** (outside git, so no commit records it): the `train_all.py` call passes
`--mem-limit-gb 14 --mem-hard-gb 16` (the runner's defaults, spelled out), and `Log()` writes the log
file before the console. Before, it wrote the console first and lost log lines when the console broke
(on 2026-11-18, under memory pressure, according to the script's comment). From
`run_queue.ps1:32-38`:

```powershell
function Log([string]$msg) {
    $line = "[queue $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"
    # File first: on 2026-11-18 the console broke under memory pressure and Write-Host threw before
    # the line reached the log. The console write may fail; the log line must not.
    Add-Content -Path $QueueLog -Value $line -Encoding utf8
    try { Write-Host $line -ForegroundColor Cyan } catch { }
}
```

The clean-slate check, from `run_queue.ps1:53-69`:

```powershell
function Get-Leftovers {
    Get-CimInstance Win32_Process | Where-Object {
        $_.Name -like 'CarlaUE4*' -or ($_.Name -eq 'python.exe' -and $_.CommandLine -match '_experiment\.py|train_all\.py')
    }
}

function Wait-CleanSlate([string]$why) {
    # Never start the next experiment while anything of the previous one is still alive.
    $deadline = (Get-Date).AddMinutes(3)
    while ((Get-Leftovers) -and (Get-Date) -lt $deadline) { Start-Sleep -Seconds 5 }
    $left = Get-Leftovers
    if ($left) {
        Log "$why - stopping leftover processes: $(($left | ForEach-Object { "$($_.Name)($($_.ProcessId))" }) -join ', ')"
        foreach ($p in $left) { & taskkill /F /T /PID $p.ProcessId 2>$null | Out-Null }
        Start-Sleep -Seconds 10
    }
}
```

The check matches **every** `CarlaUE4*` process and every `python.exe` running an
`*_experiment.py` or `train_all.py`, not only those the queue started. After a 3-minute grace period
they are force-killed.

`queue_order.txt` at the time of writing lists `2q_cnn`, `5q_cnn_resnet_layer4`, `2q_stt`, `5q_stt`.
Its comment says `5q_cnn` (NatureCNN) is parked at its 75k checkpoint and not in the queue.

State at the end of 2026-11-19: 2q_cnn reached 200k at 09:39, so the queue skips it.
5q_cnn_resnet_layer4 had started at 09:58 on the old 100% curriculum; it was stopped at 30k at the
user's request and its folder renamed to `logs/5q_cnn_resnet_layer4__old-curriculum_2026-11-19`, so
the next run of the queue starts it fresh on the 70% curriculum, followed by 2q_stt and 5q_stt. The
scheduled task is **disabled** and is re-enabled only when the user decides to. The queue does not pass
`--stall-minutes`, so the runner's default of 15 minutes applies.

---

## 6. Compatibility and gotchas

Each item was checked in the code at `dce8ace`.

**Checkpoints that cannot be resumed**

- **Checkpoints from before reward v3** (3-D action, old observation dict, image-only `CnnPolicy` or
  512-d `MlpPolicy`): `SAC.load(..., env=env)` raises `ValueError` on the space mismatch; the scripts
  print a message and return 1 (`experiments/2qCNNsac_experiment.py:130-136`).
- **STT checkpoints from before `c001d68`** (512-d vector observation) are incompatible, and an STT
  checkpoint without `stt_projection` is refused (`experiments/2qsac_stt_experiment.py:149-152`).
  The demo refuses both as well (`demo/run_model_inference.py:91-95`).
- **5q CNN image branches do not mix.** A NatureCNN checkpoint cannot be resumed with
  `--cnn resnet50` and vice versa, and ResNet modes do not mix either
  (`experiments/5qCNNsac_experiment.py:121-124`). Each branch has its own log dir and prefix, so the
  `5q_cnn_resnet_*` runs start fresh.
- **Checkpoints from before `0f2d2a2`** (4-stage curriculum) load, because the spaces did not change,
  but stage numbers now mean different traffic and crash penalties: the old S3 was "full traffic"
  (the script's `--num-npc` / `--num-pedestrians`, C = 300); the new S3 is 60/60 with C = 200. A
  resumed run is also lifted to the scheduled stage for its step count. Example from the 2q_cnn
  logs: `curriculum_state.json` records a `schedule` event from stage 2 to 4 at step 125,000.
- **5q CNN / 2q STT / 5q STT checkpoints trained before `dce8ace`** (100% traffic, reward 3.0, for example
  the parked 75k `5q_cnn` NatureCNN run) load, because the spaces did not change, but the current
  scripts resume them with the 70% curriculum and reward 3.1. To continue one exactly as before, pass
  `--traffic-fill 1.0 --steer-smoothing 0 --steer-change-penalty 0 --overtake-bonus 0 --route-corridor`;
  otherwise start fresh (the 30k 5q_cnn_resnet_layer4 run from the old curriculum was set aside, and the
  queue starts that experiment fresh). 2q_cnn checkpoints are unaffected: its script did not opt in.

**Resume semantics**

- **`--timesteps` is the target total** in both the scripts and `train_all.py`. Expert warm-up steps
  are not counted (`commons/warmup.py:50`).
- **`--lr` and `--buffer-size` apply to fresh runs only**; `SAC.load` keeps the checkpoint's values.
  The runner always passes `--buffer-size` (default 50,000); it is ignored on resume.
- **Replay buffer pickles are large and separate**: the 2q_cnn buffer at 150k is 2,128,287,063 bytes.
  Both names `<prefix>_replay_buffer_<N>_steps.pkl` and `<ckpt>_replay_buffer.pkl` are tried
  (`commons/warmup.py:131`). A truncated or mismatched file is skipped with a message and the 20k
  expert warm-up refills the buffer; the weights are restored either way.
- **A resume erases everything logged after the checkpoint step**: TensorBoard events under
  `<log-dir>/tensorboard` (rewritten in place), `episodes.jsonl` and `eval.jsonl` records, and
  `curriculum_state.json` is replaced by the checkpoint's snapshot. This runs before the model is
  loaded (`experiments/2qCNNsac_experiment.py:101-105`), so it also happens when the resume then
  fails (for example an STT checkpoint without `stt_projection`). The 5q CNN branch check runs
  before the clean-up.
- **Only one checkpoint is kept** by `LatestCheckpointCallback(keep=1)`, every 5k total steps since
  `dce8ace` (25k before). A step checkpoint saved after a caught exception or Ctrl+C is deleted at the next
  5k save. Each save also rewrites the replay buffer pickle (about 2 GB at 150k), now five times as
  often.
- **Reward and checkpoint must match.** The crash penalty C depends on the curriculum stage, so a
  stage change rescales the largest reward terms. `run_config.json` records `reward_version` ("3.0",
  or "3.1" with the steering penalty or the overtake bonus), the env constants and the
  training-curriculum options.
- **`trust_lambda` is a policy kwarg.** The experiments do not pass it, so new models use 2.0. A model
  saved with an explicit `trust_lambda` keeps it on load (`tests/unit/test_cbf.py:535-552`).

**Runtime behaviour that surprises**

- **SB3 stores the proposed action, not the CBF output**; the CBF penalty is charged only on steps
  where a constraint row was active (`commons/cbf_wrapper.py:117`).
- **Trust updates every 100 steps** in training (`tests/pipeline_carla_test.py:1817`), every step in
  the demo.
- **Town01 is not in the rotation** (`UNSTABLE_TOWNS`, section 4.1.1): `maps="all"` gives Town02,
  Town03, Town04, Town05, Town10HD here, for training, eval and the demo. An explicit map list still
  loads Town01. A server that already has Town01 loaded is left on it until the first rotation (10
  training episodes; 1 in the demo) or the next eval round.
- **The demo does not apply the training-curriculum options.** `build_env` in
  `demo/run_model_inference.py:86` constructs `CarlaGymEnv` with its defaults: 100% traffic, no
  steering low-pass, route-centred corridor, reward 3.0. A model trained on the 70% curriculum is shown
  without the steering smoothing it was trained with. Evaluation during training uses the training env,
  so it does apply them.
- **Map switches cost time and change the town after evaluation.** Each eval round sets `map_slot`
  0–9, which loads the towns in blocks (with the 5 towns: two episodes each on Town02, Town03,
  Town04, Town05 and Town10HD; before `9b93e48`, with Town01, 6 towns in blocks of 2, 2, 1, 2, 2, 1).
  `_load_map` resets the rotation counter and the auto-reset after the last eval episode passes no
  options (`stable_baselines3/common/vec_env/dummy_vec_env.py:71`), so training continues on the last
  eval town (Town10HD) for the next 10 episodes.
- **Every map switch rebuilds** the ego, sensors, traffic, route planner and traffic-rule caches, and
  since `9b93e48` the Traffic Manager (`tests/pipeline_carla_test.py:737-785`).
- **Traffic Manager ports = CARLA port + 6000 + 100·generation** (`tests/pipeline_carla_test.py:824`):
  a new Traffic Manager for every map load, 50 generations, so a server on 2000 uses 8000, 8100, …,
  12900. Keep parallel servers 10 ports apart (RUNNING.md §2); servers whose ports differ by a
  multiple of 100 would share Traffic Manager ports.
- **The queue's clean-up kills processes it did not start**: any `CarlaUE4*` and any python running
  an experiment script or `train_all.py` (section 5.3). Do not run a separate CARLA (for example
  for the demo) or training while the queue is between experiments.
- **The runner restarts CARLA before every attempt** when `--carla-exe` is given
  (`train_all.py:655-658`).
- **The runner kills experiments on purpose** (section 5.2): above `--mem-limit-gb` (default 14 GB
  private memory), after `--stall-minutes` (default 15) without output, and through the OS cap at
  `--mem-hard-gb` (default 16 GB). Each kill is a failed attempt that resumes from the latest
  checkpoint; 5 attempts in a row without a new checkpoint still end the run. Closing the runner's
  console now also ends its CARLA server and experiment (kill-on-close job). Runner messages are in
  `<log-root>/train_all.log`, the per-experiment exit summaries (decoded exit code, peak memory, kill
  reason) and `[MEM]` readings in `runner.log`.
- **`run_config.json` records `git.dirty`** from `git status --porcelain --untracked-files=no`
  (`commons/metrics_callbacks.py:985`): untracked files do not count, edits to tracked files in
  `wt-maps` do.
- **`tests/pipeline_carla_test.py` imports `models.pipeline` at import time**
  (`tests/pipeline_carla_test.py:53`) for the retired `PipelineObservationWrapper`; `models/` must
  stay importable.
- **The repo tracks some `.pyc` files.** Set `PYTHONDONTWRITEBYTECODE=1` (the queue does) to keep
  `git status` clean.
- **`make_report.py` defaults**: target 1,000,000 for the status column; automatic discovery only of
  the four standard names in `<repo>/CARLA-RL-Agents/logs` and `<repo>/logs` (section 4.20).
- **CARLA version**: server and client are both 0.9.16 (`requirements_train.txt` pins
  `carla==0.9.16`). Earlier versions of RUNNING.md and this file said the server was 0.9.15, which
  was wrong. `commons/vehicle_model.py:3` says its numbers were identified in CARLA 0.9.15 and
  `commons/navigation/` is 0.9.15's route planner; neither was re-checked against 0.9.16.

**Documentation drift found while writing this**

- The docstrings of all four experiment scripts still say "The curriculum S0..S3"
  (`experiments/2qCNNsac_experiment.py:11`, `experiments/5qCNNsac_experiment.py:16`,
  `experiments/2qsac_stt_experiment.py:17`, `experiments/5qsac_stt_experiment.py:17`); there are 6
  stages (S0..S5).
- CLAUDE.md refers to a "Vendored SB3" section below; there is no such section.
- The comment above the memory guard says the watchdog polls "every second"
  (`train_all.py:203-204`); `MemoryGuard` polls every 0.25 s (`train_all.py:394`).
- Fixed in the docs commit of 2026-11-19: METRICS.md's `curriculum/stage` row ended at "3 = full
  traffic"; since `0f2d2a2` the last stage is 5.

---

## 7. Stability incident of 2026-11-18: diagnosis and fixes

### 7.1 What happened

- On 2026-11-18, 2q_cnn was stuck at its 150k checkpoint (`sac_2q_cnn_150000_steps.zip`, written
  at 06:41). The training process (PID 22720, run_id `20261118-140610-22720`, stage S5 with about 230
  vehicles and 240 walkers on Town01_Opt) reached 75.5 GB of virtual memory according to the Windows
  Resource-Exhaustion-Detector (event 2004 at 14:38:50).
- The PC shut down unexpectedly at 06:54, 12:21 (BSOD 0xEF), 19:56 and 22:23; three of the four have a
  power-button timestamp. Two other attempts ended because the console window was closed
  (`forrtl: window-CLOSE event`; task result 0xC000013A).
- From the logs: the run's `run_config.json` records commit `bc4855d`, `dirty: false`; its last
  `episodes.jsonl` record is at step 173,356 (`run_start_step` 150,000). `training_queue.log` shows
  the queue starting 2q_cnn at 150,000 steps at 06:57, 11:57, 14:04 and 23:43, with no "runner exited"
  line after any of them. `runner.log` shows the S5 traffic on Town01_Opt as 229–230/230 vehicles
  (230 = `vehicle_cap(255)` for its 255 spawn points, `tests/pipeline_carla_test.py:287`) and
  237–247/300 walkers. With checkpoints every 25k, none of these sessions reached 175k, so each counted
  as a runner failure and lost its steps.

### 7.2 Reproduction and diagnosis (2026-11-19)

A guarded env-only harness built the training env stack at S5 and recorded memory, sensor queues and
`faulthandler` stacks.

- **Town01_Opt at S5 failed in 7 of 8 runs within 1–4 minutes.** 6 were unbounded memory growth (for
  example 2.58 → 9.87 GB in 40 s), 1 was an access violation (0xC0000005). The 8th ran 6.7 minutes
  without failing.
- **Where it fails.** Every time, the main thread was inside `world.tick()`, i.e. native CARLA client
  code. In synchronous mode the Traffic Manager and walker navigation run in-process during the tick.
- **What did not help.** Hybrid physics off, 0 walkers, 140 cars instead of 230, destroying broken
  NPCs, and starting directly on Town01 with no map switch (that run had no broken cars but still
  segfaulted).
- **Map switch vs fresh Traffic Manager.** After the switch, a Traffic Manager carried across
  `load_world()` produced 6 launched or sunk cars in the first 7 s: cars 40 m up, 60 m below the map,
  56 m/s launches. A Traffic Manager created on the already loaded map produced none.
- **Version.** Server and client are both CARLA 0.9.16.

### 7.3 Fixes

| Fix | Where | Commit |
|---|---|---|
| Town01 left out of `maps="all"` (training, eval, demo); explicit lists still load it | `UNSTABLE_TOWNS`, `town_of`, `_resolve_maps` (4.1.1) | `9b93e48` |
| A new Traffic Manager on a new port for every map load; the old one is shut down before `load_world()` | `_retire_traffic_manager`, `_setup_world`, `TM_PORT_STRIDE`, `TM_GENERATIONS` (4.1.2) | `9b93e48` |
| NPC cars that are launched, sunk or faster than 40 m/s are destroyed after the tick and replaced at the next reset (they were phantom obstacles for perception) | `npc_physics_broken`, `_remove_broken_npcs` (4.1.3) | `9b93e48` |
| OS cap on each experiment's committed memory (16 GB), watchdog kill at 14 GB, stall kill after 15 min without output, restart from the latest checkpoint; children die with the runner; `train_all.log` | `train_all.py` (5.2) | `81fc91f` |
| Latest checkpoint every 5k steps instead of 25k | `SAVE_FREQ` in the four scripts (4.19) | `dce8ace` |
| Queue log written before the console; memory-guard flags passed | `run_queue.ps1`, outside git (5.3) | — |

### 7.4 Validation

- **Tour.** With the fixes, a run touring Town10HD → Town02 → Town03 → Town04 → Town05 → Town10HD at
  S5 (2 episodes per town, 5 map switches, each with a fresh Traffic Manager) had 0 broken cars, peak
  memory 2.65 GB and no crash.
- **Guards.** An offline hog test confirmed that the watchdog kills at the soft limit and that the OS
  cap alone stops a 1 GB/s leak with `MemoryError`; another offline test confirmed the stall kill.
- **Real training.** 2q_cnn was restarted at 08:09 on 2026-11-19 from the 150k checkpoint with these
  fixes and reached 200k at 09:39: 10 map switches, no crash, peak memory 6.88 GB. Its last 50k still
  used the 100% curriculum and reward 3.0 (its script does not opt into section 4.1.4). Its
  `run_config.json` records `dirty: true` at `bc4855d`, because it started before the fixes were
  committed. It ran the stability fixes, the memory guard and the 5k interval as committed in
  `9b93e48`–`dce8ace`; the training-curriculum options (off for 2q_cnn in any case) and the stall watchdog were
  added after it had started.
- **Smoke test of the training curriculum.** With the training defaults (4.1.4): 13 episodes over 7 map
  switches at S5, memory flat at 2.64 GB. The overtake tracker armed on a real queued car through the
  CARLA lane API. The run ended with a native access violation in `world.tick()` on its second Town02
  visit, after a camera timeout and a hang of about 30 s.
- **Unit tests.** All 12 offline files pass (152 tests).

### 7.5 What remains open

- **Town01 is excluded, not repaired.** The root cause inside CARLA's native code (Traffic Manager /
  walker navigation during `world.tick()`) is unconfirmed. Town01 can still be loaded with an explicit
  map list; nothing was changed to make it stable. A server that already has Town01 loaded at start is
  only left at the first rotation or eval round (4.1.1).
- **Sporadic native crashes remain.** Across the runs of 2026-11-19 there was about 1 native crash in
  `world.tick()` per 27 map switches (the smoke test above; the real 2q_cnn session did 10 switches
  without one). They are handled, not prevented: the runner restarts from the latest
  checkpoint (at most 5k steps lost), and the stall watchdog ends hangs that would otherwise block the
  runner. Five attempts in a row without a new checkpoint still end a run.
- **The memory guard is Windows-only.** On other systems only the stall watchdog runs.
- **The 70% curriculum has only been smoke-tested.** No experiment has trained on it yet; the queue
  (5q_cnn_resnet_layer4 fresh, then 2q_stt, 5q_stt) is disabled until the user restarts it.
- **The demo ignores the training-curriculum options** (section 6), so models trained on them are
  demoed without steering smoothing and at 100% traffic. Passing the options through
  `demo/run_model_inference.py` is a follow-up.

---

## 8. Commit index

### 8.1 Rework commits (26946b8..dce8ace, plus the docs commit), oldest first

"What it changed" is taken from the diff; the branch is where the commit was made.

| Date | Hash | Branch | What it changed |
|---|---|---|---|
| 2026-11-16 | 26946b8 | merge-update-v1 | Merges the unrelated `update-v1` history (8035b57) into main 591047a, resolving conflicts in favour of `update-v1` (`tests/pipeline_carla_test.py`, the four experiments, `demo/run_model_inference.py`, `commons/cbfQP_layer.py`, `commons/feature_extractor.py`); adds `reports/` (IEEE paper); deletes `examples/policy_kwargs.py`. This tree is the "before" state |
| 2026-11-16 | 9cc0e2f | merge-update-v1 | Untracks `documentation_updates/` (removes `PHASE_5_3_OBSTACLE_DETECTOR_INTEGRATION.md` from git) and adds it to `.gitignore` |
| 2026-11-16 | a38ee61 | merge-update-v1 | Adds `commons/vehicle_model.py`, `commons/nav_layout.py`, `commons/curriculum_stages.py` (4 stages), `commons/debug.py` |
| 2026-11-16 | 3da9b1e | fix/env | Rewrites `CarlaGymEnv` in `tests/pipeline_carla_test.py` (+1815/−2039): reward v3 as a pure function, `[steer, accel]`, `{rgb_data, nav}`, frame-stamped sensor queues, ordered reset, GRP route, one snapshot per tick, stage setter; removes the v1 reward, old metric callbacks, `train_sac_agent`, `main`, obstacle sensor and depth camera; adds `tests/unit/test_env_reward.py` |
| 2026-11-16 | e383c5a | fix/cbf | Rewrites `commons/cbfQP_layer.py` (+535/−374): discrete-time CBF on `[steer, accel]`, exact collision bound, soft speed and lane rows, closed-form solve with OSQP reference, trust tightens only |
| 2026-11-16 | 82bc4be | fix/cbf | Adds `commons/cbf_wrapper.py`: penalty `0.2·min(1,‖Δu‖²)` on active steps, `info['cbf']`, per-episode CBF stats |
| 2026-11-16 | e9c072e | fix/cbf | `stable_baselines3/sac/policies.py`: trust `exp(−λ·std/(|mean|+1))` on detached values, default `trust_lambda` 0.01 → 2.0 |
| 2026-11-16 | b9de273 | fix/cbf | Adds `tests/unit/test_cbf.py` (603 lines) |
| 2026-11-17 | 77f179a | fix/integration | Merge of fix/env (no conflict changes) |
| 2026-11-17 | 7dd8289 | fix/integration | Merge of fix/cbf (no conflict changes) |
| 2026-11-17 | 328fad1 | fix/integration | Removes `import examples` from the root `__init__.py` |
| 2026-11-17 | 0470e13 | fix/exp | Adds `commons/obs_wrappers.py` (CNN and first STT nav wrappers, transformer checkpoint helpers), `commons/warmup.py` (expert prefill, buffer restore, atomic step checkpoint, stack description), `commons/curriculum.py`, `tests/unit/test_exp_glue.py` |
| 2026-11-17 | 14f9a28 | fix/exp | Rewrites the four experiments and the demo for the v3 stack (MultiInputPolicy for CNN, MlpPolicy for the first STT version), warm-up and curriculum flags, shared callbacks, 25k checkpoints with buffer, target-total `--timesteps`, save on interrupt/error; `patch_experiments.py` becomes a stub |
| 2026-11-17 | d1129ab | fix/integration | Merge of fix/exp (no conflict changes) |
| 2026-11-17 | 16ec944 | fix/metrics | Adds `commons/metrics_callbacks.py`, `make_report.py`, `METRICS.md`, `tests/unit/test_metrics_callbacks.py`; renames `train_all (1).py` → `train_all.py` (target total, episode-line parser, render default, both buffer names pruned, traffic and log-root flags, skip file); updates `FIXES_README (1).md` |
| 2026-11-17 | 10378d5 | fix/integration | Merge of fix/metrics (no conflict changes) |
| 2026-11-17 | c127036 | fix/nav | Adds `commons/navigation/` (CARLA 0.9.15 GlobalRoutePlanner, RoadOption, misc helper; only the import changed) |
| 2026-11-17 | 1dd0032 | fix/nav | Adds `commons/route.py`, `commons/perception.py`, `commons/traffic_rules.py`, `commons/expert.py`, `tests/unit/_nav_fakes.py` and four test files (37 tests) |
| 2026-11-17 | 2593c0b | fix/integration | Merge of fix/nav (no conflict changes) |
| 2026-11-17 | ba24d51 | fix/integration | `episode_step_limit` adds `LIGHT_WAIT_S` = 60 s to the drive-time budget |
| 2026-11-17 | 21c9889 | fix/integration | Updates the episode-limit unit test for the 60 s allowance |
| 2026-11-17 | 9970ebd | fix/integration | Updates CLAUDE.md for reward v3, the CBF, the wrapper stack and the runner |
| 2026-11-17 | 63c1b23 | fix/integration | Traffic Manager port from fixed 8000 to CARLA port + 6000; `train_all.py --headless` |
| 2026-11-17 | 11dc732 | fix/stt | SAC actor loss reuses `actions_pi`/`log_prob` and detaches the critic's features; single-token attention fast path and `need_weights=False` in `commons/spatiotemporal_transformer.py` |
| 2026-11-17 | c001d68 | fix/stt | Adds `commons/stt.py` and `tests/unit/test_stt.py`; STT experiments use `MultiInputPolicy` + `SttFeaturesExtractor` with the PCA projection stored in the `.zip`; removes the old STT wrapper and transformer side files from `commons/obs_wrappers.py`; demo and CLAUDE.md updated |
| 2026-11-18 | 6fdf5d4 | feat/5q-resnet | Adds `commons/resnet_extractor.py`, `--cnn` / `--resnet-train` and per-branch names in `experiments/5qCNNsac_experiment.py`, opt-in `5q_cnn_resnet_*` entries in `train_all.py`, `tests/unit/test_resnet_extractor.py` |
| 2026-11-18 | 0f2d2a2 | feat/maps-curriculum | Map rotation, population caps and `display_overlay` in the env; 6-stage curriculum with step schedule; `LatestCheckpointCallback`, `discard_logs_after`, `restore_curriculum_snapshot`, `mark_tensorboard_resume` in all four experiments; `map` in eval records; demo with one episode per town and `demo/checkpoint_rings.py`; `tests/unit/test_maps_curriculum.py` |
| 2026-11-18 | 247e89c | feat/maps-curriculum | `Route.plan`: destinations within reach, reject routes shorter than half the minimum; demo defaults to stage 5 with 300/300 maxima and credits the finish gate |
| 2026-11-18 | fc1975c | feat/maps-curriculum | Merge of feat/5q-resnet; conflict in `experiments/5qCNNsac_experiment.py` resolved by keeping the branch check and using `LatestCheckpointCallback` with the per-branch `name_prefix`; one CLAUDE.md line merged |
| 2026-11-18 | bc4855d | feat/maps-curriculum | Adds `RUNNING.md`, `requirements_train.txt`, a CLAUDE.md pointer and a finish-gate test in `tests/unit/test_maps_curriculum.py` |
| 2026-11-19 | 9b93e48 | feat/maps-curriculum | Env stability: `UNSTABLE_TOWNS` / `town_of` (Town01 out of `maps="all"`), a new Traffic Manager per map load (`_retire_traffic_manager`, `TM_PORT_STRIDE`, `TM_GENERATIONS`), NPC physics guard (`npc_physics_broken`, `_remove_broken_npcs`, replacement at reset). Opt-in training-curriculum options: `traffic_fill` / `traffic_caps`, `steer_smoothing`, `steer_change_penalty`, `overtake_bonus` (new `commons/overtake.py`), `lead_corridor_follows_ego` (`lateral_offset` in `commons/perception.py`), reward 3.1 on the instance, `add_training_curriculum_args` / `training_curriculum_kwargs`; `write_run_config` records them. Tests in `test_env_reward.py`, `test_maps_curriculum.py`, `test_perception.py` and the new `test_overtake.py` |
| 2026-11-19 | 81fc91f | feat/maps-curriculum | `train_all.py`: experiments start suspended inside a Job Object with a commit cap (`--mem-hard-gb`), `MemoryGuard` watchdog (`--mem-limit-gb`, `--mem-log-every`) and stall kill (`--stall-minutes`), runner kill-on-close job, `train_all.log`, decoded exit codes and peak memory in `runner.log` |
| 2026-11-19 | dce8ace | feat/maps-curriculum | `SAVE_FREQ` 25000 → 5000 in all four experiments; the 5q CNN, 2q STT and 5q STT scripts opt into the training curriculum (`env_kwargs`, `traffic_and_driving` in `run_config.json`) |
| 2026-11-19 | (docs commit) | feat/maps-curriculum | Adds this file; updates RUNNING.md, CLAUDE.md and METRICS.md for the changes above (Markdown only) |

### 8.2 Pre-rework commits brought in by the update-v1 merge

These are older commits of the separate `update-v1` line, not part of the rework. They are in the
range only because `26946b8` merged that line. Summaries come from the diffs and stats.

| Date | Hash | What it changed |
|---|---|---|
| 2026-04-25 | 74b22ab | Root commit of the `update-v1` history: adds the whole project tree (193 files, including the vendored `stable_baselines3`, `commons/`, `experiments/`, `tests/pipeline_carla_test.py`, `training/`) |
| 2026-04-25 | 23a2ae9 | Env: `_distance_to_nearest_vehicle`, `_select_safe_spawn_point` and reward edits; experiments log every iteration (`log_interval` 10 → 1) |
| 2026-05-08 | e4d74e0 | CBF eased (`max_accel_change` 0.3 → 1.2, smaller throttle/brake effects, fallback `[0,0,1]` → `[0,0.25,0.2]`, per-channel clipping); large env/callback rework; deletes `tests/pipeline_carla_callbacks_fixed.py` |
| 2026-05-08 | 46fe2d1 | Obstacle sensor attributes (`hit_radius` 1, `distance` 20) |
| 2026-05-14 | f3358b5 | Reward retuning (collision −300 → −600, forward-motion bonus, scaled obstacle penalties); CBF parameters (`d_min` 5 → 2, `y_max` 1.5 → 1.0, `max_accel_change` 1.9, `alpha_lane` 0.8); `--port` flag in all four experiments; deletes `examples/policy_kwargs.py` |
| 2026-05-15 | e045625 | Brake deadzone branch added to the throttle/brake mutual exclusion; opposite-lane penalty base −20 → −3. The message says the depth map was removed, but the depth camera is still attached in the merged tree (it was removed by 3da9b1e) |
| 2026-06-06 | bd92adc | Resume support in `2qsac_stt` and `5qCNN` experiments; `commons/feature_extractor.py` loads the local ResNet50 weights file; adds the first CLAUDE.md |
| 2026-06-22 | 85e5658 | Resume and replay-buffer `.pkl` saving in all four experiments |
| 2026-09-21 | 8035b57 | Adds `reports/` (IEEE paper `main.tex`, `ref.bib`, figures); trust-enabled replay in `demo/run_model_inference.py`; small resume edits in three experiments |
