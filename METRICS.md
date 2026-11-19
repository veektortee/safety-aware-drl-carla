# Metrics

Every number the training prints, logs or reports, defined once. Produced by
`commons/metrics_callbacks.py` (`EpisodeMetricsCallback`, `EvalCallback`, `write_run_config`),
read by `make_report.py`.

Two kinds of episodes:

- **Training episode**: driven while learning, with SAC's exploration noise. Noisy; used for curves.
- **Evaluation episode**: every 25,000 env steps training pauses and the current policy drives
  10 episodes **deterministically** (no noise), episode *i* from spawn point *i* with route seed *i*.
  The same 10 start/goal pairs every round, but NPC traffic is still random, so rounds are comparable,
  not identical. Evaluation steps never enter the replay buffer, `episodes.jsonl` or the curriculum.
  **Headline numbers come from evaluation.**

Direction: ↑ = higher is better, ↓ = lower is better, — = no better direction (context).

## Episode end reasons

Each episode ends for exactly one reason (`end_reason`):

| Reason | Group | Meaning |
|---|---|---|
| `endpoint` | success | Reached the end of the planned route (< 3 m left). |
| `collision` | crash | Hit a vehicle or a static object. |
| `collision_pedestrian` | crash | Hit a pedestrian (console: `pedestrian`). |
| `off_route` | rule violation | Off the drivable road, or > 3.5 m off the route and not in a same-direction lane, for > 1 s. |
| `wrong_way` | rule violation | In an oncoming lane for > 2 s. |
| `red_light` | rule violation | Crossed the stop line of a red light. |
| `blocked` | blocked | Stood still (< 0.5 m/s) for 15 s with no car ahead, no red light and no stop line. No penalty (truncation). |
| `timeout` | timeout | Ran out of the per-route time limit. No penalty (truncation). |

## TensorBoard tags

Our tags are in the run `<log_dir>/tensorboard/metrics`; SB3's `train/*` losses are in its own run
(`<log_dir>/tensorboard/<name>_N`). No tag is written per step; 39 tags in total.

### `episode/` — once per training episode (use TensorBoard's smoothing slider)

| Tag | Unit | Dir. | How computed |
|---|---|---|---|
| `episode/return_total` | reward | ↑ | Sum of every reward the agent received: env return − `cbf_penalty_sum`. This is what SAC maximises. |
| `episode/distance_m` | m | — | Distance driven (sum of speed × 0.05 s). |
| `episode/mean_speed_mps` | m/s | — | Mean forward speed over the episode. Should approach the limit (5–25 m/s) without exceeding it. |
| `episode/route_completion_frac` | 0–1 | ↑ | Route arc length covered / route length at the end of the episode. |
| `episode/speeding_frac` | 0–1 | ↓ | Share of steps with speed above the speed limit. |

### `rolling100/` — after every training episode, over the last 100 training episodes

Rates that only make sense over many episodes. (With `window=N` the namespace is `rollingN/`.)

| Tag | Unit | Dir. | How computed |
|---|---|---|---|
| `rolling100/success_frac` | 0–1 | ↑ | Share of episodes ending `endpoint`. |
| `rolling100/crash_frac` | 0–1 | ↓ | Share ending `collision` or `collision_pedestrian`. |
| `rolling100/blocked_frac` | 0–1 | ↓ | Share ending `blocked`. |
| `rolling100/timeout_frac` | 0–1 | ↓ | Share ending `timeout`. |
| `rolling100/rule_violation_frac` | 0–1 | ↓ | Share ending `off_route`, `wrong_way` or `red_light`. |
| `rolling100/collisions_per_km` | 1/km | ↓ | Σ collisions / Σ km driven over the window. Not logged until the window drove ≥ 0.1 km. |
| `rolling100/illegal_lane_crossings_per_km` | 1/km | ↓ | Σ illegal lane crossings / Σ km. Illegal = solid line, curb/grass, or into an oncoming lane; legal overtakes across broken lines do not count. |

### `eval/` — once per evaluation round (10 deterministic episodes). Headline numbers.

| Tag | Unit | Dir. | How computed |
|---|---|---|---|
| `eval/success_frac` | 0–1 | ↑ | Share of the round's episodes ending `endpoint`. |
| `eval/crash_frac` | 0–1 | ↓ | Share ending in a collision. |
| `eval/collisions_per_km` | 1/km | ↓ | Σ collisions / Σ km over the round (needs ≥ 0.1 km). |
| `eval/illegal_lane_crossings_per_km` | 1/km | ↓ | As in `rolling100/`, over the round. |
| `eval/route_completion_frac` | 0–1 | ↑ | Mean route completion of the round's episodes. |
| `eval/speeding_frac` | 0–1 | ↓ | Step-weighted share of time above the speed limit. |
| `eval/return_total_mean` | reward | ↑ | Mean `return_total` of the round's episodes. |

### `reward/` — once per training episode

| Tag | Unit | Dir. | How computed |
|---|---|---|---|
| `reward/per_step` | reward/step | ↑ | `return_total` / episode length. |
| `reward/share_progress` | 0–1 | — | \|Σ progress\| / Σ over groups of \|group sum\|. Route progress should dominate. |
| `reward/share_time` | 0–1 | — | Same for the time cost. |
| `reward/share_shaping` | 0–1 | — | Same for the driving penalties: `speeding`, `lateral`, `heading`, `illegal_lane`, `stop_sign`, and with reward 3.1 also `steer_change` (penalty) and `overtake` (bonus). |
| `reward/share_terminal` | 0–1 | — | Same for the terminal terms: `goal`, `collision`, `red_light`, `off_route`, `wrong_way`. |
| `reward/share_cbf` | 0–1 | — | Same for the CBF intervention penalty (`cbf_penalty_sum`). |

The five shares sum to 1. A group near 1 dominates what the agent learns. Per-component sums are in
`episodes.jsonl` (`reward_components`) and the report's reward-split chart.

### `safety/` — once per training episode

| Tag | Unit | Dir. | How computed |
|---|---|---|---|
| `safety/cbf_intervention_frac` | 0–1 | ↓ | Share of steps where a CBF constraint (gap, lane edge, speed) was active and the QP changed the action. Clipping does not count. |
| `safety/cbf_infeasible_count` | steps | ↓ | Steps where the CBF QP had no solution and full braking was applied. |
| `safety/min_ttc_s` | s | ↑ | Smallest time-to-collision with the object ahead during the episode, capped at 10 s (10 = never closing in). |
| `safety/min_gap_m` | m | ↑ | Smallest gap to the object ahead along the route, capped at 100 m (100 = nothing ahead). |
| `safety/trust_mean` | 0–1 | ↑ | Mean trust the CBF used: exp(−2 · std(Q)/(\|mean(Q)\|+1)) over the critic ensemble. 1 = critics agree; low trust widens safety margins. The only trust tag. |

### `curriculum/`, `perf/`

| Tag | Unit | Dir. | How computed |
|---|---|---|---|
| `curriculum/stage` | stage | ↑ | Curriculum stage of the episode (0 = empty roads, short routes … 5 = dense traffic). See `commons/curriculum_stages.py`. |
| `perf/env_steps_per_s` | steps/s | ↑ | Training env steps per wall-clock second over the last 1,000 steps (evaluation time excluded). |
| `perf/env_step_ms` | ms | ↓ | Mean time per env step spent in policy inference + CBF + CARLA tick + sensors. |
| `perf/train_step_ms` | ms | ↓ | Mean time per env step spent in SAC gradient updates. |

### SB3 `train/` (SB3's own names, every 1,000 steps)

| Tag | Unit | Dir. | How computed |
|---|---|---|---|
| `train/actor_loss` | — | — | SAC actor loss (mean over the gradient steps since the last dump). |
| `train/critic_loss` | — | ↓ | Critic (Q) loss. Steady growth = value estimates diverging. |
| `train/ent_coef` | — | — | Entropy coefficient α (exploration strength). |
| `train/ent_coef_loss` | — | — | Loss of the automatic α tuning. |
| `train/ensemble_uncertainty` | Q² | ↓ | Variance across the critic ensemble on training batches. The only uncertainty tag. |

### Removed duplicates

`EpisodeMetricsCallback` filters these from SB3's TensorBoard output (the SB3 console table still shows
them): `rollout/ep_rew_mean` (= `episode/return_total`), `rollout/ep_len_mean`, `rollout/success_rate`,
`time/fps` (= `perf/env_steps_per_s`), `train/learning_rate` (constant, in `run_config.json`),
`train/trust_score` and `cbf/trust_score` (= `safety/trust_mean`), `cbf/ensemble_uncertainty`
(= `train/ensemble_uncertainty`), SB3-logger copies of `curriculum/stage`, and anything written through
SB3's logger under the legacy or reserved namespaces `collisions/ lane/ rates/ metrics/
episode_summary/ navigation/ progress/ policy/ reward_components/ cbf/ safety/ episode/ rollout/`.
Disable with `EpisodeMetricsCallback(..., dedupe_sb3_tb=False)`.

## Files in `<log_dir>`

### `episodes.jsonl` — one JSON object per training episode

All keys of the env's `info['episode_metrics']` (minus SB3's `r`/`l`/`t`), plus:

| Field | Unit | Meaning |
|---|---|---|
| `episode` | — | Training episode number (continues across resumes). |
| `step` | steps | Total env steps when the episode ended. |
| `wall_time` | unix s | Time the episode ended. |
| `run_id`, `run_start_step` | — | Process that wrote it and the step it resumed from. After a resume, records of earlier processes with `step > run_start_step` are void (the restored checkpoint never saw them); readers drop them. |
| `return` | reward | Env return, **without** the CBF penalty. |
| `cbf_penalty_sum` | reward | Positive sum of the CBF intervention penalties. |
| `return_total` | reward | `return − cbf_penalty_sum`. |
| `length_steps` | steps | Episode length (1 step = 0.05 s). |
| `end_reason` | — | See the table above. |
| `distance_m`, `mean_speed_mps`, `route_length_m`, `route_completion_frac` | m, m/s, m, 0–1 | Driving. |
| `collisions`, `collisions_pedestrian`, `collisions_total` | count | `collisions_total` = max of the two, so a pedestrian hit counts once whether or not the env also counts it in `collisions` (an episode ends at its first collision). |
| `lane_invasions`, `illegal_lane_invasions` | count | All lane-marking crossings / only illegal ones. |
| `red_light_violations`, `stop_sign_violations` | count | Red lights run / stop signs not stopped at. |
| `speeding_frac` | 0–1 | Share of steps above the limit. |
| `min_ttc_s`, `min_gap_m` | s, m | Closest approach. The env caps `min_ttc_s` at 100 s and `min_gap_m` at 100 m (= never closing in / nothing ahead); a non-finite value is stored as null. |
| `cbf_interventions`, `cbf_intervention_frac`, `cbf_infeasible`, `trust_mean`, `cbf_rows` | count, 0–1, count, 0–1, — | From the CBF wrapper (see `safety/`). |
| `curriculum_stage` | stage | Stage during the episode. |
| `reward_components` | reward | Episode sum of each of the env's reward terms. |
| `reward_groups` | reward | The components summed into progress / time / shaping / terminal / cbf (cbf = −`cbf_penalty_sum`). |
| `reward_residual` | reward | `return_total − Σ reward_groups`; ≈ 0 unless the env's components do not add up to its return. |

### `eval.jsonl` — one JSON object per evaluation round

`step`, `wall_time`, `duration_s`, `run_id`, `run_start_step`, `deterministic`, `n_episodes`,
`curriculum_stage`, all aggregates of the round (`success_frac`, `crash_frac`, `blocked_frac`,
`timeout_frac`, `rule_violation_frac`, `end_<reason>_frac` for every end reason, `collisions_per_km`,
`illegal_lane_crossings_per_km`, `lane_invasions_per_km`, `route_completion_frac`,
`return_total_mean`, `length_steps_mean`, `mean_speed_mps`, `speeding_frac`, `cbf_intervention_frac`,
`trust_mean`, `distance_km`, `steps`, `reward_split`) and `episodes`: one compact record per evaluation
episode (`spawn_index`, `end_reason`, `return_total`, `length_steps`, `distance_m`,
`route_completion_frac`, `collisions_total`, `illegal_lane_invasions`, `speeding_frac`,
`cbf_intervention_frac`, `min_ttc_s`, `curriculum_stage`).

### `run_config.json`

Written at start: `reward_version` ("3.0", or "3.1" when the env's steering-change penalty or overtake
bonus is on: the 5q CNN, 2q STT and 5q STT scripts since 2026-11-19), git commit/branch/dirty, command line, library versions,
SAC hyperparameters, policy class and number of critics, observation/action spaces, the gym wrapper
stack, env constants (the env's UPPER_CASE class attributes) and settings (NPC/pedestrian counts,
time limit, curriculum stage, and the training-curriculum options `traffic_fill`, `steer_smoothing`,
`steer_change_penalty`, `overtake_bonus`, `lead_corridor_follows_ego`), CBF wrapper/layer parameters,
plus whatever the experiment passes in `extra` (the three scripts above add the same options as
`traffic_and_driving`). On resume the previous version is appended to `run_config_history.jsonl`.

## Console

One line per training episode:

```
ep  412 | step    98,340 |  214 steps 1.60 km | END collision  | route  34% | R  +182.4 | speed  7.9 m/s | lane×2 | cbf   3% | S2
```

episode number · total env steps · length and distance · end reason · route completion ·
`return_total` · mean speed · illegal lane crossings · CBF-active share of steps · curriculum stage.
`train_all.py` reads the `step` field for its ETA.

Every 20 episodes, a block over the last 100 episodes (rates as in `rolling100/`, the reward split as in
`reward/share_*`, throughput excluding evaluation, ETA to the target), ending with a one-sentence verdict:

```
── last 100 episodes @ 628k steps · stage S3 ───────────────────────────────────────────
 outcome   success 68% (+4) | crash 13% (ped 3%) | blocked 7% | timeout 9% | off-route 2% | wrong-way 1% | red-light 0%
 driving   route 83% (mean) | speed 5.9 m/s | speeding 5% of steps | return +5.2 (mean)
 safety    collisions 0.8 /km | illegal lane crossings 6.1 /km | cbf active 5% of steps | trust 0.89
 reward    terminal 54% · time 22% · progress 16% · shaping 6% · cbf 2%
 speed     21.4 steps/s | ETA 4h 50m
 verdict   Reaches the goal in 68% of episodes (+4 pts vs previous block); most common failure: crash (13%); covers 83% of the route on average.
─────────────────────────────────────────────────────────────────────────────────────────
```

The `(+4)` after success is the change in percentage points since the previous block. Each evaluation
round prints its 10 episode lines (`eval  3/10 | spawn   2 | ...`) and an `══ EVAL @ ... ══` block with
the same fields.
