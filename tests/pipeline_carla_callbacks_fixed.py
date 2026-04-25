"""
FIXED CALLBACKS & ENV UTILITIES — pipeline_carla_test.py
=========================================================

Fixes four compounding bugs that caused flat/constant TensorBoard logs:

  BUG 1 — logger.dump() never called:
      logger.record() only BUFFERS values. Without dump(step), nothing
      writes to TensorBoard. Every _on_step that records must dump.

  BUG 2 — VecEnv unwrapping was broken:
      SB3 auto-wraps in DummyVecEnv, so model.get_env().unwrapped returns
      the DummyVecEnv shell, NOT CarlaGymEnv. Must go through .envs[0].

  BUG 3 — Episode detection via buf_dones is unreliable:
      Use self.locals['dones'] (always populated by SB3 before _on_step).

  BUG 4 — CBFSafetyLayerWrapper traversal never found anything:
      The loop checked hasattr(wrapper, '_cbf_wrapper') but that attribute
      is set on CarlaGymEnv, not on the wrapper layers above it. Fixed by
      traversing with isinstance() checks.

Drop-in replacement: paste this file, then in train_sac_agent() swap:
    safety_callback = SafetyMetricsCallback(verbose=1)
for:
    safety_callback    = SafetyMetricsCallback(verbose=1)
    metrics_callback   = ComprehensiveMetricsLoggingCallback(verbose=1)
    callback=[checkpoint_callback, safety_callback, metrics_callback]
"""

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback


# ============================================================================
# Env Unwrapping Utilities  (fixes BUG 2 + BUG 4)
# ============================================================================

def _unwrap_vec_env(model):
    """
    Return the raw gym.Env (index-0 worker) from an SB3 VecEnv.

    SB3 always wraps user envs in DummyVecEnv / SubprocVecEnv.
    model.get_env() therefore returns a VecEnv, not your gym.Env.
    DummyVecEnv exposes its workers via .envs[]; SubprocVecEnv does not
    (you'd need env.get_attr() there).  We fall back gracefully.
    """
    vec_env = model.get_env()
    if vec_env is None:
        return None

    # DummyVecEnv (most common during single-env training)
    if hasattr(vec_env, "envs"):
        return vec_env.envs[0]

    # SubprocVecEnv — limited introspection available
    # Caller must use vec_env.get_attr() / vec_env.env_method() instead.
    return vec_env


def _find_carla_and_cbf(model):
    """
    Walk the full wrapper chain and return (CarlaGymEnv, CBFSafetyLayerWrapper).
    Either value is None if not found.

    Wrapper stack built in create_carla_env():
        DummyVecEnv
          └─ CBFSafetyLayerWrapper          ← ActionWrapper
               └─ PipelineObservationWrapper ← ObservationWrapper
                    └─ CarlaGymEnv            ← base env
    """
    # Import here to avoid circular import if placed in a separate module.
    # Replace with your actual import path if needed.
    try:
        from __main__ import CarlaGymEnv, CBFSafetyLayerWrapper
    except ImportError:
        CarlaGymEnv = None
        CBFSafetyLayerWrapper = None

    base = _unwrap_vec_env(model)
    if base is None:
        return None, None

    carla_env = None
    cbf_wrapper = None
    current = base

    while current is not None:
        # Use isinstance when classes are available, else fall back to attr checks
        if CarlaGymEnv is not None and isinstance(current, CarlaGymEnv):
            carla_env = current
        elif hasattr(current, "_collision_count") and hasattr(current, "ego_vehicle"):
            carla_env = current  # duck-typing fallback

        if CBFSafetyLayerWrapper is not None and isinstance(current, CBFSafetyLayerWrapper):
            cbf_wrapper = current
        elif hasattr(current, "cbf_layer") and hasattr(current, "correction_count"):
            cbf_wrapper = current  # duck-typing fallback

        current = getattr(current, "env", None)

    return carla_env, cbf_wrapper


# ============================================================================
# SafetyMetricsCallback  (fixes BUG 1 + BUG 2 + BUG 3)
# ============================================================================

class SafetyMetricsCallback(BaseCallback):
    """
    Log safety and navigation metrics to TensorBoard with proper episode summaries.

    Key fixes vs original:
    - Uses _find_carla_and_cbf() for reliable env unwrapping
    - Uses self.locals['dones'] for episode detection (not buf_dones)
    - Calls self.logger.dump(self.num_timesteps) so values actually flush
    - Tracks episode reward via self.locals['rewards']
    """

    def __init__(self, verbose: int = 0, log_frequency: int = 100):
        super().__init__(verbose)
        self.log_frequency = log_frequency

        # Episode accumulators
        self.episode_count = 0
        self.episode_reward = 0.0
        self.episode_length = 0
        self.episode_waypoints = 0
        self.episode_cbf_corrections = 0
        self.episode_collisions = 0

    # ------------------------------------------------------------------
    def _on_step(self) -> bool:

        # --- Accumulate reward from this step (FIX: use locals, not buf_rewards)
        rewards = self.locals.get("rewards", [0.0])
        self.episode_reward += float(rewards[0]) if len(rewards) > 0 else 0.0
        self.episode_length += 1

        # --- Resolve env references once per step
        carla_env, cbf_wrapper = _find_carla_and_cbf(self.model)

        # Sync episode-level accumulators from env state
        if carla_env is not None:
            self.episode_collisions = getattr(carla_env, "_collision_count", 0)
            self.episode_waypoints = getattr(carla_env, "waypoints_crossed", 0)
        if cbf_wrapper is not None:
            self.episode_cbf_corrections = getattr(
                cbf_wrapper, "episode_corrections", 0
            )

        # --- Periodic per-step metric logging
        if self.num_timesteps % self.log_frequency == 0:
            self._record_step_metrics(carla_env, cbf_wrapper)
            # FIX BUG 1: actually flush to TensorBoard
            self.logger.dump(self.num_timesteps)

        # --- Episode-end logging (FIX BUG 3: use locals['dones'])
        dones = self.locals.get("dones", [False])
        if any(dones):
            self._record_episode_summary()
            # Reset accumulators
            self.episode_count += 1
            self.episode_reward = 0.0
            self.episode_length = 0
            self.episode_waypoints = 0
            self.episode_cbf_corrections = 0
            self.episode_collisions = 0

        return True

    # ------------------------------------------------------------------
    def _record_step_metrics(self, carla_env, cbf_wrapper):
        """Write all per-step metrics into the logger buffer."""

        # --- CBF metrics
        if cbf_wrapper is not None:
            cbf = getattr(cbf_wrapper, "cbf_layer", None)

            self.logger.record(
                "safety/cbf_correction_mag",
                float(getattr(cbf_wrapper, "last_correction_mag", 0.0)),
            )
            self.logger.record(
                "safety/total_cbf_corrections",
                float(getattr(cbf_wrapper, "correction_count", 0)),
            )
            self.logger.record(
                "safety/cbf_corrections_episode",
                float(getattr(cbf_wrapper, "episode_corrections", 0)),
            )

            if cbf is not None:
                self.logger.record(
                    "safety/collision_prevented",
                    float(getattr(cbf, "collision_prevented", False)),
                )
                self.logger.record(
                    "safety/avoidance_efficiency",
                    float(getattr(cbf, "avoidance_efficiency", 0.0)),
                )
                violations = getattr(cbf, "constraint_violations", {})
                self.logger.record(
                    "safety/collision_violations",
                    int(violations.get("collision", 0)),
                )
                self.logger.record(
                    "safety/lane_violations",
                    int(violations.get("lane", 0)),
                )
                self.logger.record(
                    "safety/speed_violations",
                    int(violations.get("speed", 0)),
                )

        # --- Carla-env metrics
        if carla_env is not None:
            self.logger.record(
                "safety/collision_distance",
                float(carla_env._compute_collision_distance()),
            )
            self.logger.record(
                "safety/collisions_episode",
                int(getattr(carla_env, "_collision_count", 0)),
            )
            self.logger.record(
                "safety/lane_offset",
                float(carla_env._compute_lane_offset()),
            )

            # Waypoint / navigation
            wp_crossed = getattr(carla_env, "waypoints_crossed", 0)
            wp_total = getattr(carla_env, "total_waypoints", 1)
            self.logger.record("progress/waypoints_crossed", int(wp_crossed))
            self.logger.record(
                "progress/waypoints_remaining", max(0, wp_total - wp_crossed)
            )
            self.logger.record(
                "progress/progress_pct", 100.0 * wp_crossed / max(1, wp_total)
            )
            self.logger.record(
                "progress/distance_to_next_wp",
                float(carla_env._get_distance_to_next_waypoint()),
            )
            self.logger.record(
                "navigation/waypoint_completion_ratio",
                float(getattr(carla_env, "waypoint_completion_ratio", 0.0)),
            )
            self.logger.record(
                "navigation/endpoint_distance",
                float(getattr(carla_env, "endpoint_distance", 9999.0)),
            )
            self.logger.record(
                "navigation/endpoint_reached",
                1.0 if getattr(carla_env, "endpoint_reached", False) else 0.0,
            )

    # ------------------------------------------------------------------
    def _record_episode_summary(self):
        """Log end-of-episode summary and print SB3-style console line."""
        ep_len = max(self.episode_length, 1)

        self.logger.record("episode/return", self.episode_reward)
        self.logger.record("episode/length", ep_len)
        self.logger.record("episode/waypoints_crossed", self.episode_waypoints)
        self.logger.record("episode/cbf_corrections", self.episode_cbf_corrections)
        self.logger.record("episode/collisions", self.episode_collisions)
        # Flush episode summary immediately
        self.logger.dump(self.num_timesteps)

        print(
            f"Episode {self.episode_count + 1:5d} | "
            f"Return: {self.episode_reward:9.2f} | "
            f"Length: {ep_len:4d} | "
            f"Waypoints: {self.episode_waypoints:2d} | "
            f"CBF: {self.episode_cbf_corrections:3d} | "
            f"Collisions: {self.episode_collisions:1d}"
        )


# ============================================================================
# ComprehensiveMetricsLoggingCallback  (fixes all 4 bugs)
# ============================================================================

class ComprehensiveMetricsLoggingCallback(BaseCallback):
    """
    Logs all 6 requested metrics without touching any other functionality:

      1. CBF logging         — when a CBF correction is applied
      2. CBF invoke rate     — CBF invocations / episode length
      3. Lane invasion rate  — invasions / episode length
      4. Car collision rate  — collisions / episode length
      5. Ensemble uncertainty score — SAC entropy coefficient (log-scale)
      6. Trust score         — 1 - clipped(2 * ent_coef)

    All fixes applied:
      - _find_carla_and_cbf() for correct env traversal (BUG 2 + 4)
      - self.locals['dones'] for episode detection (BUG 3)
      - self.logger.dump(self.num_timesteps) after every record block (BUG 1)
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

        # Per-episode accumulators
        self.episode_number = 0
        self.episode_length = 0
        self.episode_cbf_invocations = 0
        self.episode_lane_invasions = 0
        self.episode_collisions = 0

        # Running "last seen" counters to detect increments
        self._prev_cbf_count = 0
        self._prev_collision_count = 0
        self._prev_lane_invasion_count = 0

        # Derived scalars updated each step
        self.current_trust_score = 1.0
        self.current_ent_coef = 0.0

    # ------------------------------------------------------------------
    def _on_step(self) -> bool:
        self.episode_length += 1

        # Resolve envs
        carla_env, cbf_wrapper = _find_carla_and_cbf(self.model)

        # ── 1 & 2. Ensemble uncertainty + trust score ──────────────────
        self._update_trust_score()

        # ── 3. CBF metrics ─────────────────────────────────────────────
        cbf_correction_this_step = False
        if cbf_wrapper is not None:
            current_cbf = getattr(cbf_wrapper, "correction_count", 0)
            delta_cbf = max(0, current_cbf - self._prev_cbf_count)

            if delta_cbf > 0:
                self.episode_cbf_invocations += delta_cbf
                cbf_correction_this_step = True

                self.logger.record("cbf/invoke_event", 1.0)
                self.logger.record(
                    "cbf/correction_magnitude",
                    float(getattr(cbf_wrapper, "last_correction_mag", 0.0)),
                )
                if self.verbose > 0:
                    print(
                        f"[CBF] Step {self.num_timesteps}: correction "
                        f"mag={getattr(cbf_wrapper, 'last_correction_mag', 0):.4f}"
                    )
            else:
                self.logger.record("cbf/invoke_event", 0.0)

            self.logger.record(
                "cbf/total_episode_invocations",
                float(self.episode_cbf_invocations),
            )
            self._prev_cbf_count = current_cbf

        # ── 4. Collision metrics ───────────────────────────────────────
        if carla_env is not None:
            current_col = getattr(carla_env, "_collision_count", 0)
            delta_col = max(0, current_col - self._prev_collision_count)
            if delta_col > 0:
                self.episode_collisions += delta_col
                self.logger.record("collisions/collision_event", 1.0)
            else:
                self.logger.record("collisions/collision_event", 0.0)
            self.logger.record(
                "collisions/collision_count_episode",
                float(self.episode_collisions),
            )
            self._prev_collision_count = current_col

            # ── 5. Lane invasion metrics ───────────────────────────────
            current_li = getattr(carla_env, "_lane_invasion_count", 0)
            delta_li = max(0, current_li - self._prev_lane_invasion_count)
            if delta_li > 0:
                self.episode_lane_invasions += delta_li
                self.logger.record("lane/invasion_event", 1.0)
            else:
                self.logger.record("lane/invasion_event", 0.0)
            self.logger.record(
                "lane/invasion_count_episode",
                float(self.episode_lane_invasions),
            )
            self._prev_lane_invasion_count = current_li

        # ── 6. Per-step rates ──────────────────────────────────────────
        ep_len = max(self.episode_length, 1)
        self.logger.record(
            "rates/cbf_invoke_rate",
            self.episode_cbf_invocations / ep_len,
        )
        self.logger.record(
            "rates/collision_rate",
            self.episode_collisions / ep_len,
        )
        self.logger.record(
            "rates/lane_invasion_rate",
            self.episode_lane_invasions / ep_len,
        )
        self.logger.record("metrics/ensemble_uncertainty_score", self.current_ent_coef)
        self.logger.record("metrics/trust_score", self.current_trust_score)

        # FIX BUG 1: flush every step (cheap; SB3 de-dupes on disk)
        self.logger.dump(self.num_timesteps)

        # ── Episode end (FIX BUG 3: use locals['dones']) ───────────────
        dones = self.locals.get("dones", [False])
        if any(dones):
            self._on_episode_end()

        return True

    # ------------------------------------------------------------------
    def _update_trust_score(self):
        """
        Read SAC entropy coefficient and derive trust score.
        SAC stores ent_coef as a 0-dim tensor when auto-tuned.
        """
        try:
            raw = self.model.ent_coef
            if isinstance(raw, torch.Tensor):
                # ent_coef_tensor is log-scale internally; .exp() gives actual coef
                ent = float(torch.exp(raw).detach().cpu())
            elif isinstance(raw, str):
                # "auto" before first update — fall back to the target
                ent = float(
                    getattr(self.model, "ent_coef_target", torch.tensor(0.1))
                    if isinstance(
                        getattr(self.model, "ent_coef_target", None), (int, float)
                    )
                    else 0.1
                )
            else:
                ent = float(raw)

            self.current_ent_coef = ent
            # Trust: 1 when ent_coef≈0 (confident), 0 when ent_coef≥0.5
            self.current_trust_score = float(np.clip(1.0 - 2.0 * ent, 0.0, 1.0))

        except Exception:
            pass  # Keep previous values — never crash training

    # ------------------------------------------------------------------
    def _on_episode_end(self):
        """Log episode-level summary and reset accumulators."""
        ep_len = max(self.episode_length, 1)
        self.episode_number += 1

        summary = {
            "cbf_invoke_rate":      self.episode_cbf_invocations / ep_len,
            "collision_rate":       self.episode_collisions / ep_len,
            "lane_invasion_rate":   self.episode_lane_invasions / ep_len,
            "cbf_invocations":      float(self.episode_cbf_invocations),
            "collisions":           float(self.episode_collisions),
            "lane_invasions":       float(self.episode_lane_invasions),
            "ensemble_uncertainty": self.current_ent_coef,
            "trust_score":          self.current_trust_score,
        }
        for k, v in summary.items():
            self.logger.record(f"episode_summary/{k}", v)
        self.logger.dump(self.num_timesteps)

        if self.verbose > 0:
            sep = "=" * 72
            print(f"\n{sep}")
            print(f"  EPISODE {self.episode_number} SUMMARY")
            print(f"{sep}")
            for k, v in summary.items():
                print(f"  {k:<30s}: {v:.4f}")
            print(f"{sep}\n")

        # Reset
        self.episode_length = 0
        self.episode_cbf_invocations = 0
        self.episode_lane_invasions = 0
        self.episode_collisions = 0
        self._prev_cbf_count = 0
        self._prev_collision_count = 0
        self._prev_lane_invasion_count = 0
