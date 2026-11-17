"""Gym action wrapper that filters every action through the discrete-time CBF safety layer.

    env = CBFSafetyLayerWrapper(CarlaGymEnv(...), use_trust_score=True, intervention_penalty=0.2, **cbf_params)

step(action):
    state = env.unwrapped.build_cbf_state()                 (BEFORE env.step)
    safe  = cbf_layer.compute_safe_action(action, state, trust)
    obs, r, terminated, truncated, info = env.step(safe)
    r    -= intervention_penalty * min(1, ||safe - action||^2)   only if a constraint row was active
    info['cbf'] = {active, active_rows, infeasible, correction, trust, penalty}
    last step of an episode: the env's per-episode dict (info['episode_metrics'], falling back to
    info['episode']) is updated in place with cbf_interventions, cbf_intervention_frac, cbf_infeasible,
    cbf_penalty_sum (positive magnitude), trust_mean, cbf_rows.

SB3 stores the action the POLICY proposed in the replay buffer (it never sees `safe`); the intervention
penalty is what teaches the policy to stay inside the safe set (Kasaura et al. 2023).
"""
from __future__ import annotations

import math
import traceback
import warnings

import gymnasium as gym
import numpy as np

from commons.cbfQP_layer import ROW_NAMES, CBFSafetyLayer
from commons.debug import dprint

# Constructor arguments of the pre-rewrite wrapper; accepted and ignored so stale call sites still run.
_LEGACY_WRAPPER_KWARGS = frozenset({"alpha", "correction_penalty"})


class CBFSafetyLayerWrapper(gym.ActionWrapper):
    """Applies the CBF safety filter to every action and reports what it did.

    Backward-compatible attributes read by existing callbacks:
        cbf_layer, trust_score, correction_count (all-time active steps), episode_corrections
        (active steps this episode), last_correction_mag, ensemble_uncertainty (written by
        PolicyTrustScoreCallback).
    """

    def __init__(self, env: gym.Env, use_trust_score: bool = True, intervention_penalty: float = 0.2,
                 **cbf_params):
        super().__init__(env)
        legacy = sorted(k for k in cbf_params if k in _LEGACY_WRAPPER_KWARGS)
        for k in legacy:
            cbf_params.pop(k)
        if legacy:
            warnings.warn(f"CBFSafetyLayerWrapper: ignoring legacy arguments {legacy}", stacklevel=2)

        self.cbf_layer = CBFSafetyLayer(**cbf_params)
        self.use_trust_score = bool(use_trust_score)
        self.intervention_penalty = float(intervention_penalty)
        self.trust_score = 1.0
        self.ensemble_uncertainty = 0.0

        self.correction_count = 0           # all-time number of steps with an active constraint row
        self.total_steps = 0
        self.state_errors = 0
        self.last_correction_mag = 0.0
        self.last_penalty = 0.0
        self._reset_episode_counters()

        try:                                 # some env code looks the wrapper up through this attribute
            env.unwrapped._cbf_wrapper = self
        except Exception:
            pass

    # ------------------------------------------------------------------ trust

    def set_trust_score(self, score: float):
        """Set the critic-ensemble trust in [0, 1] used for the next actions (non-finite values ignored)."""
        try:
            score = float(score)
        except (TypeError, ValueError):
            return
        if math.isfinite(score):
            self.trust_score = float(np.clip(score, 0.0, 1.0))

    def _effective_trust(self) -> float:
        return self.trust_score if self.use_trust_score else 1.0

    # ------------------------------------------------------------------ gym API

    def action(self, action):
        """Return the safe action for `action` (also fills self.cbf_layer.last_info)."""
        trust = self._effective_trust()
        try:
            state = self.env.unwrapped.build_cbf_state()
            safe = self.cbf_layer.compute_safe_action(action, state, trust_score=trust)
            self._last_error = False
        except Exception:
            # Never kill a long training run on a transient state-query failure: pass the action through
            # and make it visible (info['cbf']['error'], counter, first tracebacks printed).
            self.state_errors += 1
            self._last_error = True
            if self.state_errors <= 3 or self.state_errors % 1000 == 0:
                print(f"[CBF] safety filter failed ({self.state_errors}x), action passed through:\n"
                      f"{traceback.format_exc()}")
            safe = np.array(action, copy=True)
            self.cbf_layer.last_info = CBFSafetyLayer._blank_info()
            self.cbf_layer.last_info["trust"] = trust
        return safe

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

        self.total_steps += 1
        self.last_correction_mag = math.sqrt(du2)
        self.last_penalty = penalty
        self._ep_steps += 1
        self._ep_trust_sum += trust
        self._ep_penalty_sum += penalty
        if active:
            self.correction_count += 1
            self.episode_corrections += 1
            for name in layer_info.get("active_rows", ()):
                if name in self._ep_rows:
                    self._ep_rows[name] += 1
        if infeasible:
            self._ep_infeasible += 1

        if not isinstance(info, dict):
            info = {}
        info["cbf"] = {
            "active": active,
            "active_rows": list(layer_info.get("active_rows", [])),
            "infeasible": infeasible,
            "correction": self.last_correction_mag,
            "trust": trust,
            "penalty": penalty,
        }
        if self._last_error:
            info["cbf"]["error"] = True
        if active:
            dprint(f"[CBF-WRAP] rows={info['cbf']['active_rows']} |du|={self.last_correction_mag:.3f} "
                   f"penalty={penalty:.3f} trust={trust:.2f}")

        if terminated or truncated:
            ep = info.get("episode_metrics")
            if not isinstance(ep, dict):
                ep = info.get("episode")
            if isinstance(ep, dict):
                ep.update(self.episode_cbf_stats())
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._reset_episode_counters()
        self.last_correction_mag = 0.0
        self.last_penalty = 0.0
        self.cbf_layer.reset_metrics()
        return self.env.reset(**kwargs)

    # ------------------------------------------------------------------ metrics

    def episode_cbf_stats(self) -> dict:
        """Per-episode CBF summary merged into the env's episode_metrics dict on the last step."""
        n = max(self._ep_steps, 1)
        return {
            "cbf_interventions": int(self.episode_corrections),
            "cbf_intervention_frac": float(self.episode_corrections) / n,
            "cbf_infeasible": int(self._ep_infeasible),
            "cbf_penalty_sum": float(self._ep_penalty_sum),      # positive magnitude subtracted from reward
            "trust_mean": float(self._ep_trust_sum / self._ep_steps) if self._ep_steps else float(self._effective_trust()),
            "cbf_rows": dict(self._ep_rows),
        }

    def _reset_episode_counters(self):
        self.episode_corrections = 0
        self._ep_steps = 0
        self._ep_infeasible = 0
        self._ep_penalty_sum = 0.0
        self._ep_trust_sum = 0.0
        self._ep_rows = {name: 0 for name in ROW_NAMES}
        self._last_error = False
