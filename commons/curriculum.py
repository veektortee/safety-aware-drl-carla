"""Training curriculum: promote CarlaGymEnv through commons.curriculum_stages.STAGES.

An episode counts as a success when it reached the end of its route
(info['episode_metrics']['end_reason'] == 'endpoint'). The callback promotes stage k -> k+1 once the
current stage has run at least MIN_EPISODES_PER_STAGE episodes and at least PROMOTE_SUCCESS_RATE of
the last PROMOTE_WINDOW of them were successes. A new stage takes effect at the env's next reset;
episodes that were still running under the previous stage are not counted for the new one.

State is persisted to <log_dir>/curriculum_state.json after every episode and restored when training
resumes (model.num_timesteps > 0), so a restart never drops back to an easier stage.
"""

import json
import os
import time
from collections import deque
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from commons.curriculum_stages import (
    FINAL_STAGE,
    MIN_EPISODES_PER_STAGE,
    PROMOTE_SUCCESS_RATE,
    PROMOTE_WINDOW,
    STAGES,
)

STATE_FILE = "curriculum_state.json"


def _clip_stage(stage) -> int:
    return int(min(max(int(stage), 0), FINAL_STAGE))


def stage_name(stage: int) -> str:
    return STAGES[_clip_stage(stage)]["name"]


def load_curriculum_state(log_dir: str) -> Optional[dict]:
    path = os.path.join(log_dir, STATE_FILE)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        return state if isinstance(state, dict) and "stage" in state else None
    except (OSError, ValueError):
        return None


def resolve_start_stage(log_dir: str, start_stage: int = 0, resume: bool = False) -> int:
    """Stage to construct the env with: max(start_stage, saved stage) when resuming, else start_stage."""
    stage = _clip_stage(start_stage)
    if resume:
        saved = load_curriculum_state(log_dir)
        if saved is not None:
            stage = max(stage, _clip_stage(saved["stage"]))
    return stage


class CurriculumCallback(BaseCallback):
    """Promote the curriculum stage from training-episode outcomes (see module docstring).

    :param log_dir: run directory; holds curriculum_state.json
    :param start_stage: stage for a fresh run (use FINAL_STAGE for --no-curriculum). When resuming,
        the saved stage wins if it is higher (never demote).
    """

    def __init__(self, log_dir: str, start_stage: int = 0, verbose: int = 1):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.start_stage = _clip_stage(start_stage)
        self.stage = self.start_stage
        self.window = deque(maxlen=PROMOTE_WINDOW)
        self.episodes_in_stage = 0
        self.total_episodes = 0
        self.history = []
        self.carla_env = None

    @property
    def state_path(self) -> str:
        return os.path.join(self.log_dir, STATE_FILE)

    # --- SB3 hooks --------------------------------------------------------------------------------
    def _on_training_start(self) -> None:
        resumed = self.model.num_timesteps > 0
        saved = load_curriculum_state(self.log_dir) if resumed else None
        restored = saved is not None and _clip_stage(saved["stage"]) >= self.start_stage
        if restored:
            self.stage = _clip_stage(saved["stage"])
            self.total_episodes = int(saved.get("total_episodes", 0))
            self.history = list(saved.get("history", []))
            self.episodes_in_stage = int(saved.get("episodes_in_stage", 0))
            self.window.clear()
            self.window.extend(int(x) for x in saved.get("window", [])[-PROMOTE_WINDOW:])
        else:
            self.stage = self.start_stage
            if saved is not None:
                self.history = list(saved.get("history", []))
                self.total_episodes = int(saved.get("total_episodes", 0))
                self.history.append({"event": "set", "stage": self.stage, "step": int(self.model.num_timesteps)})

        from commons.warmup import find_carla_env
        self.carla_env = find_carla_env(self.model)
        self._apply_stage()
        self.logger.record("curriculum/stage", self.stage)
        self._save()
        if self.verbose:
            if self.stage < FINAL_STAGE:
                rule = (f"promote at >= {PROMOTE_SUCCESS_RATE:.0%} success over the last {PROMOTE_WINDOW} "
                        f"episodes (min {MIN_EPISODES_PER_STAGE} per stage)")
            else:
                rule = "final stage"
            print(f"[CURRICULUM] stage {self.stage}/{FINAL_STAGE} {stage_name(self.stage)}"
                  f"{' (restored)' if restored else ''} | {rule}")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") or []
        for info in infos:
            metrics = info.get("episode_metrics") if isinstance(info, dict) else None
            if not (isinstance(metrics, dict) and "end_reason" in metrics):
                continue
            self.total_episodes += 1
            ep_stage = metrics.get("curriculum_stage", self.stage)
            if ep_stage is not None and int(ep_stage) != self.stage:
                continue  # still ran under the previous stage
            self.window.append(1 if metrics["end_reason"] == "endpoint" else 0)
            self.episodes_in_stage += 1
            if self._should_promote():
                self._promote()
            else:
                self._save()
            self.logger.record("curriculum/stage", self.stage)
        return True

    # --- logic ------------------------------------------------------------------------------------
    def success_rate(self) -> float:
        return float(np.mean(self.window)) if len(self.window) else 0.0

    def _should_promote(self) -> bool:
        return (
            self.stage < FINAL_STAGE
            and self.episodes_in_stage >= MIN_EPISODES_PER_STAGE
            and len(self.window) >= PROMOTE_WINDOW
            and self.success_rate() >= PROMOTE_SUCCESS_RATE
        )

    def _promote(self) -> None:
        old, rate, n_eps = self.stage, self.success_rate(), self.episodes_in_stage
        self.stage = old + 1
        self.history.append({
            "event": "promote", "from": old, "to": self.stage, "step": int(self.num_timesteps),
            "success_rate": round(rate, 3), "episodes_in_stage": n_eps,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        self.window.clear()
        self.episodes_in_stage = 0
        self._apply_stage()
        self._save()
        if self.verbose:
            print(f"[CURRICULUM] step {self.num_timesteps:,}: promoted {stage_name(old)} -> "
                  f"{stage_name(self.stage)} (success {rate:.0%} over the last {PROMOTE_WINDOW} episodes, "
                  f"{n_eps} episodes in stage)")

    def _apply_stage(self) -> None:
        env = self.carla_env
        if env is None or not hasattr(env, "set_curriculum_stage"):
            if self.verbose:
                print("[CURRICULUM] WARNING: CarlaGymEnv.set_curriculum_stage not found; stage not applied")
            return
        if getattr(env, "curriculum_stage", None) != self.stage:
            env.set_curriculum_stage(self.stage)

    def _save(self) -> None:
        state = {
            "stage": self.stage,
            "stage_name": stage_name(self.stage),
            "episodes_in_stage": self.episodes_in_stage,
            "window": list(self.window),
            "window_success_rate": round(self.success_rate(), 3),
            "total_episodes": self.total_episodes,
            "num_timesteps": int(self.num_timesteps),
            "thresholds": {"success_rate": PROMOTE_SUCCESS_RATE, "window": PROMOTE_WINDOW,
                           "min_episodes": MIN_EPISODES_PER_STAGE},
            "history": self.history,
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        os.makedirs(self.log_dir, exist_ok=True)
        tmp = self.state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, self.state_path)
