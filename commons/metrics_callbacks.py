"""Training metrics: one record per episode, rolling summaries, deterministic evaluation, run config.

Every experiment script registers the same three things:

    EpisodeMetricsCallback(log_dir=..., verbose=1)            # episodes.jsonl + TB + console
    EvalCallback(eval_freq=25000, n_episodes=10, log_dir=...)  # eval.jsonl + TB eval/*
    write_run_config(log_dir, model, extra={...})              # run_config.json

What gets written (definitions of every number: METRICS.md):

  <log_dir>/episodes.jsonl        one JSON object per TRAINING episode (source of truth for make_report.py)
  <log_dir>/eval.jsonl            one JSON object per evaluation round (headline numbers)
  <log_dir>/run_config.json       reward version, git commit, hyperparameters, env/CBF parameters
  <log_dir>/tensorboard/metrics/  TensorBoard run with the tags listed in TB_TAGS below

TensorBoard rules: no per-step scalars, each concept logged once, units in the tag name.
SB3 keeps writing its own train/* losses through model.logger; EpisodeMetricsCallback removes
known duplicates (trust logged 4 times, rollout/ep_rew_mean == episode return, legacy namespaces)
from SB3's TensorBoard output only; the SB3 console table is untouched.

The env (tests/pipeline_carla_test.py) and the CBF wrapper put the per-episode dict under
info['episode_metrics'] on the last step of an episode (contract amendment A1: SB3's Monitor
replaces info['episode'] with {r, l, t}, so info['episode'] is ignored here).
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
import subprocess
import sys
import time
from collections import deque
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import KVWriter, TensorBoardOutputFormat

REWARD_VERSION = "3.0"
METRICS_KEY = "episode_metrics"     # info key holding the per-episode dict (contract amendment A1)

TTC_CAP_S = 10.0                    # min_ttc_s is capped here in TensorBoard (inf = never closing in)
GAP_CAP_M = 100.0                   # min_gap_m cap (100 m = nothing ahead)
MIN_KM_FOR_RATES = 0.1              # per-km rates are only reported once the window drove >= 100 m

# --------------------------------------------------------------------------------------------
# End reasons and how they are grouped
# --------------------------------------------------------------------------------------------
END_REASONS = (
    "endpoint", "collision", "collision_pedestrian", "blocked",
    "timeout", "off_route", "wrong_way", "red_light",
)
END_LABELS = {
    "endpoint": "endpoint", "collision": "collision", "collision_pedestrian": "pedestrian",
    "blocked": "blocked", "timeout": "timeout", "off_route": "off-route",
    "wrong_way": "wrong-way", "red_light": "red-light",
}
OUTCOME_GROUPS = {
    "success": ("endpoint",),
    "crash": ("collision", "collision_pedestrian"),
    "blocked": ("blocked",),
    "timeout": ("timeout",),
    "rule_violation": ("off_route", "wrong_way", "red_light"),
}

# --------------------------------------------------------------------------------------------
# Reward components are summarised in five groups (shares in TB, the console and the report)
# --------------------------------------------------------------------------------------------
REWARD_GROUPS = ("progress", "time", "shaping", "terminal", "cbf")
_TERMINAL_HINTS = ("goal", "endpoint", "collision", "crash", "red_light", "off_route",
                   "wrong_way", "terminal", "completion")
_NOT_A_COMPONENT = ("total", "sum", "reward", "return")


def reward_group(component: str) -> Optional[str]:
    """Map an env reward-component name to one of REWARD_GROUPS (None = not a component)."""
    name = str(component).lower()
    if name in _NOT_A_COMPONENT:
        return None
    if "cbf" in name or "intervention" in name:
        return "cbf"
    if "progress" in name:
        return "progress"
    if any(h in name for h in _TERMINAL_HINTS):
        return "terminal"
    if name == "time" or name.startswith("time_") or name.endswith("_time"):
        return "time"
    return "shaping"      # speeding, lateral, heading, illegal_lane_change, stop_sign, ...


# --------------------------------------------------------------------------------------------
# TensorBoard tag inventory (single source of truth; METRICS.md defines each one)
# --------------------------------------------------------------------------------------------
EPISODE_TB_TAGS = (
    "episode/return_total", "episode/distance_m", "episode/mean_speed_mps",
    "episode/route_completion_frac", "episode/speeding_frac",
)
REWARD_TB_TAGS = ("reward/per_step",) + tuple(f"reward/share_{g}" for g in REWARD_GROUPS)
SAFETY_TB_TAGS = (
    "safety/cbf_intervention_frac", "safety/cbf_infeasible_count", "safety/min_ttc_s",
    "safety/min_gap_m", "safety/trust_mean",
)
ROLLING_TB_FIELDS = (
    "success_frac", "crash_frac", "blocked_frac", "timeout_frac", "rule_violation_frac",
    "collisions_per_km", "illegal_lane_crossings_per_km",
)
EVAL_TB_FIELDS = (
    "success_frac", "crash_frac", "collisions_per_km", "illegal_lane_crossings_per_km",
    "route_completion_frac", "speeding_frac", "return_total_mean",
)
PERF_TB_TAGS = ("perf/env_steps_per_s", "perf/env_step_ms", "perf/train_step_ms")
CURRICULUM_TB_TAG = "curriculum/stage"
# SB3's own TensorBoard tags that survive the de-duplication filter (written by model.logger).
SB3_TB_TAGS = ("train/actor_loss", "train/critic_loss", "train/ent_coef", "train/ent_coef_loss",
               "train/ensemble_uncertainty")


def all_tb_tags(window: int = 100) -> List[str]:
    """Every TensorBoard tag a training run produces (SB3 train/* included)."""
    tags = list(SB3_TB_TAGS) + list(EPISODE_TB_TAGS) + list(REWARD_TB_TAGS) + list(SAFETY_TB_TAGS)
    tags += [f"rolling{window}/{f}" for f in ROLLING_TB_FIELDS]
    tags += [f"eval/{f}" for f in EVAL_TB_FIELDS]
    tags += list(PERF_TB_TAGS) + [CURRICULUM_TB_TAG]
    return tags


# SB3-logger keys dropped from TensorBoard (console table unaffected), each a duplicate concept:
SB3_TB_DUPLICATES = frozenset({
    "rollout/ep_rew_mean",        # = episode/return_total (smooth it in TB)
    "rollout/ep_len_mean",        # = length_steps in episodes.jsonl
    "rollout/success_rate",       # = rolling100/success_frac
    "time/fps",                   # = perf/env_steps_per_s
    "train/learning_rate",        # constant, recorded in run_config.json
    "train/trust_score",          # trust is logged once: safety/trust_mean
    "cbf/trust_score",            # (PolicyTrustScoreCallback)
    "cbf/ensemble_uncertainty",   # uncertainty is logged once: train/ensemble_uncertainty
    "curriculum/stage",           # logged once, by EpisodeMetricsCallback from the episode dict
})
# Namespaces of the old callbacks (per-step 0/1 events, snapshots); dropped if anything still writes them.
SB3_TB_LEGACY_PREFIXES = ("collisions/", "lane/", "rates/", "metrics/", "episode_summary/",
                          "navigation/", "progress/", "policy/", "reward_components/", "cbf/",
                          "safety/", "episode/", "rollout/")

_RUN_ID = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"


# --------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------
def _num(value, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out


def _finite(value) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _jsonable(value):
    """Convert numpy / non-finite / exotic values into strict-JSON-safe Python values."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        return round(f, 4) if math.isfinite(f) else None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, deque)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()] if value.size <= 64 else f"ndarray{value.shape}"
    return repr(value)


_ASCII = str.maketrans({"─": "-", "═": "=", "×": "x", "·": "|", "→": "->", "≈": "~"})


def _emit(text: str) -> None:
    """print() that never dies on a cp1252 console/pipe."""
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        print(text.translate(_ASCII).encode("ascii", "replace").decode("ascii"), flush=True)


def _pct(x, width: int = 3) -> str:
    return "n/a" if x is None else f"{100.0 * x:{width}.0f}%"


def human_duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    minutes, secs = divmod(rest, 60)
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _kilo(step: int) -> str:
    return f"{step / 1e6:.2f}M" if step >= 1_000_000 else f"{step / 1e3:.0f}k"


def append_jsonl(path: str, obj: dict) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(_jsonable(obj), separators=(",", ":"), allow_nan=False) + "\n")


def read_jsonl(path: str) -> List[dict]:
    """All parseable lines (a half-written last line is skipped)."""
    out: List[dict] = []
    if not os.path.isfile(path):
        return out
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


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


# --------------------------------------------------------------------------------------------
# Episode record and summaries (pure functions, unit-tested)
# --------------------------------------------------------------------------------------------
def is_episode_end(info) -> bool:
    if not isinstance(info, dict):
        return False
    m = info.get(METRICS_KEY)
    return isinstance(m, dict) and "end_reason" in m


def build_episode_record(metrics: dict, *, episode: int, step: int, run_id: str = _RUN_ID,
                         run_start_step: int = 0, wall_time: Optional[float] = None) -> dict:
    """Flatten info['episode_metrics'] into one episodes.jsonl record (+ derived fields)."""
    m = dict(metrics or {})
    rec: Dict[str, Any] = {
        "episode": int(episode), "step": int(step),
        "wall_time": round(time.time() if wall_time is None else wall_time, 3),
        "run_id": run_id, "run_start_step": int(run_start_step),
    }
    for key, value in m.items():
        if key in ("r", "l", "t", "reward_components", METRICS_KEY, "episode"):
            continue
        rec[str(key)] = _jsonable(value)

    ret_env = _num(m.get("return", m.get("r", 0.0)))
    penalty = abs(_num(m.get("cbf_penalty_sum", 0.0)))
    rec["return"] = round(ret_env, 4)
    rec["cbf_penalty_sum"] = round(penalty, 4)
    rec["return_total"] = round(ret_env - penalty, 4)
    rec["length_steps"] = int(_num(m.get("length_steps", m.get("l", 0))))
    rec["end_reason"] = str(m.get("end_reason", "unknown"))
    rec["collisions_total"] = int(max(_num(m.get("collisions", 0)), _num(m.get("collisions_pedestrian", 0))))
    rec["min_ttc_s"] = _finite(m.get("min_ttc_s"))

    comps: Dict[str, float] = {}
    for key, value in (m.get("reward_components") or {}).items():
        f = _finite(value)
        if f is not None:
            comps[str(key)] = round(f, 4)
    rec["reward_components"] = comps
    groups = {g: 0.0 for g in REWARD_GROUPS}
    for key, value in comps.items():
        g = reward_group(key)
        if g is not None:
            groups[g] += value
    if penalty and not any(reward_group(k) == "cbf" for k in comps):
        groups["cbf"] -= penalty
    rec["reward_groups"] = {g: round(v, 4) for g, v in groups.items()}
    if comps:
        rec["reward_residual"] = round(rec["return_total"] - sum(groups.values()), 4)
    return rec


def reward_shares(group_sums: Iterable[Dict[str, float]]) -> Dict[str, float]:
    """share_g = sum_e |G_g,e| / sum_e sum_g |G_g,e| over the given episodes."""
    totals = {g: 0.0 for g in REWARD_GROUPS}
    for groups in group_sums:
        for g in REWARD_GROUPS:
            totals[g] += abs(_num((groups or {}).get(g, 0.0)))
    denom = sum(totals.values())
    if denom <= 0:
        return {g: 0.0 for g in REWARD_GROUPS}
    return {g: totals[g] / denom for g in REWARD_GROUPS}


def summarize(records: List[dict]) -> Dict[str, Any]:
    """Aggregate a list of episode records (rolling window or one eval round)."""
    n = len(records)
    out: Dict[str, Any] = {"n_episodes": n}
    if n == 0:
        return out
    reasons = [r.get("end_reason", "unknown") for r in records]
    for reason in END_REASONS:
        out[f"end_{reason}_frac"] = reasons.count(reason) / n
    out["end_other_frac"] = sum(1 for x in reasons if x not in END_REASONS) / n
    for group, members in OUTCOME_GROUPS.items():
        out[f"{group}_frac"] = sum(1 for x in reasons if x in members) / n

    dist_km = sum(_num(r.get("distance_m")) for r in records) / 1000.0
    steps = sum(_num(r.get("length_steps")) for r in records)
    out["distance_km"] = dist_km
    out["steps"] = int(steps)

    def per_km(key):
        return sum(_num(r.get(key)) for r in records) / dist_km if dist_km >= MIN_KM_FOR_RATES else None

    out["collisions_per_km"] = per_km("collisions_total")
    out["illegal_lane_crossings_per_km"] = per_km("illegal_lane_invasions")
    out["lane_invasions_per_km"] = per_km("lane_invasions")

    def mean(key):
        vals = [_finite(r.get(key)) for r in records]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    def step_weighted(key):
        if steps <= 0:
            return None
        vals = [(_finite(r.get(key)), _num(r.get("length_steps"))) for r in records]
        vals = [(v, w) for v, w in vals if v is not None]
        w_sum = sum(w for _, w in vals)
        return sum(v * w for v, w in vals) / w_sum if w_sum > 0 else None

    out["route_completion_frac"] = mean("route_completion_frac")
    out["return_total_mean"] = mean("return_total")
    out["length_steps_mean"] = steps / n
    out["mean_speed_mps"] = mean("mean_speed_mps")
    out["speeding_frac"] = step_weighted("speeding_frac")
    out["cbf_intervention_frac"] = step_weighted("cbf_intervention_frac")
    out["trust_mean"] = mean("trust_mean")
    out["reward_split"] = reward_shares(r.get("reward_groups") for r in records)
    out["curriculum_stage"] = records[-1].get("curriculum_stage")
    return out


# --------------------------------------------------------------------------------------------
# Console formatting
# --------------------------------------------------------------------------------------------
def format_episode_line(rec: dict, prefix: Optional[str] = None) -> str:
    """ep  412 | step    98,340 |  214 steps 1.60 km | END collision  | route  34% | R  +182.4 | ..."""
    head = prefix if prefix is not None else (
        f"ep {int(rec.get('episode', 0)):>4d} | step {int(rec.get('step', 0)):>9,d}")
    reason = END_LABELS.get(rec.get("end_reason"), str(rec.get("end_reason")))
    parts = [
        head,
        f"{int(_num(rec.get('length_steps'))):>4d} steps {_num(rec.get('distance_m')) / 1000.0:4.2f} km",
        f"END {reason:<10s}",
        f"route {_pct(_finite(rec.get('route_completion_frac')))}",
        f"R {_num(rec.get('return_total')):+7.1f}",
        f"speed {_num(rec.get('mean_speed_mps')):4.1f} m/s",
        f"lane×{int(_num(rec.get('illegal_lane_invasions')))}",
        f"cbf {_pct(_finite(rec.get('cbf_intervention_frac')))}",
    ]
    if rec.get("curriculum_stage") is not None:
        parts.append(f"S{rec.get('curriculum_stage')}")
    return " | ".join(parts)


def _outcome_line(s: dict, prev: Optional[dict] = None) -> str:
    succ = f"success {_pct(s.get('success_frac'), 0)}"
    if prev and prev.get("success_frac") is not None and s.get("success_frac") is not None:
        succ += f" ({100.0 * (s['success_frac'] - prev['success_frac']):+.0f})"
    crash = f"crash {_pct(s.get('crash_frac'), 0)}"
    if s.get("end_collision_pedestrian_frac"):
        crash += f" (ped {_pct(s['end_collision_pedestrian_frac'], 0)})"
    parts = [succ, crash,
             f"blocked {_pct(s.get('end_blocked_frac'), 0)}",
             f"timeout {_pct(s.get('end_timeout_frac'), 0)}",
             f"off-route {_pct(s.get('end_off_route_frac'), 0)}",
             f"wrong-way {_pct(s.get('end_wrong_way_frac'), 0)}",
             f"red-light {_pct(s.get('end_red_light_frac'), 0)}"]
    if s.get("end_other_frac"):
        parts.append(f"other {_pct(s['end_other_frac'], 0)}")
    return " | ".join(parts)


def _rate(x, unit: str) -> str:
    return "n/a (too little distance)" if x is None else f"{x:.1f} {unit}"


def _detail_lines(s: dict) -> List[str]:
    driving = (f"route {_pct(s.get('route_completion_frac'), 0)} (mean) | "
               f"speed {_num(s.get('mean_speed_mps')):.1f} m/s | "
               f"speeding {_pct(s.get('speeding_frac'), 0)} of steps | "
               f"return {_num(s.get('return_total_mean')):+.1f} (mean)")
    safety = (f"collisions {_rate(s.get('collisions_per_km'), '/km')} | "
              f"illegal lane crossings {_rate(s.get('illegal_lane_crossings_per_km'), '/km')} | "
              f"cbf active {_pct(s.get('cbf_intervention_frac'), 0)} of steps")
    if s.get("trust_mean") is not None:
        safety += f" | trust {s['trust_mean']:.2f}"
    split = s.get("reward_split") or {}
    ordered = sorted(split.items(), key=lambda kv: -kv[1])
    reward = " · ".join(f"{name} {100.0 * share:.0f}%" for name, share in ordered if share >= 0.005) or "n/a"
    return [f" driving   {driving}", f" safety    {safety}", f" reward    {reward}"]


def verdict(s: dict, prev: Optional[dict] = None) -> str:
    """One plain-language sentence: how is the agent doing?"""
    if not s.get("n_episodes"):
        return "No episodes yet."
    succ = s.get("success_frac") or 0.0
    text = f"Reaches the goal in {100 * succ:.0f}% of episodes"
    if prev and prev.get("success_frac") is not None:
        text += f" ({100 * (succ - prev['success_frac']):+.0f} pts vs previous block)"
    failures = {name: s.get(f"{name}_frac") or 0.0
                for name in ("crash", "blocked", "timeout", "rule_violation")}
    worst, share = max(failures.items(), key=lambda kv: kv[1])
    label = {"crash": "crash", "blocked": "blocked (stopped for no reason)",
             "timeout": "timeout", "rule_violation": "traffic-rule violation"}[worst]
    if share > 0:
        text += f"; most common failure: {label} ({100 * share:.0f}%)"
    route = s.get("route_completion_frac")
    if route is not None:
        text += f"; covers {100 * route:.0f}% of the route on average"
    return text + "."


def format_rolling_block(s: dict, *, step: int, steps_per_s: Optional[float] = None,
                         eta_s: Optional[float] = None, prev: Optional[dict] = None,
                         width: int = 100) -> str:
    title = f"── last {s.get('n_episodes', 0)} episodes @ {_kilo(step)} steps"
    if s.get("curriculum_stage") is not None:
        title += f" · stage S{s['curriculum_stage']}"
    title += " "
    lines = [title + "─" * max(4, width - len(title))]
    lines.append(f" outcome   {_outcome_line(s, prev)}")
    lines += _detail_lines(s)
    speed = f"{steps_per_s:.1f} steps/s" if steps_per_s else "n/a steps/s"
    if eta_s is not None:
        speed += f" | ETA {human_duration(eta_s)}"
    lines.append(f" speed     {speed}")
    lines.append(f" verdict   {verdict(s, prev)}")
    lines.append("─" * width)
    return "\n".join(lines)


def format_eval_block(s: dict, *, step: int, duration_s: float, width: int = 100) -> str:
    n = s.get("n_episodes", 0)
    title = (f"══ EVAL @ {step:,} steps · {n} deterministic episodes · fixed spawns 0-{max(n - 1, 0)}")
    if s.get("curriculum_stage") is not None:
        title += f" · stage S{s['curriculum_stage']}"
    title += " "
    lines = [title + "═" * max(4, width - len(title)),
             f" outcome   {_outcome_line(s)}"]
    lines += _detail_lines(s)
    lines.append(f" verdict   {verdict(s)}")
    lines.append(f" took {human_duration(duration_s)} (training paused)")
    lines.append("═" * width)
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# TensorBoard plumbing
# --------------------------------------------------------------------------------------------
_WRITERS: Dict[str, Any] = {}


def tb_dir(log_dir: str) -> str:
    return os.path.abspath(os.path.join(log_dir, "tensorboard", "metrics"))


def _tb_writer(log_dir: Optional[str], resume_step: int = 0):
    """One shared SummaryWriter per log_dir (<log_dir>/tensorboard/metrics)."""
    if not log_dir:
        return None
    path = tb_dir(log_dir)
    writer = _WRITERS.get(path)
    if writer is None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception:      # tensorboard not installed: jsonl + console still work
            return None
        kwargs = {"log_dir": path, "flush_secs": 30}
        if resume_step > 0:
            # hide events an earlier process logged after the checkpoint we resumed from
            kwargs["purge_step"] = int(resume_step) + 1
        writer = SummaryWriter(**kwargs)
        _WRITERS[path] = writer
    return writer


def close_writers() -> None:
    for writer in list(_WRITERS.values()):
        try:
            writer.close()
        except Exception:
            pass
    _WRITERS.clear()


def _scalar(writer, tag: str, value, step: int) -> None:
    if writer is None:
        return
    f = _finite(value)
    if f is not None:
        writer.add_scalar(tag, f, int(step))


class _DedupTensorBoardFormat(KVWriter):
    """Wraps SB3's TensorBoardOutputFormat and drops duplicate / legacy keys (TB only)."""

    def __init__(self, inner: TensorBoardOutputFormat, drop_keys=SB3_TB_DUPLICATES,
                 drop_prefixes=SB3_TB_LEGACY_PREFIXES):
        self.inner = inner
        self.writer = getattr(inner, "writer", None)
        self.drop_keys = frozenset(drop_keys)
        self.drop_prefixes = tuple(drop_prefixes)

    def keep(self, key: str) -> bool:
        return key not in self.drop_keys and not key.startswith(self.drop_prefixes)

    def write(self, key_values, key_excluded, step: int = 0) -> None:
        kv = {k: v for k, v in key_values.items() if self.keep(k)}
        if not kv:
            return
        ke = {k: key_excluded.get(k, ("",)) for k in kv}
        self.inner.write(kv, ke, step)

    def close(self) -> None:
        self.inner.close()


def dedupe_sb3_tensorboard(logger) -> bool:
    """Install the duplicate filter on an SB3 Logger's TensorBoard output (idempotent)."""
    formats = getattr(logger, "output_formats", None)
    if not formats:
        return False
    changed = False
    for i, fmt in enumerate(formats):
        if isinstance(fmt, TensorBoardOutputFormat):
            formats[i] = _DedupTensorBoardFormat(fmt)
            changed = True
    return changed


# Wall time spent in evaluation (perf/ and steps/s exclude it so the numbers describe training).
_EVAL_SECONDS = [0.0]


def _training_clock() -> float:
    return time.perf_counter() - _EVAL_SECONDS[0]


# --------------------------------------------------------------------------------------------
# EpisodeMetricsCallback
# --------------------------------------------------------------------------------------------
class EpisodeMetricsCallback(BaseCallback):
    """Per-episode metrics from info['episode_metrics'] -> episodes.jsonl, TensorBoard, console.

    :param log_dir: run folder (episodes.jsonl and tensorboard/metrics go here)
    :param verbose: 1 = one console line per episode + a rolling block every `summary_every` episodes
    :param summary_every: episodes between rolling blocks
    :param window: rolling window size in episodes (tags are named rolling<window>/)
    :param perf_every: env steps between perf/ points
    :param dedupe_sb3_tb: drop duplicate concepts from SB3's own TensorBoard output
    """

    def __init__(self, log_dir: str, verbose: int = 1, summary_every: int = 20, window: int = 100,
                 perf_every: int = 1000, dedupe_sb3_tb: bool = True):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.summary_every = max(1, int(summary_every))
        self.window = max(1, int(window))
        self.perf_every = max(1, int(perf_every))
        self.dedupe_sb3_tb = dedupe_sb3_tb
        self.jsonl_path = os.path.join(log_dir, "episodes.jsonl")
        self.recent: deque = deque(maxlen=self.window)
        self.n_episodes = 0
        self.run_start_step = 0
        self._writer = None
        self._prev_block: Optional[dict] = None
        self._block_mark = None          # (training clock, step) at the last rolling block
        # perf accumulators (training clock: excludes evaluation)
        self._t_mark = None
        self._t_rollout_end = None
        self._env_s = 0.0
        self._train_s = 0.0
        self._perf_steps = 0
        self._perf_t0 = None
        self._perf_step0 = 0
        self._next_perf = 0
        self._last_step = 0

    # -- setup ---------------------------------------------------------------------------
    def _init_callback(self) -> None:
        os.makedirs(self.log_dir, exist_ok=True)
        step = int(self.model.num_timesteps)
        self.run_start_step = step
        self._writer = _tb_writer(self.log_dir, resume_step=step)
        if self.dedupe_sb3_tb:
            dedupe_sb3_tensorboard(getattr(self.model, "logger", None))
        # resume: continue the episode count and the rolling window from episodes.jsonl
        history = [r for r in drop_superseded(read_jsonl(self.jsonl_path)) if _num(r.get("step"), 0) <= step]
        self.n_episodes = len(history)
        self.recent.clear()
        self.recent.extend(history[-self.window:])
        now = _training_clock()
        self._block_mark = (now, step)
        self._perf_t0, self._perf_step0, self._last_step = now, step, step
        self._next_perf = (step // self.perf_every + 1) * self.perf_every
        if self.verbose and history:
            _emit(f"[metrics] resumed: {self.n_episodes} earlier episodes in {self.jsonl_path}")

    # -- timing hooks ----------------------------------------------------------------------
    def _on_rollout_start(self) -> None:
        now = _training_clock()
        if self._t_rollout_end is not None:
            self._train_s += max(0.0, now - self._t_rollout_end)
            self._t_rollout_end = None
        self._t_mark = now

    def _on_rollout_end(self) -> None:
        self._t_rollout_end = _training_clock()

    def _on_step(self) -> bool:
        now = _training_clock()
        if self._t_mark is not None:
            self._env_s += max(0.0, now - self._t_mark)
        step = int(self.num_timesteps)
        self._perf_steps += max(0, step - self._last_step)
        self._last_step = step

        for info in self.locals.get("infos") or ():
            if is_episode_end(info):
                self._record_episode(info[METRICS_KEY], step)

        if step >= self._next_perf:
            self._log_perf(step)
        self._t_mark = _training_clock()
        return True

    def _on_training_end(self) -> None:
        if self._writer is not None:
            self._writer.flush()

    # -- episode end -------------------------------------------------------------------------
    def _record_episode(self, metrics: dict, step: int) -> None:
        self.n_episodes += 1
        rec = build_episode_record(metrics, episode=self.n_episodes, step=step,
                                   run_start_step=self.run_start_step)
        try:
            append_jsonl(self.jsonl_path, rec)
        except OSError as exc:
            _emit(f"[metrics] could not write {self.jsonl_path}: {exc}")
        self.recent.append(rec)
        summary = summarize(list(self.recent))
        self._write_tb(rec, summary, step)
        if self.verbose:
            _emit(format_episode_line(rec))
            if self.n_episodes % self.summary_every == 0:
                self._print_block(summary, step)

    def _write_tb(self, rec: dict, s: dict, step: int) -> None:
        w = self._writer
        if w is None:
            return
        length = max(1, int(_num(rec.get("length_steps"), 1)))
        _scalar(w, "episode/return_total", rec.get("return_total"), step)
        _scalar(w, "episode/distance_m", rec.get("distance_m"), step)
        _scalar(w, "episode/mean_speed_mps", rec.get("mean_speed_mps"), step)
        _scalar(w, "episode/route_completion_frac", rec.get("route_completion_frac"), step)
        _scalar(w, "episode/speeding_frac", rec.get("speeding_frac"), step)
        _scalar(w, "reward/per_step", _num(rec.get("return_total")) / length, step)
        for g, share in reward_shares([rec.get("reward_groups")]).items():
            _scalar(w, f"reward/share_{g}", share, step)
        _scalar(w, "safety/cbf_intervention_frac", rec.get("cbf_intervention_frac"), step)
        _scalar(w, "safety/cbf_infeasible_count", rec.get("cbf_infeasible"), step)
        ttc = rec.get("min_ttc_s")
        _scalar(w, "safety/min_ttc_s", TTC_CAP_S if ttc is None else min(_num(ttc), TTC_CAP_S), step)
        if rec.get("min_gap_m") is not None:
            _scalar(w, "safety/min_gap_m", min(_num(rec.get("min_gap_m"), GAP_CAP_M), GAP_CAP_M), step)
        _scalar(w, "safety/trust_mean", rec.get("trust_mean"), step)
        _scalar(w, CURRICULUM_TB_TAG, rec.get("curriculum_stage"), step)
        for field in ROLLING_TB_FIELDS:
            _scalar(w, f"rolling{self.window}/{field}", s.get(field), step)

    def _print_block(self, summary: dict, step: int) -> None:
        now = _training_clock()
        rate = eta = None
        if self._block_mark is not None:
            t0, s0 = self._block_mark
            if now > t0 and step > s0:
                rate = (step - s0) / (now - t0)
        total = int(_num(getattr(self.model, "_total_timesteps", 0)))
        if rate and total > step:
            eta = (total - step) / rate
        _emit(format_rolling_block(summary, step=step, steps_per_s=rate, eta_s=eta, prev=self._prev_block))
        self._prev_block = summary
        self._block_mark = (now, step)

    def _log_perf(self, step: int) -> None:
        now = _training_clock()
        wall = now - (self._perf_t0 if self._perf_t0 is not None else now)
        steps = self._perf_steps
        if steps > 0 and wall > 0:
            _scalar(self._writer, "perf/env_steps_per_s", steps / wall, step)
            _scalar(self._writer, "perf/env_step_ms", 1000.0 * self._env_s / steps, step)
            _scalar(self._writer, "perf/train_step_ms", 1000.0 * self._train_s / steps, step)
        self._env_s = self._train_s = 0.0
        self._perf_steps = 0
        self._perf_t0 = now
        self._next_perf = (step // self.perf_every + 1) * self.perf_every


# --------------------------------------------------------------------------------------------
# EvalCallback
# --------------------------------------------------------------------------------------------
def _vec_chain(venv) -> List[Any]:
    chain, cur = [], venv
    while cur is not None and cur not in chain:
        chain.append(cur)
        cur = getattr(cur, "venv", None)
    return chain


class EvalCallback(BaseCallback):
    """Deterministic evaluation on fixed spawns, run on the (single) training env.

    Every `eval_freq` env steps (checked after the step's transition is stored, in
    _on_rollout_end) training pauses and `n_episodes` episodes are driven with the deterministic
    policy, episode i using reset options {'spawn_index': i, 'route_seed': i}. Nothing goes to the
    replay buffer and no other callback sees these steps, so EpisodeMetricsCallback and the
    curriculum never count them. Afterwards the VecEnv holds a freshly reset training episode and
    model._last_obs / _last_original_obs / _last_episode_starts point at it, so SB3 continues
    cleanly (the interrupted training episode is simply abandoned; its stored transitions stay valid).

    Note: NPC traffic is still random, so eval episodes are repeatable in route/spawn, not exactly.
    """

    def __init__(self, eval_freq: int = 25000, n_episodes: int = 10, log_dir: Optional[str] = None,
                 verbose: int = 1, deterministic: bool = True, max_episode_steps: int = 20000):
        super().__init__(verbose)
        self.eval_freq = int(eval_freq)
        self.n_episodes = int(n_episodes)
        self.log_dir = log_dir
        self.deterministic = deterministic
        self.max_episode_steps = int(max_episode_steps)
        self.is_evaluating = False
        self.last_summary: Optional[dict] = None
        self._pending = False
        self._next_eval = 0
        self._writer = None
        self.run_start_step = 0

    @property
    def jsonl_path(self) -> Optional[str]:
        return os.path.join(self.log_dir, "eval.jsonl") if self.log_dir else None

    def _init_callback(self) -> None:
        if self.log_dir is None:
            tb = getattr(self.model, "tensorboard_log", None)
            self.log_dir = os.path.dirname(os.path.abspath(tb)) if tb else None
        step = int(self.model.num_timesteps)
        self.run_start_step = step
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
        self._writer = _tb_writer(self.log_dir, resume_step=step)
        if self.eval_freq > 0:
            self._next_eval = (step // self.eval_freq + 1) * self.eval_freq

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_episodes > 0 and self.num_timesteps >= self._next_eval:
            self._pending = True
        return True

    def _on_rollout_end(self) -> None:
        if not self._pending:
            return
        self._pending = False
        step = int(self.model.num_timesteps)
        while self._next_eval <= step:
            self._next_eval += self.eval_freq
        self.run_evaluation()

    # -- the evaluation itself ---------------------------------------------------------------
    def run_evaluation(self) -> dict:
        model = self.model
        venv = model.get_env()
        step = int(model.num_timesteps)
        chain = _vec_chain(venv)
        # VecNormalize-style wrappers must not update their statistics on eval data
        frozen = [(w, w.training) for w in chain if isinstance(getattr(w, "training", None), bool)]
        for w, _ in frozen:
            w.training = False

        t_start = time.perf_counter()
        self.is_evaluating = True
        records: List[dict] = []
        obs, last_done, error = None, False, None
        try:
            for i in range(self.n_episodes):
                venv.set_options({"spawn_index": i, "route_seed": i})
                obs = venv.reset()
                last_done, metrics, agent_return, n_steps = False, None, 0.0, 0
                while n_steps < self.max_episode_steps:
                    action, _ = model.predict(obs, deterministic=self.deterministic)
                    obs, rewards, dones, infos = venv.step(action)
                    n_steps += 1
                    agent_return += float(np.asarray(rewards).reshape(-1)[0])
                    if bool(np.asarray(dones).reshape(-1)[0]):
                        last_done = True
                        info0 = infos[0] if infos else {}
                        if is_episode_end(info0):
                            metrics = info0[METRICS_KEY]
                        break
                if metrics is None:   # env gave no episode_metrics (or the safety cap hit)
                    metrics = {"end_reason": "unknown" if last_done else "eval_cap",
                               "return": agent_return, "length_steps": n_steps}
                rec = build_episode_record(metrics, episode=i + 1, step=step,
                                           run_start_step=self.run_start_step)
                rec["spawn_index"] = i
                records.append(rec)
                if self.verbose:
                    _emit(format_episode_line(
                        rec, prefix=f"  eval {i + 1:>2d}/{self.n_episodes} | spawn {i:>3d}"))
        except BaseException as exc:
            error = exc
        for w, flag in frozen:
            w.training = flag
        if not isinstance(error, KeyboardInterrupt):     # on Ctrl+C just stop, no extra reset
            try:
                self._restore_training_state(venv, obs if last_done and error is None else None)
            except Exception as exc:
                error = error or exc
        duration = time.perf_counter() - t_start
        _EVAL_SECONDS[0] += duration
        self.is_evaluating = False
        if error is not None:
            raise error

        summary = summarize(records)
        summary.update({"step": step, "wall_time": round(time.time(), 3), "duration_s": round(duration, 1),
                        "run_id": _RUN_ID, "run_start_step": self.run_start_step,
                        "deterministic": self.deterministic})
        summary["episodes"] = [
            {k: r.get(k) for k in ("spawn_index", "end_reason", "return_total", "length_steps",
                                   "distance_m", "route_completion_frac", "collisions_total",
                                   "illegal_lane_invasions", "speeding_frac", "cbf_intervention_frac",
                                   "min_ttc_s", "curriculum_stage")}
            for r in records]
        self.last_summary = summary
        if self.jsonl_path:
            try:
                append_jsonl(self.jsonl_path, summary)
            except OSError as exc:
                _emit(f"[eval] could not write {self.jsonl_path}: {exc}")
        for field in EVAL_TB_FIELDS:
            _scalar(self._writer, f"eval/{field}", summary.get(field), step)
        if self._writer is not None:
            self._writer.flush()
        if self.verbose:
            _emit(format_eval_block(summary, step=step, duration_s=duration))
        return summary

    def _restore_training_state(self, venv, fresh_obs) -> None:
        """Leave the VecEnv on a fresh training episode and point SB3 at it."""
        model = self.model
        # DummyVecEnv auto-resets after `done`, so after a completed eval episode `fresh_obs` is
        # already the first observation of a new (random-spawn) training episode.
        obs = fresh_obs if fresh_obs is not None else venv.reset()
        model._last_obs = obs
        original = obs
        for w in _vec_chain(venv):
            if hasattr(w, "get_original_obs") and hasattr(w, "obs_rms"):
                original = w.get_original_obs()
                break
        if hasattr(model, "_last_original_obs"):
            model._last_original_obs = original
        n_envs = int(getattr(venv, "num_envs", 1))
        model._last_episode_starts = np.ones((n_envs,), dtype=bool)
        noise = getattr(model, "action_noise", None)
        if noise is not None:
            try:
                noise.reset()
            except Exception:
                pass
        # With optimize_memory_usage the next observation is stored implicitly at pos+1 and would
        # become the post-eval obs; end the abandoned trajectory there (as SB3 does on truncation).
        rb = getattr(model, "replay_buffer", None)
        if rb is not None and getattr(rb, "optimize_memory_usage", False) and (rb.full or rb.pos > 0):
            rb.dones[(rb.pos - 1) % rb.buffer_size] = True


# --------------------------------------------------------------------------------------------
# run_config.json
# --------------------------------------------------------------------------------------------
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read_git_head(repo: str) -> Dict[str, Optional[str]]:
    """Commit/branch straight from the .git files (no git executable needed)."""
    out = {"commit": None, "branch": None}
    try:
        git = os.path.join(repo, ".git")
        if os.path.isfile(git):                       # worktree: ".git" is "gitdir: <path>"
            with open(git, encoding="utf-8") as fh:
                git = fh.read().split("gitdir:", 1)[1].strip()
            if not os.path.isabs(git):
                git = os.path.normpath(os.path.join(repo, git))
        common = git
        if os.path.isfile(os.path.join(git, "commondir")):
            with open(os.path.join(git, "commondir"), encoding="utf-8") as fh:
                common = os.path.normpath(os.path.join(git, fh.read().strip()))
        with open(os.path.join(git, "HEAD"), encoding="utf-8") as fh:
            head = fh.read().strip()
        if not head.startswith("ref:"):
            out["commit"] = head
            return out
        ref = head.split(":", 1)[1].strip()
        out["branch"] = ref.rsplit("refs/heads/", 1)[-1]
        for base in (git, common):
            path = os.path.join(base, *ref.split("/"))
            if os.path.isfile(path):
                with open(path, encoding="utf-8") as fh:
                    out["commit"] = fh.read().strip()
                return out
        packed = os.path.join(common, "packed-refs")
        if os.path.isfile(packed):
            with open(packed, encoding="utf-8") as fh:
                for line in fh:
                    parts = line.strip().split(" ")
                    if len(parts) == 2 and parts[1] == ref:
                        out["commit"] = parts[0]
    except Exception:
        pass
    return out


def git_info(repo: str = _REPO) -> Dict[str, Any]:
    info: Dict[str, Any] = {"commit": None, "branch": None, "dirty": None}
    try:
        def run(*args):
            res = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, timeout=10)
            return res.stdout.strip() if res.returncode == 0 else None
        info["commit"] = run("rev-parse", "HEAD")
        if info["commit"]:
            info["branch"] = run("rev-parse", "--abbrev-ref", "HEAD")
            status = run("status", "--porcelain", "--untracked-files=no")
            info["dirty"] = None if status is None else bool(status)
            return info
    except Exception:
        pass
    info.update(_read_git_head(repo))
    return info


def _wrapper_stack(model) -> List[Any]:
    """[outermost gym wrapper, ..., base env] of env 0 (VecEnv wrappers skipped)."""
    venv = model.get_env() if hasattr(model, "get_env") else None
    chain = _vec_chain(venv) if venv is not None else []
    base = chain[-1] if chain else None
    envs = getattr(base, "envs", None)
    out, cur = [], (envs[0] if envs else None)
    while cur is not None and cur not in out:
        out.append(cur)
        cur = getattr(cur, "env", None) if "env" in getattr(cur, "__dict__", {}) else None
    return out


def _class_constants(obj) -> Dict[str, Any]:
    """UPPER_CASE scalar class attributes (the env keeps its reward constants there)."""
    out: Dict[str, Any] = {}
    for klass in reversed(type(obj).__mro__):
        for key, value in vars(klass).items():
            if key.isupper() and isinstance(value, (bool, int, float, str, tuple)) and not key.startswith("_"):
                out[key] = _jsonable(value)
    return out


def _scalar_attrs(obj, names: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    attrs = getattr(obj, "__dict__", {})
    keys = names if names is not None else [k for k in attrs if not k.startswith("_")]
    out = {}
    for key in keys:
        if key in attrs and isinstance(attrs[key], (bool, int, float, str, np.integer, np.floating)):
            out[key] = _jsonable(attrs[key])
    return out


def _deep_merge(base: dict, extra: dict) -> dict:
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def write_run_config(log_dir: str, model, extra: Optional[dict] = None) -> str:
    """Write <log_dir>/run_config.json (previous version appended to run_config_history.jsonl)."""
    os.makedirs(log_dir, exist_ok=True)
    cfg: Dict[str, Any] = {
        "written_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "reward_version": REWARD_VERSION,
        "run_id": _RUN_ID,
        "git": git_info(),
        "argv": list(sys.argv),
        "python": sys.version.split()[0],
    }
    try:
        import torch
        cfg["torch"] = torch.__version__
    except Exception:
        pass
    try:
        import stable_baselines3
        cfg["stable_baselines3"] = getattr(stable_baselines3, "__version__", None)
    except Exception:
        pass

    lr = getattr(model, "learning_rate", None)
    train_freq = getattr(model, "train_freq", None)
    if hasattr(train_freq, "frequency"):
        train_freq = [train_freq.frequency, getattr(train_freq.unit, "value", str(train_freq.unit))]
    ent_coef = getattr(model, "ent_coef", None)
    log_ent = getattr(model, "log_ent_coef", None)
    if log_ent is not None:
        try:
            ent_coef = {"setting": ent_coef, "current": float(log_ent.detach().exp().cpu().item())}
        except Exception:
            pass
    policy = getattr(model, "policy", None)
    critic = getattr(policy, "critic", None)
    cfg["algorithm"] = type(model).__name__
    cfg["num_timesteps_at_start"] = int(getattr(model, "num_timesteps", 0))
    cfg["hyperparameters"] = _jsonable({
        "learning_rate": lr if isinstance(lr, (int, float)) else repr(lr),
        "buffer_size": getattr(model, "buffer_size", None),
        "batch_size": getattr(model, "batch_size", None),
        "learning_starts": getattr(model, "learning_starts", None),
        "tau": getattr(model, "tau", None),
        "gamma": getattr(model, "gamma", None),
        "train_freq": train_freq,
        "gradient_steps": getattr(model, "gradient_steps", None),
        "ent_coef": ent_coef,
        "target_entropy": getattr(model, "target_entropy", None),
        "target_update_interval": getattr(model, "target_update_interval", None),
        "seed": getattr(model, "seed", None),
        "device": str(getattr(model, "device", "")),
        "n_envs": getattr(model, "n_envs", None),
        "policy_kwargs": getattr(model, "policy_kwargs", None),
    })
    cfg["policy"] = _jsonable({
        "class": type(policy).__name__ if policy is not None else None,
        "n_critics": getattr(critic, "n_critics", None),
        "features_extractor": type(getattr(policy, "features_extractor", None)).__name__
        if getattr(policy, "features_extractor", None) is not None else None,
    })
    cfg["spaces"] = {"observation": repr(getattr(model, "observation_space", None)),
                     "action": repr(getattr(model, "action_space", None))}

    stack = _wrapper_stack(model)
    cfg["wrapper_stack"] = [type(w).__name__ for w in stack]
    env_section: Dict[str, Any] = {}
    cbf_section: Dict[str, Any] = {}
    for w in stack:
        attrs = getattr(w, "__dict__", {})
        if "cbf_layer" in attrs:
            cbf_section.update(_scalar_attrs(w, ("intervention_penalty", "use_trust_score", "trust_score")))
            layer = attrs.get("cbf_layer")
            if layer is not None:
                cbf_section["layer_class"] = type(layer).__name__
                cbf_section["params"] = _scalar_attrs(layer)
        if hasattr(type(w), "build_cbf_state") or type(w).__name__ == "CarlaGymEnv":
            env_section["class"] = type(w).__name__
            env_section["constants"] = _class_constants(w)
            env_section.update(_scalar_attrs(w, (
                "host", "port", "time_limit", "render_mode", "curriculum_stage", "num_npc_vehicles",
                "num_pedestrians", "seed", "show_sensor_data", "fixed_delta_seconds")))
            if isinstance(env_section["constants"].get("REWARD_VERSION"), str):
                cfg["reward_version"] = env_section["constants"]["REWARD_VERSION"]
    cfg["env"] = env_section
    cfg["cbf"] = cbf_section
    if extra:
        extra = _jsonable(extra)
        known = {k: v for k, v in extra.items() if k in cfg and isinstance(cfg[k], dict) and isinstance(v, dict)}
        _deep_merge(cfg, known)
        cfg["extra"] = {k: v for k, v in extra.items() if k not in known}

    path = os.path.join(log_dir, "run_config.json")
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as fh:
                previous = json.load(fh)
            append_jsonl(os.path.join(log_dir, "run_config_history.jsonl"), previous)
        except (OSError, ValueError):
            pass
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(cfg), fh, indent=2, allow_nan=False)
    os.replace(tmp, path)
    return path


__all__ = [
    "EpisodeMetricsCallback", "EvalCallback", "write_run_config",
    "build_episode_record", "summarize", "format_episode_line", "format_rolling_block",
    "format_eval_block", "verdict", "reward_group", "reward_shares", "all_tb_tags",
    "read_jsonl", "drop_superseded", "dedupe_sb3_tensorboard", "close_writers",
    "REWARD_VERSION", "METRICS_KEY", "END_REASONS", "REWARD_GROUPS",
]
