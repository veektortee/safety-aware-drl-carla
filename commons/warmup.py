"""Replay-buffer warm-up with the expert driver, plus the resume / run-config helpers the four
experiment scripts share.

Why the warm-up exists: a zero-centred random policy barely moves the car (coasting decelerates at
~3 m/s^2), so SB3's usual random-action warm-up fills the buffer with standing-still transitions.
Instead, `prefill_replay_buffer` drives with CarlaGymEnv.expert_action() through the FULL training
stack (VecEnv -> obs wrapper -> CBF wrapper -> env), so the stored observations, rewards, CBF
penalties and episode ends are exactly what the policy will see later. The stored action is the
expert's proposed action (before the CBF), like the policy's proposed action during training.
"""

import io
import json
import os
import re
import shutil
import time
import zipfile
from collections import Counter
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from commons.debug import dprint


def find_carla_env(model):
    """The CarlaGymEnv under an SB3 model's VecEnv (None if the model has no env)."""
    from tests.pipeline_carla_test import _find_carla_and_cbf, _unwrap_vec_env

    carla_env, _ = _find_carla_and_cbf(model)
    if carla_env is None:
        base = _unwrap_vec_env(model)
        carla_env = base.unwrapped if base is not None else None
    return carla_env


def prefill_replay_buffer(model, n_steps: int, log_every: int = 1000, verbose: int = 1) -> dict:
    """Fill `model.replay_buffer` with `n_steps` expert transitions.

    Each step: action = carla_env.expert_action() (shape (2,), [steer, accel] in [-1, 1]), stepped
    through model.get_env() (so the CBF and observation wrappers run, and the VecEnv auto-resets),
    then stored with SB3's own OffPolicyAlgorithm._store_transition (scaled action, next_obs taken
    from infos[i]['terminal_observation'] on episode ends, dict observations deep-copied,
    TimeLimit.truncated -> timeout so truncated episodes bootstrap).

    Leaves model._last_obs (and _last_original_obs) at the latest observation, so a following
    model.learn(..., reset_num_timesteps=False) continues the running episode instead of resetting.
    model.num_timesteps is NOT advanced: --timesteps counts policy steps only.

    Returns a summary dict (steps, episodes, end_reasons, buffer_size, seconds).
    """
    n_steps = int(n_steps)
    if n_steps <= 0:
        return {"steps": 0, "episodes": 0, "end_reasons": {}, "buffer_size": model.replay_buffer.size(),
                "seconds": 0.0}

    vec_env = model.get_env()
    if vec_env is None:
        raise RuntimeError("prefill_replay_buffer: the model has no environment")
    if vec_env.num_envs != 1:
        raise RuntimeError("prefill_replay_buffer supports a single environment (one CARLA server)")
    carla_env = find_carla_env(model)
    if carla_env is None or not hasattr(carla_env, "expert_action"):
        raise RuntimeError("prefill_replay_buffer: CarlaGymEnv.expert_action() not found in the env stack")

    replay_buffer = model.replay_buffer
    if model._last_obs is None:
        model._last_obs = vec_env.reset()
        model._last_episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
        model._last_original_obs = model._last_obs

    action_space = model.action_space
    low, high = action_space.low, action_space.high

    if verbose:
        print(f"[WARMUP] prefilling the replay buffer with {n_steps:,} expert steps "
              f"(buffer {replay_buffer.size():,}/{replay_buffer.buffer_size:,})")

    t0 = time.time()
    episodes = 0
    end_reasons = Counter()
    returns = []
    ep_return = 0.0
    for step in range(1, n_steps + 1):
        expert = np.asarray(carla_env.expert_action(), dtype=np.float32).reshape(action_space.shape)
        expert = np.clip(expert, low, high)
        actions = expert[None, :]                          # (n_envs=1, 2) for the VecEnv
        buffer_actions = model.policy.scale_action(actions)  # what SB3 stores ([-1, 1] -> identity)

        new_obs, rewards, dones, infos = vec_env.step(actions)
        # Exactly SB3's storage path (terminal_observation, TimeLimit.truncated, deep copies,
        # updates model._last_obs / _last_original_obs).
        model._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)

        ep_return += float(rewards[0])
        if dones[0]:
            episodes += 1
            metrics = infos[0].get("episode_metrics")
            reason = metrics.get("end_reason") if isinstance(metrics, dict) else None
            end_reasons[reason or "unknown"] += 1
            returns.append(ep_return)
            dprint(f"[WARMUP] episode {episodes} ended ({reason}) return {ep_return:+.1f}")
            ep_return = 0.0

        if verbose and log_every and (step % log_every == 0 or step == n_steps):
            elapsed = max(time.time() - t0, 1e-6)
            mean_ret = f"{np.mean(returns[-20:]):+.1f}" if returns else "n/a"
            reasons = ", ".join(f"{k} {v}" for k, v in end_reasons.most_common()) or "none yet"
            print(f"[WARMUP] {step:,}/{n_steps:,} expert steps | {episodes} episodes ({reasons}) | "
                  f"mean return (last 20) {mean_ret} | {step / elapsed:.1f} steps/s")

    summary = {
        "steps": n_steps,
        "episodes": episodes,
        "end_reasons": dict(end_reasons),
        "buffer_size": int(replay_buffer.size()),
        "seconds": round(time.time() - t0, 1),
    }
    if verbose:
        print(f"[WARMUP] done: {summary['buffer_size']:,} transitions in the buffer, "
              f"{episodes} expert episodes, {summary['seconds']:.0f} s")
    return summary


# --------------------------------------------------------------------------------------------------
# Resume helpers
# --------------------------------------------------------------------------------------------------

def replay_buffer_candidates(checkpoint_path: str) -> list:
    """Possible replay-buffer files for a SAC checkpoint .zip.

    CheckpointCallback(save_replay_buffer=True) writes '<prefix>_replay_buffer_<N>_steps.pkl' next to
    '<prefix>_<N>_steps.zip'; older code (and save_checkpoint below for non-step names) uses
    '<ckpt>_replay_buffer.pkl'.
    """
    candidates = [
        re.sub(r"_(\d+)_steps\.zip$", r"_replay_buffer_\1_steps.pkl", checkpoint_path),
        re.sub(r"\.zip$", "", checkpoint_path) + "_replay_buffer.pkl",
    ]
    return [c for i, c in enumerate(candidates) if c != checkpoint_path and c not in candidates[:i]]


def restore_replay_buffer(model, checkpoint_path: str, verbose: int = 1) -> bool:
    """Load the replay buffer saved beside `checkpoint_path`. True if one was restored.

    A missing, truncated (e.g. killed mid-save) or incompatible file is reported and skipped; the
    caller then refills the buffer with the expert warm-up.
    """
    empty_buffer = model.replay_buffer
    for path in replay_buffer_candidates(checkpoint_path):
        if not os.path.isfile(path):
            continue
        try:
            model.load_replay_buffer(path)
        except Exception as exc:  # noqa: BLE001 - any unreadable buffer just means "start empty"
            print(f"[RESUME] could not load replay buffer {path}: {exc}")
            model.replay_buffer = empty_buffer
            continue
        rb = model.replay_buffer
        obs_space = getattr(rb, "observation_space", None)
        if obs_space is not None and obs_space != model.observation_space:
            print(f"[RESUME] replay buffer {path} has a different observation space; ignored")
            model.replay_buffer = empty_buffer
            continue
        if verbose:
            print(f"[RESUME] replay buffer restored from {path} ({rb.size():,} transitions)")
        return True
    if verbose:
        print("[RESUME] no usable replay buffer beside the checkpoint; it will be refilled "
              "(expert warm-up). Network weights are restored either way.")
    return False


def checkpoint_num_timesteps(checkpoint_path: str) -> Optional[int]:
    """num_timesteps stored in an SB3 .zip, read without building the model (None if unreadable)."""
    try:
        with zipfile.ZipFile(checkpoint_path) as zf:
            data = json.loads(zf.read("data").decode("utf-8"))
        return int(data["num_timesteps"])
    except Exception:  # noqa: BLE001
        return None


def save_checkpoint(model, ckpt_dir: str, name_prefix: str, save_replay_buffer: bool = True,
                    verbose: int = 1) -> str:
    """Save '<prefix>_<N>_steps.zip' (+ '<prefix>_replay_buffer_<N>_steps.pkl'), the same names as
    CheckpointCallback, so train_all.py and --load-checkpoint pick it up. Each file is written to a
    temporary name first and then renamed, so an interrupted save never leaves a corrupt file.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    n = int(model.num_timesteps)
    zip_path = os.path.join(ckpt_dir, f"{name_prefix}_{n}_steps.zip")
    tmp_zip = zip_path + ".tmp.zip"
    model.save(tmp_zip)
    os.replace(tmp_zip, zip_path)
    if save_replay_buffer and getattr(model, "replay_buffer", None) is not None:
        rb_path = os.path.join(ckpt_dir, f"{name_prefix}_replay_buffer_{n}_steps.pkl")
        tmp_rb = rb_path + ".tmp.pkl"
        model.save_replay_buffer(tmp_rb)
        os.replace(tmp_rb, rb_path)
    if verbose:
        print(f"[CHECKPOINT] saved {zip_path}")
    return zip_path


# --------------------------------------------------------------------------------------------------
# Keep-latest checkpoints and resume clean-up
# --------------------------------------------------------------------------------------------------

def _checkpoints(ckpt_dir: str, name_prefix: str) -> list:
    """[(N, path)] of '<prefix>_<N>_steps.zip' in ckpt_dir, newest first."""
    if not os.path.isdir(ckpt_dir):
        return []
    pattern = re.compile(rf"^{re.escape(name_prefix)}_(\d+)_steps\.zip$")
    found = [(int(m.group(1)), os.path.join(ckpt_dir, name)) for name in os.listdir(ckpt_dir)
             for m in [pattern.match(name)] if m]
    return sorted(found, reverse=True)


def curriculum_snapshot_path(ckpt_dir: str, name_prefix: str, n: int) -> str:
    return os.path.join(ckpt_dir, f"{name_prefix}_curriculum_{int(n)}_steps.json")


def prune_checkpoints(ckpt_dir: str, name_prefix: str, keep: int = 1, verbose: int = 1) -> list:
    """Delete all but the newest `keep` checkpoints with their replay buffers and curriculum snapshots."""
    removed = []
    for n, zip_path in _checkpoints(ckpt_dir, name_prefix)[max(int(keep), 1):]:
        for path in [zip_path, curriculum_snapshot_path(ckpt_dir, name_prefix, n)] + replay_buffer_candidates(zip_path):
            if os.path.isfile(path):
                try:
                    os.remove(path)
                    removed.append(path)
                except OSError as exc:
                    print(f"[CHECKPOINT] could not delete {path}: {exc}")
    if verbose and removed:
        print(f"[CHECKPOINT] removed {len(removed)} older checkpoint file(s)")
    return removed


class LatestCheckpointCallback(BaseCallback):
    """Checkpoint at every multiple of `save_freq` total steps and keep only the newest `keep`.

    Each save writes '<prefix>_<N>_steps.zip' and its replay buffer atomically (save_checkpoint) plus a
    copy of <log_dir>/curriculum_state.json, so a resume restores the curriculum exactly as it was at
    the checkpoint. Older checkpoints are deleted only after the new one is complete.
    """

    def __init__(self, save_freq: int, save_path: str, name_prefix: str, log_dir: Optional[str] = None,
                 keep: int = 1, verbose: int = 1):
        super().__init__(verbose)
        self.save_freq = int(save_freq)
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.log_dir = log_dir
        self.keep = int(keep)
        self._next = 0

    def _init_callback(self) -> None:
        self._next = (int(self.model.num_timesteps) // self.save_freq + 1) * self.save_freq

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


def restore_curriculum_snapshot(log_dir: str, checkpoint_path: str) -> bool:
    """Put the curriculum state saved with a checkpoint back as <log_dir>/curriculum_state.json."""
    m = re.match(r"^(.*)_(\d+)_steps\.zip$", os.path.basename(checkpoint_path))
    if not m:
        return False
    snap = curriculum_snapshot_path(os.path.dirname(checkpoint_path), m.group(1), int(m.group(2)))
    if not os.path.isfile(snap):
        return False
    os.makedirs(log_dir, exist_ok=True)
    shutil.copyfile(snap, os.path.join(log_dir, "curriculum_state.json"))
    return True


def truncate_tensorboard(tb_root: str, max_step: int) -> dict:
    """Physically remove every TensorBoard event with step > max_step under `tb_root`.

    Files are rewritten in place (open 'r+b', write, truncate), which works even while a TensorBoard
    server has them open for reading on Windows. Event files left with nothing but their header are
    deleted when possible. A half-written last record (killed process) is dropped as well.
    """
    from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.summary.writer.record_writer import RecordWriter

    out = {"files_changed": 0, "files_removed": 0, "events_dropped": 0, "errors": []}
    if not os.path.isdir(tb_root):
        return out
    for root, _, names in os.walk(tb_root):
        for name in names:
            if "tfevents" not in name:
                continue
            path = os.path.join(root, name)
            try:
                records = list(RawEventFileLoader(path).Load())
                keep = [r for r in records if Event.FromString(r).step <= max_step]
                if len(keep) == len(records):
                    continue
                buf = io.BytesIO()
                writer = RecordWriter(buf)
                for rec in keep:
                    writer.write(rec)
                data = buf.getvalue()
                with open(path, "r+b") as fh:
                    fh.write(data)
                    fh.truncate()
                out["files_changed"] += 1
                out["events_dropped"] += len(records) - len(keep)
                if all(not Event.FromString(r).HasField("summary") for r in keep):
                    try:
                        os.remove(path)
                        out["files_removed"] += 1
                    except OSError:
                        pass
            except Exception as exc:  # noqa: BLE001 - report, never stop a resume over a log file
                out["errors"].append(f"{path}: {exc}")
    return out


def truncate_jsonl(path: str, max_step: int) -> int:
    """Drop records with step > max_step (and unparseable lines) from a jsonl log. Returns lines removed."""
    if not os.path.isfile(path):
        return 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    kept = []
    for line in lines:
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict) and float(rec.get("step", 0) or 0) <= max_step:
            kept.append(line if line.endswith("\n") else line + "\n")
    if len(kept) != len(lines):
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.writelines(kept)
        os.replace(tmp, path)
    return len(lines) - len(kept)


def discard_logs_after(log_dir: str, max_step: int, verbose: int = 1) -> dict:
    """On resume from a checkpoint at `max_step`: erase what an earlier process logged after it
    (TensorBoard events under <log_dir>/tensorboard, episodes.jsonl, eval.jsonl)."""
    summary = truncate_tensorboard(os.path.join(log_dir, "tensorboard"), max_step)
    summary["jsonl_records_dropped"] = sum(truncate_jsonl(os.path.join(log_dir, f), max_step)
                                           for f in ("episodes.jsonl", "eval.jsonl"))
    if verbose:
        print(f"[RESUME] discarded logs after step {max_step:,}: {summary['events_dropped']} TensorBoard events "
              f"in {summary['files_changed']} file(s), {summary['jsonl_records_dropped']} jsonl records")
        for err in summary["errors"]:
            print(f"[RESUME] WARNING could not clean {err}")
    return summary


def mark_tensorboard_resume(logger, step: int) -> None:
    """Write a SessionLog.START at step+1 into SB3's TensorBoard output: a running TensorBoard then
    drops any step > `step` it still holds in memory (what SummaryWriter(purge_step=...) does)."""
    from tensorboard.compat.proto.event_pb2 import Event, SessionLog

    for fmt in getattr(logger, "output_formats", None) or []:
        writer = getattr(fmt, "writer", None)
        if writer is not None and hasattr(writer, "_get_file_writer"):
            writer._get_file_writer().add_event(Event(step=int(step) + 1,
                                                      session_log=SessionLog(status=SessionLog.START)))


def _public_scalars(obj, class_constants: bool = False) -> Optional[dict]:
    """JSON-friendly public scalar attributes of `obj` (UPPER_CASE class constants if requested)."""
    if obj is None:
        return None
    if class_constants:
        items = {}
        for cls in reversed(type(obj).__mro__):
            items.update({k: v for k, v in vars(cls).items() if k.isupper()})
    else:
        items = vars(obj)
    out = {}
    for k, v in items.items():
        if k.startswith("_"):
            continue
        if isinstance(v, np.generic):
            v = v.item()
        if isinstance(v, (bool, int, float, str)):
            out[k] = v
    return out


def describe_env_stack(model) -> dict:
    """Env / CBF parameters for run_config.json (reward constants are CarlaGymEnv class attributes)."""
    from tests.pipeline_carla_test import _find_carla_and_cbf, _unwrap_vec_env

    carla_env, cbf_wrapper = _find_carla_and_cbf(model)
    if carla_env is None:
        carla_env = find_carla_env(model)
    chain, env = [], _unwrap_vec_env(model)
    while env is not None:
        chain.append(type(env).__name__)
        env = getattr(env, "env", None)
    return {
        "wrapper_chain": chain,
        "env_constants": _public_scalars(carla_env, class_constants=True),
        "env": _public_scalars(carla_env),
        "cbf_wrapper": _public_scalars(cbf_wrapper),
        "cbf_layer": _public_scalars(getattr(cbf_wrapper, "cbf_layer", None)),
    }
