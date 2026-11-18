"""Offline tests for commons/metrics_callbacks.py, train_all.py's progress parser and make_report.py.

No CARLA: synthetic info['episode_metrics'] dicts are fed through the callbacks with a minimal fake
model and a real SB3 logger writing to a temporary folder.

    python tests/unit/test_metrics_callbacks.py          (plain python)
    python -m pytest tests/unit/test_metrics_callbacks.py
"""

import contextlib
import io
import json
import math
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
from stable_baselines3.common.callbacks import CallbackList  # noqa: E402
from stable_baselines3.common.logger import configure  # noqa: E402

import commons.metrics_callbacks as mc  # noqa: E402
import train_all  # noqa: E402

REASONS = ["collision", "endpoint", "blocked", "timeout", "off_route", "wrong_way", "red_light",
           "collision_pedestrian", "endpoint", "collision"]
UNITLESS_OK = {"episode/return_total", "reward/per_step", "curriculum/stage", "eval/return_total_mean",
               "safety/trust_mean", "safety/cbf_infeasible_count"}
UNIT_SUFFIXES = ("_frac", "_per_km", "_s", "_m", "_mps", "_ms", "_per_s", "_count")


def fake_metrics(i, stage=1):
    reason = REASONS[i % len(REASONS)]
    crash = reason.startswith("collision")
    terminal = {"collision": -150.0} if crash else ({"goal": 50.0} if reason == "endpoint" else {})
    # the env's 12 component keys (only the terminal one that fired is non-zero)
    comps = {"progress": 80.0 + i, "time": -20.0, "speeding": -1.0, "lateral": -2.0, "heading": -1.0,
             "illegal_lane": -1.0, "stop_sign": 0.0, "goal": 0.0, "collision": 0.0, "red_light": 0.0,
             "off_route": 0.0, "wrong_way": 0.0, **terminal}
    ret = sum(comps.values())
    length = 200 + i
    return {
        "return": ret, "length_steps": length, "distance_m": 250.0 + i, "mean_speed_mps": 6.0,
        "route_length_m": 300.0, "route_completion_frac": 1.0 if reason == "endpoint" else 0.4,
        "end_reason": reason, "terminated": reason not in ("blocked", "timeout"),
        "collisions": int(crash), "collisions_pedestrian": int(reason == "collision_pedestrian"),
        "lane_invasions": 3, "illegal_lane_invasions": 1, "red_light_violations": int(reason == "red_light"),
        "stop_sign_violations": 0, "speeding_frac": 0.1, "min_ttc_s": float("inf") if i % 2 else 2.5,
        "min_gap_m": 12.0, "curriculum_stage": stage, "reward_components": comps, "spawn_index": None,
        "cbf_interventions": 5, "cbf_intervention_frac": 0.05, "cbf_infeasible": 0,
        "cbf_penalty_sum": 0.4, "trust_mean": 0.9, "cbf_rows": {"collision": 3, "lane": 2},
        "r": ret, "l": length, "t": 1.0,
    }


def episode_info(i):
    m = fake_metrics(i)
    # Monitor replaces info['episode'] with its own dict; the callbacks must use 'episode_metrics'
    return {"episode_metrics": m, "episode": {"r": m["return"] - 0.4, "l": m["length_steps"], "t": 1.0}}


class FakeModel:
    def __init__(self, logger=None, env=None, total=100_000):
        self.num_timesteps = 0
        self._total_timesteps = total
        self.logger = logger
        self.env = env
        self.tensorboard_log = None
        self._last_obs = "sentinel"
        self._last_original_obs = "sentinel"
        self.action_noise = None
        self.replay_buffer = None
        self.predict_calls = []

    def get_env(self):
        return self.env

    def predict(self, obs, deterministic=False):
        self.predict_calls.append(deterministic)
        return np.zeros((1, 2), dtype=np.float32), None


class FakeVecEnv:
    """DummyVecEnv look-alike: auto-reset on done, reset options used once."""
    num_envs = 1

    def __init__(self, ep_len=4):
        self.ep_len = ep_len
        self.t = 0
        self.resets = []          # options of every reset ('auto' = DummyVecEnv auto-reset)
        self._options = None
        self.episode = 0

    def set_options(self, options):
        self._options = options

    def _obs(self):
        return np.full((1, 3), float(len(self.resets)), dtype=np.float32)

    def reset(self):
        self.resets.append(self._options)
        self._options = None
        self.t = 0
        return self._obs()

    def step(self, action):
        self.t += 1
        done = self.t >= self.ep_len
        info = {}
        if done:
            info = episode_info(self.episode)
            self.episode += 1
            self.resets.append("auto")
            self.t = 0
        return self._obs(), np.array([1.0], dtype=np.float32), np.array([done]), [info]


def drive(callback, model, n_episodes, steps_per_episode=3, infos_fn=episode_info):
    for ep in range(n_episodes):
        for k in range(steps_per_episode):
            model.num_timesteps += 1
            done = k == steps_per_episode - 1
            callback.on_rollout_start()
            callback.update_locals({"infos": [infos_fn(ep) if done else {"speed_mps": 5.0}],
                                    "dones": np.array([done])})
            callback.on_step()
            callback.on_rollout_end()


def tb_tags(folder):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    acc = EventAccumulator(folder, size_guidance={"scalars": 0})
    acc.Reload()
    return {tag: len(acc.Scalars(tag)) for tag in acc.Tags().get("scalars", [])}


def _tmp():
    return tempfile.TemporaryDirectory(ignore_cleanup_errors=True)


# ------------------------------------------------------------------------------------------------
def test_episode_metrics_callback_end_to_end():
    with _tmp() as tmp:
        sb3_dir = os.path.join(tmp, "tensorboard", "SAC_1")
        logger = configure(sb3_dir, ["tensorboard"])
        model = FakeModel(logger=logger)
        cb = mc.EpisodeMetricsCallback(log_dir=tmp, verbose=1, summary_every=20, window=100, perf_every=10)
        cb.init_callback(model)
        cb.on_training_start({}, {})
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            drive(cb, model, 45)
        # SB3's own logger: train/* survives, duplicates are filtered from TensorBoard
        for key in ("train/actor_loss", "train/trust_score", "cbf/trust_score", "curriculum/stage",
                    "rollout/ep_rew_mean", "time/fps", "train/learning_rate"):
            logger.record(key, 1.0)
        logger.dump(step=model.num_timesteps)
        logger.close()
        cb.on_training_end()
        mc.close_writers()

        # episodes.jsonl
        recs = mc.read_jsonl(os.path.join(tmp, "episodes.jsonl"))
        assert len(recs) == 45, len(recs)
        first = recs[0]
        m0 = fake_metrics(0)
        assert first["episode"] == 1 and first["step"] == 3
        assert first["end_reason"] == "collision"
        assert math.isclose(first["return"], m0["return"], abs_tol=1e-3)
        assert math.isclose(first["return_total"], m0["return"] - 0.4, abs_tol=1e-3)
        assert first["reward_groups"]["cbf"] == -0.4
        assert first["reward_groups"]["terminal"] == -150.0
        assert first["reward_groups"]["progress"] == 80.0
        assert first["min_ttc_s"] == 2.5 and recs[1]["min_ttc_s"] is None      # inf -> null (strict JSON)
        assert abs(first["reward_residual"]) < 1e-3
        for key in ("run_id", "run_start_step", "wall_time", "distance_m", "speeding_frac",
                    "cbf_intervention_frac", "trust_mean", "curriculum_stage", "collisions_total"):
            assert key in first, key
        assert "r" not in first and "l" not in first

        # console: one line per episode + a rolling block every 20 episodes
        text = out.getvalue()
        ep_lines = [ln for ln in text.splitlines() if ln.startswith("ep ")]
        assert len(ep_lines) == 45
        assert [train_all.parse_step(ln) for ln in ep_lines] == [3 * (i + 1) for i in range(45)]
        assert "END collision" in ep_lines[0] and "lane×1" in ep_lines[0] and "S1" in ep_lines[0]
        blocks = text.split("── last ")[1:]
        assert len(blocks) == 2, len(blocks)
        block = blocks[0]
        for word in ("success", "crash", "blocked", "timeout", "off-route", "wrong-way", "red-light",
                     "collisions", "/km", "illegal lane crossings", "steps/s", "ETA", "reward", "progress",
                     "verdict", "route"):
            assert word in block, word
        assert block.startswith("20 episodes @ 0k steps")
        assert "(+" in blocks[1] or "(-" in blocks[1] or "(0" in blocks[1]   # trend vs previous block
        # end-reason shares over the first 20 episodes: 2x each of the 10-cycle
        assert "success 20%" in block and "crash 30% (ped 10%)" in block

        # TensorBoard: our own run + SB3's run
        ours = tb_tags(mc.tb_dir(tmp))
        sb3 = tb_tags(sb3_dir)
        assert set(sb3) == {"train/actor_loss"}, sb3
        expected = set(mc.all_tb_tags(100))
        assert set(ours) <= expected, set(ours) - expected
        assert "episode/return_total" in ours and "rolling100/success_frac" in ours
        assert ours["episode/return_total"] == 45           # once per episode, not per step
        assert "perf/env_steps_per_s" in ours and ours["perf/env_steps_per_s"] <= 14
        all_tags = list(ours) + list(sb3)
        assert len(all_tags) == len(set(all_tags)), "duplicate tag across runs"
        assert len(expected) < 40, len(expected)
        assert [t for t in expected if "trust" in t] == ["safety/trust_mean"]


def test_tag_inventory_has_units_and_no_duplicates():
    tags = mc.all_tb_tags(100)
    assert len(tags) == len(set(tags))
    assert len(tags) < 40, len(tags)
    for tag in tags:
        if tag.startswith("train/") or "/share_" in tag or tag in UNITLESS_OK:
            continue
        assert tag.endswith(UNIT_SUFFIXES), f"{tag} has no unit suffix"
    # every concept once: the last path element (minus eval/rolling aggregation) is unique per namespace
    for ns in ("episode/", "safety/", "reward/", "perf/"):
        names = [t.split("/", 1)[1] for t in tags if t.startswith(ns)]
        assert len(names) == len(set(names))
    for dup in mc.SB3_TB_DUPLICATES - {mc.CURRICULUM_TB_TAG}:   # curriculum/stage: our copy is the one kept
        assert dup not in tags


def test_summary_and_verdict():
    recs = [mc.build_episode_record(fake_metrics(i), episode=i + 1, step=100 * (i + 1)) for i in range(10)]
    s = mc.summarize(recs)
    assert s["n_episodes"] == 10
    assert math.isclose(s["success_frac"], 0.2) and math.isclose(s["crash_frac"], 0.3)
    assert math.isclose(s["rule_violation_frac"], 0.3)
    assert math.isclose(s["end_collision_pedestrian_frac"], 0.1)
    km = sum(250.0 + i for i in range(10)) / 1000.0
    assert math.isclose(s["collisions_per_km"], 3 / km, rel_tol=1e-6)
    assert math.isclose(s["illegal_lane_crossings_per_km"], 10 / km, rel_tol=1e-6)
    assert math.isclose(sum(s["reward_split"].values()), 1.0)
    v = mc.verdict(s)
    assert v.startswith("Reaches the goal in 20% of episodes") and "crash (30%)" in v
    # too little distance -> no per-km rate rather than a huge number
    tiny = [dict(r, distance_m=1.0) for r in recs]
    assert mc.summarize(tiny)["collisions_per_km"] is None


def test_reward_groups():
    assert mc.reward_group("progress") == "progress"
    assert mc.reward_group("time") == "time"
    for key in ("speeding", "lateral", "heading", "illegal_lane", "stop_sign"):     # env v3 keys
        assert mc.reward_group(key) == "shaping", key
    for key in ("goal", "collision", "red_light", "off_route", "wrong_way"):
        assert mc.reward_group(key) == "terminal", key
    assert mc.reward_group("cbf_penalty") == "cbf"
    assert mc.reward_group("total") is None


def test_eval_callback_restores_state_and_is_not_counted():
    with _tmp() as tmp:
        venv = FakeVecEnv(ep_len=4)
        model = FakeModel(logger=configure(None, [""]), env=venv)
        episodes = mc.EpisodeMetricsCallback(log_dir=tmp, verbose=0)
        evaluator = mc.EvalCallback(eval_freq=10, n_episodes=3, log_dir=tmp, verbose=1)
        callbacks = CallbackList([episodes, evaluator])
        callbacks.init_callback(model)
        callbacks.on_training_start({}, {})
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            drive(callbacks, model, 3, steps_per_episode=3)      # 9 steps, 3 training episodes: no eval
            assert venv.resets == [] and not os.path.exists(evaluator.jsonl_path)
            drive(callbacks, model, 1, steps_per_episode=3)      # crosses step 10 -> eval at rollout end
        mc.close_writers()
        assert evaluator.is_evaluating is False
        # fixed spawns: options {spawn_index: i, route_seed: i} plus the eval map slot (i of n);
        # DummyVecEnv auto-resets after each done
        opts = lambda i: {"spawn_index": i, "route_seed": i, "map_slot": i, "map_slots": 3}  # noqa: E731
        assert venv.resets == [opts(0), "auto", opts(1), "auto", opts(2), "auto"], venv.resets
        # training continues from the fresh auto-reset episode (no extra reset)
        assert np.array_equal(model._last_obs, np.full((1, 3), 6.0, dtype=np.float32))
        assert model._last_original_obs is model._last_obs
        assert model._last_episode_starts.tolist() == [True]
        assert model.predict_calls and all(model.predict_calls)   # deterministic only
        evals = mc.read_jsonl(evaluator.jsonl_path)
        assert len(evals) == 1 and evals[0]["n_episodes"] == 3 and evals[0]["step"] == 10
        assert [e["spawn_index"] for e in evals[0]["episodes"]] == [0, 1, 2]
        assert "success_frac" in evals[0] and "collisions_per_km" in evals[0]
        # eval episodes never reach episodes.jsonl (only the 4 training episodes)
        assert len(mc.read_jsonl(episodes.jsonl_path)) == 4
        text = out.getvalue()
        assert "══ EVAL @ 10 steps" in text and "eval  1/3 |" in text and "| spawn   0" in text
        assert all(train_all.parse_step(ln) is None for ln in text.splitlines() if "eval" in ln.lower())
        tags = tb_tags(mc.tb_dir(tmp))
        assert {f"eval/{f}" for f in ("success_frac", "crash_frac", "return_total_mean")} <= set(tags)


def test_resume_drops_superseded_and_restores_window():
    with _tmp() as tmp:
        path = os.path.join(tmp, "episodes.jsonl")
        old = [mc.build_episode_record(fake_metrics(i), episode=i + 1, step=1000 * (i + 1),
                                       run_id="old", run_start_step=0) for i in range(10)]
        for r in old:
            mc.append_jsonl(path, r)
        model = FakeModel(logger=configure(None, [""]))
        model.num_timesteps = 6000       # resumed from the 6k checkpoint: episodes at 7k..10k are void
        cb = mc.EpisodeMetricsCallback(log_dir=tmp, verbose=0)
        cb.init_callback(model)
        assert cb.n_episodes == 6 and len(cb.recent) == 6
        drive(cb, model, 1)
        mc.close_writers()
        recs = mc.drop_superseded(mc.read_jsonl(path))
        assert [r["step"] for r in recs] == [1000, 2000, 3000, 4000, 5000, 6000, 6003]
        assert recs[-1]["episode"] == 7 and recs[-1]["run_start_step"] == 6000


def test_write_run_config():
    with _tmp() as tmp:
        model = FakeModel(logger=configure(None, [""]), env=FakeVecEnv())
        model.batch_size, model.gamma, model.learning_rate = 64, 0.99, 3e-4
        path = mc.write_run_config(tmp, model, extra={"experiment": "5q_cnn", "env": {"num_npc_vehicles": 30}})
        with open(path, encoding="utf-8") as fh:
            cfg = json.load(fh)
        assert cfg["reward_version"] == "3.0"
        assert cfg["hyperparameters"]["batch_size"] == 64 and cfg["hyperparameters"]["learning_rate"] == 3e-4
        assert cfg["env"]["num_npc_vehicles"] == 30 and cfg["extra"]["experiment"] == "5q_cnn"
        assert "commit" in cfg["git"]
        mc.write_run_config(tmp, model)
        assert len(mc.read_jsonl(os.path.join(tmp, "run_config_history.jsonl"))) == 1


def test_train_all_progress_parser_and_command():
    episode_line = mc.format_episode_line(mc.build_episode_record(fake_metrics(0), episode=412, step=98340))
    assert episode_line.startswith("ep  412 | step    98,340 |")
    assert train_all.parse_step(episode_line) == 98340
    assert train_all.parse_step("ep  412 | step  98,340 | 214 steps 1.6 km | END collision") == 98340
    assert train_all.parse_step("|    total_timesteps | 4000     |") == 4000
    assert train_all.parse_step("| total_timesteps | 1e+05 |") == 100000
    assert train_all.parse_step("── last 100 episodes @ 98k steps ──") is None
    assert train_all.parse_step("  eval  3/10 | spawn   2 |  100 steps") is None
    assert train_all.parse_step("[CBF] Step 12: correction mag=0.1") is None

    args = train_all.build_parser().parse_args([])
    assert args.render is True
    assert train_all.build_parser().parse_args(["--no-render"]).render is False
    args.num_npc, args.num_pedestrians = 10, None
    cmd = train_all.build_command("exp.py", 2000, "logs/x", None, args)
    assert cmd[cmd.index("--timesteps") + 1] == "1000000"          # target total, unchanged
    assert "--render" in cmd and cmd[cmd.index("--num-npc") + 1] == "10" and "--num-pedestrians" not in cmd
    with contextlib.redirect_stdout(io.StringIO()):
        carla = train_all.carla_command("CarlaUE4.exe", 2000, ["-RenderOffScreen", "-nullrhi", "-fps=20"])
    assert "-RenderOffScreen" not in carla and "-nullrhi" not in carla and "-fps=20" in carla


def test_make_report_on_synthetic_logs():
    import make_report
    with _tmp() as tmp:
        runs = []
        for name, shift in (("2q_cnn", 0), ("5q_cnn", 3)):
            folder = os.path.join(tmp, name)
            os.makedirs(folder)
            recs = [mc.build_episode_record(fake_metrics(i + shift, stage=min(3, i // 40)), episode=i + 1,
                                            step=250 * (i + 1), run_start_step=0) for i in range(150)]
            for r in recs:
                mc.append_jsonl(os.path.join(folder, "episodes.jsonl"), r)
            for k in range(1, 4):
                s = mc.summarize(recs[(k - 1) * 30: k * 30])
                s.update({"step": 12500 * k, "run_id": "x", "run_start_step": 0})
                mc.append_jsonl(os.path.join(folder, "eval.jsonl"), s)
            with open(os.path.join(folder, "episodes.jsonl"), "a", encoding="utf-8") as fh:
                fh.write('{"episode": 151, "step"')              # half-written last line is ignored
            runs += ["--run", f"{name}={folder}"]
        out = os.path.join(tmp, "report")
        make_report.DEFAULT_LOG_ROOTS = []
        with contextlib.redirect_stdout(io.StringIO()):
            path = make_report.main(runs + ["--out", out])
        assert path and os.path.isfile(path)
        with open(path, encoding="utf-8") as fh:
            page = fh.read()
        for text in ("Headline numbers", "Success rate", "Collisions per km", "Glossary",
                     "training episodes that ended that way", "evaluation @ 12,500"):
            assert text in page, text
        import pandas as pd
        summary = pd.read_csv(os.path.join(out, "summary.csv"))
        assert list(summary["experiment"]) == ["2q_cnn", "5q_cnn"]
        assert summary["eval_rounds"].tolist() == [3, 3]
        pngs = os.listdir(os.path.join(out, "plots"))
        assert "comparison.png" in pngs and "2q_cnn_outcomes.png" in pngs and "2q_cnn_reward_split.png" in pngs
        assert os.path.isfile(os.path.join(out, "2q_cnn_episodes.csv"))


def test_real_sac_learn_with_eval_and_metrics():
    """Vendored SB3 SAC + Monitor + DummyVecEnv on a toy env: eval mid-training, SB3 keeps going."""
    import gymnasium as gym
    from stable_baselines3 import SAC

    class ToyEnv(gym.Env):
        observation_space = gym.spaces.Box(-10.0, 10.0, (3,), np.float32)
        action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)

        def __init__(self):
            self.t, self.n_ep, self.reset_options = 0, 0, []

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.reset_options.append(options)
            self.t = 0
            return np.zeros(3, np.float32), {}

        def step(self, action):
            self.t += 1
            done = self.t >= 7
            info = {}
            if done:
                m = fake_metrics(self.n_ep)
                m["length_steps"] = self.t
                info["episode_metrics"] = m
                self.n_ep += 1
            return np.full(3, self.t, np.float32), float(action[0]), done, False, info

    with _tmp() as tmp:
        env = ToyEnv()
        model = SAC("MlpPolicy", env, learning_starts=20, batch_size=16, buffer_size=1000,
                    policy_kwargs={"net_arch": [16, 16]}, tensorboard_log=os.path.join(tmp, "tensorboard"),
                    verbose=0, seed=0)
        episodes = mc.EpisodeMetricsCallback(log_dir=tmp, verbose=1, summary_every=5)
        evaluator = mc.EvalCallback(eval_freq=50, n_episodes=2, log_dir=tmp, verbose=1)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            model.learn(total_timesteps=120, callback=[episodes, evaluator], log_interval=None)
        mc.close_writers()
        evals = mc.read_jsonl(os.path.join(tmp, "eval.jsonl"))
        assert [e["step"] for e in evals] == [50, 100], [e["step"] for e in evals]
        slots = evaluator.n_episodes
        assert {"spawn_index": 0, "route_seed": 0, "map_slot": 0, "map_slots": slots} in env.reset_options
        assert {"spawn_index": 1, "route_seed": 1, "map_slot": 1, "map_slots": slots} in env.reset_options
        assert model.num_timesteps == 120
        recs = mc.read_jsonl(os.path.join(tmp, "episodes.jsonl"))
        # every training step is accounted for by complete or abandoned training episodes; eval episodes
        # (2 per round x 2 rounds) never show up, and the replay buffer only holds training transitions
        assert all(r["length_steps"] == 7 for r in recs)
        assert model.replay_buffer.size() == 120
        assert len(recs) + 2 * 2 == env.n_ep, (len(recs), env.n_ep)
        assert "══ EVAL @ 50 steps" in out.getvalue()


def _run_all():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            failed += 1
            import traceback
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
