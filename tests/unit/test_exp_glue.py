"""Offline tests for the experiment glue (no CARLA server needed).

Covers commons/obs_wrappers.py (CNN observation shapes, occlusion on the image only; the STT wrapper
is tested in test_stt.py),
commons/warmup.py (expert prefill of a real SB3 SAC MultiInputPolicy replay buffer, resume helpers)
and commons/curriculum.py (promotion logic and state persistence), using a small fake env that
exposes the CarlaGymEnv contract: Dict(rgb_data, nav) observation, Box(2) action,
info['episode_metrics'] on the last step, expert_action(), set_curriculum_stage().

Run:  python tests/unit/test_exp_glue.py      or      python -m pytest tests/unit/test_exp_glue.py
"""

import os
import sys

sys.dont_write_bytecode = True  # tracked __pycache__/*.pyc files must not change
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import json
import tempfile
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from commons.nav_layout import NAV_DIM

EXPERT = np.array([0.25, 0.5], dtype=np.float32)


class FakeCarlaEnv(gym.Env):
    """Episodes of `episode_len` steps; even episodes end at the route end (terminated), odd ones
    time out (truncated). nav[0] = step within the episode, nav[1] = episode index."""

    metadata = {"render_modes": []}

    def __init__(self, episode_len: int = 7):
        super().__init__()
        self.observation_space = spaces.Dict({
            "rgb_data": spaces.Box(0, 255, (360, 640, 3), np.uint8),
            "nav": spaces.Box(-np.inf, np.inf, (NAV_DIM,), np.float32),
        })
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.episode_len = episode_len
        self.curriculum_stage = 0
        self.stage_calls = []
        self.reset_count = 0
        self.episode_count = 0
        self.actions = []
        self._t = 0
        self._ret = 0.0

    def _obs(self):
        rgb = np.full((360, 640, 3), (self._t * 10) % 256, dtype=np.uint8)
        nav = np.zeros(NAV_DIM, dtype=np.float32)
        nav[0], nav[1] = self._t, self.episode_count
        return {"rgb_data": rgb, "nav": nav}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_count += 1
        self._t, self._ret = 0, 0.0
        return self._obs(), {}

    def step(self, action):
        self.actions.append(np.array(action, dtype=np.float32))
        self._t += 1
        reward = 1.0
        self._ret += reward
        terminated = truncated = False
        info = {"terminated_reason": None, "truncated_reason": None}
        if self._t >= self.episode_len:
            reason = "endpoint" if self.episode_count % 2 == 0 else "timeout"
            terminated, truncated = reason == "endpoint", reason != "endpoint"
            metrics = {"end_reason": reason, "return": self._ret, "length_steps": self._t,
                       "route_completion_frac": 1.0 if terminated else 0.5, "collisions": 0,
                       "curriculum_stage": self.curriculum_stage, "r": self._ret, "l": self._t, "t": 0.0}
            info["episode_metrics"] = metrics
            info["episode"] = metrics
            self.episode_count += 1
        return self._obs(), reward, terminated, truncated, info

    def expert_action(self):
        return EXPERT.copy()

    def set_curriculum_stage(self, stage):
        self.curriculum_stage = int(stage)
        self.stage_calls.append(int(stage))


def _tmpdir():
    return tempfile.TemporaryDirectory(ignore_cleanup_errors=True)


# --------------------------------------------------------------------------------------------------
# Observation wrappers
# --------------------------------------------------------------------------------------------------

def test_cnn_wrapper_shapes_and_nav():
    from commons.obs_wrappers import CnnNavObservationWrapper

    env = CnnNavObservationWrapper(FakeCarlaEnv(), size=84, occlusion="none")
    assert env.observation_space["image"].shape == (3, 84, 84)
    assert env.observation_space["image"].dtype == np.uint8
    assert env.observation_space["nav"].shape == (NAV_DIM,)
    obs, _ = env.reset()
    assert obs["image"].shape == (3, 84, 84) and obs["image"].dtype == np.uint8
    assert obs["nav"].shape == (NAV_DIM,) and obs["nav"].dtype == np.float32
    assert env.observation_space.contains(obs)
    obs, *_ = env.step(EXPERT)
    assert np.all(obs["image"] == 10) and obs["nav"][0] == 1.0
    assert obs["image"].flags["C_CONTIGUOUS"]

    bad_nav = np.zeros(NAV_DIM, dtype=np.float32)
    bad_nav[0], bad_nav[1], bad_nav[2] = np.nan, np.inf, -1e9
    out = env.observation({"rgb_data": np.zeros((360, 640, 3), np.uint8), "nav": bad_nav})
    assert np.all(np.isfinite(out["nav"])) and out["nav"][0] == 0.0
    assert out["nav"][1] == 10.0 and out["nav"][2] == -10.0


def test_cnn_wrapper_occludes_image_only():
    from commons.obs_wrappers import CnnNavObservationWrapper, _occlude

    env = CnnNavObservationWrapper(FakeCarlaEnv(), size=84, occlusion="black_bars")
    env.reset()
    occ = env.occlusion
    occ.active, occ.bar_height, occ.bar_position = True, 90, "top"   # force an occluded episode
    obs, *_ = env.step(EXPERT)                                        # camera value 10 everywhere
    rows = 90 * 84 // 360
    assert np.all(obs["image"][:, : rows - 1, :] == 0), "top bar must be black"
    assert np.all(obs["image"][:, rows + 1:, :] == 10), "rest of the image untouched"
    assert obs["nav"][0] == 1.0, "nav vector is never occluded"

    rgb = np.full((360, 640, 3), 7, np.uint8)
    out = _occlude(occ, rgb)
    assert np.all(rgb == 7), "the env's array must not be modified"
    assert np.all(out[:90] == 0)

    # reset re-rolls the occlusion for the new episode (30% chance of being active)
    np.random.seed(1)
    actives = []
    for _ in range(200):
        env.reset()
        actives.append(env.occlusion.active)
    assert 0.15 < np.mean(actives) < 0.45


# --------------------------------------------------------------------------------------------------
# Expert warm-up and resume helpers
# --------------------------------------------------------------------------------------------------

def _make_sac(env):
    from stable_baselines3 import SAC

    return SAC("MultiInputPolicy", env, buffer_size=200, learning_starts=0, batch_size=8,
               policy_kwargs={"net_arch": [32, 32], "n_critics": 2}, device="cpu", verbose=0, seed=0)


def test_prefill_replay_buffer_and_resume_helpers():
    from stable_baselines3 import SAC
    from commons.obs_wrappers import CnnNavObservationWrapper
    from commons.warmup import (checkpoint_num_timesteps, find_carla_env, prefill_replay_buffer,
                                replay_buffer_candidates, restore_replay_buffer, save_checkpoint)

    fake = FakeCarlaEnv(episode_len=7)
    model = _make_sac(CnnNavObservationWrapper(fake, occlusion="none"))
    assert find_carla_env(model) is fake

    summary = prefill_replay_buffer(model, 25, log_every=10, verbose=0)
    rb = model.replay_buffer
    assert summary["steps"] == 25 and summary["episodes"] == 3
    assert summary["end_reasons"] == {"endpoint": 2, "timeout": 1}
    assert rb.size() == 25 and model.num_timesteps == 0
    assert np.allclose(rb.actions[:25, 0], EXPERT), "stored action = scaled expert action"
    assert all(np.allclose(a, EXPERT) for a in fake.actions)
    assert np.allclose(rb.rewards[:25, 0], 1.0)

    done_idx = [6, 13, 20]
    assert set(np.flatnonzero(rb.dones[:25, 0])) == set(done_idx)
    # episode 2 (index 13) was truncated -> timeout flag so it bootstraps; terminated ones do not
    assert set(np.flatnonzero(rb.timeouts[:25, 0])) == {13}
    for i in done_idx:
        assert rb.next_observations["nav"][i, 0, 0] == 7.0, "next_obs at an episode end = terminal obs"
        assert rb.observations["nav"][i + 1, 0, 0] == 0.0, "next transition starts the new episode"
    assert rb.next_observations["image"][0].shape == (1, 3, 84, 84)

    # learn() continues the warm-up's running episode (no reset) and keeps the counter consistent
    assert model._last_obs["nav"][0, 0] == 4.0
    resets = fake.reset_count
    model.learn(total_timesteps=2, log_interval=None, reset_num_timesteps=False)
    assert fake.reset_count == resets
    assert model.num_timesteps == 2 and rb.size() == 27
    assert rb.observations["nav"][25, 0, 0] == 4.0

    with _tmpdir() as tmp:
        zip_path = os.path.join(tmp, "sac_t_2_steps.zip")
        model.save(zip_path)
        assert checkpoint_num_timesteps(zip_path) == 2
        assert checkpoint_num_timesteps(os.path.join(tmp, "missing.zip")) is None

        cands = replay_buffer_candidates(zip_path)
        assert cands == [os.path.join(tmp, "sac_t_replay_buffer_2_steps.pkl"),
                         os.path.join(tmp, "sac_t_2_steps_replay_buffer.pkl")]
        for name in (os.path.basename(c) for c in cands):       # both file-name conventions work
            model.save_replay_buffer(os.path.join(tmp, name))
            resumed = SAC.load(zip_path, env=CnnNavObservationWrapper(FakeCarlaEnv(), occlusion="none"),
                               device="cpu")
            assert restore_replay_buffer(resumed, zip_path, verbose=0)
            assert resumed.replay_buffer.size() == 27
            os.remove(os.path.join(tmp, name))

        with open(cands[0], "wb") as f:                         # truncated / corrupt file
            f.write(b"not a pickle")
        resumed = SAC.load(zip_path, env=CnnNavObservationWrapper(FakeCarlaEnv(), occlusion="none"), device="cpu")
        empty = resumed.replay_buffer
        assert not restore_replay_buffer(resumed, zip_path, verbose=0)
        assert resumed.replay_buffer is empty and empty.size() == 0

        ckpt = save_checkpoint(model, os.path.join(tmp, "ck"), "sac_x", verbose=0)
        assert os.path.basename(ckpt) == "sac_x_2_steps.zip"
        files = sorted(os.listdir(os.path.join(tmp, "ck")))
        assert files == ["sac_x_2_steps.zip", "sac_x_replay_buffer_2_steps.pkl"], files


# --------------------------------------------------------------------------------------------------
# Curriculum
# --------------------------------------------------------------------------------------------------

def _fake_model(env, num_timesteps=0):
    from stable_baselines3.common.logger import Logger
    from stable_baselines3.common.vec_env import DummyVecEnv

    vec = DummyVecEnv([lambda: env])
    return SimpleNamespace(get_env=lambda: vec, num_timesteps=num_timesteps, logger=Logger(None, []))


def _feed(cb, reason, stage=None, monitor_only=False):
    if monitor_only:
        info = {"episode": {"r": 1.0, "l": 5, "t": 0.1}}           # what Monitor alone leaves behind
    else:
        info = {"episode_metrics": {"end_reason": reason,
                                    "curriculum_stage": cb.stage if stage is None else stage}}
    cb.update_locals({"infos": [info]})
    cb.on_step()


def test_curriculum_promotion_and_persistence():
    from commons.curriculum import STATE_FILE, CurriculumCallback, resolve_start_stage
    from commons.curriculum_stages import FINAL_STAGE, MIN_EPISODES_PER_STAGE, PROMOTE_SUCCESS_RATE, PROMOTE_WINDOW

    assert MIN_EPISODES_PER_STAGE <= PROMOTE_WINDOW, "test assumes the window is the binding constraint"
    with _tmpdir() as log_dir:
        fake = FakeCarlaEnv()
        model = _fake_model(fake)
        cb = CurriculumCallback(log_dir, start_stage=0, verbose=0)
        cb.init_callback(model)
        cb.on_training_start({}, {})
        assert cb.stage == 0 and fake.stage_calls == []
        assert os.path.isfile(os.path.join(log_dir, STATE_FILE))

        for _ in range(PROMOTE_WINDOW - 1):
            _feed(cb, "endpoint")
        assert cb.stage == 0, "needs a full window"
        _feed(cb, "endpoint", monitor_only=True)                   # Monitor-only info is not an episode end
        assert cb.stage == 0 and cb.episodes_in_stage == PROMOTE_WINDOW - 1
        _feed(cb, "endpoint")
        assert cb.stage == 1 and fake.stage_calls == [1] and cb.episodes_in_stage == 0
        with open(os.path.join(log_dir, STATE_FILE), encoding="utf-8") as f:
            state = json.load(f)
        assert state["stage"] == 1 and state["history"][-1]["event"] == "promote"

        _feed(cb, "endpoint", stage=0)                              # still ran under the old stage
        assert cb.episodes_in_stage == 0

        # rolling window: PROMOTE_WINDOW failures, then successes until the window reaches the rate
        for _ in range(PROMOTE_WINDOW):
            _feed(cb, "collision")
        need = int(np.ceil(PROMOTE_SUCCESS_RATE * PROMOTE_WINDOW))
        for _ in range(need - 1):
            _feed(cb, "endpoint")
        assert cb.stage == 1, f"{need - 1}/{PROMOTE_WINDOW} successes is below the threshold"
        _feed(cb, "endpoint")
        assert cb.stage == 2 and fake.stage_calls == [1, 2]

        for _ in range(PROMOTE_WINDOW // 2):                        # partial progress in stage 2
            _feed(cb, "endpoint")
        assert resolve_start_stage(log_dir, 0, resume=True) == 2
        assert resolve_start_stage(log_dir, 0, resume=False) == 0
        assert resolve_start_stage(log_dir, FINAL_STAGE, resume=True) == FINAL_STAGE

        # resume (num_timesteps > 0): stage, window and counters restored and applied to the env
        fake2 = FakeCarlaEnv()
        cb2 = CurriculumCallback(log_dir, start_stage=0, verbose=0)
        cb2.init_callback(_fake_model(fake2, num_timesteps=5000))
        cb2.on_training_start({}, {})
        assert cb2.stage == 2 and fake2.stage_calls == [2]
        assert cb2.episodes_in_stage == PROMOTE_WINDOW // 2 and len(cb2.window) == PROMOTE_WINDOW // 2

        # fresh run (num_timesteps == 0) in the same log dir ignores the old state
        fake3 = FakeCarlaEnv()
        cb3 = CurriculumCallback(log_dir, start_stage=0, verbose=0)
        cb3.init_callback(_fake_model(fake3, num_timesteps=0))
        cb3.on_training_start({}, {})
        assert cb3.stage == 0 and fake3.stage_calls == []

    with _tmpdir() as log_dir:                                      # --no-curriculum: final stage, never beyond
        fake = FakeCarlaEnv()
        cb = CurriculumCallback(log_dir, start_stage=FINAL_STAGE, verbose=0)
        cb.init_callback(_fake_model(fake))
        cb.on_training_start({}, {})
        assert fake.stage_calls == [FINAL_STAGE]
        for _ in range(PROMOTE_WINDOW + 10):
            _feed(cb, "endpoint")
        assert cb.stage == FINAL_STAGE and fake.stage_calls == [FINAL_STAGE]


if __name__ == "__main__":
    import traceback

    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
