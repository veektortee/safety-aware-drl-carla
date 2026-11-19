"""Offline tests: map rotation helpers, the 6-stage curriculum with its 200k-step schedule, keep-latest
checkpoints with curriculum snapshots, erasing logs past a checkpoint on resume, and the demo's
checkpoint-ring geometry. No CARLA server needed.

Run:  python tests/unit/test_maps_curriculum.py      or      python -m pytest tests/unit/test_maps_curriculum.py
"""

import os
import sys

sys.dont_write_bytecode = True  # tracked __pycache__/*.pyc files must not change
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
for path in (REPO_ROOT, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

import json
import math
import tempfile

import numpy as np

from test_exp_glue import FakeCarlaEnv, _fake_model, _feed, _make_sac
from commons.obs_wrappers import CnnNavObservationWrapper


def _tmp():
    return tempfile.TemporaryDirectory(ignore_cleanup_errors=True)


# --------------------------------------------------------------------------------------------------
# Maps
# --------------------------------------------------------------------------------------------------

def test_map_selection_helpers():
    from tests.pipeline_carla_test import eval_map, map_basename, next_map, pick_town_maps, vehicle_cap

    available = ["/Game/Carla/Maps/Town10HD_Opt", "/Game/Carla/Maps/Town01", "/Game/Carla/Maps/Town01_Opt",
                 "/Game/Carla/Maps/Town02", "/Game/Carla/Maps/Town05_Opt", "/Game/Carla/Maps/Town10HD",
                 "/Game/Carla/TestMaps/EmptyMap", "/Game/Carla/Maps/OpenDriveMap", "/Game/Carla/Maps/Town03"]
    maps = pick_town_maps(available)
    assert maps == ["Town01_Opt", "Town02", "Town03", "Town05_Opt", "Town10HD_Opt"], maps
    assert map_basename("Carla/Maps/Town10HD_Opt") == "Town10HD_Opt"
    assert next_map(maps, "Carla/Maps/Town03") == "Town05_Opt"
    assert next_map(maps, "Town10HD_Opt") == "Town01_Opt", "wraps around"
    assert next_map(maps, "Town07") == "Town01_Opt", "unknown current map -> first"
    # an eval round of 10 episodes covers all 5 maps in order, 2 episodes each
    assert [eval_map(maps, i, 10) for i in range(10)] == [m for m in maps for _ in range(2)]
    assert {eval_map(maps, i, 3) for i in range(3)} <= set(maps)
    assert vehicle_cap(372) == 335 and vehicle_cap(101) == 91 and vehicle_cap(5) == 0


# --------------------------------------------------------------------------------------------------
# Curriculum
# --------------------------------------------------------------------------------------------------

def test_six_stages_up_to_300_and_200k_schedule():
    from commons import curriculum_stages as cs

    assert len(cs.STAGES) == 6 and cs.FINAL_STAGE == 5
    assert cs.STAGES[-1]["num_npc"] == 300 and cs.STAGES[-1]["num_walkers"] == 300
    npcs = [s["num_npc"] for s in cs.STAGES]
    assert npcs == sorted(npcs), "traffic only grows"
    assert [s["crash_penalty"] for s in cs.STAGES[:3]] == [50.0, 100.0, 150.0], "earlier stages unchanged"
    assert cs.PROMOTE_BY_STEP[-1] < cs.CURRICULUM_TOTAL_STEPS == 200_000
    assert [cs.scheduled_stage(s) for s in (0, 14_999, 15_000, 75_000, 125_000, 140_000, 200_000)] == [0, 0, 1, 3, 4, 5, 5]


def test_curriculum_promotes_on_schedule_and_resumes_at_scheduled_stage():
    from commons.curriculum import CurriculumCallback, resolve_start_stage
    from commons.curriculum_stages import PROMOTE_BY_STEP

    with _tmp() as log_dir:
        fake = FakeCarlaEnv()
        model = _fake_model(fake, num_timesteps=0)
        cb = CurriculumCallback(log_dir, start_stage=0, verbose=0)
        cb.init_callback(model)
        cb.on_training_start({}, {})
        _feed(cb, "collision")
        assert cb.stage == 0, "no success, schedule not due"
        model.num_timesteps = PROMOTE_BY_STEP[0]
        _feed(cb, "collision")
        assert cb.stage == 1 and fake.stage_calls == [1], "promoted by the step schedule"
        with open(os.path.join(log_dir, "curriculum_state.json"), encoding="utf-8") as f:
            assert json.load(f)["history"][-1]["reason"] == "schedule"

        # resume at 125k with a saved stage of 1: the env starts at the scheduled stage 4
        assert resolve_start_stage(log_dir, 0, resume=True, step=125_000) == 4
        assert resolve_start_stage(log_dir, 0, resume=False, step=125_000) == 0
        fake2 = FakeCarlaEnv()
        cb2 = CurriculumCallback(log_dir, start_stage=0, verbose=0)
        cb2.init_callback(_fake_model(fake2, num_timesteps=125_000))
        cb2.on_training_start({}, {})
        assert cb2.stage == 4 and fake2.stage_calls == [4] and cb2.episodes_in_stage == 0


# --------------------------------------------------------------------------------------------------
# Checkpoints and resume clean-up
# --------------------------------------------------------------------------------------------------

def test_latest_checkpoint_callback_keeps_one_with_curriculum_snapshot():
    from commons.warmup import LatestCheckpointCallback, restore_curriculum_snapshot

    model = _make_sac(CnnNavObservationWrapper(FakeCarlaEnv(), occlusion="none"))
    with _tmp() as tmp:
        log_dir, ckpt = tmp, os.path.join(tmp, "checkpoints")
        cb = LatestCheckpointCallback(10, ckpt, "sac_t", log_dir=log_dir, keep=1, verbose=0)
        cb.init_callback(model)
        for step in range(1, 26):
            model.num_timesteps = step
            with open(os.path.join(log_dir, "curriculum_state.json"), "w", encoding="utf-8") as f:
                json.dump({"stage": step // 10, "num_timesteps": step}, f)
            cb.on_step()
        assert sorted(os.listdir(ckpt)) == ["sac_t_20_steps.zip", "sac_t_curriculum_20_steps.json",
                                           "sac_t_replay_buffer_20_steps.pkl"], os.listdir(ckpt)
        with open(os.path.join(log_dir, "curriculum_state.json"), "w", encoding="utf-8") as f:
            json.dump({"stage": 9}, f)
        assert restore_curriculum_snapshot(log_dir, os.path.join(ckpt, "sac_t_20_steps.zip"))
        with open(os.path.join(log_dir, "curriculum_state.json"), encoding="utf-8") as f:
            assert json.load(f) == {"stage": 2, "num_timesteps": 20}
        assert not restore_curriculum_snapshot(log_dir, os.path.join(ckpt, "missing_5_steps.zip"))


def _tb_steps(tb_dir, tag):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(tb_dir)
    acc.Reload()
    return [e.step for e in acc.Scalars(tag)] if tag in acc.Tags()["scalars"] else []


def test_discard_logs_after_checkpoint_erases_tensorboard_and_jsonl():
    from torch.utils.tensorboard import SummaryWriter

    from commons.warmup import discard_logs_after

    with _tmp() as log_dir:
        sb3_dir = os.path.join(log_dir, "tensorboard")
        metrics_dir = os.path.join(sb3_dir, "metrics")
        for d, tag in ((sb3_dir, "train/critic_loss"), (metrics_dir, "metrics/success")):
            w = SummaryWriter(log_dir=d)
            for step in range(0, 1400, 100):
                w.add_scalar(tag, step / 100.0, step)
            w.close()
        held_open = open(next(os.path.join(sb3_dir, f) for f in os.listdir(sb3_dir) if "tfevents" in f), "rb")
        try:   # like a running TensorBoard server holding the file
            for name, steps in (("episodes.jsonl", range(0, 1400, 50)), ("eval.jsonl", (500, 1000))):
                with open(os.path.join(log_dir, name), "w", encoding="utf-8") as f:
                    for s in steps:
                        f.write(json.dumps({"step": s}) + "\n")
                    f.write('{"step": 13')                      # half-written last line
            out = discard_logs_after(log_dir, 1250, verbose=0)
        finally:
            held_open.close()
        assert out["errors"] == [] and out["files_changed"] == 2 and out["events_dropped"] == 2
        assert _tb_steps(sb3_dir, "train/critic_loss") == list(range(0, 1300, 100))
        assert _tb_steps(metrics_dir, "metrics/success") == list(range(0, 1300, 100))
        with open(os.path.join(log_dir, "episodes.jsonl"), encoding="utf-8") as f:
            assert [json.loads(l)["step"] for l in f] == list(range(0, 1300, 50))
        with open(os.path.join(log_dir, "eval.jsonl"), encoding="utf-8") as f:
            assert [json.loads(l)["step"] for l in f] == [500, 1000]
        again = discard_logs_after(log_dir, 1250, verbose=0)
        assert again["files_changed"] == 0 and again["jsonl_records_dropped"] == 0, "idempotent"


# --------------------------------------------------------------------------------------------------
# Demo checkpoint rings
# --------------------------------------------------------------------------------------------------

def test_checkpoint_ring_geometry():
    from demo.checkpoint_rings import checkpoint_positions, project_to_image, ring_points

    assert checkpoint_positions(100.0, 30.0) == [30.0, 60.0, 100.0], "no gate in the last 10 m, then the finish"
    assert checkpoint_positions(0.0) == []
    gate = ring_points((10.0, 0.0, 2.0), 0.0, 2.6, 1.8, upright=True)      # road along +x
    assert gate.shape == (33, 3) and np.allclose(gate[:, 0], 10.0), "upright gate spans the road (y, z)"
    assert math.isclose(gate[:, 1].max(), 2.6, abs_tol=1e-9) and math.isclose(gate[:, 2].max(), 3.8, abs_tol=1e-6)
    flat = ring_points((0.0, 0.0, 0.3), math.pi / 2, 2.6, 1.6, upright=False)   # road along +y
    assert np.allclose(flat[:, 2], 0.3) and math.isclose(np.abs(flat[:, 0]).max(), 2.6, abs_tol=1e-9)

    # camera at the origin looking along +x: a point straight ahead projects to the image centre,
    # one to the right lands right of centre, one behind is NaN
    world_to_cam = np.eye(4)
    uv = project_to_image(np.array([[10.0, 0.0, 0.0], [10.0, 2.0, 1.0], [-5.0, 0.0, 0.0]]), world_to_cam,
                          640, 360, 90.0)
    assert np.allclose(uv[0], [320.0, 180.0]) and uv[1, 0] > 320 and uv[1, 1] < 180 and np.isnan(uv[2]).all()


def test_checkpoint_rings_count_the_finish_gate():
    from types import SimpleNamespace

    from demo.checkpoint_rings import CheckpointRings

    route = SimpleNamespace(length_m=100.0, point_at=lambda s: (s, 0.0), z_at=lambda s: 0.0, yaw_at=lambda s: 0.0)
    env = SimpleNamespace(route=route, _snapshot={"s": 0.0}, display_overlay=None, rgb_camera=None)
    rings = CheckpointRings(env, overview=False)
    assert env.display_overlay == rings.overlay
    rings.start_episode()
    assert len(rings.gates) == 3 and rings.gates[-1]["finish"]
    env._snapshot["s"] = 65.0
    rings.update()
    assert rings.collected == 2
    env._snapshot["s"] = 98.0           # the env ends the episode a little before the finish gate
    rings.update()
    assert rings.collected == 2
    rings.finish_episode("endpoint")
    assert rings.collected == 3, "a finished route shows N/N, not N-1/N"
    rings.start_episode()
    env._snapshot["s"] = 40.0
    rings.update()
    rings.finish_episode("collision")
    assert rings.collected == 1, "only a finished route collects the remaining gates"


# --------------------------------------------------------------------------------------------------
# NPC physics guard (Traffic Manager memory runaway on dense Town01)
# --------------------------------------------------------------------------------------------------

def test_traffic_caps_are_70_percent_of_each_town():
    from tests.pipeline_carla_test import TRAINING_TRAFFIC_FILL as FILL, traffic_caps

    assert FILL == 0.7
    spawn_points = {"Town02": 101, "Town03": 265, "Town04": 372, "Town05": 302, "Town10HD": 155}
    cars = {t: traffic_caps(n, 300, FILL)[0] for t, n in spawn_points.items()}
    assert cars == {"Town02": 63, "Town03": 167, "Town04": 234, "Town05": 190, "Town10HD": 98}, cars
    assert traffic_caps(155, 300, FILL)[1] == 210, "70% of the 300-pedestrian maximum"
    assert traffic_caps(155, 300) == (140, 300), "default fill 1.0 = the original 100% curriculum"


def test_all_maps_leaves_out_unstable_towns_but_explicit_lists_keep_them():
    from types import SimpleNamespace
    from tests.pipeline_carla_test import CarlaGymEnv, UNSTABLE_TOWNS, town_of

    assert "Town01" in UNSTABLE_TOWNS and town_of("Carla/Maps/Town01_Opt") == "Town01"
    assert town_of("Town10HD_Opt") == "Town10HD"
    env = CarlaGymEnv.__new__(CarlaGymEnv)
    env.client = SimpleNamespace(get_available_maps=lambda: [
        "/Game/Carla/Maps/Town01_Opt", "/Game/Carla/Maps/Town01", "/Game/Carla/Maps/Town02_Opt",
        "/Game/Carla/Maps/Town10HD_Opt"])
    env.map = SimpleNamespace(name="Carla/Maps/Town10HD_Opt")
    assert env._resolve_maps("all") == ["Town02_Opt", "Town10HD_Opt"], "rotation and eval skip Town01"
    assert env._resolve_maps(["Town01_Opt"]) == ["Town01_Opt"], "an explicit map list is honoured"


def test_npc_physics_guard_flags_launched_sunk_and_absurdly_fast_cars():
    from tests.pipeline_carla_test import npc_physics_broken

    flat = (0.0, 0.5)   # Town01-like road heights
    assert not npc_physics_broken(8.0, 0.0, 0.0, 0.3, *flat), "normal driving"
    assert not npc_physics_broken(20.0, 10.0, 2.5, 0.3, *flat), "fast car over a crest"
    assert npc_physics_broken(7.7, 0.0, 0.0, -61.0, *flat), "under the map (seen on Town01 at z = -61)"
    assert npc_physics_broken(1.0, 0.0, 0.0, 41.0, *flat), "launched 40 m up (seen on Town01)"
    assert npc_physics_broken(5.0, 0.0, -12.0, 2.0, *flat), "falling"
    assert npc_physics_broken(56.0, 0.0, 0.0, 0.3, *flat), "56 m/s launch (seen on Town01)"
    assert npc_physics_broken(float("nan"), 0.0, 0.0, 0.3, *flat), "NaN velocity"
    assert npc_physics_broken(1.0, 0.0, 0.0, float("inf"), *flat), "non-finite position"
    hilly = (-5.0, 30.0)
    assert not npc_physics_broken(10.0, 0.0, 1.0, 29.0, *hilly), "on the highest road of a hilly town"
    assert not npc_physics_broken(10.0, 0.0, -1.0, -6.0, *hilly), "on the lowest road of a hilly town"


def test_env_destroys_only_broken_npcs_after_a_tick():
    from types import SimpleNamespace
    from tests.pipeline_carla_test import CarlaGymEnv

    def actor(v, z):
        return SimpleNamespace(get_velocity=lambda: SimpleNamespace(x=v[0], y=v[1], z=v[2]),
                               get_transform=lambda: SimpleNamespace(location=SimpleNamespace(z=z)))

    state = {1: actor((8, 0, 0), 0.3), 2: actor((7.7, 0, 0), -61.0), 3: actor((0, 0, 0), 0.2),
             4: actor((56, 0, 0), 0.3)}   # 5: already gone (None)
    destroyed = []
    env = CarlaGymEnv.__new__(CarlaGymEnv)      # no CARLA: only the attributes the guard uses
    env.npc_vehicles = [1, 2, 3, 4, 5]
    env.world = SimpleNamespace(get_snapshot=lambda: SimpleNamespace(find=lambda i: state.get(i)))
    env.client = SimpleNamespace(apply_batch_sync=lambda cmds, tick: destroyed.extend(c.actor_id for c in cmds))
    env.map = SimpleNamespace(name="Carla/Maps/Town01_Opt")
    env._road_z_range = (0.0, 0.5)
    env._npc_removed_on_map = env._npc_removed_pending = 0

    assert env._remove_broken_npcs() == 2
    assert sorted(destroyed) == [2, 4], destroyed
    assert env.npc_vehicles == [1, 3, 5] and env._npc_removed_on_map == 2
    assert env._npc_removed_pending == 2, "replaced at the next reset"
    assert env._remove_broken_npcs() == 0, "nothing left to remove"


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
