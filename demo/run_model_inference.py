"""
Demo: watch a trained SAC model drive in CARLA
==============================================

Loads a model saved by one of the four experiments and runs deterministic episodes with the SAME
wrapper stack as training:

    CarlaGymEnv -> CBFSafetyLayerWrapper -> CnnNavObservationWrapper   (CNN models, Dict observation)
    CarlaGymEnv -> CBFSafetyLayerWrapper -> SttNavObservationWrapper   (STT models, vector observation)

The variant is detected from the model's observation space. The camera window is shown and the
spectator follows the car (disable with --no-render). One line is printed per episode.

With the CBF enabled, the critic-ensemble trust score is computed from the loaded model at every
step and fed to the safety layer, as PolicyTrustScoreCallback does during training (--no-trust
keeps trust at 1.0). STT models need the run's transformer weights: they are found automatically in
<log-dir>/checkpoints/transformer/ (or pass --transformer-path).

A CARLA server must be running on --port.

Usage:
    python demo/run_model_inference.py --model-path logs/5q_cnn/sac_5q_cnn_final.zip --episodes 3
    python demo/run_model_inference.py --model-path logs/2q_stt/checkpoints/sac_2q_stt_500000_steps.zip
    python demo/run_model_inference.py --model 5cnn --checkpoint-dir demos     # demos/5cnn.zip
    python demo/run_model_inference.py --model-path <zip> --no-cbf --fixed-spawns
    python demo/run_model_inference.py --list-models
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from gymnasium import spaces
from stable_baselines3 import SAC

from tests.pipeline_carla_test import CarlaGymEnv, CBFSafetyLayerWrapper, _compute_live_policy_metrics
from commons.curriculum_stages import FINAL_STAGE, STAGES
from commons.nav_layout import NAV_DIM
from commons.obs_wrappers import CnnNavObservationWrapper, SttNavObservationWrapper, find_transformer_checkpoint

LEGACY_MODELS = {
    "5cnn": "SAC 5 critics, CNN",
    "2cnn": "SAC 2 critics, CNN",
    "5stt": "SAC 5 critics, SpatioTemporal Transformer",
    "2stt": "SAC 2 critics, SpatioTemporal Transformer",
}


def _resolve(path: str) -> str:
    """Relative paths: working directory first, then the repo root."""
    p = Path(path)
    if p.is_absolute() or p.exists():
        return str(p)
    alt = REPO_ROOT / path
    return str(alt) if alt.exists() else str(p)


def list_models(checkpoint_dir: str):
    print(f"\nModels in {checkpoint_dir}:")
    for key, desc in LEGACY_MODELS.items():
        path = os.path.join(checkpoint_dir, f"{key}.zip")
        status = f"{os.path.getsize(path) / 2**20:.1f} MB" if os.path.isfile(path) else "not found"
        print(f"  {key:5s} | {desc:45s} | {status}")
    print("\nAny experiment output also works: --model-path <log-dir>/sac_<tag>_final.zip\n")


def find_cbf_wrapper(env) -> Optional[CBFSafetyLayerWrapper]:
    while env is not None:
        if isinstance(env, CBFSafetyLayerWrapper):
            return env
        env = getattr(env, "env", None)
    return None


def build_env(model, args, model_path: str):
    """Same stack as training, chosen from the model's observation space."""
    obs_space = model.observation_space
    is_cnn = isinstance(obs_space, spaces.Dict) and "image" in obs_space.spaces
    is_stt = isinstance(obs_space, spaces.Box) and len(obs_space.shape) == 1 and obs_space.shape[0] > NAV_DIM
    if not (is_cnn or is_stt):
        raise SystemExit(f"[ERROR] {model_path} has observation space {obs_space}; it was trained with the old "
                         "env (before image+nav observations and [steer, accel] actions) and cannot run here.")
    env = CarlaGymEnv(
        host="localhost",
        port=args.port,
        timeout=10.0,
        time_limit=3000,
        render_mode=None if args.no_render else "human",
        num_npc_vehicles=args.num_npc,
        num_pedestrians=args.num_pedestrians,
        show_sensor_data=not args.no_render,
        curriculum_stage=args.stage,
    )
    if not args.no_cbf:
        env = CBFSafetyLayerWrapper(env, use_trust_score=not args.no_trust)
    if is_cnn:
        size = int(obs_space.spaces["image"].shape[-1])
        env = CnnNavObservationWrapper(env, size=size, occlusion=args.occlusion)
        variant = f"CNN (image {size}x{size} + nav)"
    else:
        embed_dim = int(obs_space.shape[0]) - NAV_DIM
        env = SttNavObservationWrapper(env, embed_dim=embed_dim, num_frames=8, occlusion=args.occlusion)
        tpath = args.transformer_path or find_transformer_checkpoint(model_path)
        if tpath:
            env.load_transformer(tpath)
            print(f"[SETUP] transformer weights: {tpath}")
        else:
            print("[SETUP] WARNING: no transformer_checkpoint_*.pth found for this model; the STT has new "
                  "weights and the policy will see different embeddings than in training (use --transformer-path)")
        variant = f"STT (embedding {embed_dim} + nav)"
    return env, variant


def run_episode(model, env, cbf_wrapper, use_trust: bool, reset_options: Optional[dict]) -> dict:
    obs, info = env.reset(options=reset_options)
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
        ep_return += float(reward)
        steps += 1
        done = bool(terminated or truncated)

    metrics = info.get("episode_metrics")
    if not (isinstance(metrics, dict) and "end_reason" in metrics):
        metrics = {}
    end_reason = (metrics.get("end_reason") or info.get("terminated_reason")
                  or info.get("truncated_reason") or "unknown")
    return {
        "end_reason": end_reason,
        "route_completion_frac": float(metrics.get("route_completion_frac", info.get("route_completion_frac", 0.0))),
        "collisions": int(metrics.get("collisions", 0)),
        "collisions_pedestrian": int(metrics.get("collisions_pedestrian", 0)),
        "return": float(metrics.get("return", ep_return)),
        "length_steps": int(metrics.get("length_steps", steps)),
        "distance_m": float(metrics.get("distance_m", 0.0)),
        "mean_speed_mps": float(metrics.get("mean_speed_mps", 0.0)),
        "route_length_m": float(metrics.get("route_length_m", 0.0)),
        "red_light_violations": int(metrics.get("red_light_violations", 0)),
        "cbf_intervention_frac": float(metrics.get("cbf_intervention_frac", 0.0)),
        "trust_mean": float(np.mean(trust_values)) if trust_values else None,
    }


def format_line(i: int, n: int, r: dict) -> str:
    trust = f" | trust {r['trust_mean']:.2f}" if r["trust_mean"] is not None else ""
    ped = f" (pedestrian {r['collisions_pedestrian']})" if r["collisions_pedestrian"] else ""
    return (f"ep {i}/{n} | END {r['end_reason']:<20s} | route {r['route_completion_frac']:4.0%} | "
            f"collisions {r['collisions']}{ped} | R {r['return']:+8.1f} | {r['length_steps']} steps "
            f"{r['distance_m'] / 1000:.2f} km | {r['mean_speed_mps']:.1f} m/s | cbf {r['cbf_intervention_frac']:.0%}{trust}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run a trained SAC model in CARLA (deterministic, visible)",
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--model-path", type=str, default=None, help="SAC .zip saved by an experiment")
    parser.add_argument("--model", type=str, choices=list(LEGACY_MODELS), default="5cnn",
                        help="Legacy name, used when --model-path is not given: loads "
                             "<checkpoint-dir>/<model>.zip (default 5cnn)")
    parser.add_argument("--checkpoint-dir", type=str, default="demos", help="Folder for --model / --list-models")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port")
    parser.add_argument("--num-npc", type=int, default=50, help="NPC vehicles (final stage)")
    parser.add_argument("--num-pedestrians", type=int, default=15, help="Pedestrians (final stage)")
    parser.add_argument("--stage", type=int, default=FINAL_STAGE, choices=range(FINAL_STAGE + 1),
                        help=f"Curriculum stage of the scenario (default {FINAL_STAGE}: full traffic)")
    parser.add_argument("--fixed-spawns", action="store_true",
                        help="Episode i uses spawn_index=i, route_seed=i (repeatable, like evaluation)")
    parser.add_argument("--occlusion", type=str, default="none",
                        choices=["none", "mixed", "black_bars", "white_noise"],
                        help="Camera occlusions as in training (default none)")
    parser.add_argument("--transformer-path", type=str, default=None, help="STT weights (auto-detected if omitted)")
    parser.add_argument("--no-cbf", action="store_true", help="Run without the CBF safety layer")
    parser.add_argument("--no-trust", action="store_true", help="CBF with a fixed trust of 1.0")
    parser.add_argument("--no-render", action="store_true", help="No camera window / spectator follow")
    parser.add_argument("--cpu", action="store_true", help="Run the policy on the CPU")
    parser.add_argument("--list-models", action="store_true", help="List legacy models in --checkpoint-dir")
    parser.add_argument("--quiet", action="store_true", help="Only print the per-episode lines and the summary")
    parser.add_argument("--results", type=str, default=None,
                        help="JSON output (default demo/results_<model>_<episodes>eps.json)")
    args = parser.parse_args(argv)

    checkpoint_dir = _resolve(args.checkpoint_dir)
    if args.list_models:
        list_models(checkpoint_dir)
        return 0
    if args.model_path:
        model_path = _resolve(args.model_path)
    else:
        model_path = os.path.join(checkpoint_dir, f"{args.model}.zip")
    if not os.path.isfile(model_path):
        print(f"[ERROR] model not found: {model_path} (pass --model-path <zip>)")
        return 1

    device = "cpu" if args.cpu else "auto"
    print(f"[SETUP] loading {model_path}")
    model = SAC.load(model_path, device=device)
    env, variant = build_env(model, args, model_path)
    cbf_wrapper = find_cbf_wrapper(env)
    if not args.quiet:
        print("=" * 100)
        print(f"model {Path(model_path).name} | {variant} | critics {model.critic.n_critics} | "
              f"stage {args.stage} {STAGES[args.stage]['name']} | CBF {'off' if args.no_cbf else 'on'}"
              f"{'' if args.no_cbf else (' (trust fixed 1.0)' if args.no_trust else ' (live trust)')} | "
              f"occlusion {args.occlusion}")
        print("=" * 100)

    results = []
    try:
        for i in range(args.episodes):
            options = {"spawn_index": i, "route_seed": i} if args.fixed_spawns else None
            t0 = time.time()
            r = run_episode(model, env, cbf_wrapper, use_trust=not args.no_trust, reset_options=options)
            r["episode"] = i + 1
            r["wall_s"] = round(time.time() - t0, 1)
            results.append(r)
            print(format_line(i + 1, args.episodes, r), flush=True)
    except KeyboardInterrupt:
        print("\n[INTERRUPT] stopped by user")
    finally:
        env.close()

    if results:
        n = len(results)
        success = sum(r["end_reason"] == "endpoint" for r in results)
        print("-" * 100)
        print(f"{n} episodes | success {success}/{n} | route {np.mean([r['route_completion_frac'] for r in results]):.0%} "
              f"(mean) | collisions {sum(r['collisions'] for r in results)} | "
              f"return {np.mean([r['return'] for r in results]):+.1f} (mean)")
        out = args.results or str(REPO_ROOT / "demo" / f"results_{Path(model_path).stem}_{n}eps.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"model": model_path, "variant": variant, "stage": args.stage, "cbf": not args.no_cbf,
                       "trust": not (args.no_cbf or args.no_trust), "episodes": results}, f, indent=2)
        print(f"[OK] results saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
