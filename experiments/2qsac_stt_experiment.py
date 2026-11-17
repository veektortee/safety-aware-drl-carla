"""
2 Q-network SAC with the SpatioTemporal Transformer (STT)

Stack:
    CarlaGymEnv              obs Dict(rgb_data 360x640x3, nav), action [steer, accel] in [-1, 1]^2
      -> CBFSafetyLayerWrapper    discrete-time CBF QP; trust from the critic ensemble tightens it
      -> SttNavObservationWrapper ResNet50 -> STT over the last 8 frames (512-d) + nav vector
                                  -> Box(512 + NAV_DIM); occlusions in 30% of episodes
    SAC 'MlpPolicy' on the concatenated vector, ensemble of 2 critics.

The transformer weights are not part of the SAC .zip: they are saved to
<log-dir>/checkpoints/transformer/ and restored automatically on --load-checkpoint.

Training: the replay buffer is first filled with expert-driver transitions (--warmup-steps), then
SAC learns (learning_starts=1000). The curriculum S0..S3 (commons/curriculum_stages.py) is promoted
automatically unless --no-curriculum. A CARLA server must be running on --port.

Usage:
    python experiments/2qsac_stt_experiment.py --render
    python experiments/2qsac_stt_experiment.py --timesteps 1000000 --log-dir ./logs/2q_stt
    python experiments/2qsac_stt_experiment.py --load-checkpoint ./logs/2q_stt/checkpoints/sac_2q_stt_250000_steps.zip

--timesteps is the TARGET TOTAL: resuming a 250k checkpoint with --timesteps 1000000 trains 750k
more steps. Expert warm-up steps are not counted.
"""

import argparse
import os
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, LogEveryNTimesteps
from stable_baselines3.common.logger import configure

from tests.pipeline_carla_test import CarlaGymEnv, CBFSafetyLayerWrapper, PolicyTrustScoreCallback
from commons.metrics_callbacks import EpisodeMetricsCallback, EvalCallback, write_run_config
from commons.obs_wrappers import (
    SttNavObservationWrapper,
    TransformerCheckpointCallback,
    find_transformer_checkpoint,
)
from commons.curriculum import CurriculumCallback, resolve_start_stage
from commons.curriculum_stages import FINAL_STAGE
from commons.warmup import (
    checkpoint_num_timesteps,
    describe_env_stack,
    prefill_replay_buffer,
    restore_replay_buffer,
    save_checkpoint,
)

EXPERIMENT = "2q_stt"
NAME_PREFIX = "sac_2q_stt"
POLICY = "MlpPolicy"
N_CRITICS = 2
DEFAULT_PORT = 2004
DEFAULT_ENCODER = None     # optional pretrained SpatioTemporalEncoder state_dict
EMBED_DIM = 512
NUM_FRAMES = 8
TIME_LIMIT = 3000          # upper bound; the env derives a shorter per-route limit
LEARNING_STARTS = 1000
SAVE_FREQ = 25000
EVAL_FREQ = 25000
EVAL_EPISODES = 10
SB3_TABLE_EVERY = 1000     # steps between SB3 stdout tables (train/* losses)
OCCLUSION = "mixed"


def _resolve_encoder_path(path):
    """Relative paths are tried from the working directory, then from the repo root."""
    if not path:
        return None
    for candidate in (Path(path), REPO_ROOT / path):
        if candidate.is_file():
            return str(candidate)
    print(f"[ENV] encoder checkpoint {path} not found; the SpatioTemporalEncoder keeps its initial weights")
    return None


def create_stt_env(port: int = DEFAULT_PORT, render: bool = False, num_npc: int = 200,
                   num_pedestrians: int = 200, curriculum_stage: int = 0, encoder_path=None):
    """CarlaGymEnv -> CBFSafetyLayerWrapper -> SttNavObservationWrapper."""
    print(f"[ENV] Connecting to CARLA on port {port} (curriculum stage {curriculum_stage})...")
    env = CarlaGymEnv(
        host="localhost",
        port=port,
        timeout=10.0,
        time_limit=TIME_LIMIT,
        render_mode="human" if render else None,   # spectator follows the ego car
        num_npc_vehicles=num_npc,
        num_pedestrians=num_pedestrians,
        show_sensor_data=render,                    # camera window
        curriculum_stage=curriculum_stage,
    )
    env = CBFSafetyLayerWrapper(env)
    env = SttNavObservationWrapper(env, embed_dim=EMBED_DIM, num_frames=NUM_FRAMES,
                                   encoder_path=_resolve_encoder_path(encoder_path), occlusion=OCCLUSION)
    print("[ENV] CarlaGymEnv -> CBFSafetyLayerWrapper -> SttNavObservationWrapper ready")
    return env


def train(args) -> int:
    log_dir = args.log_dir
    tb_dir = os.path.join(log_dir, "tensorboard")
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    transformer_dir = os.path.join(ckpt_dir, "transformer")
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    resume = bool(args.load_checkpoint)
    if resume:
        if not os.path.isfile(args.load_checkpoint):
            raise FileNotFoundError(f"--load-checkpoint not found: {args.load_checkpoint}")
        done = checkpoint_num_timesteps(args.load_checkpoint)
        if done is not None and done >= args.timesteps:
            print(f"[RESUME] checkpoint is at {done:,} steps; target {args.timesteps:,} already reached. Nothing to do.")
            return 0

    start_stage = FINAL_STAGE if args.no_curriculum else args.start_stage
    env_stage = resolve_start_stage(log_dir, start_stage, resume=resume)

    print("=" * 80)
    print(f"SAC {EXPERIMENT}: {POLICY}, {N_CRITICS} critics, STT embedding {EMBED_DIM} + nav")
    print(f"target {args.timesteps:,} steps | lr {args.lr} | batch {args.batch_size} | buffer {args.buffer_size:,}")
    print(f"warm-up {args.warmup_steps:,} expert steps | curriculum {'off (final stage)' if args.no_curriculum else 'on'}"
          f" | stage {env_stage} | NPC {args.num_npc} | pedestrians {args.num_pedestrians}")
    print(f"render {args.render} | port {args.port} | log dir {log_dir}")
    if resume:
        print(f"resume from {args.load_checkpoint}")
    print("=" * 80)

    env = create_stt_env(port=args.port, render=args.render, num_npc=args.num_npc,
                         num_pedestrians=args.num_pedestrians, curriculum_stage=env_stage,
                         encoder_path=args.encoder_path)
    try:
        policy_kwargs = {
            "net_arch": {"pi": [256, 256], "qf": [256, 256]},
            "n_critics": N_CRITICS,
            "normalize_images": True,
        }
        buffer_restored = False
        transformer_restored = None
        if resume:
            # The policy was trained on embeddings of THIS run's transformer: restore it before any
            # observation is computed.
            transformer_restored = (find_transformer_checkpoint(args.load_checkpoint)
                                    or find_transformer_checkpoint(args.load_checkpoint, transformer_dir))
            if transformer_restored:
                env.load_transformer(transformer_restored)
                print(f"[RESUME] transformer weights restored from {transformer_restored}")
            else:
                print("[RESUME] WARNING: no transformer_checkpoint_*.pth found for this checkpoint; the "
                      "STT starts from new weights, so the policy sees different embeddings than in training")
            try:
                model = SAC.load(args.load_checkpoint, env=env, tensorboard_log=tb_dir)
            except ValueError as exc:
                print(f"[ERROR] cannot resume from {args.load_checkpoint}: {exc}\n"
                      "        Checkpoints from before the v3 env (3-D action, old observation) are "
                      "incompatible; start a fresh run.")
                return 1
            buffer_restored = restore_replay_buffer(model, args.load_checkpoint)
            print(f"[RESUME] model restored at {model.num_timesteps:,} steps")
        else:
            model = SAC(
                POLICY,
                env,
                learning_rate=args.lr,
                buffer_size=args.buffer_size,
                learning_starts=LEARNING_STARTS,
                batch_size=args.batch_size,
                tau=0.005,
                gamma=0.99,
                train_freq=1,
                gradient_steps=1,
                ent_coef="auto",
                policy_kwargs=policy_kwargs,
                tensorboard_log=tb_dir,
                verbose=1,
            )

        remaining = max(0, args.timesteps - model.num_timesteps)
        if remaining == 0:
            print(f"[RESUME] target {args.timesteps:,} steps already reached. Nothing to do.")
            return 0

        model.set_logger(configure(tb_dir, ["stdout", "tensorboard"]))

        warmup_steps = 0 if buffer_restored else max(0, args.warmup_steps)
        write_run_config(log_dir, model, extra={
            "experiment": EXPERIMENT,
            "policy": POLICY,
            "n_critics": N_CRITICS,
            "net_arch": policy_kwargs["net_arch"],
            "observation": f"STT embedding ({EMBED_DIM}, {NUM_FRAMES} frames) + nav, concatenated",
            "encoder_path": args.encoder_path,
            "transformer_restored_from": transformer_restored,
            "occlusion": OCCLUSION,
            "timesteps_target": args.timesteps,
            "timesteps_at_start": int(model.num_timesteps),
            "warmup_steps": warmup_steps,
            "learning_starts": LEARNING_STARTS,
            "time_limit": TIME_LIMIT,
            "curriculum": {"enabled": not args.no_curriculum, "start_stage": start_stage, "env_stage": env_stage},
            "num_npc": args.num_npc,
            "num_pedestrians": args.num_pedestrians,
            "port": args.port,
            "render": args.render,
            "resumed_from": args.load_checkpoint,
            "replay_buffer_restored": buffer_restored,
            "stack": describe_env_stack(model),
        })

        if warmup_steps > 0:
            try:
                prefill_replay_buffer(model, warmup_steps, log_every=1000)
            except KeyboardInterrupt:
                print("\n[INTERRUPT] stopped during the expert warm-up; nothing saved")
                return 1

        callbacks = [
            CheckpointCallback(save_freq=SAVE_FREQ, save_path=ckpt_dir, name_prefix=NAME_PREFIX,
                               save_replay_buffer=True),
            TransformerCheckpointCallback(save_freq=SAVE_FREQ, save_path=transformer_dir, verbose=1),
            EvalCallback(eval_freq=EVAL_FREQ, n_episodes=EVAL_EPISODES, log_dir=log_dir, verbose=1),
            PolicyTrustScoreCallback(update_freq=100),
            EpisodeMetricsCallback(log_dir=log_dir, verbose=1),
            CurriculumCallback(log_dir, start_stage=start_stage, verbose=1),
            LogEveryNTimesteps(n_steps=SB3_TABLE_EVERY),
        ]

        print(f"[TRAIN] {remaining:,} steps to go (from {model.num_timesteps:,} to {args.timesteps:,})")
        try:
            # reset_num_timesteps=False: SB3 adds `remaining` to the current counter, and the episode
            # left running by the warm-up continues (no extra reset).
            model.learn(total_timesteps=remaining, callback=callbacks, log_interval=None,
                        progress_bar=False, reset_num_timesteps=False)
        except KeyboardInterrupt:
            print(f"\n[INTERRUPT] training interrupted at {model.num_timesteps:,} steps")
            save_checkpoint(model, ckpt_dir, NAME_PREFIX)
            env.save_transformer(os.path.join(transformer_dir, f"transformer_checkpoint_{model.num_timesteps}.pth"),
                                 timesteps=model.num_timesteps)
            return 1
        except Exception as exc:  # noqa: BLE001 - keep the run resumable, then report
            print(f"\n[ERROR] {exc}")
            traceback.print_exc()
            try:
                save_checkpoint(model, ckpt_dir, NAME_PREFIX)
                env.save_transformer(os.path.join(transformer_dir, f"transformer_checkpoint_{model.num_timesteps}.pth"),
                                     timesteps=model.num_timesteps)
            except Exception as save_exc:  # noqa: BLE001
                print(f"[ERROR] could not save a checkpoint after the failure: {save_exc}")
            return 1

        final_path = os.path.join(log_dir, f"{NAME_PREFIX}_final.zip")
        model.save(final_path)
        env.save_transformer(os.path.join(transformer_dir, f"transformer_checkpoint_{model.num_timesteps}.pth"),
                             timesteps=model.num_timesteps)
        print(f"[OK] final model saved: {final_path}")
        print(f"View curves: tensorboard --logdir {tb_dir}")
        return 0
    finally:
        env.close()
        print("[OK] environment closed")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="2 Q-network SAC, STT embedding + nav (MlpPolicy)")
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                        help="TARGET TOTAL policy steps (on resume: trains target - checkpoint steps)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="CARLA server port")
    parser.add_argument("--log-dir", type=str, default="./logs/2q_stt", help="Run directory")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (fresh runs; resume keeps the checkpoint's)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--buffer-size", type=int, default=50000, help="Replay buffer size (fresh runs)")
    parser.add_argument("--render", action="store_true", help="Camera window + spectator follows the car")
    parser.add_argument("--num-npc", type=int, default=200, help="NPC vehicles in the full-traffic stage")
    parser.add_argument("--num-pedestrians", type=int, default=200, help="Pedestrians in the full-traffic stage")
    parser.add_argument("--warmup-steps", type=int, default=20000,
                        help="Expert transitions put in the replay buffer before learning "
                             "(skipped when a replay buffer is restored on resume)")
    parser.add_argument("--start-stage", type=int, default=0, choices=range(FINAL_STAGE + 1),
                        help="Curriculum stage to start from (a higher saved stage wins on resume)")
    parser.add_argument("--no-curriculum", action="store_true",
                        help=f"Train directly on the final stage ({FINAL_STAGE})")
    parser.add_argument("--encoder-path", type=str, default=DEFAULT_ENCODER,
                        help="Optional pretrained SpatioTemporalEncoder state_dict (fresh runs; on resume "
                             "the run's saved transformer weights take precedence)")
    parser.add_argument("--load-checkpoint", type=str, default=None,
                        help="SAC .zip to resume from (replay buffer and transformer weights restored "
                             "if saved beside it)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(train(parse_args()))
