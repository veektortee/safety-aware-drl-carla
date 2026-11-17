"""
patch_experiments.py - no longer needed.

This script used to patch the four experiment scripts (save_freq 25000, replay-buffer filename fix
on resume, progress_bar=False, --timesteps default 1,000,000). All of those fixes are now part of
experiments/*_experiment.py, so there is nothing left to patch. It is kept only so that old
instructions that say "python patch_experiments.py" still run cleanly.
"""


def main():
    print("patch_experiments.py: nothing to patch - the experiment scripts already include these fixes:")
    print("  - CheckpointCallback(save_freq=25000, save_replay_buffer=True)")
    print("  - replay buffer found on resume under both '<ckpt>_replay_buffer.pkl' and")
    print("    '<prefix>_replay_buffer_<N>_steps.pkl'")
    print("  - progress_bar=False (SB3 table at most every 1,000 steps)")
    print("  - --timesteps default 1,000,000 (target total)")
    print("Start training with train_all.py or experiments/<variant>_experiment.py.")


if __name__ == "__main__":
    main()
