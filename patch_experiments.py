"""
patch_experiments.py - small, safe edits to the four CARLA SAC experiment scripts.

Put this file in the repo root (the folder holding CARLA-RL-Agents and
experiments), then run:

    python patch_experiments.py

or point it at the experiments folder:

    python patch_experiments.py D:\\safety-aware-drl-carla\\safety-aware-drl-carla-update-v1\\experiments

Safe to run more than once. Each file it changes is backed up as *.bak first.

What it changes and why
-----------------------
1. save_freq -> 25000 (all four)
   The CNN runs keep a 50,000-step replay buffer of 84x84 RGB frames, so each
   saved buffer is ~2.1 GB. At save_freq=10000 a 1,000,000-step run writes
   ~212 GB of checkpoints and stalls training on every write. 25000 cuts that
   to ~85 GB, and train_all.py prunes old ones so only the newest few survive.
   2qCNNsac_experiment.py was at save_freq=100000, which also meant a crash
   before 100k steps left NO checkpoint at all - that is the main reason the
   agent kept starting over from random weights.

2. Replay-buffer filename fix (2qCNNsac_experiment.py only)
   It looked for "<ckpt>_replay_buffer.pkl", but CheckpointCallback actually
   writes "<prefix>_replay_buffer_<N>_steps.pkl", so the buffer was never
   found on resume. The other three files already handle both names.

3. progress_bar -> False (all four)
   The tqdm bar writes thousands of carriage returns into the log file when
   output is captured. The ETA from train_all.py replaces it.

4. --timesteps default -> 1000000 (all four)
   So running a script by hand targets 1M as well.
"""

import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE

FILES = [
    "2qCNNsac_experiment.py",
    "5qCNNsac_experiment.py",
    "2qsac_stt_experiment.py",
    "5qsac_stt_experiment.py",
]

SAVE_FREQ = 25000

RB_OLD = (
    '        rb_path = load_checkpoint.replace(".zip", "_replay_buffer.pkl")\n'
    "        if os.path.isfile(rb_path):\n"
)
RB_NEW = (
    "        import re\n"
    "        rb_candidates = [\n"
    '            load_checkpoint.replace(".zip", "_replay_buffer.pkl"),\n'
    '            re.sub(r"_(\\d+)_steps\\.zip$", r"_replay_buffer_\\1_steps.pkl", load_checkpoint),\n'
    "        ]\n"
    "        rb_path = next((p for p in rb_candidates if os.path.isfile(p)), None)\n"
    "        if rb_path:\n"
)


def locate(name):
    """Find the script, whether ROOT is the repo root or the experiments folder."""
    for candidate in (ROOT / name, ROOT / "experiments" / name):
        if candidate.is_file():
            return candidate
    hits = [p for p in ROOT.rglob(name) if "venv" not in p.parts]
    return hits[0] if hits else None


def patch_save_freq(text):
    new, n = re.subn(r"save_freq\s*=\s*\d+", f"save_freq={SAVE_FREQ}", text)
    return new, (f"save_freq -> {SAVE_FREQ} ({n} place(s))" if n else None)


def patch_progress_bar(text):
    new, n = re.subn(r"progress_bar\s*=\s*True", "progress_bar=False", text)
    return new, (f"progress_bar -> False ({n})" if n else None)


def patch_timesteps_default(text):
    new, n = re.subn(
        r'(add_argument\(\s*"--timesteps",\s*type=int,\s*default=)\d+',
        r"\g<1>1000000",
        text,
    )
    return new, ("--timesteps default -> 1000000" if n else None)


def patch_replay_buffer_name(text):
    if "rb_candidates" in text:
        return text, None
    if RB_OLD not in text:
        return text, None
    return text.replace(RB_OLD, RB_NEW, 1), "replay-buffer filename fix"


PATCHES = [
    patch_save_freq,
    patch_progress_bar,
    patch_timesteps_default,
    patch_replay_buffer_name,
]


def main():
    print(f"Looking under: {ROOT}\n")
    any_missing = False

    for name in FILES:
        path = locate(name)
        if path is None:
            print(f"[NOT FOUND] {name}")
            any_missing = True
            continue

        original = path.read_text(encoding="utf-8")
        text = original
        applied = []
        for fn in PATCHES:
            text, note = fn(text)
            if note:
                applied.append(note)

        if text == original:
            print(f"[ALREADY OK] {path}")
            continue

        shutil.copy2(path, str(path) + ".bak")
        path.write_text(text, encoding="utf-8")
        print(f"[PATCHED]    {path}")
        for note in applied:
            print(f"               - {note}")

    print("\nBackups of every changed file are saved next to it as *.bak")
    if any_missing:
        print("Some files were not found - check the folder you passed on the command line.")
    print("\nNext: run the training with train_all.py (see its header for usage).")


if __name__ == "__main__":
    main()
