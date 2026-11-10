# Windows setup, fixes and training runner — safety-aware-drl-carla (update-v1)

Tested on Windows 11, CARLA 0.9.15 (`D:\WORK\Carla`), Python 3.12, CPU training.

## Files in this folder

| File | What it does |
|---|---|
| `apply_fixes.py` | Applies the three source fixes below to a fresh clone |
| `patch_experiments.py` | Applies the four experiment-script fixes below |
| `train_all.py` | Runs training with auto-resume, crash recovery and an ETA |
| `requirements_working.txt` | The exact package versions that work |

## A. Source fixes (`python apply_fixes.py`)

1. **`CARLA-RL-Agents/configuration.py`** — `ENV_SCENARIOS_FILE` was hardcoded to
   `E:/Sarosh/...`, a path from the original author's machine. Now relative to the file.
2. **`CARLA-RL-Agents/env/environment.py`** — `reset()` did `options['scenario_name']`,
   but Stable-Baselines3 calls `reset()` with `options=None`, raising
   `TypeError: 'NoneType' object is not subscriptable`. Added a `None` check.
3. **`CARLA-RL-Agents/src/map_control.py`** — `load_world('/Game/Carla/Maps/Town01')`
   failed with `Map not found` on 0.9.15, which expects just `Town01`. Now passes the
   bare name.

## B. Experiment-script fixes (`python patch_experiments.py`)

Applies to all four `experiments/*_experiment.py` files.

1. **`save_freq` → 25000.** `2qCNNsac_experiment.py` had `save_freq=100000`, so a crash
   before 100k steps left **no checkpoint at all** — this is why the CNN appeared to
   "re-initialise again and again". It was not re-initialising mid-run; every launch was
   building a fresh agent because there was nothing to resume from.
2. **Replay-buffer filename fix (`2qCNNsac_experiment.py`).** It looked for
   `<ckpt>_replay_buffer.pkl`, but `CheckpointCallback` writes
   `<prefix>_replay_buffer_<N>_steps.pkl`, so the buffer was never restored.
3. **`progress_bar` → `False`.** The tqdm bar floods the log file when output is captured.
4. **`--timesteps` default → 1,000,000.**

### Disk-space warning

The CNN variants keep a 50,000-step replay buffer of 84x84 RGB frames, so **each saved
buffer is 2.12 GB**. At the original `save_freq=10000`, a 1M-step run would write ~212 GB
and nothing deletes it. `train_all.py` keeps only the newest 3 checkpoints
(`--keep-checkpoints`). The STT variants store 512-d embeddings instead, ~205 MB per save.

## C. First-time setup

Run from the repo root (the folder holding `CARLA-RL-Agents` and `experiments`):

    python -m venv venv
    venv\Scripts\activate
    python -m pip install -r requirements_working.txt
    python apply_fixes.py
    python patch_experiments.py

Packages that are needed but missing from the original `requirements.txt` (which lists
`cv2` — not a real package name, the correct one is `opencv-python`):

    matplotlib pandas opencv-python torchvision scikit-learn scipy seaborn osqp

## D. Running

Start CARLA, then from `CARLA-RL-Agents` with the venv active:

    python ..\train_all.py --only 2q_cnn --render --carla-exe "D:\WORK\Carla\CarlaUE4.exe"

Drop `--only 2q_cnn` to run all four in turn. Useful flags:

| Flag | Meaning |
|---|---|
| `--timesteps N` | Target total steps (default 1,000,000) |
| `--carla-exe PATH` | Let the runner start and restart CARLA itself |
| `--parallel 2` | Run two experiments at once, on ports 2000 and 2010 |
| `--port-base 3000` | Use a different CARLA port |
| `--keep-checkpoints N` | Keep the newest N checkpoints (0 = keep all) |

`train_all.py` exists because running a script directly builds a brand-new agent every
launch. It finds the newest checkpoint and passes `--load-checkpoint`, so training always
continues. It also converts the target into *steps remaining* — Stable-Baselines3 **adds**
`total_timesteps` to the restored counter when `reset_num_timesteps=False`, so asking for
1,000,000 while sitting at 300,000 would otherwise run to 1,300,000.

Press Ctrl+C any time; rerun the same command to continue from the last checkpoint.

## E. CARLA troubleshooting

| Symptom | Cause and fix |
|---|---|
| `time-out of 10000ms while waiting for the simulator` | Usually a **zombie CARLA process**. Run `netstat -ano \| findstr :2000`; if two PIDs are LISTENING, a dead instance is stealing connections. `tasklist \| findstr /i carla` — the small one (~190 MB) is the zombie, the live one uses several GB. `taskkill /F /PID <zombie>`. Hard `taskkill` is what creates these, so prefer closing CARLA with its X button. |
| `UnicodeEncodeError: 'charmap' codec can't encode '\u2192'` | The scripts print `→`; piped stdout on Windows falls back to cp1252. `train_all.py` sets `PYTHONIOENCODING=utf-8` for the child process. |
| `CARLA SERVER PATH NOT FOUND` | Set `CARLA_SERVER` to the folder holding `CarlaUE4.exe`. Find it with `powershell "Get-Process *carla* \| Select-Object -Property Path -Unique"`. |
| `Map '/Game/Carla/Maps/Town01' not found` | Fix 3 above. Confirm maps exist with `python -c "import carla; c=carla.Client('localhost',2000); c.set_timeout(20); print(c.get_available_maps())"`. |
| `WinError 10013` during `pip install` | Firewall/antivirus blocking the socket. Retry — it usually succeeds on a second attempt. |

Quick health check:

    python -c "import carla; c=carla.Client('127.0.0.1',2000); c.set_timeout(20); print(c.get_world().get_map().name)"

## F. Known issues and open questions

- **Training runs on CPU.** The log says `Using cpu device`, because the default Windows
  pip wheels for torch are CPU-only. On a machine with an NVIDIA GPU, reinstalling torch
  from the CUDA index (see pytorch.org) is by far the biggest speed-up available.
  At 10 steps/s, 1M steps takes ~28 hours; at 2 steps/s, ~6 days — per experiment.
- **Do not use `SubprocVecEnv` workers without reworking the callbacks.**
  `SafetyMetricsCallback` reads `self.training_env.envs[0].unwrapped` inside a bare
  `except: pass`, and the trust-score callback walks `env.envs` to call `set_trust_score`
  on the CBF wrapper. With subprocess workers those wrappers live in other processes:
  `.envs` disappears, the exception is swallowed, and **safety metrics and CBF trust
  scores silently stop working**. Use `--parallel` to run separate experiments instead.
- **Open question — is the SpatioTemporal Transformer actually trained?** In the STT
  variants the transformer lives in `PipelineObservationWrapper`, i.e. in the environment,
  not in `model.policy`. SAC builds its optimiser from the policy's parameters, so no
  gradient should reach the transformer. If that is right, the STT acts as a frozen random
  feature extractor and the CNN-vs-STT comparison is not measuring what it claims. Worth
  confirming against `models/pipeline.py` and `tests/pipeline_carla_test.py`.
- **STT weights are not restored on resume.** `TransformerCheckpointCallback` saves
  `transformer_checkpoint_*.pth`, but nothing loads it back; `--encoder-path` only restores
  `st_encoder`, not `stacked_transformer`. Unresolved pending the question above.
- **Version note.** `requirements_working.txt` pins `carla==0.9.16` while the server is
  0.9.15. It works, but matching them is safer.
