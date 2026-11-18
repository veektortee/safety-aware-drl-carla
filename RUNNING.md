# Running the CARLA SAC experiments

How to set the project up on Windows, start CARLA, train each experiment (NatureCNN or ResNet-50, STT, resume, ports, parallel runs), use the self-removing training queue, watch TensorBoard and run the demo.

All commands are PowerShell. Paths that are specific to a machine are set once as variables:

```powershell
$REPO  = 'D:\safety-aware-drl-carla\wt-maps'        # your clone / worktree of this repo
$CARLA = 'D:\WORK\Carla\CarlaUE4.exe'               # CARLA 0.9.15 server
$PY    = "$REPO\.venv\Scripts\python.exe"           # on the original PC: D:\safety-aware-drl-carla\safety-aware-drl-carla-update-v1\venv\Scripts\python.exe
$env:PYTHONDONTWRITEBYTECODE = '1'                  # the repo tracks some .pyc files; keep them unchanged
```

---

## 1. One-time setup

**Requirements:** Windows 10/11, an NVIDIA GPU with 8 GB or more, Python 3.12, Git, and CARLA 0.9.15. The packaged Windows build gives you `CarlaUE4.exe`. The extra towns Town06/07 need CARLA's "AdditionalMaps" package; `D:\WORK\Carla\HDMaps` only holds point clouds, not drivable maps.

```powershell
git clone https://github.com/veektortee/safety-aware-drl-carla.git $REPO     # private repo: sign in when asked
cd $REPO
git checkout feat/maps-curriculum                  # or main once the PRs are merged

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements_train.txt              # carla 0.9.16 client, gymnasium, osqp, opencv, tensorboard, ...
```

- **Never install `stable-baselines3` from pip.** `stable_baselines3/` in the repo is a modified copy (critic ensemble and trust score).
- **ImageNet ResNet-50 weights** for STT and the 5q ResNet option go to `pretrained/feature_extractor/pretrained_resnet50.pth`, which git ignores. They download automatically the first time they are needed. To fetch them up front:
  ```powershell
  & $PY -c "import commons.stt as s; s._resnet50_state_dict()"
  ```
- **Check the install** with the offline unit tests. No CARLA is needed; each file is its own runner:
  ```powershell
  Get-ChildItem tests\unit\test_*.py | ForEach-Object { & $PY $_.FullName }
  ```

---

## 2. Starting a CARLA server

Every training run or demo needs its own CARLA server. A server uses three ports:

| What | Port |
|---|---|
| RPC (`-carla-rpc-port`) | the port you choose |
| Streaming | that port + 1 |
| Traffic Manager (set by the code) | that port + 6000 |

Keep parallel servers 10 apart: 2000, 2010, 2020.

```powershell
# visible window (tests, demos, recording)
& $CARLA -carla-rpc-port=2000 -quality-level=Low -windowed -ResX=1280 -ResY=720

# headless (long training; the experiment's camera window still shows)
& $CARLA -carla-rpc-port=2000 -quality-level=Low -RenderOffScreen

# is it up? (True once it accepts connections, ~20-60 s after start)
(New-Object Net.Sockets.TcpClient).ConnectAsync('127.0.0.1', 2000).Wait(2000)
```

Instead of starting CARLA yourself, you can let the runner `train_all.py` start and restart it with `--carla-exe $CARLA` (add `--headless` for no server window).

---

## 3. Training

Every experiment trains SAC with the CBF safety layer and the critic-ensemble trust score. They differ in the camera branch and the number of critics:

| Name | Script | Camera branch | Critics |
|---|---|---|---|
| `2q_cnn` | `experiments/2qCNNsac_experiment.py` | SB3 NatureCNN on 84×84 | 2 |
| `5q_cnn` | `experiments/5qCNNsac_experiment.py` | NatureCNN (default), or ResNet-50 with `--cnn resnet50` | 5 |
| `2q_stt` | `experiments/2qsac_stt_experiment.py` | cached ResNet-50 codes → trainable SpatioTemporal Transformer | 2 |
| `5q_stt` | `experiments/5qsac_stt_experiment.py` | same | 5 |

All of them also get the 26-dim navigation vector.

### 3.1 With the runner (recommended)

`train_all.py` resumes from the newest checkpoint, restarts CARLA and the script after a crash, and stops at the target. **Run it from the `CARLA-RL-Agents` folder**: it finds `../experiments` from there and writes `logs/<name>` inside it.

```powershell
cd "$REPO\CARLA-RL-Agents"

# 2q CNN, 200k steps, runner starts a headless CARLA on port 2000
& $PY ..\train_all.py --only 2q_cnn --timesteps 200000 --carla-exe $CARLA --headless

# 5q CNN with SB3's NatureCNN (default)
& $PY ..\train_all.py --only 5q_cnn --timesteps 200000 --carla-exe $CARLA --headless

# 5q CNN with the complete ResNet-50 instead of NatureCNN (same 256-d output, same decision layers)
& $PY ..\train_all.py --only 5q_cnn_resnet_layer4 --timesteps 200000 --carla-exe $CARLA --headless   # trains layer4 + fc (default)
& $PY ..\train_all.py --only 5q_cnn_resnet_head   --timesteps 200000 --carla-exe $CARLA --headless   # trains only the fc head (fastest)
& $PY ..\train_all.py --only 5q_cnn_resnet_all    --timesteps 200000 --carla-exe $CARLA --headless   # trains everything (slowest)

# STT experiments (a fresh run first fits the frame projection on ~1000 expert frames, then the 20k expert warm-up)
& $PY ..\train_all.py --only 2q_stt --timesteps 200000 --carla-exe $CARLA --headless
& $PY ..\train_all.py --only 5q_stt --timesteps 200000 --carla-exe $CARLA --headless

# a different port (e.g. another run already uses 2000)
& $PY ..\train_all.py --only 2q_cnn --timesteps 200000 --carla-exe $CARLA --headless --port-base 2010

# two runs in parallel (ports 2000 and 2010). On 8 GB of VRAM two runs fit; three do not.
& $PY ..\train_all.py --only 2q_cnn 5q_cnn --parallel 2 --timesteps 200000 --carla-exe $CARLA --headless

# CARLA already running (you started it): leave out --carla-exe; the runner waits for the port
& $PY ..\train_all.py --only 2q_cnn --timesteps 200000 --port-base 2000

# logs somewhere else (e.g. the folder TensorBoard watches)
& $PY ..\train_all.py --only 2q_cnn --timesteps 200000 --carla-exe $CARLA --headless --log-root D:\runs\logs
```

Runner options:

| Option | Meaning |
|---|---|
| `--only <names>` | Which experiments. Without it, all four standard ones run. The `5q_cnn_resnet_*` entries run only when named. |
| `--timesteps N` | Target **total** steps. Resuming a 125k checkpoint with 200000 trains 75k more. |
| `--carla-exe`, `--headless` | Runner starts and restarts CARLA; `--headless` gives no server window. |
| `--port-base`, `--port-step` | First port, and the spacing between parallel runs (default 10). |
| `--parallel K` | Run K experiments at the same time. |
| `--keep-checkpoints N` | Checkpoints kept when a run attempt ends (the scripts keep only the latest anyway). |
| `--num-npc`, `--num-pedestrians` | Maximum traffic. The curriculum goes up to 300 / 300. |
| `--no-render` | No camera window (it is on by default). |
| `--log-root DIR` | Put `logs/<name>` under DIR. |
| `--max-retries`, `--restart-delay` | Give up after N attempts with no progress (default 5); seconds between attempts (default 20). |

### 3.2 Running a script directly

You need your own CARLA server on `--port`. Run from the repo root:

```powershell
cd $REPO
& $PY experiments\2qCNNsac_experiment.py --port 2000 --timesteps 200000 --log-dir .\CARLA-RL-Agents\logs\2q_cnn --render
& $PY experiments\5qCNNsac_experiment.py --port 2000 --timesteps 200000 --render                          # NatureCNN
& $PY experiments\5qCNNsac_experiment.py --port 2000 --timesteps 200000 --render --cnn resnet50 --resnet-train layer4
& $PY experiments\2qsac_stt_experiment.py --port 2000 --timesteps 200000 --render
```

Script options (all four scripts unless noted):

| Option | Default | Meaning |
|---|---|---|
| `--port` | 2000 or 2004 | CARLA RPC port; the default differs per script, so pass it explicitly |
| `--timesteps` | 1,000,000 | target total steps |
| `--log-dir` | `./logs/<name>` | run folder |
| `--render` | off (the runner turns it on) | camera window, and the spectator follows the car |
| `--load-checkpoint ZIP` | | resume, see §3.3 |
| `--warmup-steps` | 20000 | expert transitions put in the replay buffer before learning; skipped when a buffer is restored |
| `--start-stage K` / `--no-curriculum` | 0 | start the curriculum at stage K, or train only the final stage |
| `--num-npc`, `--num-pedestrians` | 300 / 300 | traffic maxima |
| `--lr`, `--batch-size`, `--buffer-size` | 3e-4, 64, 50000 | apply to fresh runs only; a resume keeps the checkpoint's values |
| `--cnn {nature,resnet50}` | nature | **5q CNN only:** image branch |
| `--resnet-train {head,layer4,all}` | layer4 | **5q CNN only:** how much of ResNet-50 trains |

### 3.3 Resume, restart from scratch

- **Resume.** The runner resumes automatically. By hand:
  ```powershell
  & $PY experiments\5qCNNsac_experiment.py --port 2000 --timesteps 200000 --render `
      --log-dir .\CARLA-RL-Agents\logs\5q_cnn `
      --load-checkpoint .\CARLA-RL-Agents\logs\5q_cnn\checkpoints\sac_5q_cnn_100000_steps.zip
  ```
  - The weights, optimizers and step counter come from the `.zip`.
  - The replay buffer is loaded from `*_replay_buffer_<N>_steps.pkl` beside it.
  - The curriculum state comes from `*_curriculum_<N>_steps.json`.
  - STT runs also restore the frame projection, which is stored inside the `.zip`.
- **What happens to the logs.** A resume **erases everything an earlier process logged after the checkpoint**: TensorBoard points, `episodes.jsonl` and `eval.jsonl`. TensorBoard then continues the same curves from the checkpoint step. The curriculum resumes at least at the stage its step schedule allows (§4).
- **Restart from scratch.** Stop the run and delete its folder, e.g. `CARLA-RL-Agents\logs\2q_stt`.
- **Incompatible checkpoints.** Each image branch can only resume its own checkpoints: a NatureCNN checkpoint cannot be resumed with `--cnn resnet50`, and the scripts refuse with a message. STT checkpoints from before the trainable STT can't be resumed at all.

### 3.4 What a run writes

`logs/<name>/` contains:

- `checkpoints/`: only the **latest** checkpoint, every 25k steps. That is `sac_<name>_<N>_steps.zip`, its replay buffer `.pkl` (~2 GB) and a curriculum snapshot `.json`.
- `tensorboard/`: SB3's own curves, plus `tensorboard/metrics/` for the readable `metrics/*` and `eval/*` curves.
- `episodes.jsonl`: one line per training episode, including the town.
- `eval.jsonl`: one line per eval round, every 25k steps: 10 fixed episodes spread over all towns.
- `curriculum_state.json`, `run_config.json`, `runner.log`.
- `sac_<name>_final.zip` when the target is reached.

`python make_report.py` builds an HTML report from these files; the metric definitions are in `METRICS.md`.

---

## 4. Maps and curriculum

**Maps.** Each run rotates through every installed town: Town01, 02, 03, 04, 05 and 10HD (the `_Opt` builds). It switches to the next town every 10 training episodes; a switch takes about 5–15 s.

**Curriculum.** There are **6 stages**:

| Stage | Traffic (cars / pedestrians) | Routes | Crash penalty C | Left at the latest at |
|---|---|---|---|---|
| S0 empty short | 0 / 0 | 80–150 m | 50 | 15,000 steps |
| S1 empty full | 0 / 0 | 200–400 m | 100 | 35,000 |
| S2 light | 15 / 15 | 200–400 m | 150 | 65,000 |
| S3 medium | 60 / 60 | 200–400 m | 200 | 100,000 |
| S4 heavy | 150 / 150 | 200–400 m | 250 | 140,000 |
| S5 dense | 300 / 300 | 200–400 m | 300 | final (last 60k of 200k) |

A stage is left earlier if 80% of its last 50 episodes reach the route end (with at least 50 episodes in the stage). Car counts are also capped by each town's spawn points: Town02 holds about 90 cars, Town10HD about 140, Town04 the full 300.

---

## 5. The training queue (scheduled task)

`D:\safety-aware-drl-carla\training_queue\run_queue.ps1` is registered as the Windows scheduled task **CARLA-Training-Queue**. It works like this:

1. It runs the experiments listed in `training_queue\queue_order.txt` **one at a time**, each to 200,000 steps.
2. It starts the next experiment only after the previous runner has exited and no CARLA or experiment process from it is left.
3. It re-reads `queue_order.txt` before every experiment, so you can reorder or add experiments (e.g. `5q_cnn_resnet_layer4`) while it runs.
4. After a reboot it starts again at logon and continues from the checkpoints. It keeps the PC awake while running.
5. When every listed experiment has a 200k checkpoint, it unregisters itself and deletes its folder.

Its log is kept after that: `...\CARLA-RL-Agents\logs\training_queue.log`.

```powershell
Get-ScheduledTask -TaskName CARLA-Training-Queue | Select-Object TaskName, State      # status
Get-Content D:\safety-aware-drl-carla\safety-aware-drl-carla-main\CARLA-RL-Agents\logs\training_queue.log -Tail 20
notepad D:\safety-aware-drl-carla\training_queue\queue_order.txt                     # change the order
Start-ScheduledTask -TaskName CARLA-Training-Queue                                    # start / restart
Unregister-ScheduledTask -TaskName CARLA-Training-Queue -Confirm:$false               # remove without finishing
```

To stop the queue mid-run, close its console window, which also ends that experiment. You lose the steps since the last checkpoint (at most 25k) and the next start resumes from that checkpoint. Don't start other training on port 2000 while the queue runs.

---

## 6. TensorBoard (live)

```powershell
& $PY -m tensorboard.main --logdir D:\safety-aware-drl-carla\safety-aware-drl-carla-main\CARLA-RL-Agents\logs --port 6006 --host 127.0.0.1 --reload_interval 15
# then open http://127.0.0.1:6006
```

Useful tags:

- `eval/success_frac`, `eval/crash_frac`, `eval/route_completion_frac`
- `metrics/*`: the rolling-100 training episodes
- `curriculum/stage`
- `train/critic_loss`, `train/trust_score`
- `rollout/ep_rew_mean`

---

## 7. Demo (watch a trained model)

1. Start a **visible** CARLA server (§2). It shows the top-down overview.
2. Run the demo. It opens the car's camera window (driver view) and prints one line per episode:

```powershell
cd $REPO
$CK = 'D:\safety-aware-drl-carla\safety-aware-drl-carla-main\CARLA-RL-Agents\logs\5q_cnn\checkpoints\sac_5q_cnn_100000_steps.zip'

& $PY demo\run_model_inference.py --model-path $CK --port 2000 --episodes 6 --fixed-spawns --cpu
```

**What you see:**

- **Towns:** each episode runs in the next town. Each one runs to its natural end (route finished, crash, or the env's time limit) before the next starts.
- **Checkpoint rings:** a ring every 30 m along the route. In the server window they show as ovals on the road; in the camera window as upright rings with a HUD (`CHECKPOINTS n/N`, speed, route %).
- **Rings are display-only:** the policy's camera input is unchanged.

**Traffic:** by default the demo uses the densest stage (S5: up to 300 cars and 300 pedestrians, capped per town). A model that has not trained that far will crash more there.

| Option | Meaning |
|---|---|
| `--stage K` | Traffic and route stage 0–5 (default 5 = dense). Use `--stage 1` for empty roads. |
| `--num-npc`, `--num-pedestrians` | Traffic maxima (default 300 / 300). |
| `--episodes N` | Number of episodes. 6 = one per town. |
| `--fixed-spawns` | Repeatable spawns and routes (like evaluation). |
| `--maps current` | Stay on the town the server has loaded. |
| `--no-rings` | Hide the rings and HUD. |
| `--no-cbf`, `--no-trust` | No safety layer, or a fixed trust of 1.0. |
| `--occlusion {none,mixed,black_bars,white_noise}` | Camera occlusions as in training (default none). |
| `--cpu` | Policy on the CPU (saves GPU memory while training runs). |
| `--results FILE` | Where to write the per-episode JSON. |

STT and ResNet models work the same way: pass their `.zip`. The STT projection and all trained weights are inside it, and the demo detects the model type itself.

---

## 8. Troubleshooting

- **`RuntimeError: CARLA module not found` / connection timeouts:** the venv is not active, or no server is listening on `--port`.
- **"nothing is listening on port …":** the runner waits for a server. Start one (§2), or pass `--carla-exe`.
- **Out of GPU memory:** run one experiment at a time, use `--headless` servers, and `--cpu` for the demo.
- **A run "gives up":** it had 5 attempts with no new checkpoint. See `logs/<name>/runner.log`, fix the cause and start it again; it resumes.
- **Map switch is slow the first time:** the client downloads each town's navigation file once.
- **`git status` shows changed `.pyc` files:** set `PYTHONDONTWRITEBYTECODE=1` before running Python.
