"""
train_all.py - supervisor for the four CARLA SAC experiments.

Why this exists
---------------
Launch an experiment script on its own and it builds a brand-new SAC agent
every time, so the CNN starts from random weights on every launch - the
"re-initialising again and again" problem. This runner always looks for the
newest checkpoint first and passes --load-checkpoint, so each launch CONTINUES
the same agent instead of starting over.

It also:
  * Converts your target into "steps remaining". Stable-Baselines3 ADDS the
    --timesteps value to the restored counter when resuming, so asking for
    1,000,000 while sitting at 300,000 would actually run to 1,300,000. The
    runner asks for 700,000 instead, so you land exactly on 1,000,000.
  * Restarts automatically after a crash, continuing from the last checkpoint.
  * Prints throughput (steps/sec) and an ETA to the target.
  * Prunes old checkpoints - the CNN replay buffers are ~2.1 GB each.
  * Can run several experiments at once, each against its own CARLA server.

Usage - run from the CARLA-RL-Agents folder with the venv active:

    python ..\\train_all.py --only 2q_cnn --render
    python ..\\train_all.py                       # all four, one after another
    python ..\\train_all.py --parallel 2          # two at a time, ports 2000/2010

A CARLA server must already be listening on each port in use, unless you pass
--carla-exe, in which case the runner starts and restarts CARLA itself:

    python ..\\train_all.py --carla-exe D:\\CARLA_0.9.15\\WindowsNoEditor\\CarlaUE4.exe

Stop everything with Ctrl+C. Progress is kept in the checkpoints, so starting
the runner again simply picks up where it left off.
"""

import argparse
import os
import re
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

EXPERIMENTS = {
    "2q_cnn": {"script": "2qCNNsac_experiment.py", "log_dir": "logs/2q_cnn", "prefix": "sac_2q_cnn"},
    "5q_cnn": {"script": "5qCNNsac_experiment.py", "log_dir": "logs/5q_cnn", "prefix": "sac_5q_cnn"},
    "2q_stt": {"script": "2qsac_stt_experiment.py", "log_dir": "logs/2q_stt", "prefix": "sac_2q_stt"},
    "5q_stt": {"script": "5qsac_stt_experiment.py", "log_dir": "logs/5q_stt", "prefix": "sac_5q_stt"},
}

# Stable-Baselines3 prints its table as "|    total_timesteps | 4000     |"
STEP_RE = re.compile(r"\|\s*total_timesteps\s*\|\s*([0-9.]+(?:[eE][+-]?\d+)?)\s*\|")

PRINT_LOCK = threading.Lock()
STOP = threading.Event()


def log(msg):
    with PRINT_LOCK:
        print(f"[runner {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def human(seconds):
    seconds = int(max(seconds, 0))
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    minutes = rest // 60
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def latest_checkpoint(log_dir: Path, prefix: str):
    """Newest '<prefix>_<N>_steps.zip' in <log_dir>/checkpoints, plus its N."""
    folder = log_dir / "checkpoints"
    best_path, best_n = None, 0
    if folder.is_dir():
        pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)_steps\.zip$")
        for path in folder.iterdir():
            match = pattern.match(path.name)
            if match:
                n = int(match.group(1))
                if n > best_n:
                    best_path, best_n = path, n
    return best_path, best_n


def prune_checkpoints(log_dir: Path, prefix: str, keep: int, name: str):
    """Delete all but the newest `keep` checkpoints and their replay buffers."""
    if keep <= 0:
        return
    folder = log_dir / "checkpoints"
    if not folder.is_dir():
        return
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)_steps\.zip$")
    found = []
    for path in folder.iterdir():
        match = pattern.match(path.name)
        if match:
            found.append((int(match.group(1)), path))
    found.sort(reverse=True)
    for n, path in found[keep:]:
        buffer_path = folder / f"{prefix}_replay_buffer_{n}_steps.pkl"
        for victim in (path, buffer_path):
            try:
                if victim.exists():
                    size = victim.stat().st_size
                    victim.unlink()
                    log(f"[{name}] pruned {victim.name} ({size / 1e9:.2f} GB)")
            except OSError as exc:
                log(f"[{name}] could not delete {victim.name}: {exc}")


def kill_process_tree(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                check=False,
            )
        else:
            proc.terminate()
        proc.wait(timeout=30)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def port_is_open(port, host="localhost", timeout=2.0):
    """True once something (the CARLA server) accepts connections on this port."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def wait_for_port(port, seconds, name, note):
    """Poll the port until it answers. Returns False if it never does."""
    deadline = time.time() + seconds
    announced = False
    while time.time() < deadline and not STOP.is_set():
        if port_is_open(port):
            return True
        if not announced:
            log(f"[{name}] {note}")
            announced = True
        if STOP.wait(3):
            return False
    return False


def start_carla(exe, port, extra_args, wait_seconds, name):
    cmd = [exe, f"-carla-rpc-port={port}", "-quality-level=Low"] + list(extra_args)
    log(f"[{name}] starting CARLA on port {port}")
    proc = subprocess.Popen(cmd)
    if wait_for_port(port, wait_seconds, name, "waiting for CARLA to finish loading..."):
        log(f"[{name}] CARLA is up on port {port}")
    else:
        log(f"[{name}] CARLA did not answer on port {port} within {wait_seconds:.0f}s")
    return proc


def report_progress(name, samples, target):
    if len(samples) < 2:
        return
    t0, s0 = samples[0]
    t1, s1 = samples[-1]
    elapsed, gained = t1 - t0, s1 - s0
    if elapsed <= 0 or gained <= 0:
        return
    rate = gained / elapsed
    eta = (target - s1) / rate
    log(f"[{name}] {s1:,} / {target:,} steps | {rate:.1f} steps/s | ETA {human(eta)}")


def stream_run(cmd, cwd, env, log_file, name, echo, samples, target, eta_every):
    """Run the training script, mirror its output, and track progress."""
    last_report = time.time()
    highest_step = 0
    saw_error = False

    with open(log_file, "a", encoding="utf-8", errors="replace", buffering=1) as handle:
        handle.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} :: {' '.join(cmd)} =====\n")
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
        )
        try:
            for line in proc.stdout:
                handle.write(line)
                if echo:
                    with PRINT_LOCK:
                        sys.stdout.write(line)
                        sys.stdout.flush()

                match = STEP_RE.search(line)
                if match:
                    step = int(float(match.group(1)))
                    highest_step = max(highest_step, step)
                    samples.append((time.time(), step))
                    if time.time() - last_report >= eta_every:
                        last_report = time.time()
                        report_progress(name, samples, target)

                if "[ERROR]" in line or "Traceback (most recent call last)" in line:
                    saw_error = True

                if STOP.is_set():
                    kill_process_tree(proc)
                    break
        finally:
            try:
                proc.stdout.close()
            except Exception:
                pass
            returncode = proc.wait()

    return returncode, highest_step, saw_error


def run_experiment(name, cfg, args, port):
    log_dir = (args.cwd / cfg["log_dir"]).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    runner_log = log_dir / "runner.log"

    script = args.exp_dir / cfg["script"]
    if not script.is_file():
        log(f"[{name}] script not found: {script}")
        return False

    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.cwd)
    env["PYTHONUNBUFFERED"] = "1"
    # The scripts print characters like "→". When stdout is a pipe, Windows
    # falls back to cp1252 and those raise UnicodeEncodeError, so force UTF-8.
    env["PYTHONIOENCODING"] = "utf-8"
    if not args.wandb:
        env["WANDB_MODE"] = "offline"

    samples = deque(maxlen=40)
    failures = 0
    carla_proc = None

    try:
        while not STOP.is_set():
            checkpoint, done = latest_checkpoint(log_dir, cfg["prefix"])
            remaining = args.timesteps - done

            if remaining <= 0:
                log(f"[{name}] DONE - checkpoint at {done:,} steps meets the {args.timesteps:,} target")
                return True

            if failures >= args.max_retries:
                log(f"[{name}] GIVING UP after {failures} attempts with no progress. See {runner_log}")
                return False

            if args.carla_exe:
                kill_process_tree(carla_proc)
                carla_proc = start_carla(args.carla_exe, port, args.carla_arg,
                                         args.carla_wait, name)
                if STOP.is_set():
                    break
            elif not port_is_open(port):
                # Fail clearly instead of letting the script time out mid-setup.
                ready = wait_for_port(
                    port, args.carla_wait, name,
                    f"nothing is listening on port {port} - open the CARLA window "
                    f"and leave it running (or pass --carla-exe). Waiting...",
                )
                if not ready:
                    failures += 1
                    log(f"[{name}] still no CARLA on port {port}; attempt {failures}/{args.max_retries}")
                    if STOP.wait(args.restart_delay):
                        break
                    continue
                log(f"[{name}] CARLA found on port {port}")

            cmd = [
                sys.executable,
                str(script),
                "--timesteps", str(remaining),
                "--port", str(port),
                "--log-dir", str(log_dir),
                "--buffer-size", str(args.buffer_size),
            ]
            if args.render:
                cmd.append("--render")
            if checkpoint is not None:
                cmd += ["--load-checkpoint", str(checkpoint)]

            if checkpoint is None:
                log(f"[{name}] starting fresh, target {args.timesteps:,} steps, port {port}")
            else:
                log(f"[{name}] resuming from {checkpoint.name} ({done:,} steps), "
                    f"{remaining:,} to go, port {port}")

            started = time.time()
            returncode, highest_step, saw_error = stream_run(
                cmd, args.cwd, env, runner_log, name,
                echo=args.echo, samples=samples,
                target=args.timesteps, eta_every=args.eta_every,
            )

            if STOP.is_set():
                break

            _, done_after = latest_checkpoint(log_dir, cfg["prefix"])
            if done_after > done:
                failures = 0
                prune_checkpoints(log_dir, cfg["prefix"], args.keep_checkpoints, name)
            else:
                failures += 1

            if highest_step >= args.timesteps and not saw_error:
                log(f"[{name}] DONE - reached {highest_step:,} steps")
                return True

            ran_for = time.time() - started
            reason = "error in the script" if saw_error else f"exit code {returncode}"
            log(f"[{name}] stopped after {human(ran_for)} ({reason}); "
                f"last checkpoint {done_after:,} steps; attempt {failures}/{args.max_retries}")

            if STOP.wait(args.restart_delay):
                break
    finally:
        kill_process_tree(carla_proc)

    return False


def main():
    try:  # so mirrored output with non-ASCII characters never kills the runner
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="Run the CARLA SAC experiments to a target timestep count, resuming after crashes."
    )
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                        help="Target TOTAL steps per experiment (default 1,000,000)")
    parser.add_argument("--only", nargs="+", choices=list(EXPERIMENTS),
                        help="Run only these experiments (default: all four)")
    parser.add_argument("--parallel", type=int, default=1,
                        help="How many experiments to run at the same time (default 1). "
                             "Each needs its own CARLA server.")
    parser.add_argument("--port-base", type=int, default=2000, help="First CARLA port (default 2000)")
    parser.add_argument("--port-step", type=int, default=None,
                        help="Port gap between parallel experiments (default 0 when --parallel 1, else 10)")
    parser.add_argument("--render", action="store_true", help="Pass --render to the experiment scripts")
    parser.add_argument("--buffer-size", type=int, default=50000, help="Replay buffer size (default 50000)")
    parser.add_argument("--keep-checkpoints", type=int, default=3,
                        help="Keep only the newest N checkpoints per experiment; 0 disables pruning "
                             "(default 3 - CNN replay buffers are ~2.1 GB each)")
    parser.add_argument("--max-retries", type=int, default=5,
                        help="Stop an experiment after this many restarts that make no progress (default 5)")
    parser.add_argument("--restart-delay", type=float, default=20.0,
                        help="Seconds to wait before restarting after a crash (default 20)")
    parser.add_argument("--eta-every", type=float, default=120.0,
                        help="Seconds between ETA lines (default 120)")
    parser.add_argument("--carla-exe", default=None,
                        help="Path to CarlaUE4.exe. If given, the runner starts and restarts CARLA itself.")
    parser.add_argument("--carla-arg", action="append", default=[],
                        help="Extra argument for CARLA; repeat for several")
    parser.add_argument("--carla-wait", type=float, default=180.0,
                        help="Seconds to wait for CARLA to start answering on its port (default 180)")
    parser.add_argument("--exp-dir", default=None,
                        help="Folder holding the experiment scripts (default: ../experiments)")
    parser.add_argument("--cwd", default=None,
                        help="Folder to run from (default: the current folder, normally CARLA-RL-Agents)")
    parser.add_argument("--wandb", action="store_true",
                        help="Allow W&B online mode (default: offline, so it never prompts)")
    parser.add_argument("--quiet", action="store_true",
                        help="Do not mirror the training output to the console (it still goes to runner.log)")
    args = parser.parse_args()

    args.cwd = Path(args.cwd).resolve() if args.cwd else Path.cwd()
    args.exp_dir = Path(args.exp_dir).resolve() if args.exp_dir else (args.cwd.parent / "experiments")
    args.parallel = max(1, args.parallel)
    if args.port_step is None:
        args.port_step = 0 if args.parallel == 1 else 10
    args.echo = (not args.quiet) and args.parallel == 1

    names = args.only or list(EXPERIMENTS)
    ports = {name: args.port_base + i * args.port_step for i, name in enumerate(names)}

    log(f"working folder : {args.cwd}")
    log(f"experiments    : {args.exp_dir}")
    log(f"target         : {args.timesteps:,} steps each")
    log(f"running        : {', '.join(f'{n} (port {ports[n]})' for n in names)}")
    if args.parallel > 1:
        log(f"parallel       : {args.parallel} at a time - a CARLA server must be listening on each port")
        if args.render:
            log("warning: --render with several experiments at once is very heavy on GPU and RAM")
    if not args.carla_exe:
        log("CARLA is assumed to be running already (pass --carla-exe to let the runner manage it)")
    log(f"keeping the newest {args.keep_checkpoints} checkpoints per experiment"
        if args.keep_checkpoints else "checkpoint pruning is OFF")
    print(flush=True)

    results = {}

    def worker(name):
        try:
            results[name] = run_experiment(name, EXPERIMENTS[name], args, ports[name])
        except Exception as exc:
            log(f"[{name}] runner error: {exc}")
            results[name] = False

    try:
        if args.parallel == 1:
            for name in names:
                if STOP.is_set():
                    break
                worker(name)
        else:
            queue = list(names)
            while queue and not STOP.is_set():
                batch, queue = queue[:args.parallel], queue[args.parallel:]
                threads = [threading.Thread(target=worker, args=(n,), daemon=True) for n in batch]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    while thread.is_alive():
                        thread.join(timeout=1.0)
    except KeyboardInterrupt:
        log("Ctrl+C - stopping. Progress is saved in the checkpoints; run this again to continue.")
        STOP.set()
        time.sleep(3)

    print(flush=True)
    log("summary:")
    for name in names:
        log_dir = (args.cwd / EXPERIMENTS[name]["log_dir"]).resolve()
        _, steps = latest_checkpoint(log_dir, EXPERIMENTS[name]["prefix"])
        state = {True: "finished", False: "incomplete"}.get(results.get(name), "not started")
        log(f"  {name:8s} {state:11s} last checkpoint {steps:,} steps")


if __name__ == "__main__":
    main()
