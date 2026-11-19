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
  * Passes your TARGET total (--timesteps, default 1,000,000) unchanged. The
    experiment scripts subtract the restored step counter themselves, so a run
    resumed at 300,000 trains 700,000 more and stops exactly on 1,000,000.
  * Restarts automatically after a crash, continuing from the last checkpoint.
  * Prints throughput (steps/sec) and an ETA to the target, read from the
    per-episode console line ("ep  412 | step    98,340 | ..."), falling back
    to the Stable-Baselines3 table ("| total_timesteps | 4000 |").
  * Prunes old checkpoints - the CNN replay buffers are ~2.1 GB each.
  * Can run several experiments at once, each against its own CARLA server.
  * Shows the training by default: --render is passed to the scripts (camera
    window + spectator following the car) unless you pass --no-render, and a
    CARLA server started with --carla-exe opens in a normal visible window.

Usage - run from the CARLA-RL-Agents folder with the venv active:

    python ..\\train_all.py --only 2q_cnn
    python ..\\train_all.py                       # all four, one after another
    python ..\\train_all.py --parallel 2          # two at a time, ports 2000/2010
    python ..\\train_all.py --no-render           # faster, nothing drawn
    python ..\\train_all.py --only 5q_cnn_resnet_layer4   # opt-in: 5q CNN with a ResNet-50 image branch

A CARLA server must already be listening on each port in use, unless you pass
--carla-exe, in which case the runner starts and restarts CARLA itself:

    python ..\\train_all.py --carla-exe D:\\CARLA_0.9.15\\WindowsNoEditor\\CarlaUE4.exe

Experiments listed in skip_experiments.txt (project root, one name per line)
are skipped unless named explicitly with --only.

Per run, <log-dir> holds runner.log (console copy), episodes.jsonl (one line
per training episode), eval.jsonl (deterministic evaluation every 25k steps),
run_config.json, tensorboard/ and checkpoints/. Build the report with
make_report.py.

Stop everything with Ctrl+C. Progress is kept in the checkpoints, so starting
the runner again simply picks up where it left off.

Memory guard (Windows): on 2026-11-18 an experiment process committed 75 GB in a
few minutes and froze the PC. Every experiment now runs in a Job Object that
caps its committed memory (--mem-hard-gb, enforced by the OS), and a watchdog
kills it earlier (--mem-limit-gb) so it is restarted from the latest checkpoint.
The runner also puts itself in a kill-on-close job: if it dies (window closed,
Ctrl+C, crash), the CARLA server and the experiment it started die with it.
Runner messages, exit codes and memory readings go to <log-root>/train_all.log
and to each experiment's runner.log.
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
# 5q CNN with the ImageNet ResNet-50 image branch, one entry per --resnet-train mode. Opt-in: run only
# when named with --only (the default run stays the four experiments above).
for _mode in ("head", "layer4", "all"):
    EXPERIMENTS[f"5q_cnn_resnet_{_mode}"] = {
        "script": "5qCNNsac_experiment.py", "log_dir": f"logs/5q_cnn_resnet_{_mode}",
        "prefix": f"sac_5q_cnn_resnet_{_mode}", "args": ["--cnn", "resnet50", "--resnet-train", _mode],
        "opt_in": True,
    }

# Per-episode line from commons/metrics_callbacks.py: "ep  412 | step    98,340 | ..."
EPISODE_STEP_RE = re.compile(r"^\s*ep\s+\d+\s*\|\s*step\s+([\d,]+)\s*\|")
# Stable-Baselines3 prints its table as "|    total_timesteps | 4000     |" (fallback)
STEP_RE = re.compile(r"\|\s*total_timesteps\s*\|\s*([0-9.]+(?:[eE][+-]?\d+)?)\s*\|")

# CARLA flags that would hide the simulator window; the runner never passes them.
HIDDEN_WINDOW_FLAGS = ("-renderoffscreen", "-nullrhi")

PRINT_LOCK = threading.Lock()
STOP = threading.Event()


def parse_step(line):
    """Training step reported by a console line, or None.

    The per-episode line is tried first, the SB3 table row second."""
    match = EPISODE_STEP_RE.search(line)
    if match:
        return int(match.group(1).replace(",", ""))
    match = STEP_RE.search(line)
    if match:
        return int(float(match.group(1)))
    return None


RUNNER_LOG = None   # Path of train_all.log (set in main); runner messages are kept there too


def log(msg):
    with PRINT_LOCK:
        print(f"[runner {time.strftime('%H:%M:%S')}] {msg}", flush=True)
        if RUNNER_LOG is not None:
            try:
                with open(RUNNER_LOG, "a", encoding="utf-8", errors="replace") as handle:
                    handle.write(f"[runner {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
            except OSError:
                pass


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
        buffers = (folder / f"{prefix}_replay_buffer_{n}_steps.pkl",
                   folder / f"{path.stem}_replay_buffer.pkl")
        for victim in (path, *buffers):
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


# --------------------------------------------------------------------------- memory guard
# Layer 1, Job Objects (OS-enforced): the experiment process gets a hard cap on its committed memory,
# so a runaway fails inside the experiment instead of exhausting the page file. The runner's own
# kill-on-close job makes every child (CARLA, the experiment) die with the runner.
# Layer 2, MemoryGuard thread: polls the experiment's private bytes every second, logs them, and kills
# the process tree at the soft limit, below the hard cap, so the restart is clean and logged.
GB = 1024 ** 3
_JOB_EXTENDED_LIMIT_INFO = 9                    # JobObjectExtendedLimitInformation
_JOB_LIMIT_PROCESS_MEMORY = 0x0100
_JOB_LIMIT_DIE_ON_UNHANDLED_EXCEPTION = 0x0400  # no "has stopped working" dialog holding a dead process
_JOB_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_JOB_BASIC_PROCESS_ID_LIST = 3                  # JobObjectBasicProcessIdList
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
# venv\Scripts\python.exe is a launcher whose CHILD is the real interpreter. The experiment is created
# suspended, put in its job, then resumed, so that child is born inside the job (no race).
CREATE_SUSPENDED = 0x00000004

if os.name == "nt":
    import ctypes
    from ctypes import wintypes as _wt

    class _MEMORY_COUNTERS(ctypes.Structure):          # PROCESS_MEMORY_COUNTERS_EX
        _fields_ = [("cb", _wt.DWORD), ("PageFaultCount", _wt.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t)]

    class _BASIC_LIMITS(ctypes.Structure):             # JOBOBJECT_BASIC_LIMIT_INFORMATION
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", _wt.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", _wt.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", _wt.DWORD), ("SchedulingClass", _wt.DWORD)]

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [(f, ctypes.c_uint64) for f in ("ReadOperationCount", "WriteOperationCount",
                                                   "OtherOperationCount", "ReadTransferCount",
                                                   "WriteTransferCount", "OtherTransferCount")]

    class _EXTENDED_LIMITS(ctypes.Structure):          # JOBOBJECT_EXTENDED_LIMIT_INFORMATION
        _fields_ = [("BasicLimitInformation", _BASIC_LIMITS), ("IoInfo", _IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    class _PID_LIST(ctypes.Structure):                 # JOBOBJECT_BASIC_PROCESS_ID_LIST (room for 64)
        _fields_ = [("NumberOfAssignedProcesses", _wt.DWORD), ("NumberOfProcessIdsInList", _wt.DWORD),
                    ("ProcessIdList", ctypes.c_size_t * 64)]

    _K32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _NTDLL = ctypes.WinDLL("ntdll")
    _NTDLL.NtResumeProcess.argtypes = [_wt.HANDLE]
    _NTDLL.NtResumeProcess.restype = ctypes.c_long
    _K32.OpenProcess.argtypes = [_wt.DWORD, _wt.BOOL, _wt.DWORD]
    _K32.OpenProcess.restype = _wt.HANDLE
    _K32.TerminateJobObject.argtypes = [_wt.HANDLE, ctypes.c_uint]
    _K32.TerminateJobObject.restype = _wt.BOOL
    _K32.CreateJobObjectW.argtypes = [ctypes.c_void_p, _wt.LPCWSTR]
    _K32.CreateJobObjectW.restype = _wt.HANDLE
    _K32.SetInformationJobObject.argtypes = [_wt.HANDLE, ctypes.c_int, ctypes.c_void_p, _wt.DWORD]
    _K32.SetInformationJobObject.restype = _wt.BOOL
    _K32.QueryInformationJobObject.argtypes = [_wt.HANDLE, ctypes.c_int, ctypes.c_void_p, _wt.DWORD,
                                               ctypes.c_void_p]
    _K32.QueryInformationJobObject.restype = _wt.BOOL
    _K32.AssignProcessToJobObject.argtypes = [_wt.HANDLE, _wt.HANDLE]
    _K32.AssignProcessToJobObject.restype = _wt.BOOL
    _K32.GetCurrentProcess.restype = _wt.HANDLE
    _K32.CloseHandle.argtypes = [_wt.HANDLE]
    _K32.CloseHandle.restype = _wt.BOOL
    _K32.K32GetProcessMemoryInfo.argtypes = [_wt.HANDLE, ctypes.c_void_p, _wt.DWORD]
    _K32.K32GetProcessMemoryInfo.restype = _wt.BOOL


class Job:
    """A Windows Job Object: processes in it are killed when its last handle closes (runner exit),
    crash dialogs are suppressed, and optionally each process's committed memory is capped."""

    def __init__(self, process_handle, hard_limit_gb: float = 0.0):
        self.handle = None
        if os.name != "nt":
            return
        job = _K32.CreateJobObjectW(None, None)
        if not job:
            raise OSError(ctypes.get_last_error(), "CreateJobObject failed")
        info = _EXTENDED_LIMITS()
        flags = _JOB_LIMIT_KILL_ON_JOB_CLOSE | _JOB_LIMIT_DIE_ON_UNHANDLED_EXCEPTION
        if hard_limit_gb > 0:
            flags |= _JOB_LIMIT_PROCESS_MEMORY
            info.ProcessMemoryLimit = int(hard_limit_gb * GB)
        info.BasicLimitInformation.LimitFlags = flags
        ok = _K32.SetInformationJobObject(job, _JOB_EXTENDED_LIMIT_INFO, ctypes.byref(info), ctypes.sizeof(info))
        ok = ok and _K32.AssignProcessToJobObject(job, process_handle)
        if not ok:
            err = ctypes.get_last_error()
            _K32.CloseHandle(job)
            raise OSError(err, "could not set up the job object")
        self.handle = job

    @classmethod
    def for_popen(cls, proc, hard_limit_gb: float = 0.0):
        return cls(int(proc._handle), hard_limit_gb) if os.name == "nt" else cls(None)

    def peak_gb(self) -> float:
        """Highest committed memory of any process in the job so far."""
        if not self.handle:
            return 0.0
        info = _EXTENDED_LIMITS()
        if _K32.QueryInformationJobObject(self.handle, _JOB_EXTENDED_LIMIT_INFO, ctypes.byref(info),
                                          ctypes.sizeof(info), None):
            return info.PeakProcessMemoryUsed / GB
        return 0.0

    def process_ids(self):
        if not self.handle:
            return []
        info = _PID_LIST()
        if not _K32.QueryInformationJobObject(self.handle, _JOB_BASIC_PROCESS_ID_LIST, ctypes.byref(info),
                                              ctypes.sizeof(info), None):
            return []
        return [int(info.ProcessIdList[i]) for i in range(min(info.NumberOfProcessIdsInList, 64))]

    def max_private_gb(self) -> float:
        """Largest committed private memory of any process now in the job (the cap is per process)."""
        best = 0.0
        for pid in self.process_ids():
            handle = _K32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if handle:
                try:
                    best = max(best, _private_gb_of(handle) or 0.0)
                finally:
                    _K32.CloseHandle(handle)
        return best

    def terminate(self, exit_code: int = 1) -> bool:
        """Kill every process in the job (the launcher and the interpreter it started)."""
        return bool(self.handle) and bool(_K32.TerminateJobObject(self.handle, exit_code))

    def close(self):
        """Closing the last handle kills whatever is still running in the job."""
        if self.handle:
            _K32.CloseHandle(self.handle)
            self.handle = None


def _private_gb_of(handle):
    counters = _MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(_MEMORY_COUNTERS)
    if _K32.K32GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
        return counters.PrivateUsage / GB
    return None


def resume_process(proc):
    """Resume a child created with CREATE_SUSPENDED."""
    if os.name == "nt" and proc is not None:
        status = _NTDLL.NtResumeProcess(int(proc._handle))
        if status != 0:
            raise OSError(status, f"NtResumeProcess failed (NTSTATUS 0x{status & 0xFFFFFFFF:08X})")


def experiment_private_gb(proc, job):
    """Committed private memory of the experiment in GB - the largest process of its job, since the
    venv launcher's child is the real interpreter - or None once it has exited."""
    if os.name != "nt" or proc is None or proc.poll() is not None:
        return None
    if job is not None and job.handle:
        return job.max_private_gb()
    return _private_gb_of(int(proc._handle))


class MemoryGuard(threading.Thread):
    """Polls the experiment's private memory 4 times a second. Writes a reading every `log_every_s` and
    kills the process tree above `limit_gb` (the runner then restarts it from the latest checkpoint).
    It also kills an experiment that printed nothing for `stall_s` seconds: a native CARLA hang inside
    world.tick() (seen before a crash on 2026-11-19) neither exits nor grows."""

    def __init__(self, proc, job, name, limit_gb, log_every_s, write, stall_s=0.0):
        super().__init__(daemon=True, name=f"memguard-{name}")
        self.proc, self.job, self.name = proc, job, name
        self.limit_gb, self.log_every_s, self.write = limit_gb, log_every_s, write
        self.stall_s = float(stall_s)
        self.last_output = time.time()     # updated by stream_run for every line the experiment prints
        self.peak_gb = 0.0
        self.tripped_gb = None
        self.stalled_s = None
        self._halt = threading.Event()

    def _kill(self, msg):
        self.write(msg)
        log(msg)
        if not (self.job is not None and self.job.terminate(1)):
            kill_process_tree(self.proc)

    def run(self):
        last_report = time.time()
        while not self._halt.wait(0.25):   # a runaway reached ~2 GB/s; poll fast
            if self.proc.poll() is not None:
                return
            silent = time.time() - self.last_output
            if self.stall_s > 0 and silent > self.stall_s:
                self.stalled_s = silent
                self._kill(f"[STALL] {self.name}: no output for {silent / 60:.1f} min (limit "
                           f"{self.stall_s / 60:g} min) - killing it; it restarts from the latest checkpoint")
                return
            if os.name != "nt":
                continue
            gb = experiment_private_gb(self.proc, self.job)
            if gb is None:
                return
            self.peak_gb = max(self.peak_gb, gb)
            if self.limit_gb > 0 and gb > self.limit_gb:
                self.tripped_gb = gb
                self._kill(f"[MEMGUARD] {self.name}: experiment uses {gb:.1f} GB private memory (limit "
                           f"{self.limit_gb:g} GB) - killing it; it restarts from the latest checkpoint")
                return
            if self.log_every_s > 0 and time.time() - last_report >= self.log_every_s:
                last_report = time.time()
                self.write(f"[MEM] {self.name}: private {gb:.2f} GB (peak {self.peak_gb:.2f} GB)")

    def stop(self):
        self._halt.set()


_EXIT_CODES = {
    0xC000013A: "console closed or Ctrl+C",
    0xC0000005: "access violation (native crash)",
    0xC0000017: "out of memory",
    0xC0000409: "fail-fast / buffer overrun (native crash)",
    0xC00000FD: "stack overflow",
    0xC0000374: "heap corruption",
}


def describe_exit(code):
    if code is None:
        return "unknown"
    unsigned = int(code) & 0xFFFFFFFF
    text = _EXIT_CODES.get(unsigned, "")
    return f"{code} (0x{unsigned:08X}{', ' + text if text else ''})"


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


def carla_command(exe, port, extra_args, headless=False):
    """CARLA command line: a visible window by default, or off-screen with headless=True
    (the experiments' own camera windows stay visible either way)."""
    if headless:
        return [exe, f"-carla-rpc-port={port}", "-quality-level=Low", "-RenderOffScreen"] + list(extra_args)
    kept = []
    for arg in extra_args:
        if arg.lower().split("=", 1)[0] in HIDDEN_WINDOW_FLAGS:
            log(f"ignoring CARLA flag {arg}: use --headless for an off-screen server")
            continue
        kept.append(arg)
    return [exe, f"-carla-rpc-port={port}", "-quality-level=Low", "-windowed"] + kept


def start_carla(exe, port, extra_args, wait_seconds, name, headless=False):
    cmd = carla_command(exe, port, extra_args, headless)
    log(f"[{name}] starting CARLA on port {port} ({'headless' if headless else 'visible window'})")
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


def stream_run(cmd, cwd, env, log_file, name, echo, samples, target, eta_every,
               mem_limit_gb=0.0, mem_hard_gb=0.0, mem_log_every=300.0, stall_minutes=0.0):
    """Run the training script under the memory guard, mirror its output, and track progress.

    Returns (returncode, highest_step, saw_error, memory_kill_gb or None, stalled_seconds or None)."""
    last_report = time.time()
    highest_step = 0
    saw_error = False
    write_lock = threading.Lock()

    with open(log_file, "a", encoding="utf-8", errors="replace", buffering=1) as handle:
        def write(line):
            with write_lock:
                handle.write(f"[runner {time.strftime('%H:%M:%S')}] {line}\n")

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
            creationflags=CREATE_SUSPENDED if os.name == "nt" else 0,
        )
        job = None
        try:
            job = Job.for_popen(proc, mem_hard_gb)
            if mem_hard_gb > 0 and os.name == "nt":
                write(f"memory guard: hard cap {mem_hard_gb:g} GB (OS job limit), "
                      f"watchdog kill at {mem_limit_gb:g} GB")
        except OSError as exc:   # never run unguarded silently
            job = None
            write(f"WARNING could not attach the memory job object: {exc}")
            log(f"[{name}] WARNING could not attach the memory job object: {exc}")
        finally:
            resume_process(proc)   # only now may the launcher start the interpreter (inside the job)
        guard = MemoryGuard(proc, job, name, mem_limit_gb, mem_log_every, write, stall_s=60.0 * stall_minutes)
        guard.start()
        try:
            for line in proc.stdout:
                guard.last_output = time.time()
                with write_lock:
                    handle.write(line)
                if echo:
                    with PRINT_LOCK:
                        sys.stdout.write(line)
                        sys.stdout.flush()

                step = parse_step(line)
                if step is not None:
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
            guard.stop()
            peak = max(job.peak_gb() if job else 0.0, guard.peak_gb)
            if job:
                job.close()
            summary = f"experiment exited: code {describe_exit(returncode)}, peak memory {peak:.2f} GB"
            if guard.stalled_s is not None:
                summary += f" - killed by the stall watchdog after {guard.stalled_s / 60:.1f} min without output"
            elif guard.tripped_gb is not None:
                summary += f" - killed by the memory guard at {guard.tripped_gb:.1f} GB"
            elif mem_hard_gb > 0 and peak >= 0.98 * mem_hard_gb:
                summary += f" - hit the {mem_hard_gb:g} GB hard cap"
            write(summary)
            log(f"[{name}] {summary}")

    return returncode, highest_step, saw_error, guard.tripped_gb, guard.stalled_s


def exp_log_dir(name, cfg, args):
    if args.log_root:
        return (Path(args.log_root) / name).resolve()
    return (args.cwd / cfg["log_dir"]).resolve()


def build_command(script, port, log_dir, checkpoint, args, extra_args=()):
    """Experiment command line. --timesteps is the TARGET total; the script subtracts
    the restored step counter itself when resuming. extra_args: the experiment's own options."""
    cmd = [
        sys.executable,
        str(script),
        "--timesteps", str(args.timesteps),
        "--port", str(port),
        "--log-dir", str(log_dir),
        "--buffer-size", str(args.buffer_size),
    ] + list(extra_args)
    if args.render:
        cmd.append("--render")
    if args.num_npc is not None:
        cmd += ["--num-npc", str(args.num_npc)]
    if args.num_pedestrians is not None:
        cmd += ["--num-pedestrians", str(args.num_pedestrians)]
    if checkpoint is not None:
        cmd += ["--load-checkpoint", str(checkpoint)]
    return cmd


def run_experiment(name, cfg, args, port):
    log_dir = exp_log_dir(name, cfg, args)
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
                                         args.carla_wait, name, args.headless)
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

            cmd = build_command(script, port, log_dir, checkpoint, args, cfg.get("args", ()))

            if checkpoint is None:
                log(f"[{name}] starting fresh, target {args.timesteps:,} steps, port {port}")
            else:
                log(f"[{name}] resuming from {checkpoint.name} ({done:,} steps), "
                    f"{remaining:,} to go, port {port}")

            started = time.time()
            returncode, highest_step, saw_error, mem_kill_gb, stalled_s = stream_run(
                cmd, args.cwd, env, runner_log, name,
                echo=args.echo, samples=samples,
                target=args.timesteps, eta_every=args.eta_every,
                mem_limit_gb=args.mem_limit_gb, mem_hard_gb=args.mem_hard_gb,
                mem_log_every=args.mem_log_every, stall_minutes=args.stall_minutes,
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
            if stalled_s is not None:
                reason = f"stalled: no output for {stalled_s / 60:.1f} min"
            elif mem_kill_gb is not None:
                reason = f"memory guard at {mem_kill_gb:.1f} GB"
            elif saw_error:
                reason = "error in the script"
            else:
                reason = f"exit code {describe_exit(returncode)}"
            log(f"[{name}] stopped after {human(ran_for)} ({reason}); "
                f"last checkpoint {done_after:,} steps; attempt {failures}/{args.max_retries}")

            if STOP.wait(args.restart_delay):
                break
    finally:
        kill_process_tree(carla_proc)

    return False


def build_parser():
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
    parser.add_argument("--render", dest="render", action="store_true", default=True,
                        help="Show the training: pass --render to the experiment scripts (default ON)")
    parser.add_argument("--no-render", dest="render", action="store_false",
                        help="Do not pass --render (no camera window / spectator follow; faster)")
    parser.add_argument("--buffer-size", type=int, default=50000, help="Replay buffer size (default 50000)")
    parser.add_argument("--num-npc", type=int, default=None,
                        help="NPC vehicles per episode (default: the script's own). Fewer = faster")
    parser.add_argument("--num-pedestrians", type=int, default=None,
                        help="Pedestrians per episode (default: the script's own). Fewer = faster")
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
                        help="Path to CarlaUE4.exe. If given, the runner starts and restarts CARLA itself "
                             "(visible window unless --headless).")
    parser.add_argument("--headless", action="store_true",
                        help="Start CARLA off-screen (-RenderOffScreen); the experiments' camera windows "
                             "still open with --render")
    parser.add_argument("--carla-arg", action="append", default=[],
                        help="Extra argument for CARLA; repeat for several "
                             "(-RenderOffScreen / -nullrhi are ignored unless --headless)")
    parser.add_argument("--carla-wait", type=float, default=180.0,
                        help="Seconds to wait for CARLA to start answering on its port (default 180)")
    parser.add_argument("--exp-dir", default=None,
                        help="Folder holding the experiment scripts (default: ../experiments)")
    parser.add_argument("--log-root", default=None,
                        help="Put each experiment's logs/checkpoints in <log-root>/<name> "
                             "instead of CARLA-RL-Agents/logs/<name>")
    parser.add_argument("--cwd", default=None,
                        help="Folder to run from (default: the current folder, normally CARLA-RL-Agents)")
    parser.add_argument("--wandb", action="store_true",
                        help="Allow W&B online mode (default: offline, so it never prompts)")
    parser.add_argument("--quiet", action="store_true",
                        help="Do not mirror the training output to the console (it still goes to runner.log)")
    parser.add_argument("--mem-limit-gb", type=float, default=14.0,
                        help="Watchdog: kill an experiment whose private memory exceeds this and restart it "
                             "from the latest checkpoint (default 14; 0 disables)")
    parser.add_argument("--mem-hard-gb", type=float, default=16.0,
                        help="OS-enforced cap on an experiment's committed memory via a Windows Job Object; "
                             "allocations beyond it fail inside the experiment (default 16; 0 disables)")
    parser.add_argument("--stall-minutes", type=float, default=15.0,
                        help="Kill and restart an experiment that prints nothing for this long - a native "
                             "CARLA hang (default 15; 0 disables)")
    parser.add_argument("--mem-log-every", type=float, default=300.0,
                        help="Seconds between memory readings in runner.log (default 300)")
    return parser


def main(argv=None):
    try:  # so mirrored output with non-ASCII characters never kills the runner
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    args = build_parser().parse_args(argv)

    args.cwd = Path(args.cwd).resolve() if args.cwd else Path.cwd()
    args.exp_dir = Path(args.exp_dir).resolve() if args.exp_dir else (args.cwd.parent / "experiments")
    args.parallel = max(1, args.parallel)
    if args.port_step is None:
        args.port_step = 0 if args.parallel == 1 else 10
    args.echo = (not args.quiet) and args.parallel == 1

    global RUNNER_LOG
    log_root = Path(args.log_root).resolve() if args.log_root else (args.cwd / "logs")
    log_root.mkdir(parents=True, exist_ok=True)
    RUNNER_LOG = log_root / "train_all.log"
    if args.mem_hard_gb > 0 and 0 < args.mem_hard_gb <= args.mem_limit_gb:
        args.mem_limit_gb = 0.9 * args.mem_hard_gb   # the watchdog must fire before the OS cap
    runner_job = None
    if os.name == "nt":
        try:   # every child (CARLA, experiments, taskkill) inherits this job and dies with the runner
            runner_job = Job(_K32.GetCurrentProcess())
        except OSError as exc:
            log(f"WARNING could not create the runner's kill-on-close job: {exc}")

    names = args.only or [n for n, cfg in EXPERIMENTS.items() if not cfg.get("opt_in")]

    # skip_experiments.txt (project root, one name per line) drops experiments
    # without editing the launcher. Logged as "DONE - skipped" so the watchdog
    # treats them as finished.
    # An explicit --only always runs exactly what it names.
    skip_file = args.cwd.parent / "skip_experiments.txt"
    if skip_file.is_file() and not args.only:
        skipped = {s.strip() for s in skip_file.read_text().split() if s.strip()}
        for name in [n for n in names if n in skipped]:
            log(f"[{name}] DONE - skipped (listed in {skip_file.name}), not trained")
        names = [n for n in names if n not in skipped]
        if not names:
            log("nothing left to train")
            return

    ports = {name: args.port_base + i * args.port_step for i, name in enumerate(names)}

    log(f"working folder : {args.cwd}")
    log(f"experiments    : {args.exp_dir}")
    log(f"target         : {args.timesteps:,} steps each")
    log(f"running        : {', '.join(f'{n} (port {ports[n]})' for n in names)}")
    log(f"rendering      : {'ON (camera window + spectator follow)' if args.render else 'OFF (--no-render)'}")
    if args.parallel > 1:
        log(f"parallel       : {args.parallel} at a time - a CARLA server must be listening on each port")
        if args.render:
            log("warning: rendering several experiments at once is very heavy on GPU and RAM "
                "(add --no-render to save resources)")
    if not args.carla_exe:
        log("CARLA is assumed to be running already (pass --carla-exe to let the runner manage it)")
    log(f"keeping the newest {args.keep_checkpoints} checkpoints per experiment"
        if args.keep_checkpoints else "checkpoint pruning is OFF")
    if os.name == "nt" and (args.mem_limit_gb > 0 or args.mem_hard_gb > 0):
        log(f"memory guard   : watchdog kill at {args.mem_limit_gb:g} GB, OS hard cap {args.mem_hard_gb:g} GB "
            f"per experiment; children die with the runner ({'on' if runner_job else 'OFF'})")
    else:
        log("memory guard   : OFF")
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
        log_dir = exp_log_dir(name, EXPERIMENTS[name], args)
        _, steps = latest_checkpoint(log_dir, EXPERIMENTS[name]["prefix"])
        state = {True: "finished", False: "incomplete"}.get(results.get(name), "not started")
        log(f"  {name:8s} {state:11s} last checkpoint {steps:,} steps")


if __name__ == "__main__":
    main()
