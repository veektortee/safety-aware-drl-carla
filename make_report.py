"""
make_report.py - builds the results package from episodes.jsonl / eval.jsonl.

Every training run writes, in its log folder (see commons/metrics_callbacks.py):
  episodes.jsonl    one line per TRAINING episode (exploration noise included)
  eval.jsonl        one line per deterministic EVALUATION round (fixed spawns, no noise)
  run_config.json   reward version, git commit, hyperparameters, env/CBF parameters

Output (default <repo>/logs/report, or --out):
  report.html            self-contained report (plots embedded) - send this
  summary.csv            one row per experiment: first vs. last evaluation + recent training
  <exp>_episodes.csv     per-episode metrics vs. training step
  <exp>_eval.csv         per-evaluation-round metrics vs. training step
  plots/*.png            every figure as a separate image

The headline numbers come from eval.jsonl (deterministic policy), not from the noisy training
episodes; training episodes are used for the curves. Metric definitions: METRICS.md.
Safe to run any time, including while training: it only reads the logs.

    venv\\Scripts\\python.exe make_report.py
    venv\\Scripts\\python.exe make_report.py --log-root D:\\CARLA_trained_models --out D:\\CARLA_trained_models\\results
    venv\\Scripts\\python.exe make_report.py --run 2q_cnn=CARLA-RL-Agents\\logs\\2q_cnn
"""

import argparse
import base64
import glob
import html
import io
import json
import os
import re
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOG_ROOTS = [os.path.join(PROJECT, "CARLA-RL-Agents", "logs"), os.path.join(PROJECT, "logs")]
DEFAULT_OUT = os.path.join(PROJECT, "logs", "report")
TARGET = 1_000_000
WINDOW = 100          # rolling window (episodes) for training curves

EXPERIMENTS = {
    "2q_cnn": "SAC, 2 critics, camera CNN + navigation vector, CBF safety layer",
    "5q_cnn": "SAC, 5-critic ensemble, camera CNN + navigation vector, CBF safety layer",
    "2q_stt": "SAC, 2 critics, ResNet50 + spatio-temporal transformer + navigation vector, CBF safety layer",
    "5q_stt": "SAC, 5-critic ensemble, ResNet50 + spatio-temporal transformer + navigation vector, CBF safety layer",
}

# ---- colours (reference categorical palette, fixed order; adjacent stack pairs validated) -------
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
EXP_COLORS = {"2q_cnn": "#2a78d6", "5q_cnn": "#eb6834", "2q_stt": "#1baf7a", "5q_stt": "#eda100"}
EXTRA_COLORS = ["#e87ba4", "#008300", "#6250d6", "#e34948"]
# outcome stack, bottom -> top (crash on the baseline, success on the top edge)
OUTCOMES = [
    ("crash", ("collision", "collision_pedestrian"), "#eb6834", "crash (vehicle, object or pedestrian)"),
    ("rule_violation", ("off_route", "wrong_way", "red_light"), "#1baf7a", "rule violation (off-route, wrong-way, red light)"),
    ("blocked", ("blocked",), "#eda100", "blocked (stopped 15 s with no reason)"),
    ("timeout", ("timeout",), "#e87ba4", "timeout (too slow for the route)"),
    ("success", ("endpoint",), "#008300", "success (reached the route end)"),
]
REWARD_GROUPS = [
    ("progress", "#2a78d6", "route progress (+)"),
    ("time", "#eb6834", "time cost (-)"),
    ("shaping", "#1baf7a", "driving penalties: speeding, lane, heading, rules (-)"),
    ("terminal", "#eda100", "terminal: goal bonus / crash and violation penalties"),
    ("cbf", "#e87ba4", "CBF intervention penalty (-)"),
]

# (key in eval / summary, label, format, higher_is_better, plain-language definition)
HEADLINE = [
    ("success_frac", "Success rate", "pct", True,
     "Share of episodes that reach the end of the planned route. Higher is better."),
    ("crash_frac", "Crash rate", "pct", False,
     "Share of episodes that end in a collision (vehicle, object or pedestrian). Lower is better."),
    ("collisions_per_km", "Collisions per km", "{:.2f} /km", False,
     "Collisions divided by kilometres driven, so slow cautious driving is not rewarded. Lower is better."),
    ("route_completion_frac", "Route completion", "pct", True,
     "How much of the planned route the car covered before the episode ended (mean). Higher is better."),
    ("speeding_frac", "Speeding share", "pct", False,
     "Share of driving time spent above the speed limit. Lower is better."),
    ("illegal_lane_crossings_per_km", "Illegal lane crossings per km", "{:.2f} /km", False,
     "Crossings of solid lines or into oncoming lanes, per kilometre. Legal overtakes do not count. Lower is better."),
    ("return_total_mean", "Episode return", "{:+.1f}", True,
     "Total reward per episode, the number SAC maximises (includes the CBF penalty). Higher is better."),
]

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 10, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "legend.frameon": False,
    "legend.fontsize": 8, "lines.linewidth": 1.6,
})


# ------------------------------------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------------------------------------
def read_jsonl(path):
    out = []
    if not os.path.isfile(path):
        return out
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue   # half-written last line while training is running
            if isinstance(obj, dict):
                out.append(obj)
    return out


def drop_superseded(records):
    """After a resume, records an earlier process logged past the restored checkpoint are void."""
    kept, seen = [], set()
    for rec in records:
        run = rec.get("run_id")
        if run is not None and run not in seen:
            seen.add(run)
            start = rec.get("run_start_step")
            if start is not None:
                kept = [r for r in kept if (r.get("step") or 0) <= start]
        kept.append(rec)
    return kept


def episodes_frame(records):
    rows = []
    for r in records:
        row = {k: v for k, v in r.items() if not isinstance(v, (dict, list))}
        for g, v in (r.get("reward_groups") or {}).items():
            row[f"rg_{g}"] = v
        for c, v in (r.get("reward_components") or {}).items():
            row[f"rc_{c}"] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in ("step", "length_steps", "distance_m", "collisions_total", "illegal_lane_invasions",
                "route_completion_frac", "return_total", "mean_speed_mps", "speeding_frac",
                "cbf_intervention_frac", "trust_mean", "min_ttc_s", "curriculum_stage", "wall_time"):
        if col not in df:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "end_reason" not in df:
        df["end_reason"] = "unknown"
    return df.sort_values(["step"], kind="stable").reset_index(drop=True)


def find_runs(log_roots, explicit):
    runs = {}
    for spec in explicit or []:
        name, _, path = spec.partition("=")
        if not path:
            name, path = os.path.basename(os.path.normpath(spec)), spec
        runs[name] = os.path.abspath(path)
    roots = list(log_roots or []) + DEFAULT_LOG_ROOTS
    for name in EXPERIMENTS:
        if name in runs:
            continue
        for root in roots:
            folder = os.path.join(root, name)
            if os.path.isfile(os.path.join(folder, "episodes.jsonl")):
                runs[name] = os.path.abspath(folder)
                break
    return runs


def latest_checkpoint_step(folder):
    best = 0
    for path in glob.glob(os.path.join(folder, "checkpoints", "*_steps.zip")):
        match = re.search(r"_(\d+)_steps\.zip$", path)
        if match:
            best = max(best, int(match.group(1)))
    return best


def load_run(name, folder):
    episodes = episodes_frame(drop_superseded(read_jsonl(os.path.join(folder, "episodes.jsonl"))))
    evals = drop_superseded(read_jsonl(os.path.join(folder, "eval.jsonl")))
    config = {}
    try:
        with open(os.path.join(folder, "run_config.json"), encoding="utf-8") as fh:
            config = json.load(fh)
    except (OSError, ValueError):
        pass
    steps = max(int(episodes["step"].max()) if not episodes.empty else 0, latest_checkpoint_step(folder))
    finals = glob.glob(os.path.join(folder, "*_final.zip")) + glob.glob(os.path.join(folder, "..", f"sac_{name}_final.zip"))
    status = "FINISHED" if finals else f"training ({steps / TARGET:.0%})"
    return {"name": name, "folder": folder, "episodes": episodes, "evals": evals, "config": config,
            "steps": steps, "status": status, "final_model": finals[0] if finals else None,
            "desc": EXPERIMENTS.get(name, name)}


# ------------------------------------------------------------------------------------------------
# Derived series
# ------------------------------------------------------------------------------------------------
def rolling(series, window=WINDOW):
    return series.rolling(window, min_periods=1).mean()


def rolling_ratio(num, den, window=WINDOW, min_den=None):
    n = num.rolling(window, min_periods=1).sum()
    d = den.rolling(window, min_periods=1).sum()
    out = n / d.replace(0, np.nan)
    if min_den is not None:
        out[d < min_den] = np.nan
    return out


def outcome_shares(df, window=WINDOW):
    shares = {}
    for key, members, _, _ in OUTCOMES:
        shares[key] = rolling(df["end_reason"].isin(members).astype(float), window)
    return pd.DataFrame(shares)


def reward_split(df, window=WINDOW):
    mags = pd.DataFrame({g: df.get(f"rg_{g}", pd.Series(0.0, index=df.index)).abs().fillna(0.0)
                         for g, _, _ in REWARD_GROUPS})
    sums = mags.rolling(window, min_periods=1).sum()
    total = sums.sum(axis=1).replace(0, np.nan)
    return sums.div(total, axis=0).fillna(0.0)


def training_summary(df):
    """Same aggregates as the console rolling block, over the last WINDOW episodes."""
    tail = df.tail(WINDOW)
    if tail.empty:
        return {}
    km = tail["distance_m"].fillna(0).sum() / 1000.0
    steps = tail["length_steps"].fillna(0).sum()
    out = {"n_episodes": len(tail)}
    for key, members, _, _ in OUTCOMES:
        out[f"{key}_frac"] = float(tail["end_reason"].isin(members).mean())
    out["collisions_per_km"] = float(tail["collisions_total"].fillna(0).sum() / km) if km >= 0.1 else np.nan
    out["illegal_lane_crossings_per_km"] = (float(tail["illegal_lane_invasions"].fillna(0).sum() / km)
                                            if km >= 0.1 else np.nan)
    out["route_completion_frac"] = float(tail["route_completion_frac"].mean())
    out["speeding_frac"] = (float((tail["speeding_frac"] * tail["length_steps"]).sum() / steps)
                            if steps > 0 else np.nan)
    out["return_total_mean"] = float(tail["return_total"].mean())
    return out


def steps_per_second(df, n=200):
    tail = df.dropna(subset=["wall_time"]).tail(n)
    if len(tail) < 2:
        return np.nan
    if "run_id" in tail:          # only within the latest process (restarts leave gaps)
        tail = tail[tail["run_id"] == tail["run_id"].iloc[-1]]
    dt = tail["wall_time"].iloc[-1] - tail["wall_time"].iloc[0]
    ds = tail["step"].iloc[-1] - tail["step"].iloc[0]
    return float(ds / dt) if dt > 0 and ds > 0 else np.nan


# ------------------------------------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------------------------------------
_KFMT = matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1e6:g}M" if abs(x) >= 1e6 else f"{x / 1e3:g}k")
_PFMT = matplotlib.ticker.PercentFormatter(1.0, decimals=0)


def fig_to_b64(fig, path):
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor=SURFACE)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _label_stack_ends(ax, x_last, shares_last, keys, min_share=0.04):
    """Direct labels at the right edge of a 100% stack (ink colour, not the series colour)."""
    bottom = 0.0
    for key in keys:
        v = float(shares_last.get(key, 0.0))
        if v >= min_share:
            ax.annotate(f"{key.replace('_', ' ')} {v:.0%}", xy=(x_last, bottom + v / 2), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=8, color=INK2)
        bottom += v


def plot_stack(x, shares, spec, title, ylabel, path):
    keys = [s[0] for s in spec]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.stackplot(x, [shares[k].to_numpy() for k in keys], colors=[s[-2] if len(s) == 4 else s[1] for s in spec],
                 labels=[s[-1] for s in spec], edgecolor=SURFACE, linewidth=0.8)
    ax.set_ylim(0, 1)
    ax.set_xlim(x.iloc[0], x.iloc[-1] if len(x) > 1 else x.iloc[0] + 1)
    ax.yaxis.set_major_formatter(_PFMT)
    ax.xaxis.set_major_formatter(_KFMT)
    ax.set_title(title, loc="left")
    ax.set_xlabel("training step")
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=min(len(keys), 3))
    _label_stack_ends(ax, x.iloc[-1], shares.iloc[-1], keys)
    return fig_to_b64(fig, path)


def plot_outcomes(df, path, name):
    shares = outcome_shares(df)
    return plot_stack(df["step"], shares, OUTCOMES,
                      f"{name}: how training episodes end (share of the last {WINDOW} episodes)",
                      "share of episodes", path)


def plot_reward_split(df, path, name):
    split = reward_split(df)
    return plot_stack(df["step"], split, REWARD_GROUPS,
                      f"{name}: where the reward comes from (share of |reward|, last {WINDOW} episodes)",
                      "share of |reward|", path)


def plot_eval(evals, path, name):
    ev = pd.DataFrame(evals)
    fig, ax = plt.subplots(figsize=(11, 3.4))
    series = [("success_frac", "success rate", "#008300"), ("crash_frac", "crash rate", "#eb6834"),
              ("route_completion_frac", "route completion", "#2a78d6")]
    for key, label, color in series:
        if key in ev:
            ax.plot(ev["step"], ev[key], color=color, marker="o", markersize=5, label=label)
            last = ev[key].dropna()
            if not last.empty:
                ax.annotate(f"{label} {last.iloc[-1]:.0%}", xy=(ev.loc[last.index[-1], "step"], last.iloc[-1]),
                            xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK2)
    ax.set_ylim(-0.02, 1.02)
    ax.yaxis.set_major_formatter(_PFMT)
    ax.xaxis.set_major_formatter(_KFMT)
    ax.set_title(f"{name}: deterministic evaluation (fixed spawns, no exploration noise)", loc="left")
    ax.set_xlabel("training step")
    ax.set_ylabel("share")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    return fig_to_b64(fig, path)


DETAIL_PANELS = [
    ("Route completion (mean)", "share of route",
     "How far along the planned route the car got before the episode ended. Higher is better.",
     lambda d: rolling(d["route_completion_frac"]), "pct"),
    ("Episode return (mean)", "reward",
     "Total reward per episode including the CBF penalty (what SAC maximises). Higher is better.",
     lambda d: rolling(d["return_total"]), None),
    ("Collisions per km", "collisions / km",
     "Collisions per kilometre over the last 100 episodes; blank until 0.1 km was driven. Lower is better.",
     lambda d: rolling_ratio(d["collisions_total"].fillna(0), d["distance_m"].fillna(0) / 1000.0, min_den=0.1),
     None),
    ("Mean speed", "m/s",
     "Average forward speed per episode. Should approach the speed limit without exceeding it.",
     lambda d: rolling(d["mean_speed_mps"]), None),
    ("Speeding share", "share of steps",
     "Share of driving time above the speed limit. Lower is better.",
     lambda d: rolling_ratio(d["speeding_frac"].fillna(0) * d["length_steps"].fillna(0),
                             d["length_steps"].fillna(0)), "pct"),
    ("CBF active", "share of steps",
     "Share of steps where the safety layer had to change the policy's action (a real constraint was "
     "active). Lower means the policy is safe on its own.",
     lambda d: rolling_ratio(d["cbf_intervention_frac"].fillna(0) * d["length_steps"].fillna(0),
                             d["length_steps"].fillna(0)), "pct"),
    ("Trust (mean)", "trust 0-1",
     "Critic-ensemble trust applied by the CBF (1 = critics agree; low trust tightens the safety margins). "
     "Higher means a more confident value estimate.",
     lambda d: rolling(d["trust_mean"]), None),
    ("Closest time-to-collision", "s (capped at 10)",
     "Median over episodes of the smallest time-to-collision reached (10 s = never closing in). Higher is safer.",
     lambda d: d["min_ttc_s"].fillna(10.0).clip(upper=10.0).rolling(WINDOW, min_periods=1).median(), None),
    ("Curriculum stage", "stage",
     "Difficulty stage of the curriculum (0 = empty roads, short routes ... 3 = full traffic).",
     lambda d: d["curriculum_stage"], None),
]


def plot_details(df, path, name, color):
    fig, axes = plt.subplots(3, 3, figsize=(13, 8.2))
    for ax, (title, ylabel, _, fn, fmt) in zip(axes.ravel(), DETAIL_PANELS):
        try:
            y = fn(df)
        except Exception:
            y = None
        if y is not None and y.notna().any():
            if title == "Curriculum stage":
                ax.step(df["step"], y, where="post", color=color)
                ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            else:
                ax.plot(df["step"], y, color=color)
        else:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center", color=MUTED)
        if fmt == "pct":
            ax.yaxis.set_major_formatter(_PFMT)
        ax.xaxis.set_major_formatter(_KFMT)
        ax.set_title(title, loc="left")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("training step")
    fig.suptitle(f"{name}: training episodes, rolling mean over the last {WINDOW} episodes",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    return fig_to_b64(fig, path)


def plot_comparison(runs, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.6))
    extra = iter(EXTRA_COLORS)
    for run in runs:
        color = EXP_COLORS.get(run["name"]) or next(extra, MUTED)
        ev = pd.DataFrame(run["evals"])
        for ax, key in zip(axes, ("success_frac", "route_completion_frac")):
            if not ev.empty and key in ev:
                ax.plot(ev["step"], ev[key], color=color, marker="o", markersize=4, label=run["name"])
                ax.annotate(run["name"], xy=(ev["step"].iloc[-1], ev[key].iloc[-1]), xytext=(6, 0),
                            textcoords="offset points", va="center", fontsize=8, color=INK2)
    for ax, title in zip(axes, ("Evaluation success rate", "Evaluation route completion")):
        ax.set_ylim(-0.02, 1.02)
        ax.yaxis.set_major_formatter(_PFMT)
        ax.xaxis.set_major_formatter(_KFMT)
        ax.set_title(title, loc="left")
        ax.set_xlabel("training step")
        ax.legend(loc="lower right")
    fig.tight_layout()
    return fig_to_b64(fig, path)


# ------------------------------------------------------------------------------------------------
# Report
# ------------------------------------------------------------------------------------------------
def fmt_value(v, fmt):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    if fmt == "pct":
        return f"{v:.0%}"
    return fmt.format(v)


def headline_rows(run):
    """[(label, first, last, arrow_html, definition)] from eval.jsonl (training fallback)."""
    evals = run["evals"]
    if evals:
        first, last = evals[0], evals[-1]
        source = (f"evaluation @ {first.get('step', 0):,} &rarr; @ {last.get('step', 0):,} steps "
                  f"({len(evals)} rounds, {last.get('n_episodes', '?')} episodes each)")
        stages = (first.get("curriculum_stage"), last.get("curriculum_stage"))
    else:
        df = run["episodes"]
        first, last = training_summary(df.head(WINDOW)), training_summary(df)
        source = "training episodes (no evaluation yet - noisier, includes exploration)"
        stages = (df["curriculum_stage"].iloc[0] if not df.empty else None,
                  df["curriculum_stage"].iloc[-1] if not df.empty else None)
    rows = []
    for key, label, fmt, hib, definition in HEADLINE:
        a, b = first.get(key), last.get(key)
        arrow = ""
        if a is not None and b is not None and not (np.isnan(a) or np.isnan(b)) and abs(b - a) > 1e-9:
            good = (b > a) == hib
            arrow = (f" <span style='color:{'#006300' if good else '#d03b3b'}' "
                     f"title='{'improved' if good else 'got worse'}'>{'&#9650;' if b > a else '&#9660;'}</span>")
        rows.append((label, fmt_value(a, fmt), fmt_value(b, fmt), arrow, definition))
    return rows, source, stages


GLOSSARY = [
    ("Training episode", "An episode driven while learning, with exploration noise. Used for the curves."),
    ("Evaluation round", "Every 25,000 steps training pauses and the current policy drives 10 episodes "
                         "deterministically from the same 10 spawn points and destinations. Used for the headline table."),
    ("End reasons", "success = reached the route end; crash = collision with a vehicle, object or pedestrian; "
                    "rule violation = left the route lanes (off-route), drove against traffic (wrong-way) or ran a red "
                    "light; blocked = stood still 15 s with nothing ahead; timeout = ran out of time for the route."),
    ("Route completion", "Share of the planned route covered when the episode ended (0-100%)."),
    ("Collisions per km", "Number of collisions divided by distance driven - fair to fast and slow drivers."),
    ("Reward split", "Share of the absolute reward coming from each group; a group near 100% dominates learning."),
    ("CBF active", "Share of steps where the Control Barrier Function safety layer changed the action because a "
                   "safety constraint (gap to the car ahead, lane edge, speed limit) was active."),
    ("Trust", "exp(-2 x spread of the critic ensemble relative to its mean): 1 = critics agree; low trust makes "
              "the CBF more conservative."),
]


def run_setup_html(run):
    cfg = run["config"]
    if not cfg:
        return ""
    hp = cfg.get("hyperparameters", {})
    env = cfg.get("env", {})
    git = cfg.get("git", {}) or {}
    items = [
        f"reward version <b>{html.escape(str(cfg.get('reward_version', '?')))}</b>",
        f"git <code>{html.escape(str(git.get('commit') or '?')[:10])}</code>",
        f"learning rate {hp.get('learning_rate', '?')}, batch {hp.get('batch_size', '?')}, "
        f"buffer {hp.get('buffer_size', '?')}, gamma {hp.get('gamma', '?')}",
        f"critics: {cfg.get('policy', {}).get('n_critics', '?')}",
    ]
    if env:
        items.append(f"NPC vehicles {env.get('num_npc_vehicles', '?')}, pedestrians {env.get('num_pedestrians', '?')}")
    return "<p class='small'>Setup: " + " &middot; ".join(items) + "</p>"


def main(argv=None):
    global TARGET
    ap = argparse.ArgumentParser(description="Build report.html from episodes.jsonl / eval.jsonl")
    ap.add_argument("--log-root", action="append", default=[],
                    help="Folder holding <experiment>/episodes.jsonl (repeatable). "
                         "Defaults also searched: CARLA-RL-Agents/logs and logs/")
    ap.add_argument("--run", action="append", default=[],
                    help="Explicit run as name=folder (repeatable), e.g. 2q_cnn=D:\\runs\\2q_cnn")
    ap.add_argument("--out", default=None, help=f"Output folder (default {DEFAULT_OUT})")
    ap.add_argument("--target", type=int, default=TARGET, help="Target steps per experiment (for the status)")
    args = ap.parse_args(argv)

    TARGET = args.target
    out = args.out or DEFAULT_OUT
    plots = os.path.join(out, "plots")
    os.makedirs(plots, exist_ok=True)

    runs = []
    for name, folder in find_runs(args.log_root, args.run).items():
        run = load_run(name, folder)
        if run["episodes"].empty:
            print(f"[{name}] {folder}: no episodes yet")
            continue
        print(f"[{name}] {folder}: {len(run['episodes'])} episodes, {len(run['evals'])} evaluation rounds")
        runs.append(run)
    if not runs:
        print("no episodes.jsonl found (use --log-root or --run)")
        return None

    summary_rows, sections, headline_html = [], [], []
    extra = iter(EXTRA_COLORS)
    for run in runs:
        name, df = run["name"], run["episodes"]
        color = EXP_COLORS.get(name) or next(extra, MUTED)
        df.to_csv(os.path.join(out, f"{name}_episodes.csv"), index=False)
        if run["evals"]:
            pd.DataFrame([{k: v for k, v in e.items() if k != "episodes" and not isinstance(v, dict)}
                          for e in run["evals"]]).to_csv(os.path.join(out, f"{name}_eval.csv"), index=False)

        rows, source, stages = headline_rows(run)
        train = training_summary(df)
        summary = {"experiment": name, "status": run["status"], "steps": run["steps"],
                   "description": run["desc"], "eval_rounds": len(run["evals"]),
                   "training_steps_per_s_recent": steps_per_second(df),
                   "model_file": run["final_model"] or "(not finished yet)"}
        first_eval = run["evals"][0] if run["evals"] else {}
        last_eval = run["evals"][-1] if run["evals"] else {}
        for key, *_ in HEADLINE:
            summary[f"eval_first_{key}"] = first_eval.get(key)
            summary[f"eval_last_{key}"] = last_eval.get(key)
            summary[f"train_last{WINDOW}_{key}"] = train.get(key)
        summary_rows.append(summary)

        stage_note = ""
        if stages[0] is not None and stages[1] is not None and stages[0] != stages[1]:
            stage_note = (f"<p class='small'>Note: the first value was measured at curriculum stage S{stages[0]}, "
                          f"the last at S{stages[1]} (harder), so they are not the same difficulty.</p>")
        body = "".join(f"<tr><td title='{html.escape(d)}'>{lab}</td><td>{a}</td><td><b>{b}</b>{arr}</td>"
                       f"<td class='small'>{html.escape(d)}</td></tr>" for lab, a, b, arr, d in rows)
        headline_html.append(
            f"<h3>{name} <small>({html.escape(run['status'])}, {run['steps']:,} steps)</small></h3>"
            f"<p class='small'>Source: {source}</p>{stage_note}"
            f"<table><tr><th>Metric</th><th>Start</th><th>Now</th><th>What it means</th></tr>{body}</table>")

        figs = []
        figs.append((plot_outcomes(df, os.path.join(plots, f"{name}_outcomes.png"), name),
                     "Each band is the share of the last 100 training episodes that ended that way. A good run "
                     "shows the green success band growing from the top while the orange crash band at the "
                     "bottom shrinks."))
        if run["evals"]:
            figs.append((plot_eval(run["evals"], os.path.join(plots, f"{name}_eval.png"), name),
                         "Deterministic evaluation every 25,000 steps (same 10 spawn points and destinations each "
                         "time, no exploration noise). Success and route completion: higher is better; crash rate: "
                         "lower is better. These are the numbers in the headline table."))
        figs.append((plot_reward_split(df, os.path.join(plots, f"{name}_reward_split.png"), name),
                     "Share of the absolute reward from each group. Route progress should dominate; if a penalty "
                     "group dominates, the agent is mostly learning to avoid that penalty."))
        figs.append((plot_details(df, os.path.join(plots, f"{name}_training.png"), name, color),
                     "<ul>" + "".join(f"<li><b>{t}</b> ({u}): {html.escape(g)}</li>"
                                      for t, u, g, _, _ in DETAIL_PANELS) + "</ul>"))
        sections.append((run, figs))

    pd.DataFrame(summary_rows).to_csv(os.path.join(out, "summary.csv"), index=False)

    comp = ""
    with_eval = [r for r in runs if r["evals"]]
    if len(with_eval) > 1:
        b64 = plot_comparison(with_eval, os.path.join(plots, "comparison.png"))
        comp = (f"<h2>Comparison</h2><img alt='evaluation comparison' src='data:image/png;base64,{b64}'>"
                "<p class='caption'>Evaluation success rate and route completion of every experiment over training. "
                "Higher is better for both.</p>")

    section_html = ""
    for run, figs in sections:
        section_html += (f"<h2>{run['name']}</h2><p>{html.escape(run['desc'])}<br>Status: <b>{html.escape(run['status'])}"
                         f"</b> &middot; {run['steps']:,} / {TARGET:,} steps &middot; {len(run['episodes'])} training "
                         f"episodes &middot; logs: <code>{html.escape(run['folder'])}</code></p>{run_setup_html(run)}")
        for b64, caption in figs:
            section_html += (f"<img alt='{html.escape(run['name'])} chart' src='data:image/png;base64,{b64}'>"
                             f"<div class='caption'>{caption}</div>")
    glossary = "".join(f"<tr><td><b>{t}</b></td><td>{html.escape(d)}</td></tr>" for t, d in GLOSSARY)
    models = "".join(f"<li><b>{r['name']}</b>: <code>{html.escape(r['final_model'] or '(not finished yet)')}</code></li>"
                     for r in runs)

    page = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Safety-aware DRL results</title>
<style>
:root{{--ink:{INK};--ink2:{INK2};--muted:{MUTED};--line:{GRID};--surface:{SURFACE};--plane:#f9f9f7}}
body{{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;max-width:1200px;margin:24px auto;padding:0 16px;
color:var(--ink);background:var(--plane)}}
table{{border-collapse:collapse;width:100%;margin:8px 0 16px;background:var(--surface)}}
th,td{{border:1px solid var(--line);padding:6px 8px;font-size:14px;text-align:left;vertical-align:top}}
th{{background:#f0efec}} td:nth-child(2),td:nth-child(3){{font-variant-numeric:tabular-nums;white-space:nowrap}}
img{{max-width:100%;border:1px solid var(--line);margin:12px 0 4px;background:var(--surface)}}
code{{background:#f0efec;padding:1px 4px}} .small{{font-size:13px;color:var(--ink2)}}
.caption{{font-size:13px;color:var(--ink2);margin:0 0 18px}} .note{{background:var(--surface);border:1px solid var(--line);padding:10px 14px}}
</style></head><body>
<h1>Safety-aware Deep RL for Autonomous Driving (CARLA) - Training Results</h1>
<p>Generated {datetime.now():%Y-%m-%d %H:%M}. Soft Actor-Critic agents learn to drive a planned route in the CARLA
simulator (synchronous, 20 FPS) from a front camera plus a navigation vector. A Control Barrier Function (CBF) safety
layer corrects unsafe actions; its margins tighten when the critic ensemble disagrees (low trust).</p>
<div class="note small"><b>How to read this report.</b> The headline tables use the deterministic evaluation rounds
(fixed spawn points, no exploration noise) - these are the numbers to quote. The charts below them use the training
episodes, which are noisier because the agent is still exploring. Green &#9650;/&#9660; = improved, red = got worse.
Every metric is defined in <code>METRICS.md</code>.</div>
<h2>Headline numbers</h2>
{''.join(headline_html)}
<h2>Glossary</h2><table>{glossary}</table>
<h2>Trained model files</h2><ul>{models}</ul>
{comp}{section_html}
</body></html>"""
    path = os.path.join(out, "report.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(page)
    print(f"report written: {path}")
    return path


if __name__ == "__main__":
    main()
