"""Applies the Windows fixes to a fresh copy of safety-aware-drl-carla (update-v1).

Put this file in the repo root (the folder that contains CARLA-RL-Agents), then run:
    python apply_fixes.py
or point it at the repo root:
    python apply_fixes.py D:\\path\\to\\safety-aware-drl-carla-update-v1

Safe to run more than once. Originals are saved next to each file as *.bak.

Fixes:
 1. CARLA-RL-Agents/configuration.py : ENV_SCENARIOS_FILE used a hardcoded E:/Sarosh/... path
                                       -> now relative to the file, works on any PC
 2. CARLA-RL-Agents/env/environment.py : reset() crashed when options is None
                                       -> added a None check
 3. CARLA-RL-Agents/src/map_control.py : load_world got '/Game/Carla/Maps/Town01' and CARLA
                                       0.9.15 said 'not found' -> now passes just 'Town01'
"""
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent


def locate(rel):
    path = ROOT / "CARLA-RL-Agents" / rel
    if path.exists():
        return path
    name = Path(rel).name
    hits = [p for p in ROOT.rglob(name) if "venv" not in p.parts and "CARLA-RL-Agents" in p.parts]
    return hits[0] if hits else None


def patch(rel, fn):
    path = locate(rel)
    if path is None:
        print(f"[NOT FOUND] {rel}")
        return
    text = path.read_text(encoding="utf-8")
    new_text, status = fn(text)
    if new_text == text:
        print(f"[{status}] {path}")
        return
    shutil.copy2(path, str(path) + ".bak")
    path.write_text(new_text, encoding="utf-8")
    print(f"[PATCHED] {path}")


def fix_configuration(text):
    new = "ENV_SCENARIOS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'env', 'simple_scenarios.json')"
    pattern = re.compile(r"^[ \t]*ENV_SCENARIOS_FILE[ \t]*=.*$", re.M)
    if not pattern.search(text):
        return text, "NO ENV_SCENARIOS_FILE LINE"
    if new in text:
        return text, "ALREADY FIXED"
    text = pattern.sub(new, text, count=1)
    if not re.search(r"^import os\b", text, re.M):
        text = "import os\n" + text
    return text, "PATCHED"


def fix_environment(text):
    old = re.compile(r"if\s+options\[\s*['\"]scenario_name['\"]\s*\]\s+is\s+not\s+None\s*:")
    new = "if options is not None and options.get('scenario_name') is not None:"
    if new in text:
        return text, "ALREADY FIXED"
    if not old.search(text):
        return text, "LINE NOT FOUND (already different?)"
    return old.sub(new, text, count=1), "PATCHED"


def fix_map_control(text):
    if "map_path.split('/')[-1]" in text:
        return text, "ALREADY FIXED"
    old = re.compile(r"load_world\(\s*map_path\s*\)")
    if not old.search(text):
        return text, "LINE NOT FOUND (already different?)"
    return old.sub("load_world(map_path.split('/')[-1])", text, count=1), "PATCHED"


print(f"Repo root: {ROOT}\n")
patch("configuration.py", fix_configuration)
patch("env/environment.py", fix_environment)
patch("src/map_control.py", fix_map_control)
print("\nDone. Now install packages:  python -m pip install -r requirements_working.txt")
