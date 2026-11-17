"""Task fixtures: workspace snapshots, the agent-facing check command, and hidden-test evaluation."""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks"
OUTPUT_CAP = 1500
IGNORE = shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc")


def load_task(task_id):
    task_dir = TASKS_DIR / task_id
    task = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    task.update(dir=task_dir, workspace=task_dir / "workspace",
                hidden_tests=task_dir / "hidden_tests", reference=task_dir / "reference")
    return task


def _copy(src, dest):
    shutil.copytree(src, dest, ignore=IGNORE, dirs_exist_ok=True)
    return Path(dest)


def make_workspace(task_id, dest_dir):
    return _copy(load_task(task_id)["workspace"], dest_dir)


def apply_reference(task_id, name, workspace_dir):
    """Write reference state `name` (and the states it builds on) over a snapshot workspace."""
    reference = load_task(task_id)["reference"]
    parent = json.loads((reference / "parents.json").read_text(encoding="utf-8"))[name]
    if parent:
        apply_reference(task_id, parent, workspace_dir)
    _copy(reference / name, workspace_dir)


def _run(argv, cwd, timeout_s, **env):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", **env)
    try:
        proc = subprocess.run(argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace", timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout_s}s"
    return proc.returncode, proc.stdout


def _compact(text):
    kept = []
    for line in text.splitlines():
        line = re.sub(r"^[=_]{3,} (.+?) [=_]{3,}$", r"-- \1 --", line.rstrip())
        line = re.sub(r"\s*\[\s*\d+%\]$", "", line)
        line = re.sub(r"^[\w.-]+(?:\\[\w.-]+)+(?=:\d+)", lambda m: m[0].replace("\\", "/"), line)
        if line.strip() and not re.fullmatch(r"\s*[\^~]+\s*", line):
            kept.append(line)
    return "\n".join(kept)


def _cap(text):
    if len(text) <= OUTPUT_CAP:
        return text
    marker = "[... output truncated ...]\n"
    return marker + text[len(text) - OUTPUT_CAP + len(marker):]


def run_check(task_id, workspace_dir, timeout_s=60):
    """Run the task's check steps in order, stopping at the first failure. Returns (code, text)."""
    parts = []
    for command in load_task(task_id)["check_cmd"]:
        argv = [sys.executable] + command.split()[1:]  # "python ..." -> this interpreter
        code, out = _run(argv, workspace_dir, timeout_s, PYTEST_ADDOPTS="--tb=short -p no:cacheprovider")
        parts += ["$ " + command, _compact(out)]
        if code != 0:
            return code, _cap("\n".join(parts + [f"checks FAILED (exit {code})"]))
    return 0, _cap("\n".join(parts + ["all checks passed"]))


def evaluate(task_id, stage, workspace_dir, baseline_keys=None, timeout_s=120):
    """Run a stage's hidden tests on a regenerated copy of the workspace."""
    task = load_task(task_id)
    hidden = task["hidden_tests"] / stage
    source = "".join(p.read_text(encoding="utf-8") for p in sorted(hidden.glob("test_*.py")))
    checks = dict.fromkeys(re.findall(r"^def (test_\w+)", source, re.M), False)
    env = {} if baseline_keys is None else {"DW_BASELINE_KEYS": json.dumps(list(baseline_keys))}
    with tempfile.TemporaryDirectory() as tmp:
        work = _copy(workspace_dir, Path(tmp) / "ws")
        _copy(task["workspace"] / "tests", work / "tests")  # restore the original visible tests
        _copy(hidden, Path(tmp) / "hidden")
        code, out = _run([sys.executable, "tools/gen.py"], work, timeout_s)
        if code != 0:
            return {"passed": False, "checks": checks, "error": "gen failed: " + out[-500:]}
        xml = Path(tmp) / "junit.xml"
        code, out = _run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                          "--junitxml", str(xml), str(Path(tmp) / "hidden")], work, timeout_s, **env)
        if code not in (0, 1) or not xml.exists():
            return {"passed": False, "checks": checks, "error": f"pytest exit {code}: " + out[-500:]}
        for case in ET.parse(xml).iter("testcase"):
            if case.get("name") in checks:
                checks[case.get("name")] = len(case) == 0 or all(c.tag in ("system-out", "system-err")
                                                                 for c in case)
    return {"passed": all(checks.values()), "checks": checks}
