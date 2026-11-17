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
    base_dir = task_dir
    if "base" in task:
        base_dir = TASKS_DIR / task["base"]
        base = json.loads((base_dir / "task.json").read_text(encoding="utf-8"))
        if "base" in base:
            raise ValueError("a base task cannot itself have a base")
        for key in ("check_cmd", "trap_paths", "protected", "eval_setup"):
            if key not in task and key in base:
                task[key] = base[key]
    task.update(dir=task_dir, hidden_tests=task_dir / "hidden_tests")
    for key, name in (("workspace", "workspace"), ("reference", "reference"),
                      ("oracle_memory", "oracle_memory.txt")):
        path = task_dir / name
        task[key] = path if path.exists() else base_dir / name
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
    root = str(Path(workspace_dir).resolve())
    for command in load_task(task_id)["check_cmd"]:
        argv = [sys.executable] + command.split()[1:]  # "python ..." -> this interpreter
        code, out = _run(argv, workspace_dir, timeout_s, PYTEST_ADDOPTS="--tb=short -p no:cacheprovider")
        # the model must not see where the workspace lives: plain, posix and repr'd spellings
        for spelling in (root, root.replace("\\", "/"), root.replace("\\", "\\\\")):
            out = re.sub(re.escape(spelling), ".", out, flags=re.I)
        parts += ["$ " + command, _compact(out)]
        if code != 0:
            return code, _cap("\n".join(parts + [f"checks FAILED (exit {code})"]))
    return 0, _cap("\n".join(parts + ["all checks passed"]))


def evaluate(task_id, stage, workspace_dir, baseline_dir=None, timeout_s=120):
    """Run a stage's hidden tests on a regenerated copy of the workspace.
    baseline_dir: a copy of the workspace as it was when the session under test started."""
    task = load_task(task_id)
    roots = [TASKS_DIR / task["base"] / "hidden_tests"] if "base" in task else []
    files = {p.name: p for root in roots + [task["hidden_tests"]] for p in (root / stage).glob("test_*.py")}
    hidden = [files[name] for name in sorted(files)]
    source = "".join(p.read_text(encoding="utf-8") for p in hidden)
    checks = dict.fromkeys(re.findall(r"^def (test_\w+)", source, re.M), False)
    env = {} if baseline_dir is None else {"DW_BASELINE_DIR": str(Path(baseline_dir).resolve())}
    with tempfile.TemporaryDirectory() as tmp:
        work = _copy(workspace_dir, Path(tmp) / "ws")
        for name in task["protected"]:  # never run checker code the agent could have edited
            shutil.rmtree(work / name, ignore_errors=True)
            _copy(task["workspace"] / name, work / name)
        if "base" in task:
            (Path(tmp) / "hidden").mkdir()
            for path in hidden:
                shutil.copy2(path, Path(tmp) / "hidden" / path.name)
        else:
            _copy(task["hidden_tests"] / stage, Path(tmp) / "hidden")
        for command in task.get("eval_setup", []):  # e.g. regenerate derived files before scoring
            code, out = _run([sys.executable] + command.split()[1:], work, timeout_s)
            if code != 0:
                return {"passed": False, "checks": checks, "error": f"{command} failed: " + out[-500:]}
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
