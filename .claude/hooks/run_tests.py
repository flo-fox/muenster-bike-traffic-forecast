"""Runs the full pytest suite and reports failures via systemMessage.

Invoked from three sites - before a `git commit` (PreToolUse/Bash, gated
by the `if` filter in settings.json to only fire on commit commands), and
before/after the bike-forecast-reviewer subagent runs (SubagentStart/
SubagentStop) - rather than after every single Write/Edit, which produced
a full-suite run on every file change regardless of how small. Warn-only
in all three cases - never blocks, matching this project's existing
PreToolUse pattern (check_review_reminder.py).
"""

import json
import subprocess
import sys
from pathlib import Path

# Must stay below the hook's own "timeout" in .claude/settings.json (60s):
# that outer limit kills this whole process, so a subprocess timeout at or
# above it can never actually fire - this leaves a safety margin so the
# graceful "pytest timed out" report below has a chance to run and be
# reported before the harness's own hard kill would otherwise pre-empt it.
PYTEST_TIMEOUT_SECONDS = 50


def _resolve_python(repo_root: Path) -> str:
    """Return the project's venv interpreter, or fall back to this one.

    A git worktree has no .venv of its own (it's gitignored, so
    `git worktree add` never creates one there) - falling back keeps the
    hook running instead of crashing, even though pytest may then be
    unavailable there.
    """
    venv_python = (
        repo_root
        / ".venv"
        / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    )
    return str(venv_python) if venv_python.exists() else sys.executable


def main() -> None:
    """Run pytest unconditionally and print a systemMessage banner on failure."""
    # Every trigger site for this hook (pre-commit, subagent start/stop)
    # always wants a full run - unlike the old per-edit version, there's
    # nothing left to gate on, so the stdin payload's shape doesn't matter;
    # it's only read so an unconsumed pipe never blocks the caller.
    sys.stdin.read()

    repo_root = Path(__file__).resolve().parents[2]
    python = _resolve_python(repo_root)
    try:
        result = subprocess.run(
            [python, "-m", "pytest", "-q"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=PYTEST_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        print(
            json.dumps(
                {"systemMessage": f"pytest timed out after {PYTEST_TIMEOUT_SECONDS}s"}
            )
        )
        return
    except OSError as exc:
        print(
            json.dumps({"systemMessage": f"Could not run pytest via {python}: {exc}"})
        )
        return

    if result.returncode != 0:
        tail = "\n".join((result.stdout + result.stderr).splitlines()[-15:])
        print(json.dumps({"systemMessage": f"pytest failed:\n{tail}"}))


if __name__ == "__main__":
    main()
