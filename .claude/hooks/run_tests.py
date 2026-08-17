"""PostToolUse hook: runs the full pytest suite after a .py/.ipynb change.

Silent on pass; on failure, feeds the tail of the pytest output back to
Claude (hookSpecificOutput.additionalContext) and surfaces a short banner
to the user (systemMessage).
"""

import json
import subprocess
import sys
from pathlib import Path


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


def _report(system_message: str, context: str) -> None:
    """Emit the hookSpecificOutput/systemMessage JSON payload."""
    print(
        json.dumps(
            {
                "systemMessage": system_message,
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": context,
                },
            }
        )
    )


def main() -> None:
    """Read the PostToolUse payload from stdin and run pytest if relevant."""
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return
    tool_input = payload.get("tool_input", {})
    file_path = (
        tool_input.get("file_path")
        or tool_input.get("notebook_path")
        or payload.get("tool_response", {}).get("filePath")
        or ""
    )
    if not file_path.endswith((".py", ".ipynb")):
        return

    repo_root = Path(__file__).resolve().parents[2]
    python = _resolve_python(repo_root)
    try:
        result = subprocess.run(
            [python, "-m", "pytest", "-q"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        _report(
            "pytest timed out after 120s - see details",
            "The full test suite did not finish within 120s after this "
            "edit. It may be hanging, or has grown too slow to run on "
            "every edit.",
        )
        return
    except OSError as exc:
        _report(
            "Could not run pytest after this change - see details",
            f"Failed to launch pytest via {python}: {exc}",
        )
        return

    if result.returncode != 0:
        tail = "\n".join((result.stdout + result.stderr).splitlines()[-40:])
        _report(
            "pytest failed after this change - see details",
            f"Full test suite failed after this edit:\n{tail}",
        )


if __name__ == "__main__":
    main()
