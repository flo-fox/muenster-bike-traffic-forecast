"""PreToolUse hook: reminds Claude to run bike-forecast-reviewer before committing.

Informational only (never blocks) - a shell hook cannot verify that the
agent actually ran or what it found, so this can only nudge, matching
CLAUDE.md's review checklist and the bike-forecast-reviewer agent's own
"use proactively ... before treating a step as done" directive. Fires on
any Bash call that looks like `git commit`, and checks two sources for
notebook/src changes since either alone can miss a match:
- `git diff --cached --name-only` (the normal case: `git add` already
  ran as its own prior tool call).
- The command string itself (covers a chained `git add X && git commit`
  in one Bash call, where nothing is staged yet when this hook fires).
"""

import json
import re
import subprocess
import sys

COMMIT_PATTERN = re.compile(r"\bgit\s+commit(?:\s|$)")
REVIEWED_PATH_PATTERN = re.compile(
    r"(?:^|[\s\"'])(notebooks/[\w./-]+\.ipynb|src/muenster_bike_forecast/[\w./-]+\.py)"
)


def _staged_files() -> list[str]:
    """Return currently staged file paths, or an empty list on any failure."""
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if line]


def _touches_reviewed_paths(command: str) -> bool:
    """True if staged files or the command text reference notebooks/ or src/."""
    candidates = _staged_files() + REVIEWED_PATH_PATTERN.findall(command)
    return any(
        path.startswith("notebooks/") and path.endswith(".ipynb")
        or path.startswith("src/muenster_bike_forecast/") and path.endswith(".py")
        for path in candidates
    )


def main() -> None:
    """Read the PreToolUse payload from stdin and emit a reminder if warranted."""
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return
    command = payload.get("tool_input", {}).get("command", "")
    if not command or not COMMIT_PATTERN.search(command):
        return
    if not _touches_reviewed_paths(command):
        return

    reason = (
        "This commit touches notebooks/ or src/muenster_bike_forecast/ files. "
        "CLAUDE.md's review checklist and the bike-forecast-reviewer agent's own "
        "\"use proactively\" directive both call for running that agent before "
        "treating this change as done - run it now if it hasn't run yet this turn."
    )
    print(
        json.dumps(
            {
                "systemMessage": "Reminder: run bike-forecast-reviewer before this commit.",
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "allow",
                    "permissionDecisionReason": reason,
                    "additionalContext": reason,
                },
            }
        )
    )


if __name__ == "__main__":
    main()
