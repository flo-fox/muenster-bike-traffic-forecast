"""PostToolUse hook: flags violations of CLAUDE.md's security-analyst checklist.

Checks, informational only (never blocks), all regex-based heuristics
rather than full static analysis - they can be bypassed (aliased
imports, indirect calls, unconventional variable names) and are meant
to catch the common case, not guarantee absence:
- Hardcoded credential-shaped string literals: plain assignments
  (`api_key = "..."`), quoted dict/JSON keys (`{"api-key": "..."}`),
  and `os.environ.get("API_KEY", "...")`/`os.getenv(...)`-style
  hardcoded fallbacks - excluding obvious placeholders.
- Literal eval/exec/pickle.load/pickle.loads calls, which CLAUDE.md
  singles out as unsafe on the fetched CSV/HTTP content this project
  treats as untrusted input. This only matches those exact spellings -
  an aliased import or an indirect call via getattr will not be caught.
"""

import json
import re
import sys
from pathlib import Path

_CRED_WORD = r"api[_-]?key|secret|token|password|passwd|access[_-]?key"
SECRET_ASSIGNMENT_PATTERN = re.compile(
    rf"(?i)(?:[\"']([\w-]*(?:{_CRED_WORD})[\w-]*)[\"']|\b(\w*(?:{_CRED_WORD})\w*)\b)"
    r"\s*[:=]\s*[\"']([^\"']{8,})[\"']"
)
ENV_DEFAULT_PATTERN = re.compile(
    rf"(?i)\.get(?:env)?\(\s*[\"'][\w-]*(?:{_CRED_WORD})[\w-]*[\"']\s*,\s*"
    r"[\"']([^\"']{8,})[\"']"
)
PLACEHOLDER_VALUE_PATTERN = re.compile(
    r"(?i)^(your|xxx|changeme|placeholder|example|dummy|fake|test|<|\{\{|\$)"
)
DANGEROUS_CALL_PATTERN = re.compile(r"\b(eval|exec)\s*\(|pickle\.(load|loads)\s*\(")


def check_source(name: str, source: str) -> list[str]:
    """Return security issues found in a block of Python source text."""
    issues = []
    for match in SECRET_ASSIGNMENT_PATTERN.finditer(source):
        cred_name = match.group(1) or match.group(2)
        value = match.group(3)
        if PLACEHOLDER_VALUE_PATTERN.match(value):
            continue
        line = source.count("\n", 0, match.start()) + 1
        issues.append(f"{name}:{line} possible hardcoded credential (`{cred_name}`)")
    for match in ENV_DEFAULT_PATTERN.finditer(source):
        value = match.group(1)
        if PLACEHOLDER_VALUE_PATTERN.match(value):
            continue
        line = source.count("\n", 0, match.start()) + 1
        issues.append(f"{name}:{line} possible hardcoded credential default")
    for match in DANGEROUS_CALL_PATTERN.finditer(source):
        line = source.count("\n", 0, match.start()) + 1
        issues.append(
            f"{name}:{line} unsafe call on potentially untrusted data: `{match.group(0)}`"
        )
    return issues


def check_python_file(path: Path) -> list[str]:
    """Check a .py file's full source text."""
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return []
    return check_source(path.name, source)


def check_notebook_file(path: Path) -> list[str]:
    """Check a notebook's code cells' source text."""
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    issues = []
    for i, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        issues.extend(check_source(f"{path.name} cell {i}", source))
    return issues


def main() -> None:
    """Read the PostToolUse payload from stdin and report any issues."""
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
    if not file_path:
        return
    path = Path(file_path)

    if path.suffix == ".py":
        issues = check_python_file(path)
    elif path.suffix == ".ipynb":
        issues = check_notebook_file(path)
    else:
        return

    if not issues:
        return

    bullets = "\n".join(f"- {issue}" for issue in issues)
    print(
        json.dumps(
            {
                "systemMessage": f"Security check flagged {len(issues)} issue(s) in {path.name}",
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": (
                        f"CLAUDE.md security-analyst check found potential issues in {path}:\n{bullets}"
                    ),
                },
            }
        )
    )


if __name__ == "__main__":
    main()
