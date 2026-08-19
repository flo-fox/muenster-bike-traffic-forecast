"""PostToolUse hook: flags violations of CLAUDE.md's code/notebook conventions.

Checks, informational only (never blocks):
- src/ Python functions and classes missing a docstring, argument type
  hints, or a return type hint.
- Any file (notebook cell or plain .py, e.g. Streamlit dashboard pages)
  that creates a chart (matplotlib/plotly/streamlit) without a nearby
  data-source caption.
"""

import ast
import json
import re
import sys
from pathlib import Path

CHART_PATTERN = re.compile(
    r"plt\.(plot|scatter|bar|barh|hist|imshow)|\.plot\(|"
    r"go\.(Figure|Scatter|Bar|Scattermap|Histogram|Box|Heatmap|Pie)|"
    r"px\.\w+\(|add_trace|"
    r"st\.(plotly_chart|pyplot|line_chart|bar_chart|area_chart|map)"
)
CAPTION_PATTERN = re.compile(
    r"source|quelle|dwd|od-ms|radverkehr-zaehlstellen", re.IGNORECASE
)


def _defs_to_check(tree: ast.Module) -> list[ast.AST]:
    """Return top-level functions/classes and methods of top-level classes.

    Deliberately skips nested/closure functions, which aren't part of any
    public API surface and shouldn't be held to the same convention.
    """
    nodes: list[ast.AST] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nodes.append(node)
        elif isinstance(node, ast.ClassDef):
            nodes.append(node)
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    nodes.append(child)
    return nodes


def check_python_docstrings(path: Path, tree: ast.Module) -> list[str]:
    """Flag missing docstrings/type hints on top-level defs and methods."""
    issues = []
    for node in _defs_to_check(tree):
        if node.name.startswith("_") and node.name != "__init__":
            continue
        if ast.get_docstring(node) is None:
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            issues.append(
                f"{path.name}:{node.lineno} `{node.name}` ({kind}) missing a docstring"
            )
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        all_args = (
            list(node.args.posonlyargs)
            + list(node.args.args)
            + list(node.args.kwonlyargs)
        )
        missing_hints = [
            a.arg
            for a in all_args
            if a.arg not in ("self", "cls") and a.annotation is None
        ]
        if missing_hints:
            issues.append(
                f"{path.name}:{node.lineno} `{node.name}` missing type hints for: "
                f"{', '.join(missing_hints)}"
            )
        if node.returns is None and node.name != "__init__":
            issues.append(
                f"{path.name}:{node.lineno} `{node.name}` missing a return type hint"
            )
    return issues


def _function_source_blocks(source: str, tree: ast.Module) -> list[str]:
    """Splits `source` into one block per top-level function/method, plus a
    final block for whatever module-level code sits outside any function.

    A whole-file caption check is too coarse for a file with more than one
    chart-creating function (e.g. `dashboard_common.py`'s
    `render_forecast_chart` and `render_station_map`): a caption anywhere
    in the file would satisfy the check for *every* chart, including a
    future chart-creating function added with no caption of its own. This
    mirrors `check_notebook_file`'s per-cell "nearby" window, at
    function-body granularity instead of cell granularity.

    Files with no function defs at all (e.g. the Streamlit `pages/*.py`
    scripts, which are flat top-level code) fall back to one block for the
    whole file - equivalent to the previous whole-file check, which is
    already correct for that shape of file (one chart, one nearby caption).
    """
    lines = source.splitlines(keepends=True)
    func_line_ranges: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", None) or node.lineno
            func_line_ranges.append((node.lineno, end))

    if not func_line_ranges:
        return [source]

    blocks = ["".join(lines[start - 1 : end]) for start, end in func_line_ranges]
    covered = {ln for start, end in func_line_ranges for ln in range(start, end + 1)}
    module_level = "".join(
        line for i, line in enumerate(lines, start=1) if i not in covered
    )
    blocks.append(module_level)
    return blocks


def check_chart_caption(path: Path, source: str) -> list[str]:
    """Flag a chart-creating function/module-level block with no nearby
    data-source caption - see `_function_source_blocks`."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        blocks = [source]
    else:
        blocks = _function_source_blocks(source, tree)

    if any(
        CHART_PATTERN.search(block) and not CAPTION_PATTERN.search(block)
        for block in blocks
    ):
        return [f"{path.name}: chart created without a nearby data-source caption"]
    return []


def check_python_file(path: Path) -> list[str]:
    """Check a .py file: caption near each chart, docstrings/hints only under src/."""
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return []

    issues = check_chart_caption(path, source)
    if "src" in path.parts:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return issues
        issues = check_python_docstrings(path, tree) + issues
    return issues


def check_notebook_file(path: Path) -> list[str]:
    """Check a notebook's code cells for uncaptioned charts."""
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    cells = notebook.get("cells", [])
    issues = []
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if not CHART_PATTERN.search(source):
            continue
        window = [source]
        if i > 0:
            window.append("".join(cells[i - 1].get("source", [])))
        if i + 1 < len(cells):
            window.append("".join(cells[i + 1].get("source", [])))
        if not any(CAPTION_PATTERN.search(w) for w in window):
            issues.append(
                f"{path.name} cell {i}: chart created without a nearby data-source caption"
            )
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
                "systemMessage": f"Convention check flagged {len(issues)} issue(s) in {path.name}",
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": (
                        f"CLAUDE.md convention check found potential issues in {path}:\n{bullets}"
                    ),
                },
            }
        )
    )


if __name__ == "__main__":
    main()
