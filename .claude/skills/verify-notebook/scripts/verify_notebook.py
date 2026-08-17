"""Verify a notebook still runs cleanly top to bottom (Restart & Run All).

Usage: python verify_notebook.py notebooks/08_gradient_boosting_model.ipynb [timeout_seconds]

Executes the notebook via `jupyter nbconvert --execute` into a throwaway
copy (never in-place, so committed cell outputs are never touched by a
verification run), reports pass/fail, and on failure prints the failing
cell's error name/value.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def verify(notebook_path: Path, timeout_seconds: int) -> tuple[bool, str]:
    """Execute `notebook_path` into a temp copy; return (ok, message).

    `--ExecutePreprocessor.timeout` only bounds a single cell's runtime,
    not kernel startup or the nbconvert process as a whole - a stalled
    kernel (a real, common Jupyter/Windows failure mode) would otherwise
    hang this script indefinitely, so `subprocess.run` also gets its own
    timeout with generous headroom over the per-cell one.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir) / notebook_path.name
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "notebook",
                    "--execute",
                    f"--ExecutePreprocessor.timeout={timeout_seconds}",
                    "--output",
                    str(out_path),
                    str(notebook_path),
                ],
                capture_output=True,
                text=True,
                timeout=timeout_seconds + 60,
            )
        except subprocess.TimeoutExpired:
            return False, (
                f"nbconvert did not finish within {timeout_seconds + 60}s "
                "(kernel likely stalled rather than a cell exceeding its "
                "own timeout)."
            )
        if result.returncode == 0:
            return True, "Restart & Run All succeeded."

        error_cell = _find_error_cell(out_path)
        tail = "\n".join(result.stderr.splitlines()[-15:])
        detail = f"\nFailing cell: {error_cell}" if error_cell else ""
        return False, f"Execution failed.{detail}\n\n{tail}"


def _find_error_cell(out_path: Path) -> str | None:
    """Return a short description of the first cell with an error output."""
    if not out_path.exists():
        return None
    notebook = json.loads(out_path.read_text(encoding="utf-8"))
    for i, cell in enumerate(notebook.get("cells", [])):
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                ename = output.get("ename", "?")
                evalue = output.get("evalue", "")
                return f"cell {i}: {ename}: {evalue}"
    return None


def main() -> None:
    """Parse CLI args and run verification against the given notebook."""
    if len(sys.argv) not in (2, 3):
        print("Usage: python verify_notebook.py <notebook_path> [timeout_seconds]")
        sys.exit(1)

    notebook_path = Path(sys.argv[1])
    timeout_seconds = int(sys.argv[2]) if len(sys.argv) == 3 else 300
    if not notebook_path.exists():
        print(f"No such notebook: {notebook_path}")
        sys.exit(1)

    ok, message = verify(notebook_path, timeout_seconds)
    print(message)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
