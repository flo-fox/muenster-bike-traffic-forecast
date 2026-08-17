"""Scaffold the next numbered notebook under `notebooks/`.

Usage: python scaffold_notebook.py "Title Of The Notebook"

Writes `notebooks/NN_slug.ipynb` with a title cell, the project's
standard `src/`-import bootstrap cell, and a conventions-checklist
reminder cell - following the numbering/structure already established
by `notebooks/01_...` through `notebooks/17_...`.
"""

import json
import re
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

BOOTSTRAP_SOURCE = """\
import sys
from pathlib import Path

import pandas as pd

# Make `src/` importable regardless of whether this notebook is run from
# `notebooks/` (the normal case) or the project root.
_cwd = Path.cwd().resolve()
PROJECT_ROOT = _cwd.parent if _cwd.name == "notebooks" else _cwd
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
"""

CHECKLIST_SOURCE = """\
## Conventions checklist

- [ ] `src/` functions this notebook calls have type hints on every
  signature and a Google-style docstring.
- [ ] Every chart (matplotlib/plotly/streamlit) has a data-source
  caption naming the underlying source(s), e.g. bike counts:
  `od-ms/radverkehr-zaehlstellen`; weather: DWD Open Data.
- [ ] A fresh Restart & Run All succeeds top to bottom.
- [ ] Code cells are formatted with black (`line-length = 88`).
"""


def next_notebook_number() -> int:
    """Return the next unused two-digit notebook prefix."""
    numbers = [
        int(m.group(1))
        for p in NOTEBOOKS_DIR.glob("*.ipynb")
        if (m := re.match(r"(\d+)_", p.name))
    ]
    return max(numbers, default=0) + 1


def slugify(title: str) -> str:
    """Convert a title into a lowercase, underscore-separated slug."""
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    if not slug:
        raise ValueError(f"Title {title!r} has no usable characters for a slug")
    return slug


def build_notebook(number: int, title: str) -> dict:
    """Build the notebook JSON structure for the given number/title."""
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "id": uuid.uuid4().hex[:8],
                "metadata": {},
                "source": [f"# {number:02d} - {title}\n", "\n", "TODO: motivation.\n"],
            },
            {
                "cell_type": "code",
                "id": uuid.uuid4().hex[:8],
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": BOOTSTRAP_SOURCE.splitlines(keepends=True),
            },
            {
                "cell_type": "markdown",
                "id": uuid.uuid4().hex[:8],
                "metadata": {},
                "source": CHECKLIST_SOURCE.splitlines(keepends=True),
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    """Scaffold the next numbered notebook from a CLI title argument."""
    if len(sys.argv) != 2:
        print('Usage: python scaffold_notebook.py "Title Of The Notebook"')
        sys.exit(1)

    title = sys.argv[1]
    number = next_notebook_number()
    slug = slugify(title)
    out_path = NOTEBOOKS_DIR / f"{number:02d}_{slug}.ipynb"
    if out_path.exists():
        raise FileExistsError(f"{out_path} already exists")

    out_path.write_text(
        json.dumps(build_notebook(number, title), indent=1), encoding="utf-8"
    )
    print(f"Created {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
