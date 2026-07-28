"""Every notebook code cell must be syntactically valid Python.

Notebooks are nbstripped in git, so a broken cell carries no traceback in the diff
and only surfaces when someone runs it — potentially long after the change that
introduced it. These notebooks are also partly machine-generated, and a quoting slip
at the generator layer emits a cell that looks fine in review and fails at execution
(an unterminated f-string shipped exactly this way).

Parsing is cheap and catches that whole class. It does NOT execute anything: the
walkthroughs train models and take tens of minutes, which belongs in a manual run,
not the unit suite.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))


def _code_cells(path: Path):
    nb = json.loads(path.read_text())
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") == "code":
            yield i, "".join(cell.get("source", []))


@pytest.mark.skipif(not NOTEBOOKS, reason="no notebooks checked in")
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_code_cells_parse(path):
    failures = []
    for index, source in _code_cells(path):
        try:
            ast.parse(source)
        except SyntaxError as exc:
            line = source.splitlines()[exc.lineno - 1] if exc.lineno else ""
            failures.append(f"cell {index} line {exc.lineno}: {exc.msg}\n    {line.strip()}")
    assert not failures, f"{path.name} has unparseable code cells:\n" + "\n".join(failures)


@pytest.mark.skipif(not NOTEBOOKS, reason="no notebooks checked in")
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_outputs_are_stripped(path):
    """nbstripout runs as a pre-commit hook; a notebook committed around it would
    carry megabytes of embedded PNGs and make every later diff unreadable."""
    nb = json.loads(path.read_text())
    dirty = [
        i for i, cell in enumerate(nb.get("cells", []))
        if cell.get("cell_type") == "code"
        and (cell.get("outputs") or cell.get("execution_count") is not None)
    ]
    assert not dirty, f"{path.name} has un-stripped outputs in cells {dirty}"
