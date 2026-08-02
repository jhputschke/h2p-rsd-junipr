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


def _dataset_calls_in_comprehensions(source: str):
    """`SomeDataset(...)[k] for k in ...` — a dataset CONSTRUCTED inside a comprehension
    over its own items, so it is rebuilt once per item.

    Matched on the AST, not the text, so a comment or a docstring quoting the pattern
    (this repo has several, explaining it) is not a hit."""
    out = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.ListComp, ast.GeneratorExp, ast.SetComp, ast.DictComp)):
            continue
        for sub in ast.walk(node.elt if hasattr(node, "elt") else node.value):
            if not (isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Call)):
                continue
            fn = sub.value.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if name.endswith("Dataset"):
                out.append(f"line {sub.lineno}: {name}(...)[...] inside a comprehension")
    return out


@pytest.mark.skipif(not NOTEBOOKS, reason="no notebooks checked in")
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_no_dataset_is_rebuilt_once_per_item(path):
    """The O(B^2) shape that cost `prod_test_v0.ipynb` ~60 min of its ~109 minutes
    (docs/PLAN_prod_test_speedup.md). It is invisible in review because every character
    of it looks like batching, it changes no number, and profiling the primitives it
    calls exonerates every one of them — the dataset build and the forward pass are both
    fast, and only the loop that wraps them is not."""
    hits = [f"cell {i}: {h}" for i, src in _code_cells(path)
            for h in _dataset_calls_in_comprehensions(src)]
    assert not hits, (
        f"{path.name} constructs a Dataset inside a comprehension over its own items, so "
        f"it is rebuilt once per item (B datasets of B jets):\n  " + "\n  ".join(hits)
        + "\n  Hoist it: `ds_chunk = MatchedLundDataset(chunk, ...)` then index that."
    )
