"""`lund_distribution_closure_prod_test_v0.ipynb` is generated, and must stay that way.

The production test needs closure_v2 run against the held-out file with five constants
changed. A hand-edited copy would be a SECOND definition of the same headline improvement
ratios, and this repo has already been burned by two closure populations drifting apart
(docs/PLAN_prod_test_v0.md §7). So the variant is produced by
`scripts/make_prod_closure_nb.py` and these tests pin the two properties that make it
safe: it is byte-identical to v2 everywhere except the title and the parameter cell, and
the committed file matches what the generator produces right now.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
V2 = REPO / "notebooks" / "lund_distribution_closure_v2.ipynb"
PROD = REPO / "notebooks" / "lund_distribution_closure_prod_test_v0.ipynb"
GEN = REPO / "scripts" / "make_prod_closure_nb.py"

TITLE_CELL = 0
PARAM_CELL = 2


def _gen():
    spec = importlib.util.spec_from_file_location("make_prod_closure_nb", GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sources(path):
    return ["".join(c["source"]) for c in json.loads(path.read_text())["cells"]]


@pytest.mark.skipif(not PROD.exists(), reason="generated notebook not present")
def test_committed_notebook_is_what_the_generator_produces():
    """The `--check` mode, as a unit test: a stale committed notebook is a silently
    wrong one, because nothing about it looks stale in a diff."""
    assert _gen().main(["--check"]) == 0, (
        "run `python scripts/make_prod_closure_nb.py` — the committed notebook no longer "
        "matches the generator (v2 changed, or the generator did)"
    )


@pytest.mark.skipif(not PROD.exists(), reason="generated notebook not present")
def test_only_the_title_and_parameters_differ_from_v2():
    """The whole point: every analysis cell is v2's, byte for byte. If this fails, the
    two notebooks can disagree about the headline ratios."""
    v2, prod = _sources(V2), _sources(PROD)
    assert len(v2) == len(prod), "the generator must not add or drop cells"
    differing = [i for i, (a, b) in enumerate(zip(v2, prod)) if a != b]
    assert differing == [TITLE_CELL, PARAM_CELL], (
        f"cells {differing} differ from v2; only the title ({TITLE_CELL}) and the "
        f"parameter cell ({PARAM_CELL}) may"
    )


@pytest.mark.skipif(not PROD.exists(), reason="generated notebook not present")
def test_the_parameter_cell_changes_exactly_the_five_settings():
    """Named individually, because each one is a way to get a wrong answer quietly:
    the reference file instead of the held-out one, a self-fitted tau, an uncalibrated
    length head."""
    cell = _sources(PROD)[PARAM_CELL]
    for key in ("CKPT_PATH", "ROOT_PATH", "EMPTY_THRESHOLD",
                "LENGTH_TEMPERATURE", "LENGTH_TILT"):
        assert f'_IN["{key}"]' in cell, f"{key} must be read from the run's artifact"
    # ...and none of them may be a literal carried over from v2
    assert "cpp/test_data/jets.root" not in cell, "still pointed at the reference file"
    assert "calibration_v2_walkthrough" not in cell, "still pointed at v2's checkpoint"
    # the values come from their PRIMARY record, not a summary block duplicating them
    for path in ('_M["run"]["checkpoint"]', '_M["run"]["test_path"]',
                 '_M["empty_tree"]["tau"]["value"]',
                 '_M["empty_tree"]["recalibration"]["T"]',
                 '_M["empty_tree"]["recalibration"]["tilt"]'):
        assert path in cell, f"{path} is where the value is recorded; read it there"
    assert "distribution_closure_inputs" not in cell, (
        "a summary block duplicating values recorded elsewhere is a second thing that "
        "can go stale"
    )
    # tau is a quantile of q(0|x), so it only means anything on the scale it was fitted
    # to — and (T, tilt) move that scale's mean by ~3x
    assert 'fitted_under' in cell, (
        "the notebook must check that EMPTY_THRESHOLD was fitted at the (T, tilt) it is "
        "about to apply; otherwise the cut lands in the wrong place with nothing to say so"
    )
    # the guards that catch a v2 default changing underneath this variant
    assert 'MBR_BACKEND != "surrogate"' in cell
    assert "REQUIRE_TRUTH_SPLITTING is False" in cell
    assert "PLANE_NB % 30" in cell


@pytest.mark.skipif(not PROD.exists(), reason="generated notebook not present")
def test_parameter_cell_parses_and_fails_loudly_without_the_artifact(tmp_path):
    """With no `prod_test_v0_metrics.json` reachable it must raise a FileNotFoundError
    naming the notebook to run — not fall back to v2's defaults and quietly report on
    `cpp/test_data/jets.root`, which is the file the checkpoint trained on."""
    import os

    cell = _sources(PROD)[PARAM_CELL]
    compile(cell, "params", "exec")          # syntax, independent of the cwd

    cwd = os.getcwd()
    os.chdir(tmp_path)                        # an empty tree: no runs/ at all
    try:
        with pytest.raises(FileNotFoundError) as exc:
            exec(compile(cell, "params", "exec"), {})
        assert "prod_test_v0.ipynb" in str(exc.value)
    finally:
        os.chdir(cwd)


@pytest.mark.skipif(not PROD.exists(), reason="generated notebook not present")
def test_a_tau_without_its_scale_is_refused(tmp_path):
    """An artifact whose `tau` carries no `fitted_under` predates the scale fix, and its
    tau was fitted on the RAW head. Applying it here, where the head is recalibrated,
    leaves the ranking untouched and the cut in the wrong place — rate ~3x truth, with
    nothing in either notebook to say why. It must refuse, not proceed."""
    import json as _json
    import os

    art = tmp_path / "runs" / "prod_test_v0" / "r" / "prod_test_v0"
    art.mkdir(parents=True)
    (art / "prod_test_v0_metrics.json").write_text(_json.dumps({
        "run": {"checkpoint": "runs/x/best.ckpt", "test_path": "data/test.root",
                "train_path": "data/train.root"},
        "empty_tree": {"tau": {"value": 0.1},                   # no `fitted_under`
                       "recalibration": {"T": 1.372, "tilt": -0.511}},
    }))

    cell = _sources(PROD)[PARAM_CELL]
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        with pytest.raises(AssertionError, match="no scale for its tau"):
            exec(compile(cell, "params", "exec"), {})
    finally:
        os.chdir(cwd)


@pytest.mark.skipif(not PROD.exists(), reason="generated notebook not present")
def test_generated_notebook_is_output_free():
    """It is committed like every other notebook here: stripped."""
    nb = json.loads(PROD.read_text())
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            assert c.get("outputs") == [], f"cell {i} carries outputs"
            assert c.get("execution_count") is None, f"cell {i} carries an execution count"
