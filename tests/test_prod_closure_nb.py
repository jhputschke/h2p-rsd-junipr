"""`lund_distribution_closure_prod_test_v*.ipynb` are generated, and must stay that way.

The production test needs closure_v2 run against the held-out file with five constants
changed (six for v1, which also switches the MBR backend). A hand-edited copy would be a
SECOND definition of the same headline improvement ratios, and this repo has already been
burned by two closure populations drifting apart (docs/PLAN_prod_test_v0.md §7). So the
variants are produced by `scripts/make_prod_closure_nb.py` and these tests pin the two
properties that make that safe: each is byte-identical to v2 everywhere except the title
and the parameter cell, and each committed file matches what the generator produces now.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
V2 = REPO / "notebooks" / "lund_distribution_closure_v2.ipynb"
GEN = REPO / "scripts" / "make_prod_closure_nb.py"
TAGS = ["v0", "v1", "edit"]

# Which artifact each variant reads, and which notebook writes it. v0 and v1 each have a
# production-test notebook of their own; `edit` does NOT — the edit grid reuses
# `notebooks/prod_test_v1.ipynb` pointed at a different run root, so its artifact is still
# named `prod_test_v1_metrics.json` while living under `runs/prod_test_edit/`. Deriving the
# artifact name from the variant name is exactly the assumption that breaks there.
READS = {"v0": ("prod_test_v0", "prod_test_v0.ipynb"),
         "v1": ("prod_test_v1", "prod_test_v1.ipynb"),
         "edit": ("prod_test_v1", "prod_test_v1.ipynb")}

# A directory each variant's OWN artifact search will actually find, relative to a tmp
# cwd. `edit` pins both of its globs inside runs/prod_test_edit/e_v2_s0/, so an artifact
# written where v0's or v1's would go is invisible to it — which is the point of the pin.
def art_dir(tag):
    metrics_tag = READS[tag][0]
    root = "runs/prod_test_edit/e_v2_s0" if tag == "edit" else "runs/prod_test_v0"
    return f"{root}/r/{metrics_tag}"

TITLE_CELL = 0
PARAM_CELL = 2


def _gen():
    spec = importlib.util.spec_from_file_location("make_prod_closure_nb", GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prod(tag: str) -> Path:
    return REPO / "notebooks" / f"lund_distribution_closure_prod_test_{tag}.ipynb"


def _sources(path):
    return ["".join(c["source"]) for c in json.loads(path.read_text())["cells"]]


def _skip_missing(tag):
    if not prod(tag).exists():
        pytest.skip(f"generated notebook {prod(tag).name} not present")


def test_committed_notebooks_are_what_the_generator_produces():
    """The `--check` mode, as a unit test: a stale committed notebook is a silently
    wrong one, because nothing about it looks stale in a diff."""
    for tag in TAGS:
        _skip_missing(tag)
    assert _gen().main(["--check"]) == 0, (
        "run `python scripts/make_prod_closure_nb.py` — a committed notebook no longer "
        "matches the generator (v2 changed, or the generator did)"
    )


def test_every_variant_the_generator_knows_about_is_committed():
    """A variant added to the generator but never written is a notebook that exists in
    review and not on disk."""
    for tag in _gen().VARIANTS:
        assert prod(tag).exists(), (
            f"the generator defines variant {tag!r} but "
            f"{prod(tag).name} is not committed — run the generator"
        )


@pytest.mark.parametrize("tag", TAGS)
def test_only_the_title_and_parameters_differ_from_v2(tag):
    """The whole point: every analysis cell is v2's, byte for byte. If this fails, the
    notebooks can disagree about the headline ratios."""
    _skip_missing(tag)
    v2, p = _sources(V2), _sources(prod(tag))
    assert len(v2) == len(p), "the generator must not add or drop cells"
    differing = [i for i, (a, b) in enumerate(zip(v2, p)) if a != b]
    assert differing == [TITLE_CELL, PARAM_CELL], (
        f"cells {differing} differ from v2; only the title ({TITLE_CELL}) and the "
        f"parameter cell ({PARAM_CELL}) may"
    )


@pytest.mark.parametrize("tag", TAGS)
def test_the_parameter_cell_changes_exactly_the_five_settings(tag):
    """Named individually, because each one is a way to get a wrong answer quietly:
    the reference file instead of the held-out one, a self-fitted tau, an uncalibrated
    length head."""
    _skip_missing(tag)
    cell = _sources(prod(tag))[PARAM_CELL]
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
    # ...and the same discipline for every other fitted inference-layer scalar
    # (docs/PLAN_prod_test_v1.md WP-D.4). `continue_temperature` is fitted against a
    # specific N-marginal, so applying one without its provenance repeats the tau bug.
    assert "continue_temperature" in cell, (
        "a fitted continue_temperature must not be applied without a fitted_under record "
        "— the same failure tau had"
    )
    # the ln z head the artifact was written for must match the checkpoint it names: the
    # support audit's zeros mean different things under `legacy` and `physical`
    assert "lnz_support" in cell
    # the guards that catch a v2 default changing underneath this variant
    assert 'MBR_BACKEND != "surrogate"' in cell
    assert "REQUIRE_TRUTH_SPLITTING is False" in cell
    assert "PLANE_NB % 30" in cell


@pytest.mark.parametrize("tag", TAGS)
def test_each_variant_reads_its_own_artifact(tag):
    """v0 and v1 are different RNG regimes of the same assessment, so pointing one at the
    other's `prod_test_v*_metrics.json` would silently mix them. `edit` shares v1's
    artifact NAME but not its run root, so what has to be pinned there is the ROOT."""
    _skip_missing(tag)
    cell = _sources(prod(tag))[PARAM_CELL]
    metrics_tag, source_nb = READS[tag]
    assert f"{metrics_tag}_metrics.json" in cell
    assert f"notebooks/{source_nb}" in cell
    for other, (other_tag, _nb) in READS.items():
        if other_tag != metrics_tag:
            assert f"{other_tag}_metrics.json" not in cell


def test_the_edit_variant_is_pinned_to_the_arm_gate_e8_selected():
    """`edit` is the one variant whose run root holds NINE arms, so "newest artifact under
    the root" is not a choice — it is whichever arm happened to be deep-passed last, and
    this run deep-passed the LOSING stage (`e_v1_s0`) first. Both globs must also stay
    inside `runs/prod_test_edit/`: the default fallback is `runs/prod_test_v*`, which does
    not match it, so an unpinned fallback would land on an `ar_junipr` checkpoint — a
    wrong-FAMILY silent repoint, which is the failure this whole cell exists to prevent."""
    _skip_missing("edit")
    cell = _sources(prod("edit"))[PARAM_CELL]
    assert 'runs/prod_test_edit/e_v2_s0/**' in cell, "not pinned to the E8 winner"
    assert "runs/prod_test_v*/**" not in cell, (
        "the fallback glob can reach an ar_junipr run root — a different FAMILY, not "
        "merely a different arm"
    )
    for g in ('_own = _find_artifacts("runs/prod_test_edit/',
              '_any = _find_artifacts("runs/prod_test_edit/'):
        assert g in cell, f"missing or unpinned: {g}"


def test_only_v1_switches_the_mbr_backend():
    """v0 stays on `pot` deliberately: EnergyFlow reports the same EMD on its
    R-normalised scale, so `mbr_risk_mean` would change by 1/8.485 under the committed
    v0 artifact — same tree, same ratios, a different recorded risk."""
    for tag in TAGS:
        _skip_missing(tag)
    v0, v1 = (_sources(prod(t))[PARAM_CELL] for t in ("v0", "v1"))
    assert 'MBR_BACKEND           = "pot"' in v0
    assert 'find_spec("energyflow")' in v1 and '"pot"' in v1, (
        "v1 must prefer energyflow AND still fall back to pot where it is absent"
    )


@pytest.mark.parametrize("tag", TAGS)
def test_parameter_cell_parses_and_fails_loudly_without_the_artifact(tag, tmp_path):
    """With no `prod_test_v*_metrics.json` reachable it must raise a FileNotFoundError
    naming the notebook to run — not fall back to v2's defaults and quietly report on
    `cpp/test_data/jets.root`, which is the file the checkpoint trained on."""
    import os

    _skip_missing(tag)
    cell = _sources(prod(tag))[PARAM_CELL]
    compile(cell, "params", "exec")          # syntax, independent of the cwd

    cwd = os.getcwd()
    os.chdir(tmp_path)                        # an empty tree: no runs/ at all
    try:
        with pytest.raises(FileNotFoundError) as exc:
            exec(compile(cell, "params", "exec"), {})
        # the notebook it names is the one that WRITES the artifact, which for `edit` is
        # v1's — the edit grid has no production-test notebook of its own
        assert READS[tag][1] in str(exc.value)
    finally:
        os.chdir(cwd)


@pytest.mark.parametrize("tag", TAGS)
def test_a_tau_without_its_scale_is_refused(tag, tmp_path):
    """An artifact whose `tau` carries no `fitted_under` predates the scale fix, and its
    tau was fitted on the RAW head. Applying it here, where the head is recalibrated,
    leaves the ranking untouched and the cut in the wrong place — rate ~3x truth, with
    nothing in either notebook to say why. It must refuse, not proceed."""
    import json as _json
    import os

    _skip_missing(tag)
    art = tmp_path / art_dir(tag)
    art.mkdir(parents=True)
    (art / f"{READS[tag][0]}_metrics.json").write_text(_json.dumps({
        "run": {"checkpoint": "runs/x/best.ckpt", "test_path": "data/test.root",
                "train_path": "data/train.root"},
        "empty_tree": {"tau": {"value": 0.1},                   # no `fitted_under`
                       "recalibration": {"T": 1.372, "tilt": -0.511}},
    }))

    cell = _sources(prod(tag))[PARAM_CELL]
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        with pytest.raises(AssertionError, match="no scale for its tau"):
            exec(compile(cell, "params", "exec"), {})
    finally:
        os.chdir(cwd)


@pytest.mark.parametrize("tag", TAGS)
def test_generated_notebook_is_output_free(tag):
    """It is committed like every other notebook here: stripped."""
    _skip_missing(tag)
    nb = json.loads(prod(tag).read_text())
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            assert c.get("outputs") == [], f"cell {i} carries outputs"
            assert c.get("execution_count") is None, f"cell {i} carries an execution count"
