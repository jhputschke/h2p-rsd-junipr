"""NAMING: the `v0`/`v1` tags here are NOTEBOOK versions of the SAME v0 assessment
(`v1` is the sped-up rerun, docs/PLAN_prod_test_speedup.md). They have nothing to do
with docs/PLAN_prod_test_v1.md, which is a grid of 11 trainings under runs/prod_test_v1/
assessed by scripts/prod_test_v1_gates.py. The collision is historical.

Generate the production-test variants of the v2 distribution-closure notebook.

The production test needs `lund_distribution_closure_v2.ipynb` pointed at the held-out
file with a handful of constants changed, three of which are not obvious and one of which
(`EMPTY_THRESHOLD`) is circular if left at its default. Asking a reader to apply those by
hand invites exactly the mistake the run is trying to avoid.

The alternative — committing a hand-edited copy — is worse: it would be a SECOND
definition of the headline improvement ratios, and this repo has already been burned by
two closure populations drifting apart (docs/PLAN_prod_test_v0.md §7). So the variants are
GENERATED: every analysis cell is copied byte-for-byte from v2, and only the section-0
parameter cell differs. `tests/test_prod_closure_nb.py` re-runs this generator and fails
if a committed notebook does not match, so they cannot diverge silently.

The values are NOT baked in as literals either. They are read at runtime from the
`prod_test_v*_metrics.json` the matching `notebooks/prod_test_v*.ipynb` writes in its §9 —
so the frozen `tau` and the fitted `(temperature, tilt)` can never disagree with the §6
fit that produced them.

TWO variants, one per production-test notebook:

* **v0** — pairs with `notebooks/prod_test_v0.ipynb`, and changes nothing but those five
  settings. It stays on `mbr_backend="pot"` deliberately: the committed
  `dist_closure_metrics.json` beside the v0 checkpoint records `mbr_risk_mean` on POT's
  scale, and EnergyFlow reports the same distance R-normalised (1/8.485 of it). Same
  tree, same ratios, different recorded risk — not a thing to change under an artifact.
* **v1** — pairs with `notebooks/prod_test_v1.ipynb` and additionally switches
  `MBR_BACKEND` to `energyflow` where it is installed, which is Part B of
  docs/PLAN_prod_test_speedup.md: ~3.5x on the MBR stage, ~1.55x on the whole pass, and
  measured to pick a bit-identical MBR tree on 100% of 200 held-out jets.

    python scripts/make_prod_closure_nb.py            # write both notebooks
    python scripts/make_prod_closure_nb.py --check    # verify they are up to date
    python scripts/make_prod_closure_nb.py --variant v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "notebooks" / "lund_distribution_closure_v2.ipynb"


def out_path(tag: str) -> Path:
    return REPO / "notebooks" / f"lund_distribution_closure_prod_test_{tag}.ipynb"


# The section-0 lines every variant replaces, and what they become. Each must match
# EXACTLY once: a silent zero-match would ship a notebook still pointed at the reference
# file, which is the failure mode the generator exists to prevent.
SUBS = [
    ('CKPT_PATH   = "runs/calibration_v2_walkthrough/ar_junipr_v4/best.ckpt"',
     'CKPT_PATH   = _IN["CKPT_PATH"]'),
    ('ROOT_PATH   = "cpp/test_data/jets.root"   # None -> synthetic fallback',
     'ROOT_PATH   = _IN["ROOT_PATH"]            # the INDEPENDENT file, not the trained-on one'),
    ('EMPTY_THRESHOLD       = None',
     'EMPTY_THRESHOLD       = float(_IN["EMPTY_THRESHOLD"])   # FROZEN, not rate-matched here'),
    ('LENGTH_TEMPERATURE    = None',
     'LENGTH_TEMPERATURE    = float(_IN["LENGTH_TEMPERATURE"])'),
    ('LENGTH_TILT           = None',
     'LENGTH_TILT           = float(_IN["LENGTH_TILT"])'),
]

# v1 only. The line is left alone in v0 — see this module's docstring for why.
MBR_SUB = (
    'MBR_BACKEND           = "pot"',
    'MBR_BACKEND           = "energyflow" if _ilu.find_spec("energyflow") else "pot"\n'
    '#                       ^ Part B of docs/PLAN_prod_test_speedup.md. The SAME\n'
    '#                       perturbative-Lund EMD: measured ~3.5x faster on the MBR stage\n'
    '#                       and ~1.55x on the whole pass, choosing a BIT-IDENTICAL MBR tree\n'
    '#                       on 100% of 200 held-out jets. Falls back to "pot" where\n'
    '#                       energyflow is not installed, and the artifact records which one\n'
    '#                       ran — `mbr_risk_mean` is on the backend\'s own scale (EnergyFlow\n'
    '#                       normalises by R), even though the SELECTION is identical.'
)

PRELUDE = '''# ===========================================================================
# GENERATED — do not hand-edit. Regenerate with:
#     python scripts/make_prod_closure_nb.py
# Every cell below section 0 is byte-identical to
# notebooks/lund_distribution_closure_v2.ipynb; only this cell differs, so the two
# notebooks cannot drift into two definitions of the same headline number.
# ===========================================================================
#
# The production-test settings are READ FROM THE RUN'S OWN ARTIFACT rather than pasted
# in, so the frozen `tau` and the fitted `(temperature, tilt)` cannot disagree with the
# section-6 fit that produced them. Run notebooks/{NB} first.
import importlib.util as _ilu
import json as _json
from pathlib import Path as _Path

# None -> the newest {TAG}_metrics.json, preferring {RUNROOT}/ (see below).
PROD_METRICS_PATH = None

_REPO = _Path.cwd().parent if _Path.cwd().name == "notebooks" else _Path.cwd()


def _find_artifacts(pattern):
    return sorted(_REPO.glob(pattern), key=lambda q: q.stat().st_mtime)


if PROD_METRICS_PATH:
    _mp = _Path(PROD_METRICS_PATH)
    _mp = _mp if _mp.is_absolute() else _REPO / _mp
else:
    # `**`, not `*`: the run root's DEPTH is not fixed. A single-arm run writes
    # runs/prod_test_v0/<stamp>/prod_test_v0/..., but a GRID gives each arm its own root
    # (scripts/run_prod_test_v1.sh: run_root=<root>/<arm>), which adds a level. The old
    # fixed-depth pattern missed every grid arm — and did not fail, it silently resolved
    # to whatever single-arm artifact happened to be newest, so this notebook would have
    # reported on a DIFFERENT CHECKPOINT than its caller intended.
    #
    # And prefer this tag's OWN run root before falling back to any: {RUNROOT}/ wins,
    # which is what makes this notebook usable against the production test of the same
    # name rather than against whichever artifact happened to be written most recently
    # anywhere. Without it the choice rests on mtime alone, so re-running a different
    # notebook would silently repoint this one at another checkpoint.
    _own = _find_artifacts("{RUNROOT}/**/{TAG}/{TAG}_metrics.json")
    _any = _find_artifacts("runs/prod_test_v*/**/{TAG}/{TAG}_metrics.json")
    _found = _own or _any
    if not _found:
        raise FileNotFoundError(
            "no {TAG}_metrics.json under runs/. This notebook takes its checkpoint, its "
            "test file, the frozen empty-tree tau and the fitted length recalibration "
            "from that artifact — run notebooks/{NB} first, or set PROD_METRICS_PATH."
        )
    _mp = _found[-1]
    if _own:
        print(f"[{TAG}] using {RUNROOT}/ (this tag's own run root); "
              f"{len(_any) - len(_own)} artifact(s) elsewhere were NOT used")

_M = _json.loads(_mp.read_text())
# Read the PRIMARY record of each value, not a summary block duplicating it: the tau and
# the (T, tilt) live where section 6 fitted them, and a second copy is a second thing that
# can be stale. This also means any prod_test artifact works, including ones written
# before this notebook existed.
try:
    _IN = {
        "CKPT_PATH": _M["run"]["checkpoint"],
        "ROOT_PATH": _M["run"]["test_path"],
        "EMPTY_THRESHOLD": _M["empty_tree"]["tau"]["value"],
        "LENGTH_TEMPERATURE": _M["empty_tree"]["recalibration"]["T"],
        "LENGTH_TILT": _M["empty_tree"]["recalibration"]["tilt"],
    }
except KeyError as _e:
    raise KeyError(
        f"{_mp} has no {_e} — it is not a {TAG} metrics file, or predates the "
        f"section-6 recalibration. Re-run notebooks/{NB}."
    ) from None
print(f"[{TAG}] settings from {_mp.relative_to(_REPO)}")
for _k, _v in _IN.items():
    print(f"    {_k:<20} = {_v!r}")

'''

# Constants this variant relies on but does NOT set — they are already correct in v2, and
# an assertion is what keeps a future edit to v2's defaults from silently breaking this
# run rather than failing it.
EPILOGUE = '''
# --- what this variant relies on v2 already getting right --------------------
# Asserted, not assumed: these are v2 defaults today, and a future change to them would
# otherwise silently alter the production-test numbers instead of failing here.
assert MBR_BACKEND != "surrogate", (
    "the surrogate is a different risk function and, before the n_bins fix, a coarser "
    "one — no reported number may use it"
)
assert REQUIRE_TRUTH_SPLITTING is False, (
    "the production test reports on the DEPLOYABLE population (len(x) > 0); requiring a "
    "truth splitting selects on the answer"
)
assert PLANE_NB % 30 == 0, (
    f"PLANE_NB={PLANE_NB} must stay a multiple of geometry.n_bins (30) so the Lund plane "
    f"edges remain a strict subset of the model's own cells"
)
assert str(_IN["ROOT_PATH"]) != str(_M["run"]["train_path"]), (
    "the eval file is the file this checkpoint TRAINED on — that is not a closure test"
)

# THE scale check. `EMPTY_THRESHOLD` is a QUANTILE of q(0|x), so it only means anything
# on the distribution it was fitted to — and (T, tilt) move that distribution's mean by
# ~3x. Fitting on the raw head and applying here, where the head is recalibrated, leaves
# the RANKING untouched and the CUT in the wrong place: the rate goes to ~3x truth and
# precision collapses, with nothing in either notebook to say why.
_under = _M["empty_tree"]["tau"].get("fitted_under")
assert _under is not None, (
    "this artifact records no scale for its tau (a prod_test run predating the fix). "
    "Re-run notebooks/{NB}: a tau without its scale cannot be applied."
)
assert (abs(float(_under["length_temperature"]) - LENGTH_TEMPERATURE) < 1e-9
        and abs(float(_under["length_tilt"]) - LENGTH_TILT) < 1e-9), (
    f"EMPTY_THRESHOLD was fitted at (T, tilt) = "
    f"({_under['length_temperature']}, {_under['length_tilt']}) but this notebook applies "
    f"({LENGTH_TEMPERATURE}, {LENGTH_TILT}). A tau is a quantile of q(0|x); on a "
    f"different scale it cuts in the wrong place."
)

# The SAME rule for the sampling-time continue temperature (docs/PLAN_prod_test_v1.md
# WP-B.2/WP-D.4). It is fitted by matching a held-out N-marginal mean, so it is as
# scale-dependent as tau is: a T fitted on the training-val split and applied to a test
# file whose N-marginal has drifted is measuring transfer, and must SAY so rather than be
# quoted as a fit. Absent from the artifact means "never fitted", which is fine only as
# long as the notebook is not about to apply one.
_ct = _M.get("continue_temperature")
_CONTINUE_TEMPERATURE = float((_ct or {}).get("value", 1.0))
if _CONTINUE_TEMPERATURE != 1.0:
    assert _ct is not None and _ct.get("fitted_under") is not None, (
        f"this notebook would apply continue_temperature={_CONTINUE_TEMPERATURE} with no "
        f"record of what it was fitted under. Like tau, it is fitted against a specific "
        f"N-marginal; without its provenance it cannot be applied."
    )
    print(f"    {'CONTINUE_TEMPERATURE':<20} = {_CONTINUE_TEMPERATURE!r}"
          f"   (fitted under {_ct['fitted_under']})")

# The `ln z` head this artifact was produced by. The support audit's zeros mean two
# different things — "the head cannot leave the interval" under `physical`, "it happened
# not to" under `legacy` — so a closure report that names one while the checkpoint carries
# the other is describing a model that does not exist.
_ART_LNZ = _M.get("lnz_support")
if _ART_LNZ is not None:
    from omegaconf import OmegaConf as _OC

    from h2p_rsd_junipr.train.checkpoint import load_for_inference as _lfi
    _ckpt_lnz = str(_OC.select(_OC.create(_lfi(str(_IN["CKPT_PATH"]),
                                               map_location="cpu")["config"]),
                               "model.lnz_support") or "legacy")
    assert _ckpt_lnz == str(_ART_LNZ), (
        f"the artifact was written for lnz_support={_ART_LNZ!r} but the checkpoint it "
        f"names carries {_ckpt_lnz!r}. The support audit and every ln z panel below mean "
        f"different things under the two heads (docs/PLAN_prod_test_v1.md WP-A)."
    )
    print(f"    {'LNZ_SUPPORT':<20} = {_ckpt_lnz!r}")
'''

# Per-variant: which extra substitutions apply, and the one paragraph of the title cell
# that says what is different about it.
VARIANTS = {
    "v0": {
        "subs": [],
        "backend_row": "",
    },
    "v1": {
        "subs": [MBR_SUB],
        "backend_row": (
            "| `MBR_BACKEND` | `energyflow` where installed: the **same**"
            " perturbative-Lund EMD, measured ~1.55x faster over the whole pass and"
            " bit-identical in its chosen tree on 100% of 200 held-out jets"
            " ([`docs/PLAN_prod_test_speedup.md`](../docs/PLAN_prod_test_speedup.md) Part"
            " B). Falls back to `pot`; the artifact records which ran |\n"
        ),
    },
}


def header(tag: str) -> str:
    v = VARIANTS[tag]
    return (
        f"# Lund distribution closure — production test {tag}\n"
        "\n"
        "[`lund_distribution_closure_v2.ipynb`](lund_distribution_closure_v2.ipynb),"
        " run against the **held-out** file of\n"
        "[`docs/PLAN_prod_test_v0.md`](../docs/PLAN_prod_test_v0.md) with every"
        " setting already applied.\n"
        "\n"
        "**Generated** by [`scripts/make_prod_closure_nb.py`](../scripts/make_prod_closure_nb.py);"
        " every cell below\n"
        "section 0 is byte-identical to v2. Edit v2 (or the generator) and regenerate"
        " — a hand-edited\n"
        "copy would be a second definition of the same headline ratios, which is"
        " exactly how two\n"
        "closure populations drifted apart before.\n"
        "\n"
        "These settings differ from v2's defaults, and the first five are read from"
        f" `prod_test_{tag}_metrics.json`\n"
        "rather than pasted in, so they cannot disagree with the fit that produced"
        " them:\n"
        "\n"
        "| constant | why it must change |\n"
        "|---|---|\n"
        "| `CKPT_PATH` | the production-test checkpoint |\n"
        "| `ROOT_PATH` | the independent test file — v2 defaults to"
        " `cpp/test_data/jets.root` |\n"
        "| `EMPTY_THRESHOLD` | the **frozen** tau from the training val split; v2's"
        " `None` rate-matches on the sample it reports on, which is circular |\n"
        "| `LENGTH_TEMPERATURE` | fitted on the training val split; reaches"
        " `length_pmf` **and** `sample` |\n"
        "| `LENGTH_TILT` | the same fit. A scalar temperature is symmetric about"
        " the mode, so it cannot move `q(0\\|x)` the way a monotone ramp in `n`"
        " requires — whenever the head needs correcting at all |\n"
        + v["backend_row"] +
        "\n"
        f"Run [`notebooks/prod_test_{tag}.ipynb`](prod_test_{tag}.ipynb) first; this reads"
        " its artifact.\n"
    )


def build(tag: str) -> dict:
    nb = json.loads(SRC.read_text())
    # keyed on the ASSIGNMENT, not the name: the loader cell downstream mentions
    # `CKPT_PATH` too, and matching that would pick two cells
    params = [c for c in nb["cells"]
              if c["cell_type"] == "code" and SUBS[0][0] in "".join(c["source"])]
    if len(params) != 1:
        raise SystemExit(f"expected exactly one parameter cell in {SRC.name}, found {len(params)}")
    cell = params[0]

    body = "".join(cell["source"])
    for old, new in SUBS + VARIANTS[tag]["subs"]:
        if body.count(old) != 1:
            raise SystemExit(
                f"{SRC.name} parameter line changed or vanished — {body.count(old)} matches "
                f"for:\n  {old}\nUpdate scripts/make_prod_closure_nb.py to match."
            )
        body = body.replace(old, new)
    fill = {"{TAG}": f"prod_test_{tag}", "{NB}": f"prod_test_{tag}.ipynb",
            # the tag's own run root. NOT "runs/prod_test_{TAG}" -- {TAG} is
            # already `prod_test_<tag>`, so that spells prod_test_prod_test_v1.
            "{RUNROOT}": f"runs/prod_test_{tag}"}
    text = PRELUDE + body.rstrip("\n") + "\n" + EPILOGUE
    for k, v in fill.items():
        text = text.replace(k, v)
    cell["source"] = text.splitlines(keepends=True)

    # REPLACE v2's title rather than inserting above it: two H1s naming the notebook
    # differently is worse than one, and keeping the cell count equal to v2's makes the
    # "everything else is byte-identical" check a straight positional comparison.
    if nb["cells"][0]["cell_type"] != "markdown":
        raise SystemExit(f"expected {SRC.name} to open with a markdown title cell")
    nb["cells"][0] = {
        "cell_type": "markdown",
        "id": nb["cells"][0].get("id", "0"),
        "metadata": {},
        "source": header(tag).splitlines(keepends=True),
    }
    return nb


def rendered(tag: str) -> str:
    # ensure_ascii=False, because nbformat writes notebooks as raw UTF-8 and this repo's
    # prose is full of em dashes. With the default, the generator emits `—` where
    # every other writer emits the character, so the first save in Jupyter or the first
    # nbstripout smudge rewrites the file and `--check` goes red on an encoding
    # difference with no content behind it -- which is exactly the false alarm that
    # teaches people to ignore this check.
    return json.dumps(build(tag), indent=1, ensure_ascii=False) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if a committed notebook is stale")
    ap.add_argument("--variant", choices=sorted(VARIANTS), default=None,
                    help="only this variant (default: all of them)")
    args = ap.parse_args(argv)

    tags = [args.variant] if args.variant else sorted(VARIANTS)
    rc = 0
    for tag in tags:
        out, text = out_path(tag), rendered(tag)
        if args.check:
            if not out.exists():
                print(f"[make_prod_closure_nb] {out.name} does not exist")
                rc = 1
            elif out.read_text() != text:
                print(f"[make_prod_closure_nb] {out.name} is STALE — "
                      f"run `python scripts/make_prod_closure_nb.py`")
                rc = 1
            else:
                print(f"[make_prod_closure_nb] {out.name} is up to date")
            continue
        out.write_text(text)
        n = len(json.loads(text)["cells"])
        print(f"[make_prod_closure_nb] wrote {out.relative_to(REPO)} ({n} cells)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
