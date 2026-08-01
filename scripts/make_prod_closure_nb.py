"""Generate the production-test variants of the v2 distribution-closure notebook.

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

# None -> newest runs/prod_test_v*/*/{TAG}/{TAG}_metrics.json
PROD_METRICS_PATH = None

_REPO = _Path.cwd().parent if _Path.cwd().name == "notebooks" else _Path.cwd()
if PROD_METRICS_PATH:
    _mp = _Path(PROD_METRICS_PATH)
    _mp = _mp if _mp.is_absolute() else _REPO / _mp
else:
    _found = sorted(_REPO.glob("runs/prod_test_v*/*/{TAG}/{TAG}_metrics.json"),
                    key=lambda q: q.stat().st_mtime)
    if not _found:
        raise FileNotFoundError(
            "no {TAG}_metrics.json under runs/. This notebook takes its checkpoint, its "
            "test file, the frozen empty-tree tau and the fitted length recalibration "
            "from that artifact — run notebooks/{NB} first, or set PROD_METRICS_PATH."
        )
    _mp = _found[-1]

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
    fill = {"{TAG}": f"prod_test_{tag}", "{NB}": f"prod_test_{tag}.ipynb"}
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
