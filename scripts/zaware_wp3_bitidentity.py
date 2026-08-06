"""Is WP-3's coordinate threading bit-identical with the switch off? — the DoD #2 check.

`docs/PLAN_next_steps.md` §2/A1 asks for the same verification WP-1 was held to: run the
closure metric dict before and after on a fixed synthetic setup and diff it, and require
that **the diff contains additions only**. This script is the "after" half — it dumps the
dict as JSON — and `--compare a.json b.json` is the differ, so the "before" half is the
same file run from a pristine checkout of the parent commit:

    git worktree add /tmp/base HEAD
    python /tmp/base/scripts/zaware_wp3_bitidentity.py --out /tmp/base.json
    python scripts/zaware_wp3_bitidentity.py --out /tmp/head.json
    python scripts/zaware_wp3_bitidentity.py --compare /tmp/base.json /tmp/head.json

**Both RNG streams are reset between runs**, and that is not a detail: `decode_generator`
is private to the decode layer, advances per call and persists on the model
(`models/base.py`), so `torch.manual_seed` alone does not make two runs in one process
comparable. `tests/test_zaware_ruler.py::_closure` resets the same pair for the same
reason.

The synthetic jets come from `data.synthetic.synthetic_matched_dataset` — the same
generator `tests/conftest.py`'s `small_jets` uses — so nothing here depends on
`data/*.root` being present.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))


def _metrics(arms: tuple[str, ...]) -> dict:
    """The closure metric dict for each named arm, from a reset RNG state."""
    import torch

    from h2p_rsd_junipr.config import decode_params, load_config
    from h2p_rsd_junipr.data.dataset import MatchedLundDataset
    from h2p_rsd_junipr.data.synthetic import synthetic_matched_dataset
    from h2p_rsd_junipr.eval.closure import run_closure
    from h2p_rsd_junipr.geometry import Geometry
    from h2p_rsd_junipr.models.base import build_model

    dev = torch.device("cpu")
    jets = synthetic_matched_dataset(128, seed=0)[:24]
    out: dict = {}
    for arm in arms:
        overrides = [
            "model=ar_junipr_v4", "encoder=gru", "data.n_jets=64",
            "decode.mbr_backend=pot", "decode.min_emissions=0",
            # The cluster layer needs a SQUARE D, so the candidate cap is off on that arm
            # (`assert_cluster_metric_ok`); the two MBR arms keep the cap, which is what
            # the fielded decode tier uses.
            f"decode.mbr_n_candidates={0 if arm == 'clusters' else 6}",
            f"decode.point_estimator={'map' if arm == 'map' else 'mbr'}",
        ]
        cfg = load_config(overrides)
        geom = Geometry.from_config(cfg.geometry)
        torch.manual_seed(0)
        model = build_model(cfg, geom).to(dev).eval()
        ds = MatchedLundDataset(jets, geom)
        dec = decode_params(cfg)
        torch.manual_seed(0)
        model.__dict__.pop("_decode_generators", None)
        if arm == "clusters":
            from h2p_rsd_junipr.eval.clusters import run_cluster_diagnostics

            out[arm] = run_cluster_diagnostics(
                model, ds, jets, geom, dev, K=12, n_jets=16, decode=dec,
                verbose=False, null_reps=3)
        else:
            out[arm] = run_closure(model, ds, jets, geom, dev, K=12, n_closure=16,
                                   verbose=False, decode=dec, continuous=True,
                                   per_jet=True)
    return out


# ---------------------------------------------------------------------------
def _flatten(obj, prefix="") -> dict:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
        return out
    if isinstance(obj, (list, tuple)):
        out = {}
        for i, v in enumerate(obj):
            out.update(_flatten(v, f"{prefix}[{i}]"))
        return out
    return {prefix: obj}


def _equal(a, b) -> bool:
    """Exact equality with `NaN == NaN` — the metric dict is full of honest NaNs."""
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    return a == b


def compare(before: dict, after: dict) -> int:
    fa, fb = _flatten(before), _flatten(after)
    removed = sorted(set(fa) - set(fb))
    added = sorted(set(fb) - set(fa))
    moved = sorted(k for k in set(fa) & set(fb) if not _equal(fa[k], fb[k]))
    print(f"keys before {len(fa)}   after {len(fb)}")
    print(f"  ADDED    {len(added)}")
    for k in added[:40]:
        print(f"      + {k} = {fb[k]!r}")
    if len(added) > 40:
        print(f"      ... and {len(added) - 40} more")
    print(f"  REMOVED  {len(removed)}")
    for k in removed:
        print(f"      - {k}")
    print(f"  MOVED    {len(moved)}")
    for k in moved[:40]:
        print(f"      ~ {k}: {fa[k]!r} -> {fb[k]!r}")
    ok = not removed and not moved
    print("\n" + ("PASS — additions only, every pre-existing value bit-identical"
                  if ok else
                  "FAIL — the switch-off path moved; see REMOVED/MOVED above"))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="")
    ap.add_argument("--arms", default="mbr,map")
    ap.add_argument("--compare", nargs=2, default=None, metavar=("BEFORE", "AFTER"))
    args = ap.parse_args(argv)

    if args.compare:
        b = json.loads(Path(args.compare[0]).read_text())
        a = json.loads(Path(args.compare[1]).read_text())
        return compare(b, a)

    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    m = _metrics(arms)
    text = json.dumps(m, indent=1, allow_nan=True) + "\n"
    if args.out:
        Path(args.out).write_text(text)
        print(f"[bitid] wrote {args.out}  ({len(_flatten(m))} leaf keys)")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
