"""check_disjoint — is a "held-out" RNTuple actually held out?

Two files generated from the same binary and the same card differ only by the seed,
and `pythia_driver` does `Random:seed = seed % 900000000`. If the two streams overlap,
"held-out" is false and every number computed on the second file is a training number.
Nothing else in the repo checks this (docs/PLAN_prod_test_v0.md, "Risks").

Three assertions, in the order they can fail:

1. **Same physics.** `(z_cut, beta, kt_floor, kt_floor_sec, generator)` must be EQUAL.
   Different cards make the comparison a covariate shift, not a generalisation test.
2. **Different sample.** The `LundDataModule._fingerprint` of the two files must DIFFER
   — the cheap tell that someone passed the same path twice.
3. **Disjoint content.** Hash each jet's (x, y) sequence buffers and require the two
   hash sets to have an empty intersection.

On (3), read the exit criterion carefully. A hash collision here is a *jet* collision,
and a jet is only identifying if it carries enough content to name an event. On this
sample the mean groomed sequence is ~1.8 nodes, so hashing the eight (x, y) buffers
ALONE identifies almost nothing: a jet with an empty groomed tree hashes exactly like
every other empty-tree jet in both files, and ~93% of jets fall below a 3-emission bar.
Reporting "0 overlaps" out of the 7% that survive would be a far weaker statement than
it looks.

So two sets are compared, and both are reported:

* **`full`** — the (x, y) buffers *plus* the jet four-vector `(pt, eta, phi, m)` as
  written. Four independent float32s are overwhelmingly identifying even when the
  groomed tree is empty, so this covers EVERY jet read. This is the headline test.
* **`seq`** — the (x, y) buffers alone, restricted to jets with `>= --min-emissions`
  entries in both sequences. Narrower, but it depends on nothing but the columns the
  model actually consumes.

A genuine seed collision fails both: it reproduces the whole event stream, so the
overlap is total, not marginal.

    python scripts/check_disjoint.py data/jet_aux_asym.root data/jet_aux_asym_test.root

Exit code 0 = disjoint, 1 = overlap or provenance mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

PROVENANCE = ("z_cut", "beta", "kt_floor", "kt_floor_sec")
_SEQ_FIELDS = (
    "x_lnInvDelta", "x_lnkt", "x_lnz", "x_psi",
    "y_lnInvDelta", "y_lnkt", "y_lnz", "y_psi",
)


KIN_FIELDS = ("jet_pt", "jet_eta", "jet_phi", "jet_m")


def jet_hashes(path, ntuple="Jets", n_max=20000, min_emissions=3):
    """Per-jet SHA1s over the first `n_max` jets, in the two flavours above.

    Returns `(full, seq, provenance, n_jets_total, n_short)`: `full` hashes the eight
    (x, y) buffers plus the jet four-vector for every jet read, `seq` hashes the buffers
    alone for jets with >= `min_emissions` entries in BOTH sequences, `n_short` counts
    the rest. Buffers are hashed from the flat content array with per-jet offsets, the
    same layout `rntuple.py` builds, so no per-jet awkward indexing happens in the loop.
    """
    import awkward as ak
    import numpy as np
    import uproot

    with uproot.open(path) as f:
        arr = f[ntuple].arrays(library="ak")

    n_total = len(arr["weight"])
    n = min(n_max, n_total)

    prov = {k: float(np.asarray(arr[k])[0]) for k in PROVENANCE if k in arr.fields}
    prov["generator"] = str(ak.to_list(arr["generator"][:1])[0]) if "generator" in arr.fields else "unknown"

    flat = {}
    for name in _SEQ_FIELDS:
        counts = np.asarray(ak.num(arr[name]), dtype=np.int64)
        offsets = np.zeros(len(counts) + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        content = np.ascontiguousarray(ak.to_numpy(ak.flatten(arr[name])), dtype=np.float32)
        flat[name] = (content, offsets)

    # float32, not float64: hash the bits the file stores, so a re-read of the same
    # file reproduces the hash exactly rather than depending on the widening.
    kin = np.stack([np.asarray(arr[k]).astype(np.float32, copy=False)
                    if k in arr.fields else np.zeros(n_total, np.float32)
                    for k in KIN_FIELDS], axis=1)

    nx = np.asarray(ak.num(arr["x_lnInvDelta"]), dtype=np.int64)
    ny = np.asarray(ak.num(arr["y_lnInvDelta"]), dtype=np.int64)

    full, seq, n_short = set(), set(), 0
    for i in range(n):
        h = hashlib.sha1()
        for name in _SEQ_FIELDS:
            content, offsets = flat[name]
            h.update(content[offsets[i]:offsets[i + 1]].tobytes())
        seq_digest = h.hexdigest()
        if min(int(nx[i]), int(ny[i])) >= min_emissions:
            seq.add(seq_digest)
        else:
            n_short += 1
        h.update(kin[i].tobytes())
        full.add(h.hexdigest())
    return full, seq, prov, n_total, n_short


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("file_a")
    ap.add_argument("file_b")
    ap.add_argument("--ntuple", default="Jets")
    ap.add_argument("--n", type=int, default=20000, help="jets to hash per file")
    ap.add_argument("--min-emissions", type=int, default=3,
                    help="jets shorter than this in x or y are not identifying; skipped")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    fa_full, fa_seq, pa, na, sa = jet_hashes(args.file_a, args.ntuple, args.n, args.min_emissions)
    fb_full, fb_seq, pb, nb, sb = jet_hashes(args.file_b, args.ntuple, args.n, args.min_emissions)

    print(f"[check_disjoint] A {args.file_a}: {na} jets, read {min(args.n, na)}; "
          f"full hashes {len(fa_full)}, seq hashes {len(fa_seq)} "
          f"({sa} jets below {args.min_emissions} emissions, seq-only)")
    print(f"[check_disjoint] B {args.file_b}: {nb} jets, read {min(args.n, nb)}; "
          f"full hashes {len(fb_full)}, seq hashes {len(fb_seq)} "
          f"({sb} jets below {args.min_emissions} emissions, seq-only)")

    ok = True

    # 1. same physics
    if pa != pb:
        diff = {k: (pa.get(k), pb.get(k)) for k in set(pa) | set(pb) if pa.get(k) != pb.get(k)}
        print(f"[check_disjoint] FAIL provenance differs (A, B): {diff}\n"
              f"  The two files are different physics samples; any assessment on B "
              f"measures covariate shift, not generalisation.")
        ok = False
    else:
        print(f"[check_disjoint] OK  provenance identical: {pa}")

    # 2. different sample
    ga, gb = _fingerprint_of(fa_full), _fingerprint_of(fb_full)
    if ga == gb:
        print(f"[check_disjoint] FAIL both files fingerprint to {ga} — same file passed twice?")
        ok = False
    else:
        print(f"[check_disjoint] OK  fingerprints differ: A={ga} B={gb}")

    # 3. disjoint content — `full` is the headline, `seq` the narrower cross-check
    stats = {}
    for label, sa_, sb_, note in (
        ("full", fa_full, fb_full, "(x, y) + jet four-vector, every jet read"),
        ("seq", fa_seq, fb_seq, f"(x, y) only, >= {args.min_emissions} emissions"),
    ):
        overlap = sa_ & sb_
        denom = max(1, min(len(sa_), len(sb_)))
        frac = len(overlap) / denom
        stats[label] = {"n_a": len(sa_), "n_b": len(sb_),
                        "n_overlap": len(overlap), "overlap_fraction": frac}
        if overlap:
            print(f"[check_disjoint] FAIL [{label}] {len(overlap)} identical jets "
                  f"({frac:.3%} of the smaller set, {denom} compared) appear in BOTH "
                  f"files — {note}. The seeds collided: 'held-out' is false and every "
                  f"number computed on B is a training number.")
            ok = False
        else:
            print(f"[check_disjoint] OK  [{label}] 0 shared jets of {denom} compared "
                  f"— {note}.")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({
                "file_a": args.file_a, "file_b": args.file_b,
                "n_jets_a": na, "n_jets_b": nb,
                "n_read_a": min(args.n, na), "n_read_b": min(args.n, nb),
                "n_short_a": sa, "n_short_b": sb,
                "min_emissions": args.min_emissions,
                "provenance_a": pa, "provenance_b": pb,
                "provenance_equal": pa == pb,
                "fingerprint_a": ga, "fingerprint_b": gb,
                "overlap": stats,
                "disjoint": stats["full"]["n_overlap"] == 0 and stats["seq"]["n_overlap"] == 0,
                "passed": ok,
            }, f, indent=2)
        print(f"[check_disjoint] wrote {args.json_out}")

    print(f"[check_disjoint] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def _fingerprint_of(hashes):
    """Content fingerprint of a hashed set — the sorted hashes and nothing else, so two
    different paths holding identical content still fingerprint the same (which is the
    duplicate-file case this is meant to catch)."""
    h = hashlib.sha1()
    for x in sorted(hashes):
        h.update(x.encode())
    return h.hexdigest()[:12]


if __name__ == "__main__":
    sys.exit(main())
