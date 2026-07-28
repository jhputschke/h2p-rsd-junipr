"""RNTuple reader (was `load_rntuple`): the real-data path written by the C++
`write_lund_rntuple` stage. Retains the per-jet `generator` tag for the §8
systematic and the grooming provenance for downstream reporting.

Also carries the aux conditioning columns (`jet_pt`, `jet_eta`, `x_mg`, `x_ptg`,
`x_nsec`, `x_kt_sec_*`, `x_sec_attach`; docs/PLAN_Input.md). Their defaults are
SENTINELS — NaN and -1 — that `features.aux_vector` rejects, so a file written before
those columns existed fails loud at dataset build time when aux is requested, rather
than training on NaNs.

PERFORMANCE NOTE. Every column is converted to numpy ONCE, up front; the per-jet loop
then only does numpy scalar reads and buffer slices. The natural-looking alternative —
indexing the awkward array per jet, `arr[name][i]` — costs ~30-40 us *per access*
because awkward rebuilds a layout slice each time. At ~22 columns x 54k jets that is
~1.2M such calls, i.e. ~47 s of pure Python for a 3.5 MB file whose I/O takes 0.07 s.
Hoisting makes the same read ~0.2 s. Keep it that way: no `arr[...]` inside the loop.
"""

from __future__ import annotations

# Mandatory per-jet sequence columns; absent ones are a corrupt/foreign file, not an
# older schema, so they raise rather than defaulting.
_SEQ_FIELDS = (
    "x_lnInvDelta", "x_lnkt", "x_lnz", "x_psi",
    "y_lnInvDelta", "y_lnkt", "y_lnz", "y_psi",
)


def load_rntuple(path: str = "jets.root", ntuple: str = "Jets"):
    """Read matched (x, y) primary Lund sequences from a ROOT RNTuple via uproot.
    Returns a list of per-jet dicts or None if the file/tool is unavailable."""
    try:
        import awkward as ak
        import numpy as np
        import uproot

        # awkward, not numpy: the x_*/y_* columns are jagged (variable-length
        # vector<float>) and cannot be coerced to a regular numpy array.
        with uproot.open(path) as f:
            arr = f[ntuple].arrays(library="ak")
        fields = set(arr.fields)
    except Exception as exc:  # missing file, no RNTuple support, etc.
        print(f"[load_rntuple] could not read {path}:{ntuple} ({exc}); using synthetic data.")
        return None

    n = len(arr["weight"])

    missing = [f for f in _SEQ_FIELDS if f not in fields]
    if missing:
        raise KeyError(
            f"{path}:{ntuple} is missing the primary-sequence columns {missing}; "
            f"present: {sorted(fields)}"
        )

    def column(name, default, dtype):
        """One vectorized conversion for the whole column. `default` fills a column the
        file does not have — the sentinel that `features.aux_vector` rejects."""
        if name not in fields:
            return np.full(n, default, dtype=dtype)
        return np.asarray(arr[name]).astype(dtype, copy=False)

    def jagged(name):
        """`(content, offsets)`: the column flattened into ONE contiguous buffer plus
        per-jet offsets, so a jet's sequence is the slice `content[off[i]:off[i+1]]`.

        The slices are VIEWS into `content`, which every consumer treats as read-only
        (`node_features` stacks into a new array, `seq_cells` builds one, the
        fingerprint only hashes). The buffer is marked non-writeable so a future write
        raises here instead of silently corrupting a neighbouring jet's sequence."""
        counts = np.asarray(ak.num(arr[name]), dtype=np.int64)
        offsets = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        content = np.ascontiguousarray(ak.to_numpy(ak.flatten(arr[name])), dtype=np.float32)
        content.flags.writeable = False
        return content, offsets

    weight = column("weight", 1.0, np.float64)
    event = column("event", 0, np.int64) if "event" in fields else None
    generator = ak.to_list(arr["generator"]) if "generator" in fields else ["unknown"] * n
    z_cut, beta, kt_floor = (column(k, np.nan, np.float64)
                             for k in ("z_cut", "beta", "kt_floor"))
    # --- aux conditioning sources (sentinels when the columns are absent) ---
    jet_pt, jet_eta = (column(k, np.nan, np.float64) for k in ("jet_pt", "jet_eta"))
    x_mg, x_ptg = (column(k, np.nan, np.float64) for k in ("x_mg", "x_ptg"))
    x_nsec, x_sec_attach = (column(k, -1, np.int64) for k in ("x_nsec", "x_sec_attach"))
    # secondary-plane kinematics: 0 is a LEGITIMATE value ("no secondary"), so the
    # absent-column sentinel has to be negative, not 0 or NaN.
    x_kt_sec_max, x_kt_sec_sum = (column(k, -1.0, np.float64)
                                  for k in ("x_kt_sec_max", "x_kt_sec_sum"))
    seqs = {name: jagged(name) for name in _SEQ_FIELDS}

    def seq(name, i):
        content, offsets = seqs[name]
        return content[offsets[i]:offsets[i + 1]]

    jets = [
        dict(
            weight=float(weight[i]),
            event=(int(event[i]) if event is not None else None),
            generator=str(generator[i]),
            z_cut=float(z_cut[i]),
            beta=float(beta[i]),
            kt_floor=float(kt_floor[i]),
            jet_pt=float(jet_pt[i]),
            jet_eta=float(jet_eta[i]),
            x_mg=float(x_mg[i]),
            x_ptg=float(x_ptg[i]),
            x_nsec=int(x_nsec[i]),
            x_kt_sec_max=float(x_kt_sec_max[i]),
            x_kt_sec_sum=float(x_kt_sec_sum[i]),
            x_sec_attach=int(x_sec_attach[i]),
            x=(seq("x_lnInvDelta", i), seq("x_lnkt", i), seq("x_lnz", i), seq("x_psi", i)),
            y=(seq("y_lnInvDelta", i), seq("y_lnkt", i), seq("y_lnz", i), seq("y_psi", i)),
        )
        for i in range(n)
    ]
    print(f"[load_rntuple] read {len(jets)} jets from {path}:{ntuple}.")
    return jets
