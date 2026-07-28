"""RNTuple reader (was `load_rntuple`): the real-data path written by the C++
`write_lund_rntuple` stage. Retains the per-jet `generator` tag for the §8
systematic and the grooming provenance for downstream reporting.

Also carries the aux conditioning columns (`jet_pt`, `x_mg`, `x_nsec`;
docs/PLAN_Input.md) through the same tolerant `scalar()` reads. Their defaults are
SENTINELS — NaN and -1 — that `features.aux_vector` rejects, so a file written
before those columns existed fails loud at dataset build time when aux is requested,
rather than training on NaNs.
"""

from __future__ import annotations


def load_rntuple(path: str = "jets.root", ntuple: str = "Jets"):
    """Read matched (x, y) primary Lund sequences from a ROOT RNTuple via uproot.
    Returns a list of per-jet dicts or None if the file/tool is unavailable."""
    try:
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

    def seq(name, i):
        return np.asarray(arr[name][i], dtype=np.float32)

    def scalar(name, i, default):
        return arr[name][i] if name in fields else default

    jets = []
    n = len(arr["weight"])
    for i in range(n):
        jets.append(
            dict(
                weight=float(arr["weight"][i]),
                event=(int(arr["event"][i]) if "event" in fields else None),
                generator=str(scalar("generator", i, "unknown")),
                z_cut=float(scalar("z_cut", i, float("nan"))),
                beta=float(scalar("beta", i, float("nan"))),
                kt_floor=float(scalar("kt_floor", i, float("nan"))),
                # aux conditioning sources (sentinels when the columns are absent)
                jet_pt=float(scalar("jet_pt", i, float("nan"))),
                jet_eta=float(scalar("jet_eta", i, float("nan"))),
                x_mg=float(scalar("x_mg", i, float("nan"))),
                x_ptg=float(scalar("x_ptg", i, float("nan"))),
                x_nsec=int(scalar("x_nsec", i, -1)),
                # secondary-plane kinematics: 0 is a LEGITIMATE value ("no secondary"),
                # so the absent-column sentinel has to be negative, not 0 or NaN.
                x_kt_sec_max=float(scalar("x_kt_sec_max", i, -1.0)),
                x_kt_sec_sum=float(scalar("x_kt_sec_sum", i, -1.0)),
                x_sec_attach=int(scalar("x_sec_attach", i, -1)),
                x=(seq("x_lnInvDelta", i), seq("x_lnkt", i), seq("x_lnz", i), seq("x_psi", i)),
                y=(seq("y_lnInvDelta", i), seq("y_lnkt", i), seq("y_lnz", i), seq("y_psi", i)),
            )
        )
    print(f"[load_rntuple] read {len(jets)} jets from {path}:{ntuple}.")
    return jets
