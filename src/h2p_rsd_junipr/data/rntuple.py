"""RNTuple reader (was `load_rntuple`): the real-data path written by the C++
`write_lund_rntuple` stage. Retains the per-jet `generator` tag for the §8
systematic and the grooming provenance for downstream reporting.
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
                x=(seq("x_lnInvDelta", i), seq("x_lnkt", i), seq("x_lnz", i), seq("x_psi", i)),
                y=(seq("y_lnInvDelta", i), seq("y_lnkt", i), seq("y_lnz", i), seq("y_psi", i)),
            )
        )
    print(f"[load_rntuple] read {len(jets)} jets from {path}:{ntuple}.")
    return jets
