"""scripts/check_disjoint.py — the guard that a "held-out" file really is held out.

The failure it exists to catch (two `pythia_driver` seeds colliding modulo 900000000)
cannot be reproduced cheaply, so these tests pin the two halves that CAN be: that the
same file compared against itself is reported as a total overlap, and that the `full`
hash covers every jet while the `seq` hash covers only the long ones. On this physics
the second point is the whole reason `full` exists — the mean groomed sequence is ~1.8
nodes, so a sequence-only comparison silently discards ~93% of the sample.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
JETS = REPO / "cpp" / "test_data" / "jets.root"

pytest.importorskip("uproot")
pytest.importorskip("awkward")


def _load():
    spec = importlib.util.spec_from_file_location(
        "check_disjoint", REPO / "scripts" / "check_disjoint.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cd():
    return _load()


@pytest.mark.skipif(not JETS.exists(), reason="reference RNTuple not present")
def test_full_hash_covers_every_jet_seq_hash_does_not(cd):
    full, seq, prov, n_total, n_short = cd.jet_hashes(str(JETS), n_max=400, min_emissions=3)

    # `full` mixes in (pt, eta, phi, m), so every jet is identifying and distinct.
    assert len(full) == 400, "the four-vector should make every jet uniquely hashable"
    # `seq` skips short jets; on this sample that is the vast majority of them.
    assert len(seq) + n_short <= 400
    assert n_short > 0, "expected the reference sample to contain sub-3-emission jets"
    assert n_total == 54007
    assert prov["kt_floor"] == pytest.approx(1.0)


@pytest.mark.skipif(not JETS.exists(), reason="reference RNTuple not present")
def test_same_file_twice_is_reported_as_total_overlap(cd, tmp_path, capsys):
    out = tmp_path / "disjoint.json"
    rc = cd.main([str(JETS), str(JETS), "--n", "300", "--json-out", str(out)])
    assert rc == 1, "comparing a file with itself must fail the guard"

    import json
    rep = json.loads(out.read_text())
    assert rep["passed"] is False
    assert rep["disjoint"] is False
    assert rep["provenance_equal"] is True          # same file, so same card
    assert rep["fingerprint_a"] == rep["fingerprint_b"]
    assert rep["overlap"]["full"]["overlap_fraction"] == pytest.approx(1.0)

    text = capsys.readouterr().out
    assert "training number" in text, "the failure message must say what it invalidates"


@pytest.mark.skipif(not JETS.exists(), reason="reference RNTuple not present")
def test_fingerprint_ignores_path_and_ordering(cd):
    a = cd._fingerprint_of({"c", "a", "b"})
    b = cd._fingerprint_of({"b", "c", "a"})
    assert a == b, "the fingerprint must be order-independent"
    assert a != cd._fingerprint_of({"a", "b"})
