"""The N-information ceiling probe (docs/PLAN_NCeilingProbe.md WP-A).

The oracle-N decode reaches **1.67** perturbative-Lund EMD against the plain medoid's
**2.33** — the largest single lever measured anywhere in this line of work — but the
generative length belief `q(N|x)` puts its median on the right multiplicity for only
**0.448** of jets, identically at K=200 and K=1000. Calibrated, not sharp. The open
question this script answers:

    is 0.448 an INFORMATION CEILING of x, or an EXTRACTION FAILURE of the length model?

The instrument is a **discriminative** predictor of `n_true` from `(x, aux)`. Predicting a
label is a far easier task than carrying a correct generative posterior over trees, so the
probe's accuracy is a **lower bound** on the multiplicity information `x` carries — it can
only under-state the ceiling, never over-state it. That asymmetry is what makes the two
readings actionable in opposite directions, and it is why "the probe ties" is written below
as *no evidence of headroom* rather than as proof of a ceiling.

Everything is measured on the population the 0.448 was measured on: the first
`--n-test` jets with `len(x) > 0` of the held-out file, in file order, after the same
drop-and-count aux screen the cluster notebook applies. The posterior-median baseline is
RE-MEASURED here rather than quoted, so the sanity row (0.448) says out loud whether this
script is looking at the same jets as `per_jet_clusters.json`.

Two controls make the null readable, because a null result is only ever as good as the
instrument that produced it:

  * the **trivial-predictor tests** — the probe must beat the majority class and `n_x` on
    the same jets, or "it ties the model" is a statement about a probe that cannot learn;
  * the **learning curve** — accuracy against training-set size must be flat at the full
    sample, or the probe was starved and the tie is about the training budget, not `x`.

Three arms, so the verdict cannot be an artifact of what the probe was fed:
  * `x+aux`  — every feature (the headline);
  * `x-only` — the sequence summaries alone (the aux A/B was null for NLL, but the
    `n_sec = 2-3` stratum carried signal, so aux may matter for N specifically);
  * `n_x`    — the hadron multiplicity used directly as the prediction, and the majority
    class: the two trivial predictors any real signal has to beat.

The EMD payoff row then prices the measured n-hat where it would actually be spent: one
fresh posterior pass over the same jets, `stratified_medoid` (docs/PLAN_StratifiedMBR.md
WP1) fed by each n-hat in turn, against the plain medoid and the oracle-N ceiling, with a
paired jet-bootstrap CI. An accuracy that does not move the EMD is not a lever.

Run:
    python scripts/n_ceiling_probe.py --fast          # ~1 min smoke, small subsample
    python scripts/n_ceiling_probe.py                 # the full measurement
    python scripts/n_ceiling_probe.py --no-emd        # classifier arms only (no checkpoint)

Output: the printed table plus `runs/n_ceiling_probe/<stamp>/n_ceiling_probe.json`.
No unit tests, per the `scripts/probe_map_collapse.py` precedent; every function here is
importable and side-effect free apart from `main`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:  # editable installs already have it; be explicit
    sys.path.insert(0, str(REPO / "src"))

from h2p_rsd_junipr.data.rntuple import load_rntuple  # noqa: E402
from h2p_rsd_junipr.eval.report import save_metrics  # noqa: E402
from h2p_rsd_junipr.features import AUX_FEATURES, aux_vector, node_raw  # noqa: E402
from h2p_rsd_junipr.inference.length import quantile_floor  # noqa: E402

# The `7+` bucket. Multiplicities above this are rare enough that a per-class column is
# noise, and the decode only ever needs an n-hat the posterior pool can realise anyway
# (`_nearest_populated`). Fixed here, not a flag: it is part of what the accuracy MEANS.
N_MAX = 7

# The nine registered aux columns (docs/PLAN_Input.md). Order is irrelevant to a tree
# ensemble; the checkpoint's own order differs and is cross-checked, not adopted.
AUX_NAMES = tuple(AUX_FEATURES)

# A jet whose aux sources are sentinels is dropped and counted, and above this FRACTION
# the run raises instead — the `AUX_MAX_DROP` pattern of the cluster notebook section 3.
# Dropping more than this would reshape the population every number is quoted against.
AUX_MAX_DROP = 0.01

DEFAULT_TRAIN = "data/jet_aux_asym.root"
DEFAULT_TEST = "data/jet_aux_asym_test.root"


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def x_feature_names() -> list[str]:
    """Column names of `x_features`, in the order it fills them."""
    coords = ("lnInvDelta", "lnkt", "lnz", "psi")
    names = ["n_x"]
    for stat in ("mean", "std", "min", "max"):
        names += [f"{c}_{stat}" for c in coords]
    names += ["lnkt_lead", "lnkt_sublead", "sum_kt"]
    return names


def x_features(jet: dict) -> np.ndarray:
    """Per-jet summary of the hadron sequence `x` — the probe's x-only feature block.

    Deliberately a fixed-length SUMMARY rather than the sequence itself: the question is
    whether the multiplicity information is *there*, not whether a sequence model can be
    built to read it, and a summary that already saturates the generative model's accuracy
    settles the question with no architecture in the way.

    `lnkt_sublead` is NaN on a one-node jet — genuinely undefined, not zero, and
    HistGradientBoosting has native missing-value support, so the split learns "there is no
    second emission" instead of learning a fabricated -inf. `sum_kt` is the IRC-safe scale
    the EMD's own weights use (`lund_cloud(weight="kt")`)."""
    raw = np.asarray(node_raw(*jet["x"]), dtype=float)  # (n, 4)
    n = raw.shape[0]
    if n == 0:  # the population is len(x) > 0; kept so the helper is total
        return np.full(len(x_feature_names()), np.nan)
    lnkt = np.sort(raw[:, 1])[::-1]
    return np.concatenate([
        [float(n)],
        raw.mean(axis=0), raw.std(axis=0), raw.min(axis=0), raw.max(axis=0),
        [float(lnkt[0]), float(lnkt[1]) if n > 1 else np.nan, float(np.exp(raw[:, 1]).sum())],
    ])


def build_matrix(jets, aux_names=AUX_NAMES, *, max_drop=AUX_MAX_DROP):
    """`(X, y, names, kept_jets, n_dropped)` for a list of jets.

    The aux screen is drop-and-count, exactly as in the cluster notebook: `aux_vector`
    raises on a sentinel (a column written before it existed), and letting one bad jet kill
    the run names no count. Above `max_drop` it raises anyway — that many is the wrong file,
    not stragglers, and silently dropping them would reshape the population.

    `y` is `len(y_lnInvDelta)` clipped into the `7+` bucket."""
    names = x_feature_names() + list(aux_names)
    rows, targets, kept, dropped, why = [], [], [], 0, None
    for jet in jets:
        try:
            aux = aux_vector(jet, aux_names) if aux_names else np.zeros(0)
        except (KeyError, ValueError) as exc:
            dropped += 1
            why = why or str(exc)
            continue
        rows.append(np.concatenate([x_features(jet), np.asarray(aux, dtype=float)]))
        targets.append(min(len(jet["y"][0]), N_MAX))
        kept.append(jet)
    if not rows:
        raise RuntimeError(
            f"no jet can supply the aux inputs {list(aux_names)} ({why}); this is a file "
            f"written before those columns existed — re-write it with the current cpp/ "
            f"writer (docs/PLAN_Input.md stage 1)."
        )
    frac = dropped / (dropped + len(rows))
    if frac > max_drop:
        raise RuntimeError(
            f"{frac:.2%} of jets cannot supply the aux inputs ({why}), above "
            f"max_drop = {max_drop:.2%}. Dropping that many would reshape the population "
            f"every fraction below is quoted against."
        )
    return (np.asarray(rows, dtype=float), np.asarray(targets, dtype=int), names,
            kept, dropped)


def load_jets(path: str, *, limit: int | None = None) -> list:
    """The deployable population of a file: `len(x) > 0`, in file order.

    `len(y) > 0` is deliberately NOT required — that would read the answer, and the ~17%
    of jets whose parton truth is the empty tree are exactly the N = 0 stratum the length
    channel has to get right."""
    jets = load_rntuple(str(path), "Jets")
    if not jets:
        # `load_rntuple` prints and returns None when the file/uproot is unavailable, and
        # its other callers fall back to synthetic jets. There is no fallback here: the aux
        # columns exist only on the RNTuple path, so a synthetic stand-in would be a proxy
        # built from x — exactly what aux is not.
        raise FileNotFoundError(f"no jets read from {path}:Jets")
    jets = [j for j in jets if len(j["x"][0])]
    return jets if limit is None else jets[:limit]


# ---------------------------------------------------------------------------
# Predictions and their scoring
# ---------------------------------------------------------------------------
def proba_median(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """The L1-Bayes point estimate of each row: the smallest class with cdf >= 0.5.

    Literally `quantile_floor` — the SAME function `mbr_select_stratified` applies to
    `q(N|x)` — evaluated on a dense pmf over 0..N_MAX. Using one implementation for both is
    the point: the comparison against 0.448 is then a comparison of the information in two
    beliefs, not of two different summarising rules."""
    dense = np.zeros((proba.shape[0], N_MAX + 1), dtype=float)
    dense[:, np.asarray(classes, dtype=int)] = proba
    return np.array([quantile_floor(row, 0.5) for row in dense], dtype=int)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval for a binomial proportion — never the normal approximation,
    which leaves [0, 1] at exactly the counts this script reports on."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def score(pred: np.ndarray, truth: np.ndarray) -> dict:
    """Exact accuracy (with its Wilson interval) and the mean L1 error of one n-hat."""
    pred = np.asarray(pred, dtype=int)
    truth = np.asarray(truth, dtype=int)
    hit = int((pred == truth).sum())
    lo, hi = wilson_ci(hit, truth.size)
    return {
        "n": int(truth.size),
        "exact": float(hit / truth.size),
        "exact_ci95": [float(lo), float(hi)],
        "mean_abs": float(np.abs(pred - truth).mean()),
    }


def mcnemar(pred_a: np.ndarray, pred_b: np.ndarray, truth: np.ndarray) -> dict:
    """Exact paired test of "A is right more often than B" on the SAME jets.

    The pre-registered reading is a binomial CI against 0.448, which is the unpaired
    statement; both n-hats are however evaluated on identical rows, so the paired test is
    strictly the sharper instrument and is reported beside it. Discordant pairs only
    (McNemar, *Psychometrika* **12** (1947) 153): the jets both predictors get right or both
    get wrong carry no information about which is better."""
    a = np.asarray(pred_a, dtype=int) == np.asarray(truth, dtype=int)
    b = np.asarray(pred_b, dtype=int) == np.asarray(truth, dtype=int)
    n01 = int((a & ~b).sum())  # A right, B wrong
    n10 = int((~a & b).sum())
    n = n01 + n10
    if n == 0:
        return {"n_discordant": 0, "a_only": 0, "b_only": 0, "p_two_sided": 1.0}
    # exact binomial two-sided p under H0: a discordant pair is a fair coin
    k = min(n01, n10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * 0.5**n
    return {
        "n_discordant": n,
        "a_only": n01,
        "b_only": n10,
        "p_two_sided": float(min(1.0, 2.0 * tail)),
    }


def paired_bootstrap(delta: np.ndarray, n_boot: int = 2000, seed: int = 1234) -> dict:
    """Mean of a paired per-jet difference with a jet-bootstrap 95% interval.

    Paired at the JET, which is the unit that was sampled: the same posterior pool feeds
    every arm, so an unpaired interval would price the spread between jets rather than the
    difference between estimators (docs/SUMMARY_Model_Status.md section 5, "pair the
    comparison or don't call it one")."""
    d = np.asarray(delta, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {"mean": float("nan"), "ci95": [float("nan"), float("nan")], "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(int(n_boot), d.size))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(d.mean()), "ci95": [float(lo), float(hi)], "n": int(d.size)}


def learning_curve(x_tr, y_tr, x_te, y_te, *, fracs=(0.05, 0.125, 0.25, 0.5, 1.0),
                   seed: int = 1234) -> dict:
    """Accuracy against training-set size — the check that decides whether a NULL is real.

    "The probe ties the generative model" has two explanations: `x` carries no more N
    information, or the probe was starved. They are told apart by the slope. A curve still
    climbing at the full sample means the probe is data-limited and the tie says nothing
    about the data; a flat curve means more of the same data buys nothing, which is what
    turns the tie into a statement about `x`.

    It doubles as the honest error bar on the headline. Each point is an independent fit on
    an independent subsample, so the spread ACROSS the curve is the fit-to-fit variability
    the single headline number hides — and it is the wrong thing to leave out of a null
    result, where a small difference is precisely what is being claimed not to exist."""
    rng = np.random.default_rng(int(seed))
    rows = []
    for frac in fracs:
        n = max(2, int(float(frac) * len(y_tr)))
        idx = rng.permutation(len(y_tr))[:n]
        clf = fit_probe(x_tr[idx], y_tr[idx], seed=seed)
        proba = clf.predict_proba(x_te)
        med = proba_median(proba, clf.classes_)
        arg = clf.classes_[np.argmax(proba, axis=1)].astype(int)
        rows.append({"n_train": int(n), "frac": float(frac),
                     "exact_median": float((med == y_te).mean()),
                     "exact_argmax": float((arg == y_te).mean())})
    med_vals = [r["exact_median"] for r in rows]
    return {
        "rows": rows,
        "spread_median": float(max(med_vals) - min(med_vals)),
        "slope_last_two": (float(med_vals[-1] - med_vals[-2]) if len(med_vals) > 1
                           else float("nan")),
    }


def sanity_row(record: dict | None, measured: dict) -> dict:
    """Re-measured baselines beside the ones `per_jet_clusters.json` recorded.

    The plan's verification item, made a number rather than an assertion: if this script
    is not looking at the same jets under the same decode, the posterior median and the
    plain medoid will not land where the cluster run left them, and every comparison below
    is against the wrong reference. It is a CONSISTENCY row, not a gate — the length pmf of
    a continue/stop family IS the histogram of its K draws, so a fresh posterior pass moves
    the median on the jets whose belief straddles two multiplicities, and the recorded and
    re-measured values differ by exactly that MC noise."""
    if not record:
        return {"available": False, "reason": "no per_jet_clusters.json beside the checkpoint"}
    rec_run = record.get("run") or {}
    same = (int(rec_run.get("n_jets", -1)) == int(measured["n_jets"])
            and int(rec_run.get("K_draws", -1)) == int(measured["K"]))
    if not same:
        # A 60-jet K=50 smoke against a 600-jet K=200 record is not a consistency check,
        # it is two different measurements printed next to each other.
        return {"available": False,
                "reason": (f"recorded at n_jets={rec_run.get('n_jets')} K={rec_run.get('K_draws')}, "
                           f"this pass at n_jets={measured['n_jets']} K={measured['K']}")}
    rec_n = ((record.get("n_first") or {}).get("n_accuracy") or {})
    rec_lad = ((record.get("n_first") or {}).get("ladder") or {})

    def recorded(block, key):
        return float(block[key]) if key in block else None

    return {
        "available": True,
        "artifact": "per_jet_clusters.json",
        "posterior_median_exact": {
            "recorded": (recorded(rec_n.get("n_hat (median q(N|x))", {}), "exact")),
            "remeasured": measured.get("posterior_median_exact"),
        },
        "d_medoid": {
            "recorded": recorded(rec_lad.get("d_mbr", {}), "all"),
            "remeasured": measured.get("d_medoid"),
        },
        "d_oracle_N": {
            "recorded": recorded(rec_lad.get("d_mbr_ntrue", {}), "all"),
            "remeasured": measured.get("d_oracle_N"),
        },
    }


def fit_probe(x_train, y_train, *, seed: int = 1234, max_iter: int = 400,
              learning_rate: float = 0.06):
    """The discriminative length model: a multiclass HistGradientBoosting over 0..7+.

    Chosen because it is the strongest thing that can be fitted in seconds on tabular
    features with native NaN handling, so a null result cannot be blamed on the fit. Early
    stopping on an internal split, so `max_iter` is a ceiling rather than a tuned knob —
    nothing here is tuned against the test population, which would make the ceiling verdict
    circular.

    Early stopping is dropped when a class has a single member, because sklearn's internal
    split is STRATIFIED and raises on it. That is not hypothetical: the `7+` bucket holds
    **2** jets in 460 594 (the tail is 0/1/2/3 → 79 784 / 178 932 / 138 636 / 51 840, then
    10 272 / 1 062 / 66 / 2), so any subsample can leave it with one. The fallback trains
    the full `max_iter` on a tiny sample, which is the safe direction."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    counts = np.bincount(np.asarray(y_train, dtype=int))
    stoppable = bool(counts[counts > 0].min() >= 2)
    clf = HistGradientBoostingClassifier(
        max_iter=int(max_iter),
        learning_rate=float(learning_rate),
        early_stopping=stoppable,
        validation_fraction=0.15,
        n_iter_no_change=25,
        random_state=int(seed),
    )
    clf.fit(x_train, y_train)
    return clf


# ---------------------------------------------------------------------------
# The EMD payoff row
# ---------------------------------------------------------------------------
def resolve_run(ckpt: str | None, metrics: str | None) -> tuple[Path, dict | None]:
    """`(checkpoint, prod_test_v1 artifact)` — the same resolution the cluster notebook
    does, reduced to the one route this script needs.

    The artifact is what records WHICH file a checkpoint was evaluated on and under which
    frozen tau, so it is cross-checked rather than assumed; without it the checkpoint alone
    is still enough to build a distance matrix."""
    def newest(root: Path, pattern: str):
        hits = sorted(root.rglob(pattern), key=lambda q: q.stat().st_mtime) if root.is_dir() else []
        return hits[-1] if hits else None

    art = Path(metrics) if metrics else (
        newest(REPO / "runs" / "prod_test_v1" / "v1_contstop_s0", "prod_test_v1_metrics.json")
        or newest(REPO / "runs", "prod_test_v1_metrics.json")
    )
    record = json.loads(Path(art).read_text()) if art and Path(art).exists() else None
    if ckpt:
        path = Path(ckpt)
    elif record is not None:
        path = REPO / record["run"]["checkpoint"]
    else:
        raise FileNotFoundError(
            "no prod_test_v1_metrics.json under runs/ and no --ckpt given; the EMD payoff "
            "row needs a checkpoint to draw a posterior from (use --no-emd to skip it)."
        )
    if not path.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    return path, record


def emd_payoff(jets, n_hats: dict, *, ckpt: Path, k_draws: int, seed: int,
               n_boot: int, aux_names=AUX_NAMES) -> dict:
    """One posterior pass over `jets`, pricing each n-hat in EMD against the truth.

    Everything is read off ONE `K x K` matrix per jet — the plain medoid, every stratified
    variant, and the oracle — so the comparison cannot drift: `stratified_medoid` solves
    zero additional EMDs, and the only cost beyond the medoid's own is the `1 x K` row of
    distances to the truth cloud.

    `n_hats` maps a label to a per-jet integer array in the same order as `jets`. The oracle
    (`n_true`) and the plain medoid are added here, so a caller cannot forget the ceiling
    the whole exercise is measured against."""
    import torch
    from omegaconf import OmegaConf

    from h2p_rsd_junipr.config import decode_params
    from h2p_rsd_junipr.data.dataset import MatchedLundDataset
    from h2p_rsd_junipr.geometry import Geometry
    from h2p_rsd_junipr.inference.mbr import (
        lund_cloud,
        lund_emd_matrix,
        posterior_distances,
        stratified_medoid,
    )
    from h2p_rsd_junipr.models.base import build_model
    from h2p_rsd_junipr.train.checkpoint import load_for_inference
    from h2p_rsd_junipr.train.trainer import seed_everything

    seed_everything(int(seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = load_for_inference(str(ckpt), map_location=device)
    cfg = OmegaConf.create(info["config"])
    geom = Geometry.from_config(cfg.geometry)
    model = build_model(cfg, geom).to(device)
    model.load_state_dict(info["model_state"])
    model.eval()

    ckpt_aux = tuple(model.aux_feature_names)
    if set(ckpt_aux) != set(aux_names):
        # Not fatal — the probe's feature set and the encoder's conditioning set are two
        # different things — but a silent divergence would make "with aux" mean two things
        # in one document, so it is printed with both lists named.
        print(f"[note] the checkpoint conditions on {list(ckpt_aux)}; the probe's aux block "
              f"is {list(aux_names)}")

    dec = decode_params(cfg)
    cloud_kw = dict(lnkt_cut=dec["mbr_lnkt_cut"], weight=dec["mbr_weight"],
                    coords=dec["mbr_coords"])
    try:  # the batched OpenMP path when it is available; identical numbers either way
        import energyflow  # noqa: F401

        backend = "energyflow"
    except ImportError:
        backend = "pot"
    emd_kw = dict(R=dec["mbr_R"], beta=dec["mbr_beta"], norm=dec["mbr_norm"],
                  periodic_phi=dec["mbr_periodic_phi"], phi_col=dec["mbr_phi_col"],
                  backend=backend)

    ds = MatchedLundDataset(jets, geom, aux_features=ckpt_aux)
    labels = ["medoid"] + list(n_hats) + ["n_true"]
    d_truth = {name: np.full(len(jets), np.nan) for name in labels}
    n_post = np.zeros(len(jets), dtype=int)
    n_true = np.array([min(len(j["y"][0]), N_MAX) for j in jets], dtype=int)
    realised = {name: np.zeros(len(jets), dtype=bool) for name in n_hats}
    t0 = time.time()

    with torch.inference_mode():
        for i in range(len(jets)):
            item = ds[i]
            xf = item["xf"].unsqueeze(0).to(device)
            nx = torch.tensor([item["nx"]], device=device)
            draws = model.sample(xf, nx, n=int(k_draws))
            mults = np.array([len(d) for d in draws], dtype=int)
            _d, clouds, cand_idx, D = posterior_distances(
                model, xf, nx, draws=draws, geom=geom, n_candidates=0, **cloud_kw, **emd_kw)
            if not cand_idx:
                continue
            y = np.asarray(item["yraw"].numpy(), dtype=float)
            tc = lund_cloud([row for row in y], geom, **cloud_kw)
            d_row = lund_emd_matrix([tc], clouds, **emd_kw, geom=geom)[0]

            win = int(np.argmin(D.mean(axis=1)))
            d_truth["medoid"][i] = float(d_row[win])
            # the posterior median, RE-MEASURED on this population — the sanity row that
            # says whether this script is looking at the same jets as per_jet_clusters.json
            pmf = np.asarray(model.length_pmf(xf, nx, mults=mults.tolist()), dtype=float)
            n_post[i] = int(quantile_floor(pmf, 0.5))
            for name, hats in n_hats.items():
                idx, _risk, used = stratified_medoid(D, mults, int(hats[i]))
                d_truth[name][i] = float(d_row[idx])
                realised[name][i] = bool(used == int(hats[i]))
            idx_t, _r, _u = stratified_medoid(D, mults, int(len(y)))
            d_truth["n_true"][i] = float(d_row[idx_t])
            if (i + 1) % 100 == 0:
                print(f"  [emd] {i + 1}/{len(jets)} jets   {time.time() - t0:.0f}s")

    rows, base = {}, d_truth["medoid"]
    for name in labels:
        rows[name] = {
            "d_truth": float(np.nanmean(d_truth[name])),
            "delta_vs_medoid": paired_bootstrap(base - d_truth[name], n_boot=n_boot, seed=seed),
        }
        if name in realised:
            rows[name]["n_hat_realised"] = float(realised[name].mean())
    return {
        "checkpoint": str(ckpt),
        "K": int(k_draws),
        "backend": backend,
        "device": str(device),
        "n_jets": int(len(jets)),
        "seconds": float(time.time() - t0),
        "rows": rows,
        "posterior_median": {
            "n_hat": n_post.tolist(),
            **score(np.minimum(n_post, N_MAX), n_true),
        },
    }


# ---------------------------------------------------------------------------
# The printed report
# ---------------------------------------------------------------------------
PREREGISTERED = """
--------------------------------------------------------------------------------
PRE-REGISTERED READING (printed before the numbers, so the rule is not chosen after)

  The probe is DISCRIMINATIVE, so its accuracy is a LOWER BOUND on the multiplicity
  information x carries. The reference is the generative posterior median — recorded at
  0.448 in per_jet_clusters.json, and RE-MEASURED below on these same jets so the
  comparison is paired rather than against a four-decimal constant carrying K-draw noise.

  * accuracy > 0.448, with a 95% binomial CI that EXCLUDES it
        => the length channel UNDER-EXTRACTS. `stratified_medoid` can consume a sharper
           n-hat immediately (docs/PLAN_StratifiedMBR.md WP1), and architecture/training
           work on the length model is justified.
  * accuracy ~ 0.448 (CIs overlap)
        => NO EVIDENCE OF HEADROOM with these features. A lower bound, not proof of a
           ceiling — but the calibrated ambiguity of the set layer is then the right
           product for N, and length-model work is not the place to spend effort.
        READABLE ONLY IF the two controls below hold: the probe must beat the trivial
        predictors (else it measures nothing), and its learning curve must be flat (else
        it was starved, and the tie is about the training budget rather than about x).

  Either way the EMD payoff row prices what the measured n-hat buys against the 1.67
  oracle-N ceiling and the 2.33 plain medoid.
--------------------------------------------------------------------------------
"""


def print_table(rows: dict) -> None:
    """The classifier/baseline table: exact accuracy, its Wilson interval, mean L1."""
    print("\n" + "=" * 78)
    print(f"{'predictor':<28s}{'exact':>9s}{'95% CI':>18s}{'mean |dn|':>12s}{'n':>7s}")
    print("-" * 78)
    for name, r in rows.items():
        lo, hi = r["exact_ci95"]
        print(f"{name:<28s}{r['exact']:>9.4f}{f'[{lo:.3f}, {hi:.3f}]':>18s}"
              f"{r['mean_abs']:>12.4f}{r['n']:>7d}")
    print("=" * 78)


def print_emd(emd: dict) -> None:
    """The payoff table: mean EMD to truth per n-hat, and the paired delta vs the medoid."""
    print("\n" + "=" * 78)
    print(f"EMD payoff — {emd['n_jets']} jets, K = {emd['K']}, backend {emd['backend']!r}")
    print(f"{'n-hat feeding stratified_medoid':<34s}{'d(truth)':>11s}"
          f"{'delta vs medoid':>18s}{'95% CI':>22s}")
    print("-" * 78)
    for name, r in emd["rows"].items():
        d = r["delta_vs_medoid"]
        lo, hi = d["ci95"]
        delta = "—" if name == "medoid" else f"{d['mean']:+.3f}"
        ci = "—" if name == "medoid" else f"[{lo:+.3f}, {hi:+.3f}]"
        print(f"{name:<34s}{r['d_truth']:>11.3f}{delta:>18s}{ci:>22s}")
    print("=" * 78)
    print("positive delta = CLOSER to truth than the plain medoid (d(medoid) - d(arm)).")


def verdict(clf: dict, baseline: dict) -> tuple[str, str]:
    """`(label, sentence)` under the pre-registered rule above."""
    lo, hi = clf["exact_ci95"]
    ref = baseline["exact"]
    if lo > ref:
        return ("under_extraction",
                f"the probe reaches {clf['exact']:.4f} [{lo:.3f}, {hi:.3f}], whose 95% CI "
                f"EXCLUDES the posterior median's {ref:.4f}: x carries more N information "
                f"than the generative length model extracts, and length-model work is "
                f"justified.")
    if hi < ref:
        return ("probe_below_baseline",
                f"the probe reaches only {clf['exact']:.4f} [{lo:.3f}, {hi:.3f}], BELOW the "
                f"posterior median's {ref:.4f}. A discriminative lower bound that does not "
                f"reach the generative model is no evidence of headroom — the length "
                f"channel is at least as sharp as these features can be made.")
    return ("no_evidence_of_headroom",
            f"the probe reaches {clf['exact']:.4f} [{lo:.3f}, {hi:.3f}], whose 95% CI "
            f"CONTAINS the posterior median's {ref:.4f}: no evidence of headroom with "
            f"these features. A lower bound, not proof of a ceiling.")


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--train-path", default=DEFAULT_TRAIN, help="RNTuple the probe fits on")
    p.add_argument("--test-path", default=DEFAULT_TEST, help="held-out RNTuple it is scored on")
    p.add_argument("--n-train", type=int, default=0, help="0 = every training jet")
    p.add_argument("--n-test", type=int, default=600,
                   help="first N jets with len(x)>0 — the population the 0.448 was measured on")
    p.add_argument("--k-draws", type=int, default=200, help="posterior draws per jet (EMD row)")
    p.add_argument("--n-boot", type=int, default=2000, help="paired jet-bootstrap reps")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--ckpt", default=None, help="checkpoint for the EMD row (default: newest)")
    p.add_argument("--metrics", default=None, help="prod_test_v1_metrics.json to resolve from")
    p.add_argument("--no-emd", action="store_true", help="classifier arms only")
    p.add_argument("--no-learning-curve", action="store_true",
                   help="skip the data-limitation check (it is what makes the null readable)")
    p.add_argument("--fast", action="store_true",
                   help="smoke: small train subsample, 60 test jets, K=50")
    p.add_argument("--out", default=None, help="output directory (default runs/n_ceiling_probe/…)")
    args = p.parse_args(argv)

    if args.fast:
        args.n_train = args.n_train or 4000
        args.n_test = min(args.n_test, 60)
        args.k_draws = min(args.k_draws, 50)
        args.n_boot = min(args.n_boot, 200)

    print(PREREGISTERED)
    print(f"[probe] train {args.train_path}   test {args.test_path}   "
          f"n_test={args.n_test}  K={args.k_draws}  seed={args.seed}"
          + ("   [FAST SMOKE — not a result]" if args.fast else ""))

    train_jets = load_jets(REPO / args.train_path, limit=args.n_train or None)
    test_jets = load_jets(REPO / args.test_path)
    x_tr, y_tr, names, _kept_tr, drop_tr = build_matrix(train_jets)
    x_te_all, y_te_all, _n, kept_te, drop_te = build_matrix(test_jets)
    # The cut order matters and mirrors the cluster notebook exactly: len(x) > 0, then the
    # aux screen, THEN the first n_test. Taking the head first would score a different
    # population than `per_jet_clusters.json` whenever a jet is dropped.
    x_te, y_te = x_te_all[: args.n_test], y_te_all[: args.n_test]
    kept_te = kept_te[: args.n_test]
    print(f"[probe] train {x_tr.shape[0]:,} jets x {x_tr.shape[1]} features "
          f"({drop_tr} aux-dropped);  test {x_te.shape[0]:,} jets ({drop_te} aux-dropped)")
    print(f"[probe] target: n_true clipped at {N_MAX}+   "
          f"train class rates " + "  ".join(
              f"{c}:{(y_tr == c).mean():.3f}" for c in range(N_MAX + 1)))

    n_x_col = names.index("n_x")
    aux_cols = [i for i, nm in enumerate(names) if nm in AUX_NAMES]
    x_only = [i for i in range(len(names)) if i not in aux_cols]

    rows: dict[str, dict] = {}
    arms = {}
    t0 = time.time()
    for label, cols in (("probe: x + aux", None), ("probe: x only", x_only)):
        xtr = x_tr if cols is None else x_tr[:, cols]
        xte = x_te if cols is None else x_te[:, cols]
        clf = fit_probe(xtr, y_tr, seed=args.seed)
        proba = clf.predict_proba(xte)
        argmax = clf.classes_[np.argmax(proba, axis=1)].astype(int)
        median = proba_median(proba, clf.classes_)
        rows[f"{label}  (median)"] = score(median, y_te)
        rows[f"{label}  (argmax)"] = score(argmax, y_te)
        arms[label] = {"median": median, "argmax": argmax,
                       "n_iter": int(clf.n_iter_), "n_features": int(xtr.shape[1])}
    print(f"[probe] both arms fitted in {time.time() - t0:.1f}s")

    n_x_pred = np.minimum(x_te[:, n_x_col].astype(int), N_MAX)
    rows["baseline: n_x (hadron mult)"] = score(n_x_pred, y_te)
    majority = int(np.bincount(y_tr, minlength=N_MAX + 1).argmax())
    rows[f"baseline: majority ({majority})"] = score(np.full(y_te.size, majority), y_te)

    curve = None
    if not args.no_learning_curve:
        fracs = (0.25, 1.0) if args.fast else (0.05, 0.125, 0.25, 0.5, 1.0)
        curve = learning_curve(x_tr, y_tr, x_te, y_te, fracs=fracs, seed=args.seed)
        print(f"[probe] learning curve over {len(fracs)} training sizes "
              f"({time.time() - t0:.0f}s cumulative)")

    emd = None
    if not args.no_emd:
        ckpt, record = resolve_run(args.ckpt, args.metrics)
        if record is not None and str(record["run"]["test_path"]) != str(args.test_path):
            print(f"[warn] the artifact records test_path={record['run']['test_path']!r}, "
                  f"but the probe is scored on {args.test_path!r}")
        print(f"[probe] EMD payoff row: {ckpt.relative_to(REPO) if ckpt.is_relative_to(REPO) else ckpt}")
        emd = emd_payoff(
            kept_te,
            {"probe (x+aux) median": arms["probe: x + aux"]["median"],
             "probe (x only) median": arms["probe: x only"]["median"]},
            ckpt=ckpt, k_draws=args.k_draws, seed=args.seed, n_boot=args.n_boot,
        )
        pm = emd["posterior_median"]
        rows["baseline: posterior median"] = {k: pm[k] for k in
                                              ("n", "exact", "exact_ci95", "mean_abs")}

    print_table(rows)

    # --- is the probe a WORKING instrument? --------------------------------------
    # "The probe ties the generative model" only means anything if the probe can learn at
    # all. Paired against the two trivial predictors on the same jets: a probe that beats
    # them is measuring something, and its tie with the posterior median is then a
    # statement about the information in x rather than about a weak fit.
    head_pred = arms["probe: x + aux"]["median"]
    validity = {
        "vs majority": mcnemar(head_pred, np.full(y_te.size, majority), y_te),
        "vs n_x": mcnemar(head_pred, n_x_pred, y_te),
        "vs probe x-only": mcnemar(head_pred, arms["probe: x only"]["median"], y_te),
    }
    print("\ninstrument validity — paired (McNemar), probe (x+aux) median against:")
    for name, m in validity.items():
        print(f"  {name:<16s} probe-only-right {m['a_only']:>4d}   other-only-right "
              f"{m['b_only']:>4d}   p = {m['p_two_sided']:.3g}")

    if curve is not None:
        print("\nis the probe data-limited? — accuracy vs training-set size")
        print(f"  {'n_train':>9s} {'median':>9s} {'argmax':>9s}")
        for r in curve["rows"]:
            print(f"  {r['n_train']:>9d} {r['exact_median']:>9.4f} {r['exact_argmax']:>9.4f}")
        print(f"  spread across the curve {curve['spread_median']:.3f} — a FLAT curve says "
              f"more data buys nothing,\n  so the tie below is a statement about x rather "
              f"than about the probe's training budget.")

    ref_key = "baseline: posterior median"
    label, sentence = None, None
    if ref_key in rows:
        head = rows["probe: x + aux  (median)"]
        label, sentence = verdict(head, rows[ref_key])
        paired = mcnemar(head_pred,
                         np.asarray(emd["posterior_median"]["n_hat"], dtype=int)[: y_te.size],
                         y_te)
        print(f"\nVERDICT [{label}]: {sentence}")
        print(f"  paired (McNemar) on the same {y_te.size} jets: probe-only-right "
              f"{paired['a_only']}, posterior-only-right {paired['b_only']}, "
              f"p = {paired['p_two_sided']:.3g}")
        if curve is not None:
            # Two independent sources of noise, and a null result has to price both: the
            # Wilson interval covers the TEST set, the curve's range covers the FIT. Printed
            # side by side rather than combined, because the honest use of them is the same
            # either way — this test resolves differences of about `resolution`, not of a
            # third decimal, and that bound is what a future claim of a win must clear.
            width = head["exact_ci95"][1] - head["exact_ci95"][0]
            resolution = max(width, curve["spread_median"])
            print(f"  resolution: Wilson width {width:.3f} (test set) vs fit-to-fit range "
                  f"{curve['spread_median']:.3f} (learning curve) — comparable, so this "
                  f"test separates beliefs about N only down to ~{resolution:.2f}. Any "
                  f"future claim of a win must clear that, not the third decimal.")
    else:
        paired = None
        print("\nVERDICT: not taken — the posterior-median baseline needs the EMD pass "
              "(re-run without --no-emd).")

    sanity = {"available": False}
    if emd is not None:
        clusters_art = Path(emd["checkpoint"]).parent / "per_jet_clusters.json"
        record_c = json.loads(clusters_art.read_text()) if clusters_art.exists() else None
        sanity = sanity_row(record_c, {
            "posterior_median_exact": rows[ref_key]["exact"],
            "d_medoid": emd["rows"]["medoid"]["d_truth"],
            "d_oracle_N": emd["rows"]["n_true"]["d_truth"],
            "n_jets": emd["n_jets"],
            "K": emd["K"],
        })
        if sanity.get("available"):
            print(f"\nsanity — this pass vs {clusters_art.name} (same jets, fresh draws):")
            for key, pair in sanity.items():
                if not isinstance(pair, dict) or pair.get("recorded") is None:
                    continue
                print(f"  {key:<26s} recorded {pair['recorded']:.4f}   "
                      f"re-measured {pair['remeasured']:.4f}")
        else:
            print(f"\nsanity — not comparable: {sanity.get('reason')}")

    if emd is not None:
        print_emd(emd)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out) if args.out else REPO / "runs" / "n_ceiling_probe" / stamp
    metrics = {
        "run": {
            "script": "scripts/n_ceiling_probe.py",
            "plan": "docs/PLAN_NCeilingProbe.md",
            "stamp": stamp,
            "fast": bool(args.fast),
            "train_path": str(args.train_path),
            "test_path": str(args.test_path),
            "n_train": int(x_tr.shape[0]),
            "n_test": int(x_te.shape[0]),
            "aux_dropped": {"train": int(drop_tr), "test": int(drop_te)},
            "n_max_bucket": N_MAX,
            "features": names,
            "aux_features": list(AUX_NAMES),
            "seed": int(args.seed),
            "model": "HistGradientBoostingClassifier",
            "arms": {k: {"n_iter": v["n_iter"], "n_features": v["n_features"]}
                     for k, v in arms.items()},
        },
        "prereading": PREREGISTERED.strip(),
        "accuracy": rows,
        "verdict": {"label": label, "statement": sentence, "mcnemar": paired,
                    "instrument_validity": validity},
        "learning_curve": curve,
        "sanity": sanity,
        "emd_payoff": emd,
    }
    path = save_metrics(metrics, out_dir / "n_ceiling_probe.json")
    print(f"\n[probe] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
