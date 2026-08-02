"""`h2p-rsd-junipr {generate,train,eval,export,serve}` — the single entry point.

CLI ergonomics are OmegaConf-dotted overrides (no Hydra):

    h2p-rsd-junipr train model=cinn encoder=lundnet \
        encoder.num_layers=3 geometry.n_bins=16 optim.lr=1e-3 trainer.max_epochs=100
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch

from .config import config_hash, load_config, save_config
from .data.datamodule import LundDataModule
from .data.stats import check_lnz_support, check_multiplicity_support
from .geometry import Geometry


def _run_dir(cfg) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(cfg.run_root) / f"{stamp}-{config_hash(cfg)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _setup(cfg):
    from .train.trainer import seed_everything, select_device

    seed_everything(cfg.trainer.seed, cfg.trainer.deterministic)
    device = select_device()
    geometry = Geometry.from_config(cfg.geometry)
    dm = LundDataModule(cfg, geometry).setup()
    return device, geometry, dm


def _warn_objective_is_not_nll(model, cfg) -> None:
    """Say so when the logged `train_nll` is not an NLL.

    A family may minimize something other than -log_prob (`cfm` regresses a vector
    field), and a family's `log_prob` may itself be a surrogate (`diffusion`). Both
    change what the training curve means, so both are stated once, up front, rather
    than left for a reader to infer from the column header."""
    from .models.base import PosteriorModel

    if type(model).training_objective is not PosteriorModel.training_objective:
        print(f"[train] NOTE: {cfg.model.name!r} trains a surrogate objective (not the NLL); "
              "the logged `train_nll` is that objective, while `val_nll` is the exact NLL.")
    elif not getattr(model, "exact_likelihood", True):
        print(f"[train] NOTE: {cfg.model.name!r} sets exact_likelihood=False; both logged "
              "losses are a training surrogate, comparable only within this family.")


# ---------------------------------------------------------------------------
def cmd_train(argv) -> int:
    from .train.logging import CSVJSONLLogger
    from .train.trainer import Trainer, build_components

    cfg = load_config(argv)
    device, geometry, dm = _setup(cfg)
    run_dir = _run_dir(cfg)
    save_config(cfg, run_dir / "config.yaml")
    logger = CSVJSONLLogger(run_dir, tensorboard=False)
    loaders = dm.loaders()

    print(
        f"[train] model={cfg.model.name} encoder={cfg.encoder.name} "
        f"device={device} run_dir={run_dir}"
    )
    print(f"[train] {len(dm.train_jets)} train / {len(dm.val_jets)} val jets "
          f"(fingerprint={dm.fingerprint})")
    # WP4 guard: a categorical q(N|x) head has finite support, so a truth sequence
    # past it is silently clamped. Checked against the data actually loaded, before
    # any time is spent training on it.
    check_multiplicity_support(dm.jets, cfg)
    # WP-A guard: the physical `ln z` head normalizes over the interval the FILE's
    # grooming defines, and the config is the only place the model learns it — so the
    # declared pair is verified against the data before any time is spent training on it.
    check_lnz_support(dm.jets, cfg)

    if cfg.trainer.resume_from:
        trainer = Trainer.resume(
            cfg.trainer.resume_from, geometry, loaders, logger, device, run_dir, dm.fingerprint
        )
    else:
        model, opt, sched = build_components(cfg, geometry, device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[train] {n_params/1e3:.1f}k parameters")
        _warn_objective_is_not_nll(model, cfg)
        trainer = Trainer(model, opt, sched, loaders, cfg, logger, device, run_dir, dm.fingerprint)

    best = trainer.fit()
    logger.close()
    print(f"[train] done. best val NLL/jet = {best:.3f}. checkpoints in {run_dir}")
    return 0


def _lift_onto_snapshot(snapshot, cfg, argv, group: str) -> dict:
    """Lift the `<group>` fields this invocation named explicitly (`explicit_group_keys`)
    from the composed CLI config onto a checkpoint snapshot, in place.

    Returns `{field: (old, new)}` for the ones that actually moved. Both the eval log and
    `eval_metrics.json` report it: a metrics file that does not name the jets and the
    decode it describes is not a record of anything.

    Values come from `cfg` — already schema-validated and interpolation-resolved by
    `load_config` — so the snapshot only ever receives well-typed data. `force_add`
    because a snapshot predating a field has no key to update (the same tolerance
    `decode_params` provides on the read side)."""
    from .config import OmegaConf, explicit_group_keys

    applied: dict = {}
    for key in sorted(explicit_group_keys(argv, group)):
        new = OmegaConf.select(cfg, f"{group}.{key}")
        if OmegaConf.is_config(new):
            new = OmegaConf.to_object(new)
        old = OmegaConf.select(snapshot, f"{group}.{key}")
        if OmegaConf.is_config(old):
            old = OmegaConf.to_object(old)
        if new == old:
            continue
        OmegaConf.update(snapshot, f"{group}.{key}", new, force_add=True)
        applied[key] = (old, new)
    return applied


def _describe_data(cfg) -> str:
    from .config import OmegaConf

    d = cfg.data
    source = str(OmegaConf.select(d, "source") or "synthetic")
    if source == "rntuple":
        return f"rntuple:{OmegaConf.select(d, 'path')}"
    return f"synthetic(n_jets={OmegaConf.select(d, 'n_jets')}, seed={OmegaConf.select(d, 'seed')})"


def _report_eval_inputs(cfg_eval, dm, overrides: dict, cfg_cli=None) -> None:
    """Say which jets and which decode this eval is about, before it spends any time."""
    from .config import OmegaConf

    for group, applied in overrides.items():
        if applied:
            print(f"[eval] {group} lifted over the checkpoint snapshot: "
                  + ", ".join(f"{k}: {o!r} -> {n!r}" for k, (o, n) in applied.items()))
    scope = "every jet (explicitly named eval sample)" if overrides.get("data") else "val split"
    print(f"[eval] {len(dm.val_jets)} eval jets, {scope}, from {_describe_data(cfg_eval)} "
          f"(fingerprint={dm.fingerprint})")
    if cfg_cli is None or not any(overrides.values()):
        return
    cli_geom = OmegaConf.to_container(cfg_cli.geometry, resolve=True)
    ckpt_geom = OmegaConf.to_container(cfg_eval.geometry, resolve=True)
    if cli_geom != ckpt_geom:
        # e.g. presets/decode/mbr_study.yaml's `mbr_lnkt_cut: ${geometry.ln_kt_range[0]}`
        print(f"[eval] NOTE: CLI geometry {cli_geom} differs from the checkpoint's {ckpt_geom}. "
              "The checkpoint's is used for the model; a ${geometry...} interpolation inside a "
              "lifted data/decode override resolved against the CLI one.")


def cmd_eval(argv) -> int:
    """Closure / calibration / point estimate on held-out jets.

    With a checkpoint the snapshot is the baseline for everything — architecture,
    geometry, sample, decode — and exactly two groups may be lifted over it, because both
    are inference-time choices rather than properties of the trained model:

      * `data`   — WHICH jets to report on. An explicitly named sample is treated as a
                   test set and evaluated in full, not re-split 90/10.
      * `decode` — HOW the posterior becomes a point estimate (MAP floors, MBR, beam).

    `geometry` and `encoder` are deliberately NOT liftable: they set tensor widths and the
    model contract, so changing them describes a different model, not a re-run. `experiment`
    is the eval suite's own configuration and always comes from the CLI/preset.

    Each liftable group accepts the full composition surface — `group=name`, a `base=`
    preset's `defaults:`/inline block, and dotted `group.field=value` — in `load_config`'s
    precedence order, with the CLI last."""
    from .config import OmegaConf, decode_params, experiment_params
    from .eval.calibration import run_calibration
    from .eval.closure import print_point_estimate, run_closure
    from .eval.report import inert_decode_keys, plot_calibration, save_metrics
    from .inference.mbr import mbr_kwargs_from_decode
    from .models.base import build_model
    from .train.checkpoint import load_for_inference
    from .train.trainer import seed_everything, select_device

    # first token may be a checkpoint path
    ckpt = None
    if argv and "=" not in argv[0]:
        ckpt, argv = argv[0], argv[1:]
    cfg = load_config(argv)          # schema-validates every token before anything is loaded
    seed_everything(cfg.trainer.seed, cfg.trainer.deterministic)
    device = select_device()

    if ckpt:
        info = load_for_inference(ckpt, map_location=device)
        cfg_eval = OmegaConf.create(info["config"])
        overrides = {g: _lift_onto_snapshot(cfg_eval, cfg, argv, g) for g in ("data", "decode")}
        geometry = Geometry.from_config(cfg_eval.geometry)
        model = build_model(cfg_eval, geometry).to(device)
        model.load_state_dict(info["model_state"])
        # ONE load: the sample follows the (possibly lifted) snapshot, so there is no
        # CLI-config datamodule to build and throw away.
        dm = LundDataModule(cfg_eval, geometry).setup()
        if overrides["data"]:
            # An explicitly named eval sample is a TEST set, not a training corpus: report
            # on every jet in it. Keeping the 90/10 split would silently evaluate a tenth of
            # the file — and a *different* tenth as soon as its length changed.
            dm.train_jets, dm.val_jets = [], dm.jets
        # non-fatal here: the model is already trained, so report rather than refuse
        check_multiplicity_support(dm.jets, cfg_eval, strict=False)
        check_lnz_support(dm.jets, cfg_eval, strict=False)
    else:
        from .train.trainer import build_components
        cfg_eval, overrides = cfg, {"data": {}, "decode": {}}
        geometry = Geometry.from_config(cfg.geometry)
        dm = LundDataModule(cfg, geometry).setup()
        model, _, _ = build_components(cfg, geometry, device)
        print("[eval] no checkpoint given; evaluating an untrained model (smoke).")

    _report_eval_inputs(cfg_eval, dm, overrides, cfg_cli=cfg if ckpt else None)
    _, val_ds = dm.datasets()
    dm_jets = dm.val_jets
    decode = decode_params(cfg_eval)  # tolerant of pre-decode-field checkpoint snapshots
    # `build_model` already read this from the snapshot; re-apply so a LIFTED
    # `decode.length_temperature=` wins, like every other decode override.
    model.length_temperature = float(decode["length_temperature"])
    model.length_tilt = float(decode["length_tilt"])
    model.continue_temperature = float(decode["continue_temperature"])
    model.kappa_min_mode = float(decode["kappa_min_mode"])

    model.eval()
    if not getattr(model, "exact_likelihood", True):
        print(
            f"[eval] WARNING: model family {cfg_eval.model.name!r} sets exact_likelihood=False — "
            "its `log_prob` is a training surrogate, not a normalized density. Reported "
            "NLLs and log-ratios are NOT comparable across families (use model=cfm for the "
            "exact-likelihood continuous-time family)."
        )
    exp = experiment_params(cfg)  # eval-suite config, always from the CLI/preset
    # The family reported is the one that was LOADED, not the one the CLI would have built:
    # `eval <cinn ckpt>` with no `model=` token composes the repo default, and a metrics file
    # naming the wrong family is worse than none.
    metrics = {
        "model": str(cfg_eval.model.name), "encoder": str(cfg_eval.encoder.name),
        "checkpoint": str(ckpt) if ckpt else None,
        # `select_device()` above picks cuda > mps > cpu with nothing to override it, so
        # WHICH backend produced these numbers is decided by the caller's environment
        # (`CUDA_VISIBLE_DEVICES=""` is the only lever, and it is what
        # scripts/eval_prod_test_v1.sh --device cpu sets). Backends are a different RNG
        # stream *and* different float kernels, so two runs are comparable only once both
        # name the same one — same reason lund_closure_report.py records it.
        "device": str(device),
        "data": {
            "source": str(OmegaConf.select(cfg_eval, "data.source")),
            "path": str(OmegaConf.select(cfg_eval, "data.path")),
            "fingerprint": dm.fingerprint,
            "n_eval_jets": len(dm_jets),
            "scope": "all" if overrides["data"] else "val_split",
            "overrides": {k: n for k, (_o, n) in overrides["data"].items()},
        },
        "decode": dict(decode),
        "decode_overrides": {k: n for k, (_o, n) in overrides["decode"].items()},
        # ...and which of those knobs never reached a number below. Without this the
        # JSON faithfully records settings that did nothing, and a reader comparing two
        # runs on `beam_width` compares a knob neither consulted.
        "decode_inert": inert_decode_keys(model, dict(decode)),
        "experiment": dict(exp),
    }
    if metrics["decode_inert"]:
        print("\n[eval] decode knobs that did NOT reach these numbers:")
        for e in metrics["decode_inert"]:
            print(f"    {e['key']} = {e['value']!r}   — {e['reason']}")
    metrics["closure"] = run_closure(
        model, val_ds, dm_jets, geometry, device,
        K=exp["n_closure_samples"], n_closure=exp["closure_jets"], decode=decode,
        continuous=exp["closure_continuous"],
    )
    metrics["calibration"] = run_calibration(
        model, val_ds, geometry, device,
        K=exp["n_closure_samples"], n_jets=exp["closure_jets"],
        pit_coords=exp["pit_coords"], stratify_regions=exp["stratify_regions"],
        tarp=exp["tarp"], tarp_refs=exp["tarp_refs"], tarp_reference=exp["tarp_reference"],
        mbr_kwargs=mbr_kwargs_from_decode(decode),
        tarp_null_reps=exp["tarp_null_reps"], tarp_stratify=exp["tarp_stratify"],
    )
    if exp["support_audit"]:
        from .eval.support import run_support_audit

        metrics["support_audit"] = run_support_audit(
            model, val_ds, dm_jets, geometry, device,
            n_jets=exp["closure_jets"], K=exp["n_closure_samples"],
        )
    if exp["exposure_diagnostic"]:
        from .eval.exposure import run_exposure

        metrics["exposure"] = run_exposure(
            model, val_ds, device, n_jets=exp["closure_jets"], K=exp["n_closure_samples"],
        )
    mode_audit = None
    if exp["mode_audit"]:
        from .config import audit_params
        from .eval.mode_audit import run_mode_audit

        aud = audit_params(cfg)  # the audit block is eval-suite config, like `experiment`
        mode_audit = run_mode_audit(
            model, val_ds, dm_jets, geometry, device,
            n_jets=(aud["n_jets"] or exp["closure_jets"]), K=exp["n_closure_samples"],
            audit=aud,
        )
        # Provenance, so the artifact can be read on its own — and so the cross-family
        # delta of the plan's §7.5 compares two runs rather than two file names.
        mode_audit["run"] = {
            "model": metrics["model"], "encoder": metrics["encoder"],
            "checkpoint": metrics["checkpoint"], "device": metrics["device"],
            "data": metrics["data"], "decode": metrics["decode"],
        }
    print_point_estimate(model, val_ds, dm_jets, geometry, device, decode=decode)

    if ckpt:  # artifacts land beside the checkpoint, next to the training curves
        out_dir = Path(ckpt).resolve().parent
        save_metrics(metrics, out_dir / "eval_metrics.json")
        figs = plot_calibration(metrics["calibration"], out_dir)
        print(f"\n[eval] wrote {out_dir/'eval_metrics.json'}"
              + (f" and {len(figs)} figure(s)" if figs else ""))
        if mode_audit is not None:
            # Its own file: the per-jet records are one row per jet and would swamp
            # eval_metrics.json, which every other consumer diffs.
            save_metrics(mode_audit, out_dir / "mode_audit.json")
            print(f"[eval] wrote {out_dir/'mode_audit.json'} "
                  f"({mode_audit['n_jets']} per-jet records)")
    return 0


def cmd_generate(argv) -> int:
    """Shell out to the built C++ writer (cpp/build/write_lund_rntuple) or document
    the build. Pass-through args go to the binary."""
    binary = Path(__file__).resolve().parents[2] / "cpp" / "build" / "write_lund_rntuple"
    if not binary.exists():
        print(
            "[generate] C++ writer not built. Build it with:\n"
            "  cmake -S cpp -B cpp/build && cmake --build cpp/build -j\n"
            f"  (expected binary: {binary})"
        )
        return 1
    print(f"[generate] running {binary} {' '.join(argv)}")
    return subprocess.call([str(binary), *argv])


def cmd_export(argv) -> int:
    from .config import OmegaConf
    from .models.base import build_model
    from .serving.export import export_encoder_torchscript
    from .train.checkpoint import load_for_inference

    if not argv:
        print("[export] usage: h2p-rsd-junipr export <ckpt> [out.pt]")
        return 1
    ckpt = argv[0]
    out = Path(argv[1]) if len(argv) > 1 else Path("encoder_scripted.pt")
    info = load_for_inference(ckpt, map_location="cpu")
    cfg = OmegaConf.create(info["config"])
    geometry = Geometry.from_config(cfg.geometry)
    model = build_model(cfg, geometry)
    model.load_state_dict(info["model_state"])
    # trace width follows the encoder the checkpoint actually built: 5, or 5 + n_aux
    n_in = 5 + len(getattr(model, "aux_feature_names", ()) or ())
    example = (torch.zeros(1, 3, n_in), torch.tensor([3]))
    path = export_encoder_torchscript(model, out, example=example, verify=True)
    print(f"[export] scripted encoder -> {path} (allclose-verified)")
    return 0


def cmd_serve(argv) -> int:
    if not argv:
        print("[serve] usage: h2p-rsd-junipr serve <ckpt> [host] [port]")
        return 1
    ckpt = argv[0]
    host = argv[1] if len(argv) > 1 else "127.0.0.1"
    port = int(argv[2]) if len(argv) > 2 else 8000
    import uvicorn  # optional [serve] dependency

    from .serving.api import create_app

    uvicorn.run(create_app(ckpt), host=host, port=port)
    return 0


_COMMANDS = {
    "train": cmd_train,
    "eval": cmd_eval,
    "generate": cmd_generate,
    "export": cmd_export,
    "serve": cmd_serve,
}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(
            "h2p-rsd-junipr — amortized hadron-to-parton posterior over groomed Lund trees\n\n"
            "usage: h2p-rsd-junipr <command> [overrides]\n\n"
            "commands:\n"
            "  generate   run the C++ RNTuple writer -> jets.root\n"
            "  train      train a posterior model (config-first, OmegaConf overrides)\n"
            "  eval       closure / calibration / point-estimate on held-out jets\n"
            "  export     TorchScript the encoder (allclose-verified)\n"
            "  serve      FastAPI: x -> {MAP, posterior summary}\n\n"
            "example:\n"
            "  h2p-rsd-junipr train model=ar_junipr_v2 encoder=gru trainer.max_epochs=20\n"
            "  h2p-rsd-junipr train base=presets/mbr_study.yaml   # custom top-level config\n"
        )
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd not in _COMMANDS:
        print(f"unknown command {cmd!r}; choose from {sorted(_COMMANDS)}")
        return 2
    return _COMMANDS[cmd](rest)


if __name__ == "__main__":
    raise SystemExit(main())
