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

    if cfg.trainer.resume_from:
        trainer = Trainer.resume(
            cfg.trainer.resume_from, geometry, loaders, logger, device, run_dir, dm.fingerprint
        )
    else:
        model, opt, sched = build_components(cfg, geometry, device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[train] {n_params/1e3:.1f}k parameters")
        trainer = Trainer(model, opt, sched, loaders, cfg, logger, device, run_dir, dm.fingerprint)

    best = trainer.fit()
    logger.close()
    print(f"[train] done. best val NLL/jet = {best:.3f}. checkpoints in {run_dir}")
    return 0


def cmd_eval(argv) -> int:
    from .config import OmegaConf
    from .eval.calibration import run_calibration
    from .eval.closure import print_point_estimate, run_closure
    from .models.base import build_model
    from .train.checkpoint import load_for_inference

    # first token may be a checkpoint path
    ckpt = None
    if argv and "=" not in argv[0]:
        ckpt, argv = argv[0], argv[1:]
    cfg = load_config(argv)
    device, geometry, dm = _setup(cfg)
    _, val_ds = dm.datasets()

    if ckpt:
        info = load_for_inference(ckpt, map_location=device)
        cfg2 = OmegaConf.create(info["config"])
        geometry = Geometry.from_config(cfg2.geometry)
        model = build_model(cfg2, geometry).to(device)
        model.load_state_dict(info["model_state"])
        _, val_ds = LundDataModule(cfg2, geometry).setup().datasets()
        dm_jets = LundDataModule(cfg2, geometry).setup().val_jets
    else:
        from .train.trainer import build_components
        model, _, _ = build_components(cfg, geometry, device)
        dm_jets = dm.val_jets
        print("[eval] no checkpoint given; evaluating an untrained model (smoke).")

    model.eval()
    run_closure(model, val_ds, dm_jets, geometry, device,
                K=cfg.experiment.n_closure_samples, n_closure=cfg.experiment.closure_jets)
    run_calibration(model, val_ds, geometry, device,
                    K=cfg.experiment.n_closure_samples, n_jets=cfg.experiment.closure_jets)
    print_point_estimate(model, val_ds, dm_jets, geometry, device)
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
    example = (torch.zeros(1, 3, 5), torch.tensor([3]))
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
        )
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd not in _COMMANDS:
        print(f"unknown command {cmd!r}; choose from {sorted(_COMMANDS)}")
        return 2
    return _COMMANDS[cmd](rest)


if __name__ == "__main__":
    raise SystemExit(main())
