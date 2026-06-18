"""Lightweight logging (§9): a Logger protocol with a dependency-free default
(CSV + JSONL appended to the run dir). TensorBoard is optional and only imported
if requested; the Trainer never imports a hosted backend.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Protocol


class Logger(Protocol):
    def log(self, step: int, metrics: dict) -> None: ...
    def log_artifact(self, path: str) -> None: ...
    def close(self) -> None: ...


class CSVJSONLLogger:
    """Appends metrics to `metrics.csv` and `metrics.jsonl`. No service, no
    network. Header is written from the first record's keys."""

    def __init__(self, run_dir: Path, tensorboard: bool = False):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.run_dir / "metrics.csv"
        self.jsonl_path = self.run_dir / "metrics.jsonl"
        self._fields: list[str] | None = None
        self._tb = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb = SummaryWriter(log_dir=str(self.run_dir / "tb"))
            except Exception as exc:  # tensorboard not installed -> degrade gracefully
                print(f"[logging] TensorBoard unavailable ({exc}); CSV/JSONL only.")

    def log(self, step: int, metrics: dict) -> None:
        record = {"step": step, **metrics}
        with self.jsonl_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        if self._fields is None:
            self._fields = list(record.keys())
            with self.csv_path.open("w", newline="") as f:
                csv.DictWriter(f, fieldnames=self._fields).writeheader()
        with self.csv_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self._fields, extrasaction="ignore")
            w.writerow(record)
        if self._tb is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._tb.add_scalar(k, v, step)

    def log_artifact(self, path: str) -> None:  # noqa: D401 - protocol stub for CSV backend
        pass

    def close(self) -> None:
        if self._tb is not None:
            self._tb.close()
