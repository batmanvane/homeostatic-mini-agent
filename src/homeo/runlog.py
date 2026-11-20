"""Run directory: config.yaml next to steps.jsonl and activations.npz (hidden rows and, phase 3 hybrid, state sketches)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml


class RunLog:
    def __init__(self, out_dir: Path, config: dict):
        self.dir = out_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        self._steps = (self.dir / "steps.jsonl").open("w")
        self._hidden: list[np.ndarray] = []
        self._sketch: list[np.ndarray] = []

    def step(self, record: dict, hidden: np.ndarray | None = None, sketch: np.ndarray | None = None) -> None:
        self._steps.write(json.dumps(record) + "\n")
        self._steps.flush()
        if hidden is not None:
            self._hidden.append(hidden)
        if sketch is not None:
            self._sketch.append(sketch)

    def close(self) -> Path:
        self._steps.close()
        arrays = {k: np.stack(v) for k, v in (("hidden", self._hidden), ("sketch", self._sketch)) if v}
        if arrays:
            np.savez_compressed(self.dir / "activations.npz", **arrays)
        return self.dir
