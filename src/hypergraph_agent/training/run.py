"""Run directories, manifests and append-only logs.

Each run directory holds ``manifest.json`` (rewritten as the run progresses),
``config.json``, append-only ``events.jsonl`` and ``episodes.jsonl``, the
working-tree patch when the source tree is dirty, and checkpoints. An existing
run id is never reused: a rerun is a new run.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

import gymnasium

from .. import METHOD_VERSION, __version__
from ..envs.generator import stable_hash


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout if out.returncode == 0 else None


def source_info(repo: Path) -> dict:
    commit = _git(["rev-parse", "HEAD"], repo)
    status = _git(["status", "--porcelain", "--untracked-files=no"], repo)
    diff = _git(["diff", "HEAD"], repo) or ""
    return {
        "commit": commit.strip() if commit else None,
        "dirty": bool(status and status.strip()),
        "dirty_patch_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
        "_patch": diff,
    }


def environment_info() -> dict:
    return {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "torch": torch.__version__, "numpy": np.__version__, "gymnasium": gymnasium.__version__,
        "torch_threads": torch.get_num_threads(), "device": "cpu",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "package": __version__, "method_version": METHOD_VERSION,
    }


class RunContext:
    def __init__(self, runs_dir: str | Path, run_id: str, config: dict, *, phase: str, arm: str,
                 seed: int, parent_run: str | None = None, repo: Path | None = None):
        self.run_id = run_id
        self.dir = Path(runs_dir) / run_id
        if self.dir.exists():
            raise FileExistsError(f"run {run_id} exists; start a new run or an explicit child run")
        self.dir.mkdir(parents=True)
        repo = repo or Path(__file__).resolve().parents[3]
        src = source_info(repo)
        patch = src.pop("_patch")
        if patch:
            (self.dir / "dirty.patch").write_text(patch)
        (self.dir / "config.json").write_text(json.dumps(config, indent=1, default=str))
        self.t0 = time.time()
        self.manifest = {
            "run_id": run_id, "parent_run": parent_run, "phase": phase, "arm": arm, "seed": seed,
            "status": "running", "method_version": METHOD_VERSION,
            "config_hash": stable_hash(config), "source": src, "environment": environment_info(),
            "start_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.t0)),
        }
        self.write_manifest()

    def write_manifest(self, **updates):
        self.manifest.update(updates)
        tmp = self.dir / "manifest.tmp"
        tmp.write_text(json.dumps(self.manifest, indent=1, default=str))
        tmp.replace(self.dir / "manifest.json")

    def _append(self, name: str, record: dict):
        with open(self.dir / name, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"t": round(time.time() - self.t0, 3), **record}, default=str) + "\n")

    def event(self, record: dict):
        self._append("events.jsonl", record)

    def episode(self, record: dict):
        self._append("episodes.jsonl", record)

    def finish(self, status: str, **updates):
        self.write_manifest(status=status, end_time=time.strftime("%Y-%m-%dT%H:%M:%S"),
                            wallclock_s=round(time.time() - self.t0, 1), **updates)

    @property
    def elapsed(self) -> float:
        return time.time() - self.t0
