"""Session budget ledger and per-run meters.

Every primitive environment transition executed by a training, discovery,
validation or evaluation command is charged here. The ledger persists to
``runs/ledger.json`` so separate commands and restarts share one allowance.

Adaptive interactions (exploration, skill practice, candidate validation,
pretraining, any evaluation used for selection) count against the adaptive
cap and an allocation; non-adaptive reporting evaluation has its own cap.
Shared pretraining that is executed once and copied into several arms is
recorded physically once and charged logically to every arm.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path

DEFAULT_ALLOCATIONS = {
    "smoke": 6_000,
    "p1": 36_000,
    "p2": 36_000,
    "p3": 66_000,
    "diagnostics": 16_000,
}
ADAPTIVE_CAP = 160_000
REPORTING_CAP = 20_000
PURPOSES = ("exploration", "skill_practice", "candidate_validation", "pretraining",
            "evaluation_selection", "reporting_eval")


class BudgetExhausted(RuntimeError):
    pass


class _FileLock:
    """Exclusive lock file so concurrent commands update one ledger safely."""

    def __init__(self, path: Path, timeout: float = 60.0):
        self.path, self.timeout = path.with_suffix(".lock"), timeout

    def __enter__(self):
        t0 = time.time()
        while True:
            try:
                self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                return self
            except FileExistsError:
                if time.time() - t0 > self.timeout:
                    raise TimeoutError(f"ledger lock {self.path} held too long")
                time.sleep(0.05)

    def __exit__(self, *exc):
        os.close(self.fd)
        os.remove(self.path)


class SessionLedger:
    """Persistent session ledger. With a path, every mutation is a locked
    read-modify-write of the JSON file, so separate processes share one
    allowance; without a path it lives in memory (tests)."""

    def __init__(self, path: str | Path | None = None, allocations: dict | None = None,
                 adaptive_cap: int = ADAPTIVE_CAP, reporting_cap: int = REPORTING_CAP):
        self.path = Path(path) if path is not None else None
        self.state = {
            "adaptive_cap": adaptive_cap, "reporting_cap": reporting_cap,
            "allocations": dict(allocations or DEFAULT_ALLOCATIONS),
            "adaptive_total": 0, "reporting_total": 0,
            "by_allocation": {}, "runs": {}, "logical_charges": [],
        }
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._refresh()

    def _refresh(self):
        if self.path is not None and self.path.exists():
            self.state = json.loads(self.path.read_text())

    def _mutate(self, fn):
        if self.path is None:
            fn(self.state)
            return
        with _FileLock(self.path):
            self._refresh()
            fn(self.state)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.state, indent=1))
            os.replace(tmp, self.path)

    # --------------------------------------------------------------- caps
    def remaining(self, allocation: str, kind: str = "adaptive") -> int:
        self._refresh()
        s = self.state
        if kind == "reporting":
            return s["reporting_cap"] - s["reporting_total"]
        if allocation not in s["allocations"]:
            raise KeyError(f"unknown allocation {allocation!r}")
        used = s["by_allocation"].get(allocation, 0)
        return max(0, min(s["allocations"][allocation] - used, s["adaptive_cap"] - s["adaptive_total"]))

    def register_run(self, run_id: str, allocation: str, run_cap: int, meta: dict | None = None):
        def fn(s):
            if run_id in s["runs"]:
                raise ValueError(f"run id {run_id!r} already exists in the ledger")
            s["runs"][run_id] = {"allocation": allocation, "run_cap": run_cap, "used": 0,
                                 "by_purpose": {}, "started": time.time(), "meta": meta or {}}
        self._mutate(fn)

    def record(self, run_id: str, allocation: str, n: int, by_purpose: dict, kind: str):
        def fn(s):
            run = s["runs"].setdefault(run_id, {"allocation": allocation, "used": 0, "by_purpose": {}})
            run["used"] += n
            for k, v in by_purpose.items():
                run["by_purpose"][k] = run["by_purpose"].get(k, 0) + v
            if kind == "reporting":
                s["reporting_total"] += n
            else:
                s["adaptive_total"] += n
                s["by_allocation"][allocation] = s["by_allocation"].get(allocation, 0) + n
        self._mutate(fn)

    def logical_charge(self, run_id: str, source_run: str, n: int, note: str):
        """Charge reused computation to an arm's scientific learning cost
        without consuming physical session budget again."""
        self._mutate(lambda s: s["logical_charges"].append(
            {"run_id": run_id, "source_run": source_run, "n": n, "note": note}))

    def save(self):
        self._mutate(lambda s: None)

    def summary(self) -> dict:
        self._refresh()
        s = self.state
        return {
            "adaptive_total": s["adaptive_total"], "adaptive_cap": s["adaptive_cap"],
            "reporting_total": s["reporting_total"], "reporting_cap": s["reporting_cap"],
            "by_allocation": dict(s["by_allocation"]),
            "allocations": dict(s["allocations"]),
        }


class BudgetMeter:
    """Per-run counter. The cap is fixed at construction from the run cap and
    the ledger's remaining allowance; counts are flushed to the ledger."""

    def __init__(self, ledger: SessionLedger, run_id: str, allocation: str, run_cap: int,
                 kind: str = "adaptive", flush_every: int = 512):
        if kind not in ("adaptive", "reporting"):
            raise ValueError("kind must be adaptive or reporting")
        self.ledger, self.run_id, self.allocation, self.kind = ledger, run_id, allocation, kind
        self.cap = min(run_cap, ledger.remaining(allocation, kind))
        self.used = 0
        self.by_purpose: Counter = Counter()
        self._unflushed = 0
        self._unflushed_purpose: Counter = Counter()
        self.flush_every = flush_every
        self.stop_reason: str | None = None

    def can_charge(self, n: int = 1) -> bool:
        return self.used + n <= self.cap

    def charge(self, n: int = 1, purpose: str = "exploration"):
        if purpose not in PURPOSES:
            raise ValueError(f"unknown purpose {purpose!r}")
        if self.used + n > self.cap:
            self.stop_reason = "interaction_cap"
            raise BudgetExhausted(f"run {self.run_id}: cap {self.cap} reached")
        self.used += n
        self.by_purpose[purpose] += n
        self._unflushed += n
        self._unflushed_purpose[purpose] += n
        if self._unflushed >= self.flush_every:
            self.flush()

    def flush(self):
        if self._unflushed:
            self.ledger.record(self.run_id, self.allocation, self._unflushed,
                               dict(self._unflushed_purpose), self.kind)
            self._unflushed = 0
            self._unflushed_purpose = Counter()

    @property
    def remaining(self) -> int:
        return self.cap - self.used
