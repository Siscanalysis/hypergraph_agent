"""Where benchmark files live and how their checksums are verified.

Every benchmark has a folder ``benchmarks/<name>/`` at the repository root with
a ``manifest.json``: the files that belong to it, each with its source URL,
local path (relative to the folder), SHA-256 and whether it is committed to the
repository or must be fetched (``python benchmarks/<name>/fetch.py``, into the
folder's ``data/`` subdirectory, which is not tracked).

Checksums are those of the bytes as published upstream, and only exact bytes
pass. Committed third-party files are marked ``-text`` in
``benchmarks/.gitattributes`` so that Git never converts their line endings.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

BENCHMARKS_DIR = Path(os.environ.get(
    "HYPERGRAPH_BENCHMARKS_DIR", Path(__file__).resolve().parents[3] / "benchmarks"))


class DataMissing(FileNotFoundError):
    """A fetch-only benchmark whose data has not been downloaded."""


def folder(name: str) -> Path:
    return BENCHMARKS_DIR / name


def manifest(name: str) -> dict:
    return json.loads((folder(name) / "manifest.json").read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(name: str, which: str = "all") -> dict[str, str]:
    """Status per manifest file: ``ok``, ``missing`` or ``mismatch``.
    ``which``: ``all``, ``committed`` or ``fetched``."""
    out = {}
    for f in manifest(name)["files"]:
        if which != "all" and f["status"] != which:
            continue
        p = folder(name) / f["path"]
        if not p.exists():
            out[f["path"]] = "missing"
        else:
            out[f["path"]] = "ok" if f["sha256"] == sha256(p) else "mismatch"
    return out


def require(name: str, rel_path: str) -> Path:
    """Path of a manifest file after verifying its checksum."""
    entry = next((f for f in manifest(name)["files"] if f["path"] == rel_path), None)
    if entry is None:
        raise KeyError(f"{rel_path} is not listed in benchmarks/{name}/manifest.json")
    p = folder(name) / rel_path
    if not p.exists():
        hint = (f"run `python benchmarks/{name}/{manifest(name).get('fetch_script', 'fetch.py')}`"
                if entry["status"] == "fetched" else "the committed file is missing")
        raise DataMissing(f"benchmarks/{name}/{rel_path} is absent: {hint}")
    if entry["sha256"] != sha256(p):
        raise ValueError(f"benchmarks/{name}/{rel_path}: SHA-256 does not match the manifest")
    return p
