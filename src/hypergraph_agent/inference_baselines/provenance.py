"""Provenance of the X2 offline benchmark (no heavy imports, so the ARMS
worker can stamp its results cheaply).

``code_sha256``   the package's code hash, computed exactly as
                  ``training.run.LOADED_CODE_SHA256`` (every ``*.py`` file of the
                  package with its relative path), once per process
``msgi_sha256``   the hash of the vendored MSGI files (``third_party/msgi``)
``versions``      the versions of python and of the libraries the methods use
``cache_key``     the key of the benchmark cache: code, vendored files, options
                  and versions; a cached row or ARMS result is reused only under
                  the same key
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
MSGI = Path(__file__).resolve().parents[3] / "third_party" / "msgi"
LIBRARIES = ("macq", "nnf", "python-sat", "torch", "numpy")
_CODE: str | None = None


def code_sha256(fresh: bool = False) -> str:
    global _CODE
    if _CODE is None or fresh:
        h = hashlib.sha256()
        for f in sorted(PKG.rglob("*.py")):
            h.update(str(f.relative_to(PKG)).encode())
            h.update(f.read_bytes())
        if fresh:
            return h.hexdigest()
        _CODE = h.hexdigest()
    return _CODE


def msgi_sha256() -> str | None:
    files = [MSGI / "graph" / "ILP.py", MSGI / "graph" / "graph_utils.py", MSGI / "LICENSE"]
    if not all(f.exists() for f in files):
        return None
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def versions() -> dict:
    out = {"python": platform.python_version()}
    for name in LIBRARIES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def cache_key(options: dict) -> str:
    body = {"code": code_sha256(), "msgi": msgi_sha256(), "options": options, "versions": versions()}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:16]
