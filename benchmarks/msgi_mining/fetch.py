"""Download this benchmark's source files and verify their SHA-256 checksums.

Reads ``manifest.json`` next to this script and fetches every entry with
``"status": "fetched"`` into this folder (the ``data/`` subdirectory, which is
not tracked by Git). Standard library only.

    python benchmarks/<name>/fetch.py              # required files
    python benchmarks/<name>/fetch.py --optional   # also the optional ones
    python benchmarks/<name>/fetch.py --check      # verify what is present, download nothing

A file whose checksum does not match is deleted and reported; the script then
exits with status 1. Files already present and verified are not downloaded
again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
USER_AGENT = "hypergraph-agent-benchmark-fetch/1"


def sha256(path: Path) -> str:
    """SHA-256 of the exact bytes (no line-ending normalization)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(dest)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="verify only, download nothing")
    ap.add_argument("--optional", action="store_true", help="include files marked optional")
    args = ap.parse_args(argv)
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    failed = 0
    for entry in manifest["files"]:
        if entry["status"] != "fetched" or (entry.get("optional") and not args.optional):
            continue
        dest = HERE / entry["path"]
        if dest.exists() and entry["sha256"] == sha256(dest):
            print(f"ok       {entry['path']}")
            continue
        if args.check:
            print(f"{'mismatch' if dest.exists() else 'missing'} {entry['path']}")
            failed += 1
            continue
        print(f"fetching {entry['path']} <- {entry['url']}")
        try:
            download(entry["url"], dest)
        except OSError as e:
            print(f"error    {entry['path']}: {e}")
            failed += 1
            continue
        if entry["sha256"] != sha256(dest):
            dest.unlink()
            print(f"mismatch {entry['path']}: SHA-256 differs from the manifest; file removed")
            failed += 1
        else:
            print(f"ok       {entry['path']}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
