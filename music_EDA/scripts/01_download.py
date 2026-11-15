"""Download every raw source once, resumably, and record a checksum manifest.

Raw files are never modified after download. Re-running skips complete files.
Usage: python scripts/01_download.py [source_key ...]
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sound_signal.config import DOCS_DIR, RAW_DIR, SOURCES, USER_AGENT  # noqa: E402

CHUNK = 1024 * 1024


def remote_size(url: str) -> int | None:
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Range": "bytes=0-0"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        content_range = response.headers.get("Content-Range", "")
    return int(content_range.rsplit("/", 1)[1]) if "/" in content_range else None


def fetch(url: str, target: Path, retries: int = 8) -> None:
    partial = target.with_suffix(target.suffix + ".part")
    expected = remote_size(url)
    if target.exists() and (expected is None or target.stat().st_size == expected):
        print(f"  already complete: {target.name}", flush=True)
        return
    for attempt in range(1, retries + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": USER_AGENT}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                mode = "ab" if offset and response.status == 206 else "wb"
                with partial.open(mode) as sink:
                    while chunk := response.read(CHUNK):
                        sink.write(chunk)
            if expected is None or partial.stat().st_size == expected:
                partial.replace(target)
                return
            raise OSError(f"short read: {partial.stat().st_size} of {expected}")
        except (urllib.error.URLError, OSError, TimeoutError) as error:
            print(f"  attempt {attempt} failed: {error}", flush=True)
            time.sleep(min(60, 2**attempt))
    raise SystemExit(f"Could not download {url}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = DOCS_DIR / "raw_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    wanted = sys.argv[1:] or list(SOURCES)
    # Smallest first, so a slow large file cannot hold up the others.
    for key in wanted:
        source = SOURCES[key]
        target = RAW_DIR / source["file"]
        print(f"[{key}] {source['url']}", flush=True)
        fetch(source["url"], target)
        checksum = sha256(target)
        previous = manifest.get(key, {})
        if previous.get("sha256") not in (None, checksum):
            print(f"  WARNING: {key} differs from the recorded checksum", flush=True)
        retrieved = (
            previous.get("retrieved_utc")
            if previous.get("sha256") == checksum
            else datetime.now(timezone.utc).isoformat(timespec="seconds")
        )
        manifest[key] = {
            **{name: source[name] for name in ("title", "url", "page", "licence", "role")},
            "file": source["file"],
            "bytes": target.stat().st_size,
            "sha256": checksum,
            "retrieved_utc": retrieved,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"  ok {target.stat().st_size / 1e6:.1f} MB", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
