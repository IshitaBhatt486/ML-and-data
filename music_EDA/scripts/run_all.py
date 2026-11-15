"""Rebuild the project from raw data, in order.

Usage:
  python scripts/run_all.py            everything, starting with the downloads
  python scripts/run_all.py --from-clean   reuse raw files
  python scripts/run_all.py --outputs      reuse cleaned data and rebuild results onward
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FULL = [
    ["scripts/01_download.py"],
    ["scripts/02_clean_merge.py"],
    ["scripts/03_musicbrainz.py", "300"],
]
OUTPUTS = [
    ["tests/test_stats.py"],
    ["scripts/04_analyze.py"],
    ["scripts/05_figures.py"],
    ["scripts/06_build_notebook.py"],
    ["scripts/07_build_site.py"],
]


def main() -> None:
    steps = FULL + OUTPUTS
    if "--from-clean" in sys.argv:
        steps = FULL[1:] + OUTPUTS
    if "--outputs" in sys.argv:
        steps = [["scripts/03_musicbrainz.py", "300", "--cached-only"]] + OUTPUTS
    for step in steps:
        print(f"\n>>> {' '.join(step)}", flush=True)
        subprocess.run([sys.executable, str(ROOT / step[0]), *step[1:]], check=True, cwd=ROOT)
    print("\nAll steps finished.")


if __name__ == "__main__":
    main()
