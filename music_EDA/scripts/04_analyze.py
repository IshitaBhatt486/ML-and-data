"""Run the investigations, each in its own process to keep memory use low.

Usage: python scripts/04_analyze.py [overview emotion popularity ...]
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ORDER = ["overview", "emotion", "popularity", "evolution", "genre", "sadness", "geography", "hits", "good", "dashboard"]


def run_one(name: str) -> None:
    from sound_signal import analysis, analysis_charts

    module = analysis if hasattr(analysis, name) and name not in {"read", "save"} else analysis_charts
    getattr(module, name)()


def main() -> None:
    requested = sys.argv[1:]
    if len(requested) == 2 and requested[0] == "--one":
        run_one(requested[1])
        return
    for name in requested or ORDER:
        started = time.time()
        print(f"=== {name}", flush=True)
        subprocess.run([sys.executable, __file__, "--one", name], check=True)
        print(f"    {time.time() - started:.0f}s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
