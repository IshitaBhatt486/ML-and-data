"""Write the static figures to outputs/figures after checking the written claims."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sound_signal import figures, findings  # noqa: E402

findings.check()
figures.save_all()
