#!/usr/bin/env python3
"""E7: three receipt targets and short/long incorrect resume labels."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from attacks.paper_runner import main

if __name__ == "__main__":
    main(("E7",))
