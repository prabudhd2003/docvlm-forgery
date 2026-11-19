#!/usr/bin/env python3
"""E6: separate refusal and subtotal-forgery trajectories (120 total)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from attacks.paper_runner import main

if __name__ == "__main__":
    main(("E6",))
