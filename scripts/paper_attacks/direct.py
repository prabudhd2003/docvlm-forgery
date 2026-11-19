#!/usr/bin/env python3
"""E5a/E5b/E5c: adaptive EBV attacks and budget selection."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from attacks.paper_runner import main

if __name__ == "__main__":
    main(("E5a", "E5b", "E5c"))
