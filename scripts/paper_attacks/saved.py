#!/usr/bin/env python3
"""E2a/E2b/E2c saved PNG attacks. See scripts/paper_attacks/README.md."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from attacks.paper_runner import main

if __name__ == "__main__":
    main(("E2a", "E2b", "E2c"))
