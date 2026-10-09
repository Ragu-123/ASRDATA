#!/usr/bin/env python3
"""Root-level launcher for the distributed transcription engine."""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from transcribe.main import main

if __name__ == "__main__":
    main()
