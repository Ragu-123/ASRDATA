#!/usr/bin/env python3
"""Alias launcher for python segement.py -> points to segment pipeline."""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from segment.main import main

if __name__ == "__main__":
    main()
