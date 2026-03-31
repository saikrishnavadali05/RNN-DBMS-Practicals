# -*- coding: utf-8 -*-
"""Delegate to repo-root ``main.py`` so ``python -m golden_ai`` works."""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from main import run_demo

if __name__ == "__main__":
    run_demo()
