#!/usr/bin/env python3
"""
CLI entry point for the L2R offline training pipeline.

Usage:
    python scripts/train_l2r.py
    python scripts/train_l2r.py --feedback ./data/feedback.jsonl --model ./models/l2r.pkl

Reads click-signal events from a JSONL feedback log, extracts ranking features,
trains a LogisticRegression model, and writes the resulting pickle to --model.
Requires >=10 feedback samples; exits 1 with a clear message on fewer.
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path when invoked as `python scripts/train_l2r.py`
sys.path.insert(0, str(Path(__file__).parents[1]))

from app.learning.trainer import main

if __name__ == "__main__":
    main()
