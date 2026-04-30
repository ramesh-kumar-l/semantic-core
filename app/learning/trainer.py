#!/usr/bin/env python3
"""
Offline trainer for the L2R ML model.

Usage:
    python -m app.learning.trainer
    python -m app.learning.trainer --feedback ./data/feedback.jsonl --model ./models/l2r.pkl
"""
import argparse
import sys
from pathlib import Path


def build_training_data(events: list) -> list:
    """Convert feedback events to feature-label rows."""
    rows = []
    for ev in events:
        results = ev.get("results", [])
        clicked = ev.get("clicked", "")
        for position, doc_id in enumerate(results):
            is_clicked = 1 if doc_id == clicked else 0
            rows.append({
                "vector_score": 1.0 / (position + 1),
                "bm25_score": 0.0,
                "combined_score": 1.0 / (position + 1),
                "rank_position": float(position),
                "inv_rank": 1.0 / (position + 1),
                "is_clicked": is_clicked,
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Train L2R model from feedback")
    parser.add_argument("--feedback", default="./data/feedback.jsonl")
    parser.add_argument("--model", default="./models/l2r.pkl")
    args = parser.parse_args()

    # Import here to allow running as a script
    sys.path.insert(0, str(Path(__file__).parents[2]))

    from app.learning.feedback_store import FeedbackStore
    from app.learning.l2r_model import L2RModel

    store = FeedbackStore(path=args.feedback)
    events = store.load_all()
    print(f"Loaded {len(events)} feedback events")

    data = build_training_data(events)
    print(f"Generated {len(data)} training samples ({sum(r['is_clicked'] for r in data)} positive)")

    if len(data) < 10:
        print(
            f"Insufficient training data: need ≥10 samples, got {len(data)}. "
            "Collect more feedback events and retry.",
            file=sys.stderr,
        )
        sys.exit(1)

    model = L2RModel()
    success = model.train(data)
    if success:
        model.save(args.model)
        print(f"Model saved to {args.model}")
    else:
        print("Training failed — check logs for details", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
