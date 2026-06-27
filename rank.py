#!/usr/bin/env python3
"""
Create a Redrob top-100 submission CSV.

Usage:
    python rank.py --candidates data/candidates.jsonl --out outputs/submission.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning import generate_reasoning
from scoring import rank_candidates


HEADER = ["candidate_id", "rank", "score", "reasoning"]


def write_submission(candidates_path: str | Path, out_path: str | Path, limit: int = 100) -> list:
    ranked = rank_candidates(candidates_path, limit=limit)
    if not ranked:
        raise ValueError(f"No rankable candidates found in {candidates_path}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        for rank, score in enumerate(ranked, start=1):
            writer.writerow(
                [
                    score.candidate_id,
                    rank,
                    f"{score.rank_score:.6f}",
                    generate_reasoning(score, rank),
                ]
            )
    return ranked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank Redrob candidates and write a submission CSV.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl, candidates.jsonl.gz, or sample_candidates.json")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--limit", type=int, default=100, help="Number of ranked candidates to write")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    ranked = write_submission(args.candidates, args.out, args.limit)
    elapsed = time.time() - t0
    print(f"Wrote {len(ranked)} rows to {args.out} in {elapsed:.1f}s")
    print(f"Top candidate: {ranked[0].candidate_id} score={ranked[0].rank_score:.6f}")


if __name__ == "__main__":
    main()
