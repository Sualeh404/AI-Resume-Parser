"""
load_candidates.py

Loads the Redrob candidate pool (JSONL) into a pandas DataFrame with
flattened top-level fields for fast filtering, plus the raw nested
record preserved for feature extraction that needs full structure
(career_history descriptions, skills list, redrob_signals dict).

Design note: we do NOT use a generic json_normalize on the whole file.
With 100k records and deeply nested lists (career_history, skills),
a naive normalize either explodes rows or silently drops nested data.
We extract exactly the scalar fields we need for cheap filtering/sorting,
and keep `raw` as the dict for everything else. This keeps memory bounded
and keeps every downstream feature traceable back to the source record.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterator


def iter_candidates(path: str | Path) -> Iterator[dict]:
    """Yield one parsed candidate dict at a time.

    Supports the released full pool as JSONL/JSONL.GZ and the bundled
    sample_candidates.json file as a small JSON array. The full JSONL path
    stays streaming; the JSON-array path is intended for small sandbox/demo
    inputs only.
    """
    path = Path(path)
    suffixes = [s.lower() for s in path.suffixes]

    if suffixes[-1:] == [".json"]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("candidates", [])
        for candidate in data:
            yield candidate
        return

    opener = gzip.open if suffixes[-2:] == [".jsonl", ".gz"] else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_candidates_df(path: str | Path) -> pd.DataFrame:
    """Load all candidates into a DataFrame with flattened scalar fields
    for filtering/sorting, and the full raw dict in the `raw` column for
    feature extraction.
    """
    import pandas as pd

    rows = []
    for c in iter_candidates(path):
        profile = c["profile"]
        signals = c["redrob_signals"]
        rows.append(
            {
                "candidate_id": c["candidate_id"],
                "current_title": profile["current_title"],
                "current_company": profile["current_company"],
                "current_company_size": profile["current_company_size"],
                "current_industry": profile["current_industry"],
                "years_of_experience": profile["years_of_experience"],
                "location": profile["location"],
                "country": profile["country"],
                "headline": profile["headline"],
                "summary": profile["summary"],
                "num_career_entries": len(c["career_history"]),
                "num_skills": len(c["skills"]),
                "open_to_work_flag": signals["open_to_work_flag"],
                "last_active_date": signals["last_active_date"],
                "recruiter_response_rate": signals["recruiter_response_rate"],
                "willing_to_relocate": signals["willing_to_relocate"],
                "notice_period_days": signals["notice_period_days"],
                "github_activity_score": signals["github_activity_score"],
                "raw": c,
            }
        )
    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    import sys
    import time

    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/candidates.jsonl"

    t0 = time.time()
    df = load_candidates_df(data_path)
    elapsed = time.time() - t0

    print(f"Loaded {len(df):,} candidates in {elapsed:.1f}s")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print()
    print("Columns:", list(df.columns))
    print()
    print("Sample row (excluding raw):")
    print(df.drop(columns=["raw"]).iloc[0])
