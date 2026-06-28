"""
load_candidates.py

Streams the candidate pool (JSONL or JSONL.GZ) one record at a time,
or reads the bundled sample JSON array for testing.

We don't use pandas json_normalize on the full file. With 100k records
and nested lists for career_history and skills, normalize either creates
one row per nested item or silently drops the nesting — neither is usable.
Instead we pull only the flat fields needed for cheap filtering, and keep
the full raw dict around for anything that needs the nested structure.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterator


def iter_candidates(path: str | Path) -> Iterator[dict]:
    """Yield one candidate dict at a time.

    Handles:
    - Full pool: candidates.jsonl or candidates.jsonl.gz
    - Sample: sample_candidates.json (plain JSON array)
    """
    path = Path(path)
    suffixes = [s.lower() for s in path.suffixes]

    if suffixes[-1:] == [".json"]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("candidates", [])
        yield from data
        return

    opener = gzip.open if suffixes[-2:] == [".jsonl", ".gz"] else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_candidates_df(path: str | Path):
    """Load all candidates into a DataFrame.

    Flattens the scalar fields you'd want to filter or sort on,
    and keeps the full raw dict in a `raw` column for feature extraction.
    """
    import pandas as pd

    rows = []
    for c in iter_candidates(path):
        profile = c["profile"]
        signals = c["redrob_signals"]
        rows.append({
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
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    import time

    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/candidates.jsonl"

    t0 = time.time()
    df = load_candidates_df(data_path)
    elapsed = time.time() - t0

    print(f"Loaded {len(df):,} candidates in {elapsed:.1f}s")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print(df.drop(columns=["raw"]).iloc[0])
