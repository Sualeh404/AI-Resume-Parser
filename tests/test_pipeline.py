from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from features import honeypot_flags
from load_candidates import iter_candidates
from reasoning import generate_reasoning
from scoring import rank_candidates


def test_iter_candidates_reads_sample_json_array():
    candidates = list(iter_candidates(ROOT / "data" / "sample_candidates.json"))
    assert len(candidates) == 50
    assert candidates[0]["candidate_id"] == "CAND_0000001"


def test_honeypot_flags_zero_duration_experts():
    candidate = {
        "skills": [
            {"name": "Python", "proficiency": "expert", "duration_months": 0},
            {"name": "RAG", "proficiency": "expert", "duration_months": 0},
        ]
    }
    assert honeypot_flags(candidate)


def test_rank_candidates_sample_sorted_and_reasoned():
    ranked = rank_candidates(ROOT / "data" / "sample_candidates.json", limit=5)
    assert len(ranked) == 5
    assert ranked == sorted(ranked, key=lambda s: (-s.rank_score, s.candidate_id))

    for idx, score in enumerate(ranked, start=1):
        reasoning = generate_reasoning(score, idx)
        assert score.candidate_id.startswith("CAND_")
        assert score.title in reasoning
        assert reasoning
