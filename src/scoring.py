"""
scoring.py

Composite scoring for the Redrob candidate ranking challenge.

Each feature function returns a score and an evidence string. The same
evidence is reused in reasoning.py rather than generating a separate
justification, so the explanation always matches what actually drove the rank.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

from career_templates import score_career_history
from constants import AI_TITLES, OVERRIDE_PATTERNS
from features import (
    behavioral_availability_multiplier,
    company_type_score,
    compute_real_experience_years,
    experience_band_fit,
    experience_field_is_suspect,
    honeypot_flags,
    location_fit_score,
    skill_trust_score,
    title_tier_score,
)
from load_candidates import iter_candidates

_OVERRIDE_RE = re.compile("|".join(OVERRIDE_PATTERNS), re.IGNORECASE)

BASE_WEIGHTS = {
    "career":     0.35,
    "title":      0.15,
    "experience": 0.15,
    "company":    0.10,
    "skills":     0.15,
    "location":   0.10,
}


@dataclass(frozen=True)
class CandidateScore:
    candidate_id: str
    rank_score: float
    base_fit_score: float
    behavioral_multiplier: float
    component_scores: dict[str, float]
    evidence: dict[str, str]
    concerns: list[str]
    title: str
    company: str
    location: str
    real_years: float


def score_candidate(candidate: dict, reference_date: date) -> CandidateScore | None:
    """Score one candidate. Returns None if flagged as a honeypot."""
    if honeypot_flags(candidate):
        return None

    real_years = compute_real_experience_years(candidate)
    career_score, career_evidence = score_career_history(candidate)
    title_ev   = title_tier_score(candidate)
    exp_ev     = experience_band_fit(real_years)
    company_ev = company_type_score(candidate)
    skill_ev   = skill_trust_score(candidate)
    location_ev = location_fit_score(candidate)
    behavior_ev = behavioral_availability_multiplier(candidate, reference_date)

    component_scores = {
        "career":     career_score,
        "title":      title_ev.score,
        "experience": exp_ev.score,
        "company":    company_ev.score,
        "skills":     skill_ev.score,
        "location":   location_ev.score,
    }
    base_fit = sum(component_scores[k] * BASE_WEIGHTS[k] for k in BASE_WEIGHTS)
    rank_score = base_fit * behavior_ev.score

    evidence = {
        "career":     "; ".join(career_evidence[:3]),
        "title":      title_ev.fact,
        "experience": exp_ev.fact,
        "company":    company_ev.fact,
        "skills":     skill_ev.fact,
        "location":   location_ev.fact,
        "behavior":   behavior_ev.fact,
    }

    concerns = []
    if experience_field_is_suspect(candidate):
        profile_yoe = candidate["profile"]["years_of_experience"]
        concerns.append(f"profile says {profile_yoe:.1f} yrs but career history sums to {real_years:.1f} yrs")
    for name, s in sorted(component_scores.items(), key=lambda item: item[1]):
        if s < 0.75:
            concerns.append(evidence[name])
    if behavior_ev.score < 0.82:
        concerns.append(behavior_ev.fact)

    profile = candidate["profile"]
    return CandidateScore(
        candidate_id=candidate["candidate_id"],
        rank_score=rank_score,
        base_fit_score=base_fit,
        behavioral_multiplier=behavior_ev.score,
        component_scores=component_scores,
        evidence=evidence,
        concerns=concerns,
        title=profile["current_title"],
        company=profile["current_company"],
        location=profile["location"],
        real_years=real_years,
    )


def _is_eligible(candidate: dict) -> bool:
    """Pre-filter before running expensive feature extraction.

    98.8% of the 100k candidates have titles that can never reach a
    competitive score (Accountants, HR Managers, etc.). Filtering those
    before running six feature functions cuts runtime by more than half
    with identical results.
    """
    if candidate["profile"]["current_title"] in AI_TITLES:
        return True
    full_text = " ".join(ch["description"] for ch in candidate["career_history"])
    return bool(_OVERRIDE_RE.search(full_text))


def rank_candidates(candidates_path: str | Path, limit: int = 100) -> list[CandidateScore]:
    """Stream the candidate file, score eligible candidates, return top N.

    Does one pass over the file: computes the reference date and collects
    eligible candidates simultaneously, then scores them.
    """
    eligible: list[dict] = []
    latest_active: date | None = None
    n_total = n_filtered = n_honeypot = 0

    for candidate in iter_candidates(candidates_path):
        n_total += 1
        last_active = date.fromisoformat(candidate["redrob_signals"]["last_active_date"])
        latest_active = last_active if latest_active is None else max(latest_active, last_active)

        if not _is_eligible(candidate):
            n_filtered += 1
            continue
        if honeypot_flags(candidate):
            n_honeypot += 1
            continue
        eligible.append(candidate)

    if latest_active is None:
        raise ValueError(f"No candidates found in {candidates_path}")

    scored = [s for c in eligible if (s := score_candidate(c, latest_active)) is not None]
    scored.sort(key=lambda s: (-s.rank_score, s.candidate_id))

    print(
        f"[rank_candidates] reference_date={latest_active} total={n_total} "
        f"filtered_out={n_filtered} honeypots={n_honeypot} scored={len(scored)}",
        file=sys.stderr,
    )

    return scored[:limit]
