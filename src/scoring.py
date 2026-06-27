"""
Composite scoring for the Redrob candidate-ranking challenge.

This module intentionally keeps the ranker small and explainable. Each score is
made from feature functions that return both a numeric value and an evidence
string; reasoning.py later reuses the same evidence rather than inventing a
separate justification.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

from career_templates import score_career_history
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


BASE_WEIGHTS = {
    "career": 0.35,
    "title": 0.15,
    "experience": 0.15,
    "company": 0.10,
    "skills": 0.15,
    "location": 0.10,
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


def reference_date_from_candidates(candidates_path: str | Path) -> date:
    """Use the dataset's latest activity date as a deterministic recency anchor."""
    latest: date | None = None
    for candidate in iter_candidates(candidates_path):
        last_active = date.fromisoformat(candidate["redrob_signals"]["last_active_date"])
        latest = last_active if latest is None else max(latest, last_active)
    if latest is None:
        raise ValueError(f"No candidates found in {candidates_path}")
    return latest


def score_candidate(candidate: dict, reference_date: date) -> CandidateScore | None:
    """Score one candidate. Returns None for hard-excluded honeypots."""
    flags = honeypot_flags(candidate)
    if flags:
        return None

    real_years = compute_real_experience_years(candidate)
    career_score, career_evidence = score_career_history(candidate)
    title_ev = title_tier_score(candidate)
    exp_ev = experience_band_fit(real_years)
    company_ev = company_type_score(candidate)
    skill_ev = skill_trust_score(candidate)
    location_ev = location_fit_score(candidate)
    behavior_ev = behavioral_availability_multiplier(candidate, reference_date)

    component_scores = {
        "career": career_score,
        "title": title_ev.score,
        "experience": exp_ev.score,
        "company": company_ev.score,
        "skills": skill_ev.score,
        "location": location_ev.score,
    }
    base_fit = sum(component_scores[name] * BASE_WEIGHTS[name] for name in BASE_WEIGHTS)
    rank_score = base_fit * behavior_ev.score

    evidence = {
        "career": "; ".join(career_evidence[:3]),
        "title": title_ev.fact,
        "experience": exp_ev.fact,
        "company": company_ev.fact,
        "skills": skill_ev.fact,
        "location": location_ev.fact,
        "behavior": behavior_ev.fact,
    }

    concerns = []
    if experience_field_is_suspect(candidate):
        profile_years = candidate["profile"]["years_of_experience"]
        concerns.append(
            f"profile says {profile_years:.1f} yrs but career history sums to {real_years:.1f} yrs"
        )
    for name, score in sorted(component_scores.items(), key=lambda item: item[1]):
        if score < 0.75:
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


def rank_candidates(candidates_path: str | Path, limit: int = 100) -> list[CandidateScore]:
    """Return the top ranked candidates sorted by score descending."""
    reference_date = reference_date_from_candidates(candidates_path)
    scored: list[CandidateScore] = []
    for candidate in iter_candidates(candidates_path):
        candidate_score = score_candidate(candidate, reference_date)
        if candidate_score is not None:
            scored.append(candidate_score)

    scored.sort(key=lambda item: (-item.rank_score, item.candidate_id))
    return scored[:limit]


def count_hard_honeypots(candidates: Iterable[dict]) -> int:
    return sum(1 for candidate in candidates if honeypot_flags(candidate))
