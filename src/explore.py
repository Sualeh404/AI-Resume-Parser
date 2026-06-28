"""
explore.py

Exploratory analysis scripts used to understand the dataset structure before
writing any scoring logic. Not called during ranking — run these manually
against the full candidates.jsonl to verify or update constants.py.
"""

from __future__ import annotations

import collections
import re

from constants import (
    AI_TITLES,
    AI_SKILL_NAMES,
    CV_SPEECH_TITLES,
    FICTIONAL_FILLER_COMPANIES,
    IT_SERVICES_COMPANIES,
    JD_PREFERRED_CITIES,
    OVERRIDE_PATTERNS,
    REAL_PRODUCT_COMPANIES,
)
from load_candidates import iter_candidates


def title_distribution(path: str) -> collections.Counter:
    titles = collections.Counter()
    for c in iter_candidates(path):
        titles[c["profile"]["current_title"]] += 1
    return titles


def company_distribution(path: str) -> collections.Counter:
    companies = collections.Counter()
    for c in iter_candidates(path):
        companies[c["profile"]["current_company"]] += 1
    return companies


def company_distribution_by_location(path: str, city_substrings: set[str]) -> collections.Counter:
    companies = collections.Counter()
    for c in iter_candidates(path):
        loc = (c["profile"].get("location") or "").lower()
        if any(city in loc for city in city_substrings):
            companies[c["profile"]["current_company"]] += 1
    return companies


def find_keyword_stuffers(path: str, ai_skill_names: set[str], min_ai_skills: int = 5) -> list[dict]:
    """Candidates with many AI-sounding skills but a non-AI title.

    These are exactly the profiles that would top a naive keyword-matching
    ranker, and exactly what the sample_submission.csv fell into.
    """
    stuffers = []
    for c in iter_candidates(path):
        skill_names = {s["name"] for s in c["skills"]}
        ai_count = len(skill_names & ai_skill_names)
        title = c["profile"]["current_title"].lower()
        is_ai = any(kw in title for kw in ("ml", "machine learning", "ai ", "data scientist", "nlp", "research engineer"))
        if ai_count >= min_ai_skills and not is_ai:
            stuffers.append({
                "candidate_id": c["candidate_id"],
                "title": c["profile"]["current_title"],
                "company": c["profile"]["current_company"],
                "ai_skill_count": ai_count,
            })
    return stuffers


def find_honeypot_signature(path: str, min_zero_duration_experts: int = 2) -> list[dict]:
    """Candidates claiming expert proficiency in a skill with zero months of use."""
    flagged = []
    for c in iter_candidates(path):
        zero_dur = [
            s["name"]
            for s in c["skills"]
            if s["proficiency"] == "expert" and s.get("duration_months", 1) == 0
        ]
        if len(zero_dur) >= min_zero_duration_experts:
            flagged.append({
                "candidate_id": c["candidate_id"],
                "title": c["profile"]["current_title"],
                "zero_duration_expert_skills": zero_dur,
            })
    return flagged


def find_real_ml_fits(path: str, ai_titles: set[str], product_companies: set[str]) -> list[dict]:
    """Candidates with a genuine AI/ML title at a real product company."""
    override_re = re.compile("|".join(OVERRIDE_PATTERNS), flags=re.IGNORECASE)

    def career_text(c: dict) -> str:
        return " ".join(
            ch.get("title", "") + " " + ch.get("description", "")
            for ch in c.get("career_history", [])
        ).lower()

    def has_nlp_ir(c: dict) -> bool:
        return any(k in career_text(c) for k in (
            "nlp", "retriev", "embedding", "search", "ranking", "recommendation", "information retriev"
        ))

    fits = []
    for c in iter_candidates(path):
        title = c["profile"]["current_title"]
        company = c["profile"]["current_company"]

        if company not in product_companies:
            continue

        ai_relevant = title in ai_titles
        if title in CV_SPEECH_TITLES and not has_nlp_ir(c):
            ai_relevant = False
        if not ai_relevant and override_re.search(career_text(c)):
            ai_relevant = True
        if not ai_relevant:
            continue

        loc = (c["profile"].get("location") or "").lower()
        if not any(city in loc for city in JD_PREFERRED_CITIES):
            continue

        fits.append({
            "candidate_id": c["candidate_id"],
            "title": title,
            "company": company,
            "years_of_experience": c["profile"]["years_of_experience"],
            "location": c["profile"]["location"],
        })

    return fits


if __name__ == "__main__":
    PATH = "data/candidates.jsonl"

    print("=" * 70)
    print("1. TITLE DISTRIBUTION (top 15 + AI-relevant titles)")
    print("=" * 70)
    titles = title_distribution(PATH)
    for t, n in titles.most_common(15):
        print(f"  {n:>6,}  {t}")
    ai_counts = {t: titles[t] for t in AI_TITLES if t in titles}
    for t, n in sorted(ai_counts.items(), key=lambda x: -x[1]):
        print(f"  {n:>6,}  {t}   <- AI-relevant")
    print(f"\n  Total AI-relevant: {sum(ai_counts.values()):,} / {sum(titles.values()):,}")

    print()
    print("=" * 70)
    print("2. COMPANY DISTRIBUTION")
    print("=" * 70)
    companies = company_distribution(PATH)
    companies_pref = company_distribution_by_location(PATH, JD_PREFERRED_CITIES)
    print(f"  Real product companies (preferred cities): {sum(companies_pref[c] for c in REAL_PRODUCT_COMPANIES if c in companies_pref):,}")
    print(f"  IT-services companies: {sum(companies[c] for c in IT_SERVICES_COMPANIES if c in companies):,}")
    print(f"  Fictional filler: {sum(companies[c] for c in FICTIONAL_FILLER_COMPANIES if c in companies):,}")

    print()
    print("=" * 70)
    print("3. KEYWORD STUFFERS (5+ AI skills, non-AI title)")
    print("=" * 70)
    stuffers = find_keyword_stuffers(PATH, AI_SKILL_NAMES)
    print(f"  Found {len(stuffers):,}")
    for s in stuffers[:5]:
        print(f"    {s['candidate_id']}  {s['title']!r}  @ {s['company']!r}  ai_skills={s['ai_skill_count']}")

    print()
    print("=" * 70)
    print("4. HONEYPOT SIGNATURES (2+ expert skills, 0 months duration)")
    print("=" * 70)
    honeypots = find_honeypot_signature(PATH)
    print(f"  Found {len(honeypots):,}")
    for h in honeypots[:5]:
        print(f"    {h['candidate_id']}  {h['title']!r}  {h['zero_duration_expert_skills']}")

    print()
    print("=" * 70)
    print("5. REAL ML FITS (AI title + real product company + preferred location)")
    print("=" * 70)
    fits = find_real_ml_fits(PATH, AI_TITLES, REAL_PRODUCT_COMPANIES)
    print(f"  Found {len(fits):,}")
    for f in fits[:10]:
        print(f"    {f['candidate_id']}  {f['title']} @ {f['company']}  {f['years_of_experience']}yrs  {f['location']}")
