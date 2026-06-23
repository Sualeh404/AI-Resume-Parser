"""
explore.py

Exploratory analysis of the candidate pool, focused on finding and
documenting the trap structure described in the JD and README:
  - keyword-stuffer skills vs. actual career history
  - filler/noise titles vs. genuine AI/ML titles
  - real product companies vs. fictional filler companies vs.
    IT-services-only companies (explicitly down-weighted by the JD)
  - honeypot signature: "expert" skill proficiency with ~0 months used
"""

from __future__ import annotations

import collections
import json

from constants import (
    AI_TITLES,
    FICTIONAL_FILLER_COMPANIES,
    IT_SERVICES_COMPANIES,
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


def find_keyword_stuffers(path: str, ai_skill_names: set[str], min_ai_skills: int = 5) -> list[dict]:
    """Candidates with many AI-sounding skills listed, but whose current
    title is clearly unrelated to AI/ML work. These are the trap the
    sample_submission.csv fell into."""
    stuffers = []
    for c in iter_candidates(path):
        skill_names = {s["name"] for s in c["skills"]}
        ai_skill_count = len(skill_names & ai_skill_names)
        title = c["profile"]["current_title"].lower()
        is_ai_title = any(
            kw in title
            for kw in ("ml", "machine learning", "ai ", "data scientist", "nlp", "research engineer")
        )
        if ai_skill_count >= min_ai_skills and not is_ai_title:
            stuffers.append(
                {
                    "candidate_id": c["candidate_id"],
                    "title": c["profile"]["current_title"],
                    "company": c["profile"]["current_company"],
                    "ai_skill_count": ai_skill_count,
                }
            )
    return stuffers


def find_honeypot_signature(path: str, min_zero_duration_expert_skills: int = 2) -> list[dict]:
    """Candidates claiming 'expert' proficiency in a skill with 0 (or
    near-0) months of actual use. This is the honeypot pattern called
    out explicitly in submission_spec.md section 7."""
    flagged = []
    for c in iter_candidates(path):
        zero_dur_experts = [
            s["name"]
            for s in c["skills"]
            if s["proficiency"] == "expert" and s.get("duration_months", 1) == 0
        ]
        if len(zero_dur_experts) >= min_zero_duration_expert_skills:
            flagged.append(
                {
                    "candidate_id": c["candidate_id"],
                    "title": c["profile"]["current_title"],
                    "zero_duration_expert_skills": zero_dur_experts,
                }
            )
    return flagged


def find_real_ml_fits(path: str, ai_titles: set[str], product_companies: set[str]) -> list[dict]:
    """Candidates with a genuine AI/ML title AND at a real product
    company (not IT services, not fictional filler). These are
    plausible high-tier candidates worth manually inspecting."""
    fits = []
    for c in iter_candidates(path):
        title = c["profile"]["current_title"]
        company = c["profile"]["current_company"]
        if title in ai_titles and company in product_companies:
            fits.append(
                {
                    "candidate_id": c["candidate_id"],
                    "title": title,
                    "company": company,
                    "years_of_experience": c["profile"]["years_of_experience"],
                    "location": c["profile"]["location"],
                }
            )
    return fits


AI_SKILL_NAMES = {
    "NLP", "Fine-tuning LLMs", "LoRA", "Speech Recognition", "Image Classification",
    "GANs", "Prompt Engineering", "Haystack", "Kubeflow", "Weights & Biases",
    "Milvus", "TTS", "BentoML", "Feature Engineering",
}


if __name__ == "__main__":
    PATH = "data/candidates.jsonl"

    print("=" * 70)
    print("1. TITLE DISTRIBUTION (top 15 + AI-relevant tail)")
    print("=" * 70)
    titles = title_distribution(PATH)
    for t, n in titles.most_common(15):
        print(f"  {n:>6,}  {t}")
    print("  ...")
    ai_title_counts = {t: titles[t] for t in AI_TITLES if t in titles}
    for t, n in sorted(ai_title_counts.items(), key=lambda x: -x[1]):
        print(f"  {n:>6,}  {t}   <- AI-relevant")
    print(f"\n  Total AI-relevant titled candidates: {sum(ai_title_counts.values()):,} / {sum(titles.values()):,}")

    print()
    print("=" * 70)
    print("2. COMPANY DISTRIBUTION (categorized)")
    print("=" * 70)
    companies = company_distribution(PATH)
    real_product_total = sum(companies[c] for c in REAL_PRODUCT_COMPANIES if c in companies)
    it_services_total = sum(companies[c] for c in IT_SERVICES_COMPANIES if c in companies)
    filler_total = sum(companies[c] for c in FICTIONAL_FILLER_COMPANIES if c in companies)
    print(f"  Real product companies (Swiggy, CRED, Razorpay, ...): {real_product_total:,}")
    print(f"  IT-services-only companies (TCS, Infosys, Wipro, ...): {it_services_total:,}")
    print(f"  Fictional filler companies (Wayne Ent., Stark Ind., ...): {filler_total:,}")
    print(f"  Other/unique companies: {sum(companies.values()) - real_product_total - it_services_total - filler_total:,}")

    print()
    print("=" * 70)
    print("3. KEYWORD-STUFFER TRAP (5+ AI skills listed, but non-AI title)")
    print("=" * 70)
    stuffers = find_keyword_stuffers(PATH, AI_SKILL_NAMES, min_ai_skills=5)
    print(f"  Found {len(stuffers):,} candidates matching this pattern.")
    for s in stuffers[:5]:
        print(f"    {s['candidate_id']}  title={s['title']!r}  company={s['company']!r}  ai_skills={s['ai_skill_count']}")

    print()
    print("=" * 70)
    print("4. HONEYPOT SIGNATURE (2+ 'expert' skills with 0 months duration)")
    print("=" * 70)
    honeypots = find_honeypot_signature(PATH, min_zero_duration_expert_skills=2)
    print(f"  Found {len(honeypots):,} candidates matching this pattern.")
    for h in honeypots[:5]:
        print(f"    {h['candidate_id']}  title={h['title']!r}  skills={h['zero_duration_expert_skills']}")

    print()
    print("=" * 70)
    print("5. REAL ML/AI FITS (AI title + real product company)")
    print("=" * 70)
    fits = find_real_ml_fits(PATH, AI_TITLES, REAL_PRODUCT_COMPANIES)
    print(f"  Found {len(fits):,} candidates matching this pattern.")
    for f in fits[:10]:
        print(f"    {f['candidate_id']}  {f['title']} @ {f['company']}  {f['years_of_experience']}yrs  {f['location']}")