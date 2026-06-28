"""
features.py

One function per scoring dimension, each returning a score and the
specific fact that produced it. reasoning.py pulls these facts directly,
so the reasoning text always reflects what actually drove the score.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import re

from constants import (
    AI_TITLES,
    CV_SPEECH_TITLES,
    FICTIONAL_FILLER_COMPANIES,
    IT_SERVICES_COMPANIES,
    JD_PREFERRED_CITIES,
    OVERRIDE_PATTERNS,
    REAL_PRODUCT_COMPANIES,
)

_OVERRIDE_RE = re.compile("|".join(OVERRIDE_PATTERNS), re.IGNORECASE)

# JD target band: 5–9 years.
JD_MIN_YEARS = 5.0
JD_MAX_YEARS = 9.0

DEFAULT_REFERENCE_DATE = date(2026, 5, 27)

CORE_RETRIEVAL_SKILLS = {
    "BM25", "Content Matching", "Elasticsearch", "Embeddings", "FAISS",
    "Haystack", "Information Retrieval", "Information Retrieval Systems",
    "Learning to Rank", "LlamaIndex", "Milvus", "OpenSearch", "Pinecone",
    "Qdrant", "RAG", "Ranking Systems", "Recommendation Systems",
    "Search & Discovery", "Search Backend", "Search Infrastructure",
    "Semantic Search", "Sentence Transformers", "Vector Search",
    "Weaviate", "pgvector",
}

SUPPORTING_AI_SKILLS = {
    "BentoML", "Deep Learning", "Feature Engineering", "Fine-tuning LLMs",
    "Hugging Face Transformers", "Kubeflow", "LLMs", "LoRA",
    "Machine Learning", "MLflow", "MLOps", "Natural Language Processing",
    "NLP", "PEFT", "Prompt Engineering", "PyTorch", "Python", "QLoRA",
    "scikit-learn", "TensorFlow", "Text Encoders", "Vector Representations",
    "Weights & Biases",
}

PROFICIENCY_WEIGHT = {
    "beginner": 0.25,
    "intermediate": 0.55,
    "advanced": 0.80,
    "expert": 1.00,
}


@dataclass
class Evidence:
    """Score + the human-readable fact that produced it."""
    score: float
    fact: str


def compute_real_experience_years(candidate: dict) -> float:
    """Sum career_history durations instead of trusting the profile field.

    profile.years_of_experience is wrong for ~48 candidates in the full pool
    (12 of them in the AI-titled pool, roughly 20x over-represented).
    The JD explicitly warns about this — a headline number can be inflated.
    """
    months = sum(ch["duration_months"] for ch in candidate["career_history"])
    return round(months / 12, 1)


def experience_field_is_suspect(candidate: dict, tolerance_years: float = 2.0) -> bool:
    """True if the profile's stated years disagree with the career history by more than 2 years."""
    return abs(candidate["profile"]["years_of_experience"] - compute_real_experience_years(candidate)) > tolerance_years


def experience_band_fit(real_years: float) -> Evidence:
    """Score fit against the JD's 5–9 year band.

    1.0 inside the band, linear decay outside it. A 16-year veteran isn't
    wrong — just probably overqualified for what the role needs.
    """
    if JD_MIN_YEARS <= real_years <= JD_MAX_YEARS:
        return Evidence(1.0, f"{real_years:.1f} yrs experience, within target 5-9yr band")
    if real_years < JD_MIN_YEARS:
        gap = JD_MIN_YEARS - real_years
        return Evidence(max(0.0, 1.0 - gap / 3.0), f"{real_years:.1f} yrs experience, {gap:.1f}yrs below target band")
    gap = real_years - JD_MAX_YEARS
    return Evidence(max(0.0, 1.0 - gap / 6.0), f"{real_years:.1f} yrs experience, {gap:.1f}yrs above target band")


def title_tier_score(candidate: dict) -> Evidence:
    """Title-based eligibility gate.

    AI-titled candidates pass. CV/speech titles get a partial penalty because
    the JD specifically calls out that background as risky for a search/retrieval
    role. Non-AI titles can still pass if the override pattern fires on career
    history text — but as of the current dataset, that never happens.
    """
    title = candidate["profile"]["current_title"]

    if title in AI_TITLES:
        if title in CV_SPEECH_TITLES:
            return Evidence(0.6, f"title '{title}' is AI-relevant but CV-focused; JD prefers NLP/IR background")
        return Evidence(1.0, f"title '{title}' is directly AI/ML-relevant")

    full_text = " ".join(ch["description"] for ch in candidate["career_history"])
    if _OVERRIDE_RE.search(full_text):
        return Evidence(0.7, f"title '{title}' is not AI-labeled, but career history shows direct ranking/retrieval ownership")

    return Evidence(0.0, f"title '{title}' is not AI/ML-relevant")


def company_type_score(candidate: dict) -> Evidence:
    """Score based on product vs. IT-services vs. filler company history.

    A candidate currently at TCS isn't penalized if they also worked at Swiggy —
    the JD explicitly says prior product experience is fine.
    """
    current = candidate["profile"]["current_company"]
    all_companies = {current} | {ch["company"] for ch in candidate["career_history"]}

    has_product = bool(all_companies & REAL_PRODUCT_COMPANIES)
    has_only_services = bool(all_companies & IT_SERVICES_COMPANIES) and not has_product
    filler_only = all_companies <= FICTIONAL_FILLER_COMPANIES

    if has_product:
        matched = sorted(all_companies & REAL_PRODUCT_COMPANIES)
        return Evidence(1.0, f"has product-company experience ({', '.join(matched)})")
    if has_only_services:
        matched = sorted(all_companies & IT_SERVICES_COMPANIES)
        return Evidence(0.3, f"only IT-services experience ({', '.join(matched)}), no product-company background")
    if filler_only:
        return Evidence(0.5, "no recognizable product-company or IT-services experience on record")
    return Evidence(0.6, f"current company '{current}' is a smaller/unlisted firm")


def location_fit_score(candidate: dict) -> Evidence:
    """Score based on JD-preferred cities, with partial credit for relocation willingness."""
    location = candidate["profile"]["location"].lower()
    country = candidate["profile"]["country"]
    willing = candidate["redrob_signals"]["willing_to_relocate"]

    if any(city in location for city in JD_PREFERRED_CITIES):
        return Evidence(1.0, f"based in {candidate['profile']['location']}, a preferred location")
    if country == "India" and willing:
        return Evidence(0.7, f"based in {candidate['profile']['location']}, elsewhere in India, willing to relocate")
    if country == "India":
        return Evidence(0.4, f"based in {candidate['profile']['location']}, not willing to relocate")
    return Evidence(0.2, f"based outside India ({candidate['profile']['location']})")


def _duration_weight(months: int) -> float:
    if months <= 0:
        return 0.0
    if months < 6:
        return 0.35
    if months < 12:
        return 0.55
    if months < 24:
        return 0.80
    return 1.00


def _endorsement_weight(endorsements: int) -> float:
    return min(1.0, 0.70 + endorsements / 60.0)


def skill_trust_score(candidate: dict) -> Evidence:
    """Score JD-relevant skills by proficiency, duration, and endorsements.

    Skill presence alone means nothing here — zero-duration skills score zero.
    The component is also capped so a great skills section can't substitute for
    weak career history.
    """
    relevant = []
    for skill in candidate.get("skills", []):
        name = skill["name"]
        if name in CORE_RETRIEVAL_SKILLS:
            category_weight = 1.0
        elif name in SUPPORTING_AI_SKILLS:
            category_weight = 0.45
        else:
            continue

        value = (
            category_weight
            * PROFICIENCY_WEIGHT.get(skill.get("proficiency"), 0.0)
            * _duration_weight(skill.get("duration_months", 0))
            * _endorsement_weight(skill.get("endorsements", 0))
        )
        if value > 0:
            relevant.append((value, name, skill.get("duration_months", 0), skill.get("proficiency", "unknown")))

    if not relevant:
        return Evidence(0.0, "no trusted JD-relevant skills with nonzero duration")

    relevant.sort(reverse=True)
    score = min(1.0, sum(v for v, _, _, _ in relevant) / 4.5)
    top = [f"{name} ({prof}, {months}mo)" for _, name, months, prof in relevant[:5]]
    return Evidence(score, "trusted JD skills: " + ", ".join(top))


def honeypot_flags(candidate: dict) -> list[str]:
    """Check for the honeypot pattern: expert-level skills with zero months of use.

    21 candidates in the full pool match this. All of them sit outside the
    AI-titled pool — the title filter already screens them out before this runs.
    Kept as a hard exclude rather than a penalty because zero-duration expert
    claims aren't a matter of degree; they're a fabrication signal.
    """
    zero_duration_experts = [
        s["name"]
        for s in candidate.get("skills", [])
        if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0
    ]
    if len(zero_duration_experts) >= 2:
        shown = ", ".join(zero_duration_experts[:6])
        return [f"{len(zero_duration_experts)} expert skills with 0 months duration ({shown})"]
    return []


def _recency_score(days_since_active: int) -> float:
    if days_since_active <= 7:
        return 1.0
    if days_since_active <= 30:
        return 0.90
    if days_since_active <= 90:
        return 0.75
    if days_since_active <= 180:
        return 0.55
    return 0.35


def _rate_score(rate: float, low: float = 0.15, good: float = 0.75) -> float:
    if rate >= good:
        return 1.0
    if rate <= low:
        return 0.25
    return 0.25 + 0.75 * ((rate - low) / (good - low))


def _notice_score(days: int) -> float:
    if days <= 30:
        return 1.0
    if days <= 60:
        return 0.85
    if days <= 90:
        return 0.65
    if days <= 120:
        return 0.50
    return 0.35


def behavioral_availability_multiplier(candidate: dict, reference_date: date | None = None) -> Evidence:
    """Return a 0.50–1.00x multiplier based on how reachable the candidate is.

    The JD says to down-weight unavailable candidates, not disqualify them.
    Five candidates in the pool have perfect career scores but haven't been
    active in months and barely respond to recruiters — without this multiplier
    they'd dominate the top of the ranking despite being unreachable.
    """
    reference_date = reference_date or DEFAULT_REFERENCE_DATE
    signals = candidate["redrob_signals"]
    days_since_active = max(0, (reference_date - date.fromisoformat(signals["last_active_date"])).days)

    recency = _recency_score(days_since_active)
    response = _rate_score(float(signals["recruiter_response_rate"]))
    interview = _rate_score(float(signals["interview_completion_rate"]), low=0.30, good=0.90)
    open_to_work = 1.0 if signals["open_to_work_flag"] else 0.65
    notice = _notice_score(int(signals["notice_period_days"]))
    verified = sum(bool(signals.get(k)) for k in ("verified_email", "verified_phone", "linkedin_connected")) / 3.0

    availability = (
        0.25 * recency
        + 0.25 * response
        + 0.20 * interview
        + 0.15 * open_to_work
        + 0.10 * notice
        + 0.05 * verified
    )
    multiplier = 0.50 + 0.50 * availability

    status = "open to work" if signals["open_to_work_flag"] else "not marked open to work"
    fact = (
        f"availability {multiplier:.2f}x: active {days_since_active}d ago, "
        f"response {signals['recruiter_response_rate']:.2f}, "
        f"interview {signals['interview_completion_rate']:.2f}, "
        f"{status}, notice {signals['notice_period_days']}d"
    )
    return Evidence(multiplier, fact)


if __name__ == "__main__":
    # Quick smoke test on a couple of known candidates
    from load_candidates import iter_candidates

    targets = {"CAND_0091534", "CAND_0000031"}
    for c in iter_candidates("data/candidates.jsonl"):
        if c["candidate_id"] not in targets:
            continue
        print(f"\n=== {c['candidate_id']} ({c['profile']['current_title']} @ {c['profile']['current_company']}) ===")
        real = compute_real_experience_years(c)
        print(f"  profile yoe={c['profile']['years_of_experience']}  real={real}  suspect={experience_field_is_suspect(c)}")
        for fn in (experience_band_fit(real), title_tier_score(c), company_type_score(c), location_fit_score(c)):
            print(f"  {fn.score:.2f}  {fn.fact}")
