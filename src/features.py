"""
features.py

Feature extraction functions, one concern per function, each one
traceable back to a specific fact in the candidate record. This is
what makes the reasoning-generation step honest later: every score
component has a corresponding human-readable fact already computed
here, so reasoning.py never has to invent justification separately
from what actually drove the score.

Each function takes a raw candidate dict (the `raw` column from
load_candidates.load_candidates_df) and returns either a score or a
small dict of (score, evidence_string) so downstream code can explain
itself.
"""

from __future__ import annotations

from datetime import date
import re
from dataclasses import dataclass

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

# JD target experience band: 5-9 years. Candidates outside this band
# aren't auto-disqualified (a 4.5-year candidate with a perfect
# profile is still worth a look) but get a penalty that grows with
# distance from the band.
JD_MIN_YEARS = 5.0
JD_MAX_YEARS = 9.0
DEFAULT_REFERENCE_DATE = date(2026, 5, 27)

CORE_RETRIEVAL_SKILLS = {
    "BM25",
    "Content Matching",
    "Elasticsearch",
    "Embeddings",
    "FAISS",
    "Haystack",
    "Information Retrieval",
    "Information Retrieval Systems",
    "Learning to Rank",
    "LlamaIndex",
    "Milvus",
    "OpenSearch",
    "Pinecone",
    "Qdrant",
    "RAG",
    "Ranking Systems",
    "Recommendation Systems",
    "Search & Discovery",
    "Search Backend",
    "Search Infrastructure",
    "Semantic Search",
    "Sentence Transformers",
    "Vector Search",
    "Weaviate",
    "pgvector",
}

SUPPORTING_AI_SKILLS = {
    "BentoML",
    "Deep Learning",
    "Feature Engineering",
    "Fine-tuning LLMs",
    "Hugging Face Transformers",
    "Kubeflow",
    "LLMs",
    "LoRA",
    "Machine Learning",
    "MLflow",
    "MLOps",
    "Natural Language Processing",
    "NLP",
    "PEFT",
    "Prompt Engineering",
    "PyTorch",
    "Python",
    "QLoRA",
    "scikit-learn",
    "TensorFlow",
    "Text Encoders",
    "Vector Representations",
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
    """A score paired with the human-readable fact that produced it.
    Used directly by reasoning.py so generated reasoning text can never
    drift from what actually drove the ranking."""

    score: float
    fact: str


def compute_real_experience_years(candidate: dict) -> float:
    """Sum career_history durations rather than trusting
    profile.years_of_experience, which is unreliable for ~0.05% of
    candidates (concentrated in the AI-titled pool — see constants.py).
    """
    months = sum(ch["duration_months"] for ch in candidate["career_history"])
    return round(months / 12, 1)


def experience_field_is_suspect(candidate: dict, tolerance_years: float = 2.0) -> bool:
    """True if profile.years_of_experience disagrees with the summed
    career history by more than `tolerance_years`. This is itself a
    useful signal: a large, unexplained mismatch is either a data
    error or a candidate's self-reported number inflating their
    seniority — either way, the computed value should be trusted over
    the self-reported one, and the mismatch is worth noting in the
    candidate's reasoning text as a transparency point.
    """
    profile_yoe = candidate["profile"]["years_of_experience"]
    real_yoe = compute_real_experience_years(candidate)
    return abs(profile_yoe - real_yoe) > tolerance_years


def experience_band_fit(real_years: float) -> Evidence:
    """Score how well real (career-history-derived) experience fits
    the JD's stated 5-9 year band. 1.0 inside the band, decaying
    linearly outside it. Capped at 0 beyond a 6-year miss in either
    direction (e.g. a 16-year veteran is not a junior-disguised-as-
    senior fit for this role, regardless of skill quality)."""
    if JD_MIN_YEARS <= real_years <= JD_MAX_YEARS:
        return Evidence(1.0, f"{real_years:.1f} yrs experience, within target 5-9yr band")
    if real_years < JD_MIN_YEARS:
        gap = JD_MIN_YEARS - real_years
        score = max(0.0, 1.0 - gap / 3.0)  # full penalty by 2yr under
        return Evidence(score, f"{real_years:.1f} yrs experience, {gap:.1f}yrs below target band")
    gap = real_years - JD_MAX_YEARS
    score = max(0.0, 1.0 - gap / 6.0)  # gentler decay above the band
    return Evidence(score, f"{real_years:.1f} yrs experience, {gap:.1f}yrs above target band")


def title_tier_score(candidate: dict) -> Evidence:
    """Title-based eligibility gate. Returns 0.0 for titles outside
    the AI-relevant pool UNLESS the override pattern fires on career
    history (see constants.OVERRIDE_PATTERNS — as of this dataset,
    zero non-AI-titled candidates trigger this, but the mechanism
    stays since the JD explicitly asks for this kind of reasoning).

    Within the AI-relevant pool, CV_SPEECH_TITLES get a moderate
    penalty per the JD's explicit note that pure computer-vision/
    speech/robotics background without NLP/IR exposure means
    "re-learning fundamentals" for this role — that penalty is
    overridden if their career history shows real NLP/IR/retrieval
    work (checked separately in career_narrative_score).
    """
    title = candidate["profile"]["current_title"]

    if title in AI_TITLES:
        if title in CV_SPEECH_TITLES:
            return Evidence(0.6, f"title '{title}' is AI-relevant but CV-focused; JD prefers NLP/IR background")
        return Evidence(1.0, f"title '{title}' is directly AI/ML-relevant")

    # Override path: non-AI title, check career history for strong
    # production ranking/retrieval ownership language.
    full_text = " ".join(ch["description"] for ch in candidate["career_history"])
    if _OVERRIDE_RE.search(full_text):
        return Evidence(0.7, f"title '{title}' is not AI-labeled, but career history shows direct ranking/retrieval ownership")

    return Evidence(0.0, f"title '{title}' is not AI/ML-relevant and career history shows no override signal")


def company_type_score(candidate: dict) -> Evidence:
    """Score current + prior employers against the JD's explicit
    preference for product-company experience over pure IT-services /
    consulting. A candidate currently at an IT-services firm is NOT
    auto-penalized if their career history includes prior product-
    company experience — the JD states this explicitly.
    """
    current_company = candidate["profile"]["current_company"]
    all_companies = {current_company} | {ch["company"] for ch in candidate["career_history"]}

    has_product_company = bool(all_companies & REAL_PRODUCT_COMPANIES)
    has_only_it_services = bool(all_companies & IT_SERVICES_COMPANIES) and not has_product_company
    is_filler_only = all_companies <= FICTIONAL_FILLER_COMPANIES

    if has_product_company:
        matched = sorted(all_companies & REAL_PRODUCT_COMPANIES)
        return Evidence(1.0, f"has product-company experience ({', '.join(matched)})")
    if has_only_it_services:
        matched = sorted(all_companies & IT_SERVICES_COMPANIES)
        return Evidence(0.3, f"only IT-services/consulting experience ({', '.join(matched)}), no product-company background")
    if is_filler_only:
        return Evidence(0.5, "no recognizable product-company or IT-services experience on record")
    return Evidence(0.6, f"current company '{current_company}' is a smaller/unlisted firm, no major product-company experience found")


def location_fit_score(candidate: dict) -> Evidence:
    """Score based on JD-preferred cities, with a fallback for
    candidates elsewhere in India who are explicitly willing to
    relocate."""
    location = candidate["profile"]["location"].lower()
    country = candidate["profile"]["country"]
    willing = candidate["redrob_signals"]["willing_to_relocate"]

    in_preferred_city = any(city in location for city in JD_PREFERRED_CITIES)

    if in_preferred_city:
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
    """Score JD-relevant skills using duration and endorsements.

    Skill presence alone is deliberately not enough: zero-duration skills
    contribute nothing, and the whole component is capped so keyword stuffers
    cannot outrank candidates with real career-history evidence.
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
    top = [
        f"{name} ({prof}, {months}mo)"
        for _, name, months, prof in relevant[:5]
    ]
    return Evidence(score, "trusted JD skills: " + ", ".join(top))


def honeypot_flags(candidate: dict) -> list[str]:
    """Return hard honeypot flags.

    The clean signal in this dataset is multiple expert-level skills with
    zero months of usage. Looser duration-vs-career checks were tested and
    rejected because they falsely flag legitimate strong candidates.
    """
    zero_duration_experts = [
        skill["name"]
        for skill in candidate.get("skills", [])
        if skill.get("proficiency") == "expert" and skill.get("duration_months", 0) == 0
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
    """Return a multiplicative availability modifier.

    The JD says behavioral signals should down-weight unavailable candidates,
    not replace technical fit. We therefore convert behavior into a 0.50-1.00
    multiplier and apply it after the base fit score.
    """
    reference_date = reference_date or DEFAULT_REFERENCE_DATE
    signals = candidate["redrob_signals"]
    last_active = date.fromisoformat(signals["last_active_date"])
    days_since_active = max(0, (reference_date - last_active).days)

    recency = _recency_score(days_since_active)
    response = _rate_score(float(signals["recruiter_response_rate"]))
    interview = _rate_score(float(signals["interview_completion_rate"]), low=0.30, good=0.90)
    open_to_work = 1.0 if signals["open_to_work_flag"] else 0.65
    notice = _notice_score(int(signals["notice_period_days"]))
    verified = sum(
        bool(signals.get(k))
        for k in ("verified_email", "verified_phone", "linkedin_connected")
    ) / 3.0

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
        f"interview completion {signals['interview_completion_rate']:.2f}, "
        f"{status}, notice {signals['notice_period_days']}d"
    )
    return Evidence(multiplier, fact)


if __name__ == "__main__":
    # Smoke test against the two reference candidates we found by hand:
    # CAND_0091534 (the years_of_experience trap: field says 16.6,
    # real career history says 7.1) and CAND_0000031 (the clean
    # Swiggy recommendation-systems fit).
    from load_candidates import iter_candidates

    targets = {"CAND_0091534", "CAND_0000031", "CAND_0001056"}
    for c in iter_candidates("data/candidates.jsonl"):
        if c["candidate_id"] not in targets:
            continue
        print(f"\n=== {c['candidate_id']} ({c['profile']['current_title']} @ {c['profile']['current_company']}) ===")
        print(f"  profile.years_of_experience = {c['profile']['years_of_experience']}")
        print(f"  real (career-history-derived) years = {compute_real_experience_years(c)}")
        print(f"  experience_field_is_suspect = {experience_field_is_suspect(c)}")
        eb = experience_band_fit(compute_real_experience_years(c))
        print(f"  experience_band_fit = {eb.score:.2f}  ({eb.fact})")
        tt = title_tier_score(c)
        print(f"  title_tier_score = {tt.score:.2f}  ({tt.fact})")
        ct = company_type_score(c)
        print(f"  company_type_score = {ct.score:.2f}  ({ct.fact})")
        lf = location_fit_score(c)
        print(f"  location_fit_score = {lf.score:.2f}  ({lf.fact})")
