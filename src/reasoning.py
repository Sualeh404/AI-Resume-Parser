"""
reasoning.py

Generates the reasoning column for the submission CSV.

Each string is built from the same evidence already used to compute the score,
so the explanation can't drift from what actually drove the ranking. Target is
1–2 short sentences, matching the format of the sample_submission.csv examples
(~12 words each), not the verbose prose examples in the spec document.
"""

from __future__ import annotations

from scoring import CandidateScore

MAX_WORDS = 40


def _shorten(text: str, max_chars: int = 90) -> str:
    """Trim a long evidence string to its lead clause.

    Several evidence strings list multiple items separated by commas or
    semicolons. This keeps only the first clause so a single sentence
    doesn't turn into a data dump.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return text
    for sep in (";", ":"):
        idx = text.find(sep)
        if 0 < idx <= max_chars:
            return text[:idx]
    cut = text[:max_chars]
    last_comma = cut.rfind(",")
    if last_comma > 20:
        return cut[:last_comma]
    return cut.rsplit(" ", 1)[0]


def _best_concern(score: CandidateScore, career_fact_raw: str) -> str | None:
    """Return the most relevant concern, skipping one that duplicates the opener.

    Comparing against the full untruncated career fact (not the shortened opener)
    prevents the same text from appearing in both sentences.
    """
    for item in score.concerns:
        cleaned = " ".join(item.split())
        if cleaned and cleaned != career_fact_raw:
            return _shorten(cleaned, max_chars=80)
    return None


def generate_reasoning(score: CandidateScore, rank: int) -> str:
    """Build a 1–2 sentence reasoning string.

    Sentence 1 (always): title, company, real years, strongest evidence.
    Sentence 2 (optional): one specific concern, only if it adds something new.
    """
    career_fact_raw = " ".join(score.evidence["career"].split())
    career_fact = _shorten(career_fact_raw)
    opener = f"{score.title} at {score.company}, {score.real_years:.1f} yrs; {career_fact}."

    concern = _best_concern(score, career_fact_raw)
    reasoning = f"{opener} {concern}." if concern else opener
    reasoning = " ".join(reasoning.split())

    words = reasoning.split()
    if len(words) > MAX_WORDS:
        reasoning = " ".join(words[:MAX_WORDS]).rstrip(",;:") + "."

    return reasoning
