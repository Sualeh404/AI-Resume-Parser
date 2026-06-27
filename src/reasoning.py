"""
Reasoning generation for the submission CSV.

The reasoning column is graded for factual specificity and consistency with
the rank. These strings are generated from CandidateScore evidence, not from a
separate language model, so the explanation stays tied to the score.
"""

from __future__ import annotations

from scoring import CandidateScore


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        cleaned = " ".join(item.split())
        if cleaned and cleaned not in seen:
            out.append(cleaned)
            seen.add(cleaned)
    return out


def _top_component_facts(score: CandidateScore) -> list[str]:
    priority = ["career", "company", "skills", "location", "title", "experience"]
    facts = []
    for name in priority:
        if score.component_scores.get(name, 0.0) >= 0.75:
            facts.append(score.evidence[name])
    return _dedupe(facts)


def generate_reasoning(score: CandidateScore, rank: int) -> str:
    """Return a 1-2 sentence, submission-ready explanation."""
    facts = _top_component_facts(score)
    career = score.evidence["career"]
    opener = (
        f"{score.title} at {score.company} with {score.real_years:.1f} real yrs; "
        f"career evidence: {career}."
    )

    supporting = []
    for fact in facts:
        if fact != career:
            supporting.append(fact)
        if len(supporting) == 2:
            break

    caveats = _dedupe(score.concerns)
    caveat = ""
    for item in caveats:
        if item not in opener and item not in supporting:
            caveat = item
            break

    if caveat:
        second = f"Also: {'; '.join(supporting)}; caveat: {caveat}; {score.evidence['behavior']}."
    elif supporting:
        second = f"Also: {'; '.join(supporting)}; {score.evidence['behavior']}."
    else:
        second = score.evidence["behavior"] + "."

    reasoning = f"{opener} {second}"
    return " ".join(reasoning.split())
