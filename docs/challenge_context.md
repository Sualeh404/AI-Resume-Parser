# Challenge Context Notes

This note records how each provided file shaped the implementation.

## README.docx

The bundle overview confirms the core task: rank the top 100 candidates from a
100,000-row pool, produce a CSV with reasoning, validate locally, and provide a
repo plus sandbox link. It also names the main traps: keyword stuffers,
plain-language tier-5 candidates, behavioral twins, and honeypots. The top-100
honeypot rate above 10% is a disqualification risk.

## job_description.docx

The JD is for a Senior AI Engineer on Redrob's founding AI team. The real target
profile is not "many AI keywords"; it is production ownership of retrieval,
ranking, matching, search, recommendations, embeddings, vector/hybrid search,
evaluation metrics, and product-minded shipping.

Key implementation decisions from the JD:

- Career evidence dominates skills keywords.
- Real experience is computed from career history, not the summary field.
- Product-company experience is rewarded; services-only histories are penalized.
- CV/speech/robotics-only backgrounds need NLP/IR evidence to be trusted.
- Behavioral availability must down-weight inactive or unresponsive candidates.
- Location is relevant but softer than technical fit.

## redrob_signals_doc.docx

This file defines the 23 behavioral signals. The ranker uses the most direct
hiring-availability signals as a multiplier: last active date, recruiter response
rate, interview completion rate, open-to-work flag, notice period, and account
verification. Recency is anchored to the latest `last_active_date` in the input
file so scores are reproducible across future reruns.

## submission_spec.docx

The submission must be a UTF-8 CSV with exactly 100 data rows and columns in this
order:

```text
candidate_id,rank,score,reasoning
```

Scores must be non-increasing by rank, with deterministic tie-breaking. The code
must run in 5 minutes or less, CPU-only, within 16 GB RAM, with no network calls.
The hidden score heavily rewards the top 10: `0.50 * NDCG@10 + 0.30 * NDCG@50 +
0.15 * MAP + 0.05 * P@10`.

The reasoning column is manually reviewed for specific facts, JD connection,
honest concerns, no hallucination, variation, and rank consistency. That is why
`src/reasoning.py` uses score evidence directly.

## candidate_schema.json

The schema confirmed the nested structure:

- `profile`: title, company, location, headline, summary, nominal experience.
- `career_history`: roles, durations, current flag, descriptions.
- `skills`: name, proficiency, endorsements, duration months.
- `redrob_signals`: platform behavior and availability signals.

The loader preserves raw candidate dicts for feature extraction rather than
flattening nested arrays away.

## candidates.jsonl

Full pool: 100,000 candidates, about 465 MB uncompressed.

Confirmed baseline findings:

- 1,179 candidates have AI/ML-relevant titles after including senior/staff/lead
  variants.
- 255 candidates have many AI-looking skills but non-AI titles, matching the
  keyword-stuffer trap.
- 21 candidates have the strict honeypot signature of multiple expert skills
  with zero months of usage.
- Only a small candidate slice has both AI title and real product-company
  experience, so the final ranking should be narrow rather than broad.

## sample_candidates.json

Used for quick schema inspection and sample-path tests. The first candidate is a
clear keyword-stuffer example: backend/data-engineering history with many AI
skills listed.

## sample_submission.csv

Used only as a format reference. Its ranking quality is intentionally poor: it
places unrelated titles with AI skill keywords above real ML candidates, which
matches the JD's warning.

## validate_submission.py

The validator checks CSV extension, exact header order, exactly 100 data rows,
unique candidate IDs, ranks 1 through 100, non-increasing scores, and candidate
ID formatting. The generated `outputs/submission.csv` passes it.