"""
career_templates.py

CRITICAL FINDING: career_history[].description text in this dataset is
NOT unique per candidate. Across the entire 1,179-candidate AI-titled
pool, there are exactly 23 unique description strings shared across
2,636 career_history entries (114.6x average reuse). The same exact
paragraph appears on a Junior ML Engineer at Flipkart and a Senior
Software Engineer at Rephrase.ai.

This means career-narrative signal lives at the TEMPLATE level, not
the free-text level. A fuzzy keyword/regex scorer over this text would
just be a noisy way of re-discovering which of 23 buckets a sentence
belongs to. So instead: every template was read in full and manually
tier-classified below, once, with the reasoning documented. Scoring a
candidate's career_history then becomes an exact-string lookup against
this table — deterministic, fast, fully auditable, and the documented
reasoning IS the reasoning text we can quote later.

Templates are tiered 1 (irrelevant/disqualifying) to 5 (ideal JD match)
based on how directly the work matches the JD's actual ask: ranking,
retrieval, search relevance, RAG, recommendation systems, with real
production ownership (not "modeling was secondary" / "deployment
handled by platform team" disclaimers, which several templates
explicitly include as a self-disqualifying signal).

Tier 5 templates (17, 18, 22) describe almost exactly the JD's own
"ideal candidate" language: migrating keyword search to embedding
retrieval, hybrid sparse+dense retrieval, explicit NDCG@10 evaluation.
Template 14 describes a RAG ranking pipeline for "a recruiter-facing
search product" — strikingly close to what this hackathon itself asks
candidates to build, which is either a deliberate easter egg or simply
confirms the dataset's authors used the JD itself as a seed for the
top-tier templates.

If the dataset is regenerated with new template text, this table will
silently stop matching (lookups simply return tier 0 = unknown). The
__main__ block below double-checks against the live data file —
re-run it after any dataset update to catch drift immediately rather
than silently degrading.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TemplateInfo:
    tier: int  # 1 (irrelevant) to 5 (ideal JD match)
    label: str  # short human-readable category for reasoning text
    note: str  # why this tier, for documentation / interview defense


# Keyed by exact description string. Built from a full read of every
# unique template found in the AI-titled pool (see commit history for
# the extraction script). Tier 0 is the implicit default for any
# description NOT in this table (used as a drift-detection signal).
CAREER_TEMPLATE_TIERS: dict[str, TemplateInfo] = {
    "Contributed to ML feature engineering and model deployment for a fraud-detection product. My main role was engineering: building the Flask-based prediction API, integrating with the feature store, and writing the model-serving observability layer. I worked closely with senior data scientists but my own modeling work was secondary — I was the production-side engineer.": TemplateInfo(
        1, "fraud-detection ML infra (non-ranking)",
        "Solid production engineering, but unrelated domain and self-described as 'modeling secondary'."
    ),
    "Built recommendation-style features at a mid-stage startup — lighter weight than ranking systems at FAANG, but production. Used a combination of collaborative filtering (matrix factorization in implicit-feedback library) and gradient-boosted re-ranking over engagement signals. Pure ML side of the work; production deployment was handled by the platform team.": TemplateInfo(
        2, "lightweight recommendation work",
        "Recommendation-adjacent and relevant tech, but explicitly 'lighter weight' and deployment owned by another team — partial credit only."
    ),
    "Built computer vision models for our product's image moderation feature using PyTorch — fine-tuned ResNet variants on a labeled dataset of ~200K images. Set up the training pipeline (data loading, augmentation, evaluation) and the inference service. Most of my project work has been in CV; I'm now interested in transitioning toward NLP/LLM work but my professional experience there is limited.": TemplateInfo(
        1, "computer vision (non-NLP/IR)",
        "Matches the JD's explicit CV-without-NLP-exposure exclusion; the candidate's own words admit limited NLP/LLM experience."
    ),
    "Worked on time-series forecasting models for supply-chain demand prediction at a logistics company. Built models in Prophet, LightGBM, and (for one project) a small LSTM — the LightGBM model ended up shipping. Also ran some reinforcement learning experiments for dynamic pricing but those didn't make it to production. The work was a mix of modeling, analysis, and stakeholder communication with the operations team.": TemplateInfo(
        1, "time-series/forecasting (non-ranking)",
        "Real shipped ML, but forecasting/pricing domain has no overlap with ranking, retrieval, or candidate matching."
    ),
    "Worked on customer-facing predictive modeling for an e-commerce platform — churn prediction, conversion likelihood, lifetime value estimation. Used scikit-learn and XGBoost; main models were gradient-boosted trees with ~80 hand-engineered features. The work was split roughly 60/40 between modeling and data prep / SQL. The churn model is now used by the retention team, though my role was more on the modeling side than the productionization.": TemplateInfo(
        1, "churn/LTV predictive modeling",
        "Classic applied ML, but tabular prediction, not ranking/retrieval; explicitly modeling-focused over productionization."
    ),
    "Built NLP pipelines for sentiment analysis and document classification — primarily for an internal feedback-analytics dashboard. Started with sklearn-based bag-of-words models, then moved to transformer-based classifiers (DistilBERT) for the harder classes. Comfortable with PyTorch and Hugging Face but most of my training experience has been on small datasets and pre-trained model fine-tuning, not from-scratch model design.": TemplateInfo(
        2, "NLP classification (non-retrieval)",
        "Real NLP and transformer experience (relevant building blocks), but classification/sentiment, not search/ranking/retrieval; admits limited from-scratch design experience."
    ),
    "Owned the ranking layer for an e-commerce search product, evolving it from a hand-tuned scoring function to a learning-to-rank model over 9 months. Designed the relevance labeling pipeline (mix of click-through data and explicit human judgments), the feature pipeline, and the training/eval workflow. Most of the work was infrastructure and data quality — the modeling part was almost the easy bit. Final model improved revenue-per-search by 12%.": TemplateInfo(
        3, "e-commerce ranking ownership",
        "Genuine ranking ownership with eval pipeline and measured business impact — directly relevant skillset, though e-commerce relevance ranking rather than candidate-search/retrieval."
    ),
    "Trained and shipped multiple ranking models for our product's discovery feed using XGBoost and LightGBM. Designed features across three families: content metadata, user behavior signals, and item engagement history. Owned the offline-online correlation analysis that determined which offline metrics actually predicted A/B test outcomes. Worked closely with PMs to define the optimization target (click-through vs. dwell time vs. downstream conversion) — that work was as important as the modeling itself.": TemplateInfo(
        3, "discovery-feed ranking",
        "Strong ranking-model and offline/online eval-correlation experience — core transferable skill for this JD, applied to a feed rather than search/retrieval."
    ),
    "Developed a semantic search feature for an internal knowledge base of ~500K documents. Used sentence-transformers (all-MiniLM-L6-v2 initially, later upgraded to bge-base) with FAISS for fast nearest-neighbor retrieval. Designed the query expansion module that handles vocabulary mismatch between user queries and document terms. Reported search-relevance improvement of 35% over the prior Elasticsearch BM25 setup, validated through human relevance judgments.": TemplateInfo(
        4, "semantic search / FAISS retrieval",
        "Directly on-target: embeddings, FAISS, BM25 comparison, measured relevance improvement — this is the exact technical territory the JD describes."
    ),
    "Implemented a RAG-based customer support chatbot integrated with our existing ticketing system. Built the document ingestion pipeline (chunking, embedding via OpenAI embeddings, storing in Pinecone) and the answer-generation layer (initially GPT-4, then a fine-tuned smaller model for cost control). Designed the evaluation framework with both automatic metrics (BLEU, ROUGE) and human-in-the-loop quality scores. Deployment cut average ticket resolution time by 31% for the supported categories.": TemplateInfo(
        4, "RAG pipeline ownership",
        "Full RAG stack (ingestion, embeddings, vector DB, generation, eval) shipped with measured impact — strong transferable architecture experience, applied domain (support) differs from candidate-matching."
    ),
    "Built a content recommendation system serving 10M+ users that combined collaborative filtering with content-based ranking. The system uses item-item similarity (via sentence-transformer embeddings) for cold starts and a gradient-boosted model trained on engagement signals for warm users. Most of my time went into the feature pipeline (~200 features) and the A/B testing infrastructure. The launch improved 7-day retention by 6% and time spent per session by 14%.": TemplateInfo(
        4, "large-scale recommendation system",
        "Production recommendation system at real scale (10M+ users), embeddings for cold-start, measured business impact — strong, directly relevant experience."
    ),
    "Built and operated production ML pipelines using MLflow for experiment tracking, Kubeflow for orchestration, and our internal feature store. My main project was a churn prediction model that's now used by the customer success team to prioritize outreach. Designed the model monitoring stack: data drift detection, prediction distribution checks, and alerting. Mentored a junior engineer through their first end-to-end ML project last year.": TemplateInfo(
        2, "MLOps/platform (non-ranking)",
        "Strong MLOps and mentoring signal, but the flagship project (churn) is not ranking/retrieval; tier reflects general engineering strength without domain fit."
    ),
    "Fine-tuned LLaMA-2-7B and Mistral-7B variants using LoRA and QLoRA for domain-specific candidate-JD matching. Built the data curation pipeline that generated 200K high-quality preference pairs from recruiter labels, plus the eval harness using both ranking metrics and human-quality scores. Deployed the model via BentoML on Kubernetes with sub-200ms p95 latency by quantizing to INT8 and batching at the request level. Cost per inference dropped from $0.04 with GPT-3.5-fallback to under $0.001.": TemplateInfo(
        5, "LLM fine-tuning for candidate-JD matching",
        "About as direct as it gets: literally candidate-JD matching with LoRA fine-tuning, recruiter-label data curation, ranking-metric eval, and production deployment with real latency/cost numbers."
    ),
    "Built a RAG-based ranking pipeline serving 50M+ queries per month for an internal recruiter-facing search product. The architecture combined BM25 + dense retrieval (BGE embeddings, FAISS HNSW) with an LLM-based re-ranker on the top-50, falling back to a learning-to-rank model when latency budget was tight. Designed the offline evaluation framework from scratch — NDCG, MRR, recall@K calibrated against online A/B engagement metrics. Drove the migration over 4 months including the recruiter-feedback loop that surfaced reranking edge cases.": TemplateInfo(
        5, "recruiter-facing hybrid search + LLM re-ranking",
        "Near-literal description of this hackathon's own ask: hybrid retrieval, LLM re-ranking, NDCG/MRR eval, recruiter-facing search at real scale."
    ),
    "Built and shipped a production recommendation system at a marketplace product, going from offline experimentation to live A/B test in 5 months. The system combined collaborative filtering (matrix factorization), content-based features (TF-IDF + sentence-transformer embeddings), and a behavioral re-ranking layer. The most interesting technical challenge was the cold-start problem for new users; I designed an exploration-exploitation policy using Thompson sampling that improved new-user retention by 11% in the first month.": TemplateInfo(
        5, "marketplace recommendation with behavioral re-ranking",
        "End-to-end ownership (offline to live A/B), behavioral re-ranking, and a genuinely hard problem (cold-start via Thompson sampling) solved and measured."
    ),
    "Owned the end-to-end ranking pipeline at a recommendations-heavy consumer product: candidate sourcing → embedding generation (using a fine-tuned BGE-large) → Pinecone retrieval → learning-to-rank re-scoring (XGBoost) → behavioral-signal integration. The hardest part wasn't the ML — it was the evaluation: building offline metrics that actually predicted what the recommendation would do to live engagement. After three iterations we landed on a calibration approach using simulated A/B tests that has held up over the last 18 months.": TemplateInfo(
        5, "full ranking pipeline ownership with evaluation focus",
        "Uses the word 'candidate sourcing' literally; full pipeline ownership end to end, and explicitly names evaluation (not modeling) as the hard part — exactly the JD's framing."
    ),
    "Owned the design and rollout of a large-scale semantic search system serving an internal corpus of 35M+ items. Migrated the existing BM25-only retrieval to a hybrid setup combining sparse and dense vectors (sentence-transformers, MPNet-base initially, later fine-tuned BGE-large for our domain). The new system reduced p95 retrieval latency by 60% while improving NDCG@10 by 18% on our held-out eval set. Spent substantial time on the boring-but-critical parts: incremental index refresh, embedding drift monitoring, online/offline metric correlation. Led a team of 4 engineers across the rollout.": TemplateInfo(
        5, "BM25-to-hybrid-retrieval migration at scale",
        "Directly mirrors the JD's evaluation rubric: explicitly reports NDCG@10, led a hybrid sparse+dense migration at real scale, plus leadership."
    ),
    "Led the migration from keyword-based to embedding-based search across a 30M+ candidate corpus over 8 months. Designed three successive ranker variants and ran them in A/B testing alongside the legacy keyword system. The final embedding ranker improved recruiter engagement metrics by 24% and reduced the average time-to-shortlist by 38%. Most of the engineering effort went into the boring infrastructure: index versioning, embedding versioning, rollback paths, and the dashboards that let recruiters trust the new system. Mentored two junior engineers through this rollout.": TemplateInfo(
        5, "keyword-to-embedding migration on a candidate corpus",
        "This is the JD's own thesis statement, almost verbatim: keyword search failing recruiters, replaced by embedding-based ranking, with 'time-to-shortlist' as the measured outcome."
    ),
    "Built systems that understand what users are looking for and connect them to the most relevant matches across a large dataset. Worked at the intersection of infrastructure, algorithms, and product judgment — none of the three were optional. Recent project was a complete overhaul of the matching layer; took it from a hand-tuned heuristic system to one with explicit modeling and evaluation. The team grew from just me to 6 engineers over the course of that work.": TemplateInfo(
        4, "general matching-system ownership",
        "Strong matching/relevance framing and growth-of-scope signal, but vaguer on specific techniques (no embeddings/retrieval terms named) than the tier-5 templates."
    ),
    "Shipped the personalization infrastructure: the system that learns from user behavior and improves relevance over time. Designed the offline experimentation environment, the online A/B testing framework, and the feature-engineering pipeline that connected them. Most of my time went into the boring-but-critical operational layer — feature monitoring, drift detection, retraining cadence — rather than the modeling itself. Worked closely with the product and growth teams.": TemplateInfo(
        3, "personalization infrastructure",
        "Solid infra/experimentation ownership for relevance systems, but explicitly operational-layer-focused over modeling, and no ranking/retrieval specifics named."
    ),
    "Designed the ranking layer for the company's flagship product: how do we surface the right thing at the right time, across millions of items, for millions of users. The hard problem was rarely the modeling — it was the data pipeline that fed the models, the evaluation framework that told us whether they worked, and the operational discipline of keeping all of it healthy in production. I owned all three across roughly 14 months.": TemplateInfo(
        4, "flagship-product ranking ownership",
        "Strong end-to-end ranking ownership language and the JD's 'evaluation over modeling' framing, but no specific retrieval/embedding technique named."
    ),
    "Owned the search and discovery experience end-to-end at a consumer product, from how content is represented internally through to how the most relevant results appear for each user's intent. The work spanned data infrastructure, ranking algorithms, evaluation methodology, and direct collaboration with product/PM on what 'relevance' actually means for our users. Spent a fair amount of time on the eval side — building offline metrics that actually correlated with online engagement, which turned out to be the hardest part.": TemplateInfo(
        5, "end-to-end search/discovery ownership",
        "Explicitly frames 'what relevance means' and names evaluation as the hardest part — directly echoes the JD's own framing of the problem, even without naming specific tools."
    ),
    "Led the engineering team building infrastructure to surface relevant content to users at scale. The system processed billions of documents and served millions of queries with low latency. Most of the technical effort went into the boring-but-essential parts: index refresh, query understanding, ranking calibration, and the dashboards that made the system's behavior legible to product and business teams. I had a small team of 4 across this work.": TemplateInfo(
        4, "large-scale content-surfacing infra leadership",
        "Real scale (billions of documents) and leadership, with index/query/ranking infra language, but stays high-level without naming specific retrieval techniques."
    ),
}


def score_career_history(candidate: dict) -> tuple[float, list[str]]:
    """Score a candidate's full career history by template tier.
    Returns (score 0-1, list of evidence notes for the top contributing
    entries). Uses a recency-weighted max: the current role's tier
    counts at full weight, past roles count at a 0.85x discount. This
    matters for a small minority of candidates (26/1,179 have any
    tier-5 entry at all; of those, 22 have it as their CURRENT role and
    4 have it only in a past role) — a candidate whose most relevant
    work is a job behind them should rank slightly below an otherwise-
    identical candidate doing that work right now, without being
    penalized so harshly that real, relevant past experience stops
    mattering.
    """
    tiers_found = []
    for ch in candidate["career_history"]:
        info = CAREER_TEMPLATE_TIERS.get(ch["description"])
        if info is None:
            continue
        weight = 1.0 if ch.get("is_current") else 0.85
        tiers_found.append((info.tier * weight, info.tier, info.label, ch["company"], ch.get("is_current", False)))

    if not tiers_found:
        # No recognized template at all — either a templating drift
        # (dataset changed) or this candidate's history uses free text
        # we haven't seen. Score neutral rather than penalizing, but
        # this is flagged loudly so it doesn't go unnoticed.
        return 0.5, ["career history uses no recognized template (manual review recommended)"]

    best_weighted = max(w for w, _, _, _, _ in tiers_found)
    score = min(1.0, best_weighted / 5.0)
    top_evidence = [
        f"{label} at {company}{' (current)' if is_current else ' (past role)'} (tier {tier}/5)"
        for w, tier, label, company, is_current in tiers_found
        if w == best_weighted
    ]
    return score, top_evidence


if __name__ == "__main__":
    # Drift check: confirm every template string we've classified
    # still appears in the live data, and flag any career_history
    # description in the AI-titled pool that ISN'T in our table.
    import sys

    sys.path.insert(0, ".")
    from load_candidates import iter_candidates
    from constants import AI_TITLES

    seen_known = set()
    unknown = set()
    for c in iter_candidates("data/candidates.jsonl"):
        if c["profile"]["current_title"] not in AI_TITLES:
            continue
        for ch in c["career_history"]:
            if ch["description"] in CAREER_TEMPLATE_TIERS:
                seen_known.add(ch["description"])
            else:
                unknown.add(ch["description"])

    print(f"Known templates matched in live data: {len(seen_known)} / {len(CAREER_TEMPLATE_TIERS)}")
    print(f"Unrecognized descriptions found: {len(unknown)}")
    if unknown:
        print("WARNING: dataset may have drifted. First unknown description:")
        print(" ", next(iter(unknown))[:200])