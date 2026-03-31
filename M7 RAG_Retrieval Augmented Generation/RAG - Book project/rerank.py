"""
rerank.py
─────────
Post-retrieval re-ranking.

Takes the raw DB rows (section, content, cosine_sim) returned by
retrieve_topk and adds a final_score by combining:

  1. Vector similarity  (cosine_sim from pgvector)     weight = 0.7
  2. Keyword overlap    (question terms ∩ chunk text)  weight = 0.3

Returns a list of 4-tuples so the notebook HTML loop can unpack:
    section, content, cosine_sim, final_score = row

BUG FIXED: original returned the string "scored_results" — nothing worked.
"""

import re
from typing import List, Tuple


def _keyword_score(question: str, content: str) -> float:
    """
    Fraction of question tokens (≥ 3 chars) that appear in the chunk text.
    Returns a float in [0, 1].
    """
    q_tokens = set(re.findall(r"\b\w{3,}\b", question.lower()))
    if not q_tokens:
        return 0.0
    c_lower = content.lower()
    hits = sum(1 for t in q_tokens if t in c_lower)
    return hits / len(q_tokens)


def rerank_results(
    question: str,
    results: List[Tuple],
    vec_weight: float = 0.7,
    kw_weight:  float = 0.3,
) -> List[Tuple]:
    """
    Re-rank retrieved chunks by a combined vector + keyword score.

    Parameters
    ----------
    question   : str         The user's question string.
    results    : list of 3-tuples  (section, content, cosine_sim)
                 as returned by retrieve_topk.
    vec_weight : float       Weight applied to cosine similarity (default 0.7).
    kw_weight  : float       Weight applied to keyword overlap   (default 0.3).

    Returns
    -------
    List of 4-tuples: (section, content, cosine_sim, final_score)
    Sorted by final_score descending.
    """
    scored = []
    for row in results:
        section    = row[0] if len(row) > 0 else ""
        content    = row[1] if len(row) > 1 else ""
        cosine_sim = float(row[2]) if len(row) > 2 and row[2] is not None else 0.0

        kw_score   = _keyword_score(question, content)
        final      = vec_weight * cosine_sim + kw_weight * kw_score

        scored.append((section, content, cosine_sim, final))

    return sorted(scored, key=lambda x: x[3], reverse=True)
