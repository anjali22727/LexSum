"""
evaluation/cosine_scorer.py
============================
Cosine similarity scoring between:
  1. Summary  ↔ Source chunks   (grounding score — did the summary stay faithful?)
  2. Summary  ↔ Query           (relevance score — did it answer the question?)
  3. Summary  ↔ Reference       (accuracy score  — how close to gold standard?)

Why cosine similarity here instead of ROUGE?
--------------------------------------------
ROUGE measures word overlap — it penalises valid paraphrases.
Cosine similarity over Legal-BERT embeddings captures SEMANTIC closeness,
meaning "dismissed the petition" and "rejected the writ" both score high
against a reference that uses either phrase.

We combine both:
  • ROUGE   → lexical accuracy (traditional NLP metric, expected in dissertations)
  • Cosine  → semantic accuracy (modern embedding-based metric)
  • BERTScore → hybrid (both in one)

All three together give a robust picture of summary quality.

Score interpretation
--------------------
  ≥ 0.85   Excellent — semantically very close
  0.70–0.85 Good     — captures main ideas
  0.55–0.70 Fair     — partial coverage
  < 0.55   Poor     — summary diverges from source
"""

import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class CosineSimilarityReport:
    """All cosine similarity scores for one summary."""

    # Summary ↔ individual source chunks
    chunk_scores:         list[float]   # one score per retrieved chunk
    mean_chunk_score:     float         # average over all chunks
    max_chunk_score:      float         # best-matching chunk
    min_chunk_score:      float         # worst-matching chunk

    # Summary ↔ query (relevance)
    query_relevance:      Optional[float]   # None if no query

    # Summary ↔ reference/gold (accuracy)
    reference_accuracy:   Optional[float]   # None if no reference provided

    # Combined weighted accuracy score (0–1)
    # = 0.5 × mean_chunk_score + 0.3 × query_relevance + 0.2 × reference_accuracy
    # Falls back gracefully if query/reference not available
    accuracy_score:       float
    accuracy_label:       str    # "Excellent" / "Good" / "Fair" / "Poor"
    accuracy_pct:         float  # accuracy_score × 100, for display

    def to_dict(self) -> dict:
        return asdict(self)


def _label(score: float) -> str:
    if score >= 0.85:  return "Excellent"
    if score >= 0.70:  return "Good"
    if score >= 0.55:  return "Fair"
    return "Poor"


class CosineScorer:
    """
    Computes cosine similarity scores using Legal-BERT embeddings.

    Parameters
    ----------
    embedder : LegalBERTEmbedder instance (already loaded)
               Pass the same embedder used for the FAISS index to avoid
               loading the model twice.
    """

    def __init__(self, embedder):
        self.embedder = embedder

    # ── Core method ───────────────────────────────────────────────────────────

    def score(
        self,
        summary:    str,
        chunks:     list[dict],          # list of retrieved chunk dicts with 'text' key
        query:      Optional[str] = None,
        reference:  Optional[str] = None,
    ) -> CosineSimilarityReport:
        """
        Compute all cosine similarity scores for a generated summary.

        Parameters
        ----------
        summary   : the generated summary text
        chunks    : retrieved source chunks (from RAG pipeline)
        query     : original user query (optional)
        reference : gold/reference summary (optional)

        Returns
        -------
        CosineSimilarityReport with all scores
        """
        # Embed the summary once
        summary_vec = self.embedder.embed_query(summary)    # shape (1, 768)

        # ── Chunk similarity ─────────────────────────────────────────────────
        chunk_texts = [c["text"] for c in chunks] if chunks else []

        if chunk_texts:
            chunk_vecs   = self.embedder.embed_texts(chunk_texts)  # (N, 768)
            # Dot product of L2-normalised vectors = cosine similarity
            chunk_scores = (chunk_vecs @ summary_vec.T).flatten().tolist()
        else:
            chunk_scores = [0.0]

        mean_chunk = float(np.mean(chunk_scores))
        max_chunk  = float(np.max(chunk_scores))
        min_chunk  = float(np.min(chunk_scores))

        # ── Query relevance ──────────────────────────────────────────────────
        query_rel = None
        if query and query.strip():
            query_vec = self.embedder.embed_query(query)
            query_rel = float((summary_vec @ query_vec.T).flatten()[0])

        # ── Reference accuracy ───────────────────────────────────────────────
        ref_acc = None
        if reference and reference.strip():
            ref_vec = self.embedder.embed_query(reference)
            ref_acc = float((summary_vec @ ref_vec.T).flatten()[0])

        # ── Combined accuracy score ──────────────────────────────────────────
        # Weighted combination; degrade gracefully when components missing
        if query_rel is not None and ref_acc is not None:
            accuracy = 0.50 * mean_chunk + 0.30 * query_rel + 0.20 * ref_acc
        elif query_rel is not None:
            accuracy = 0.60 * mean_chunk + 0.40 * query_rel
        elif ref_acc is not None:
            accuracy = 0.60 * mean_chunk + 0.40 * ref_acc
        else:
            accuracy = mean_chunk

        # Clamp to [0, 1]
        accuracy = float(np.clip(accuracy, 0.0, 1.0))

        return CosineSimilarityReport(
            chunk_scores       = [round(s, 4) for s in chunk_scores],
            mean_chunk_score   = round(mean_chunk, 4),
            max_chunk_score    = round(max_chunk, 4),
            min_chunk_score    = round(min_chunk, 4),
            query_relevance    = round(query_rel, 4) if query_rel is not None else None,
            reference_accuracy = round(ref_acc, 4)   if ref_acc  is not None else None,
            accuracy_score     = round(accuracy, 4),
            accuracy_label     = _label(accuracy),
            accuracy_pct       = round(accuracy * 100, 1),
        )

    # ── Batch scoring ─────────────────────────────────────────────────────────

    def score_batch(
        self,
        summaries:   list[str],
        chunks_list: list[list[dict]],
        queries:     Optional[list[str]] = None,
        references:  Optional[list[str]] = None,
    ) -> list[CosineSimilarityReport]:
        """Score a batch of summaries."""
        n = len(summaries)
        results = []
        for i in range(n):
            results.append(self.score(
                summary   = summaries[i],
                chunks    = chunks_list[i] if chunks_list else [],
                query     = queries[i]    if queries    else None,
                reference = references[i] if references else None,
            ))
        return results
