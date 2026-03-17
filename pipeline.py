"""
rag_engine/pipeline.py
=======================
The Retrieval-Augmented Generation (RAG) core of the system.

How RAG improves summarisation accuracy
-----------------------------------------
Naive LED summarisation of a full judgment:
  • May emphasise less relevant sections (procedural history, appearances)
  • Miss cross-document context (related precedents)

Our RAG pipeline:
  1. RETRIEVE  — embed the user query with Legal-BERT, pull top-k chunks
                 from the FAISS index (may span multiple documents)
  2. RERANK    — cross-encoder rescoring for higher precision
  3. DEDUPE    — remove near-duplicate chunks (Jaccard similarity)
  4. SUMMARISE — LED generates summary from the ranked context window
  5. HIGHLIGHT — return source chunks so the user can verify grounding

This follows the "RAG-Sequence" pattern (vs RAG-Token): we build
a single merged context string and generate one summary from it.
"""

import re
import hashlib
from typing import Optional

from models.embedder        import LegalBERTEmbedder, FAISSVectorStore
from models.summarizer      import LEDSummarizer
from evaluation.cosine_scorer import CosineScorer
from evaluation.validator     import OutputValidator

try:
    from sentence_transformers import CrossEncoder
    CE_AVAILABLE = True
except ImportError:
    CE_AVAILABLE = False


# ─── Cross-encoder reranker ───────────────────────────────────────────────────

class CrossEncoderReranker:
    """
    Reranks retrieved chunks using a cross-encoder (bi-encoder retrieval
    is fast but less precise; cross-encoders score (query, chunk) pairs
    jointly for higher accuracy).

    Model: cross-encoder/ms-marco-MiniLM-L-6-v2  (fast, 384-dim)
    For a legal-specialised reranker you can fine-tune this on
    (query, relevant-chunk, irrelevant-chunk) triplets.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        if not CE_AVAILABLE:
            raise ImportError("Install sentence-transformers: pip install sentence-transformers")
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
        pairs  = [(query, c["text"]) for c in chunks]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(scores, chunks),
            key=lambda x: x[0],
            reverse=True,
        )
        for score, chunk in ranked[:top_k]:
            chunk["rerank_score"] = float(score)

        return [chunk for _, chunk in ranked[:top_k]]


# ─── Deduplication ───────────────────────────────────────────────────────────

def _jaccard(a: str, b: str, n: int = 3) -> float:
    """Character n-gram Jaccard similarity."""
    def ngrams(s):
        return set(s[i:i+n] for i in range(len(s) - n + 1))
    sa, sb = ngrams(a.lower()), ngrams(b.lower())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def deduplicate(chunks: list[dict], threshold: float = 0.85) -> list[dict]:
    """
    Remove near-duplicate chunks above the Jaccard similarity threshold.
    Keeps the first occurrence (highest ranked).
    """
    kept = []
    for candidate in chunks:
        if all(_jaccard(candidate["text"], k["text"]) < threshold for k in kept):
            kept.append(candidate)
    return kept


# ─── Main RAG pipeline ────────────────────────────────────────────────────────

class RAGPipeline:
    """
    End-to-end pipeline: query → retrieved chunks → LED summary.

    Parameters
    ----------
    vector_store    : pre-built FAISSVectorStore
    summarizer      : LEDSummarizer instance
    reranker        : optional CrossEncoderReranker
    initial_k       : how many chunks to retrieve before reranking
    final_k         : how many chunks to keep after reranking/dedup
    max_context_len : word limit for the context fed to LED
    """

    def __init__(
        self,
        vector_store:    FAISSVectorStore,
        summarizer:      LEDSummarizer,
        reranker:        Optional[CrossEncoderReranker] = None,
        initial_k:       int   = 20,
        final_k:         int   = 8,
        max_context_len: int   = 4_000,
        grounding_threshold: float = 0.55,
        relevance_threshold: float = 0.50,
    ):
        self.vector_store    = vector_store
        self.summarizer      = summarizer
        self.reranker        = reranker
        self.initial_k       = initial_k
        self.final_k         = final_k
        self.max_context_len = max_context_len

        # Cosine scorer + validator share the same already-loaded embedder
        self.cosine_scorer = CosineScorer(vector_store.embedder)
        self.validator     = OutputValidator(
            grounding_threshold = grounding_threshold,
            relevance_threshold = relevance_threshold,
        )

    def run(self, query: str, mode: str = "rag") -> dict:
        """
        Run the full pipeline.

        Parameters
        ----------
        query : natural-language question or document title
        mode  : "rag"      → retrieve chunks then summarise
                "fulltext" → pass raw text directly (if provided separately)

        Returns
        -------
        {
            "query":    str,
            "summary":  str,
            "sources":  list of chunk dicts used for generation,
            "metadata": {...}
        }
        """
        # ── Step 1: Bi-encoder retrieval ────────────────────────────────────
        retrieved = self.vector_store.search(query, top_k=self.initial_k)

        if not retrieved:
            return {
                "query":   query,
                "summary": "No relevant documents found in the index.",
                "sources": [],
                "metadata": {},
            }

        # ── Step 2: Cross-encoder reranking (optional) ───────────────────────
        if self.reranker:
            retrieved = self.reranker.rerank(query, retrieved, top_k=self.final_k)
        else:
            retrieved = retrieved[:self.final_k]

        # ── Step 3: Deduplicate ──────────────────────────────────────────────
        retrieved = deduplicate(retrieved)

        # ── Step 4: Build context string ─────────────────────────────────────
        #    Prepend a task instruction so LED knows what to do.
        #    This "prompt engineering" trick improves output quality.
        context_parts = [
            f"[CHUNK {i+1} | Doc: {c.get('metadata', {}).get('title', c.get('doc_id',''))} "
            f"| Score: {c.get('rerank_score', c.get('score', 0)):.3f}]\n{c['text']}"
            for i, c in enumerate(retrieved)
        ]
        context = "\n\n---\n\n".join(context_parts)

        # Truncate context to word limit
        words = context.split()
        if len(words) > self.max_context_len:
            context = " ".join(words[:self.max_context_len])

        instruction = (
            "Summarize the following Indian legal document excerpts. "
            "Focus on: key legal issue, court's reasoning, final decision, and precedent set.\n\n"
        )
        full_input = instruction + context

        # ── Step 5: Generate summary ─────────────────────────────────────────
        summary = self.summarizer.summarise(full_input)

        # ── Step 6: Cosine similarity scoring
        cosine_report = self.cosine_scorer.score(
            summary  = summary,
            chunks   = retrieved,
            query    = query,
        )

        # ── Step 7: Output validation
        validation = self.validator.validate(
            summary       = summary,
            cosine_report = cosine_report,
            query         = query,
            source_texts  = [c["text"] for c in retrieved],
        )

        return {
            "query":      query,
            "summary":    summary,
            "sources":    retrieved,
            "cosine":     cosine_report.to_dict(),
            "validation": validation.to_dict(),
            "metadata": {
                "chunks_retrieved":      self.initial_k,
                "chunks_after_rerank":   len(retrieved),
                "context_words":         len(full_input.split()),
                "accuracy_score":        cosine_report.accuracy_score,
                "accuracy_label":        cosine_report.accuracy_label,
                "accuracy_pct":          cosine_report.accuracy_pct,
                "validation_label":      validation.overall_label,
                "validation_confidence": validation.confidence_pct,
            },
        }

    def summarise_document(self, document_text: str) -> dict:
        """
        Directly summarise a document (no retrieval).
        For when the user uploads a PDF/text file.
        """
        instruction = (
            "You are an expert Indian legal analyst. "
            "Provide a structured summary of this legal document including: "
            "1) Parties involved 2) Legal issue 3) Arguments 4) Court's reasoning 5) Decision.\n\n"
        )
        full_input = instruction + document_text
        summary    = self.summarizer.summarise(full_input)

        # Cosine scorer needs at least one "chunk" — wrap the input text
        cosine_report = self.cosine_scorer.score(
            summary = summary,
            chunks  = [{"text": document_text[:2000]}],
            query   = None,
        )
        validation = self.validator.validate(
            summary       = summary,
            cosine_report = cosine_report,
            source_texts  = [document_text[:2000]],
        )

        return {
            "summary":    summary,
            "sources":    [],
            "cosine":     cosine_report.to_dict(),
            "validation": validation.to_dict(),
            "metadata": {
                "mode":                  "direct",
                "input_words":           len(document_text.split()),
                "accuracy_score":        cosine_report.accuracy_score,
                "accuracy_label":        cosine_report.accuracy_label,
                "accuracy_pct":          cosine_report.accuracy_pct,
                "validation_label":      validation.overall_label,
                "validation_confidence": validation.confidence_pct,
            },
        }
