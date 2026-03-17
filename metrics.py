"""
evaluation/metrics.py
======================
Evaluation framework for comparing our system against baselines.

Metrics
-------
ROUGE-1  : unigram overlap (content coverage)
ROUGE-2  : bigram overlap  (fluency / phrase-level accuracy)
ROUGE-L  : longest common subsequence (structural similarity)
BERTScore: semantic similarity using BERT embeddings — captures
           meaning even when wording differs (key for legal text
           where synonyms matter: "dismissed"/"rejected"/"denied")

Baselines
---------
We compare against:
  1. BART-base     (facebook/bart-base)
  2. T5-small      (t5-small)
  3. extractive    (lead-k: first k sentences — trivial baseline)
  4. Our system    (Legal-BERT RAG + LED fine-tuned)

The research claim: our system achieves higher ROUGE-2 and BERTScore
on Indian legal judgments because of domain-specific embeddings and
full-document LED input.
"""

import json
from dataclasses import dataclass, asdict, field
from typing import Optional

import numpy as np
from rouge_score import rouge_scorer
from bert_score import score as bertscore

try:
    from transformers import pipeline as hf_pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


# ─── Result dataclass ────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    model_name:    str
    rouge1:        float = 0.0
    rouge2:        float = 0.0
    rougeL:        float = 0.0
    bertscore_f1:  float = 0.0
    avg_length:    float = 0.0
    num_samples:   int   = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"{'─'*50}\n"
            f"Model      : {self.model_name}\n"
            f"ROUGE-1    : {self.rouge1:.4f}\n"
            f"ROUGE-2    : {self.rouge2:.4f}\n"
            f"ROUGE-L    : {self.rougeL:.4f}\n"
            f"BERTScore  : {self.bertscore_f1:.4f}\n"
            f"Avg length : {self.avg_length:.1f} tokens\n"
            f"Samples    : {self.num_samples}\n"
        )


# ─── Evaluator ───────────────────────────────────────────────────────────────

class Evaluator:
    """
    Evaluates summaries against reference (gold) summaries.

    Usage
    -----
    ev = Evaluator()

    # Evaluate our system
    preds = [pipeline.run(q)["summary"] for q in queries]
    result = ev.evaluate(preds, references, model_name="LED+RAG (ours)")

    # Evaluate BART baseline
    bart_preds = ev.generate_baseline_summaries(documents, model="bart")
    bart_result = ev.evaluate(bart_preds, references, model_name="BART-base")

    # Print comparison table
    ev.compare([result, bart_result])
    """

    def __init__(self):
        self._rouge = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

    # ── Core evaluation ───────────────────────────────────────────────────────

    def evaluate(
        self,
        predictions:  list[str],
        references:   list[str],
        model_name:   str = "model",
        lang:         str = "en",
    ) -> EvalResult:
        """
        Compute ROUGE + BERTScore for a list of (prediction, reference) pairs.

        predictions : list of generated summaries
        references  : list of gold/reference summaries
        """
        assert len(predictions) == len(references), "Lists must be same length"

        r1_scores, r2_scores, rL_scores = [], [], []

        for pred, ref in zip(predictions, references):
            scores = self._rouge.score(ref, pred)
            r1_scores.append(scores["rouge1"].fmeasure)
            r2_scores.append(scores["rouge2"].fmeasure)
            rL_scores.append(scores["rougeL"].fmeasure)

        # BERTScore — uses DeBERTa-xlarge-mnli by default (rescale_with_baseline=True)
        _, _, bs_f1 = bertscore(
            predictions,
            references,
            lang=lang,
            rescale_with_baseline=True,
            verbose=False,
        )

        return EvalResult(
            model_name   = model_name,
            rouge1       = float(np.mean(r1_scores)),
            rouge2       = float(np.mean(r2_scores)),
            rougeL       = float(np.mean(rL_scores)),
            bertscore_f1 = float(bs_f1.mean().item()),
            avg_length   = float(np.mean([len(p.split()) for p in predictions])),
            num_samples  = len(predictions),
        )

    # ── Baseline generators ───────────────────────────────────────────────────

    def generate_baseline_summaries(
        self,
        documents: list[str],
        model:     str = "bart",          # "bart" | "t5" | "extractive"
        max_len:   int = 300,
    ) -> list[str]:
        """
        Generate baseline summaries from standard models for comparison.

        bart       → facebook/bart-base
        t5         → t5-small
        extractive → Lead-3 (first 3 sentences)
        """
        if model == "extractive":
            return [self._lead_k(doc, k=5) for doc in documents]

        if not HF_AVAILABLE:
            raise ImportError("transformers not installed")

        model_map = {
            "bart": "facebook/bart-base",
            "t5":   "t5-small",
        }
        hf_model = model_map.get(model, model)

        print(f"[Evaluator] Loading baseline model: {hf_model}...")
        summariser = hf_pipeline(
            "summarization",
            model=hf_model,
            device=0 if __import__("torch").cuda.is_available() else -1,
        )

        summaries = []
        for doc in documents:
            # BART/T5 max input is ~1024 tokens; truncate
            truncated = " ".join(doc.split()[:900])
            out = summariser(
                truncated,
                max_length=max_len,
                min_length=50,
                do_sample=False,
            )
            summaries.append(out[0]["summary_text"])

        return summaries

    @staticmethod
    def _lead_k(text: str, k: int = 5) -> str:
        """Extractive baseline: first k sentences."""
        sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
        return ". ".join(sentences[:k]) + "."

    # ── Comparison table ──────────────────────────────────────────────────────

    @staticmethod
    def compare(results: list[EvalResult]) -> None:
        """Print a formatted comparison table."""
        header = f"{'Model':<30} {'R-1':>8} {'R-2':>8} {'R-L':>8} {'BERTScore':>10} {'Avg Len':>8}"
        print("\n" + "═" * 80)
        print("EVALUATION RESULTS")
        print("═" * 80)
        print(header)
        print("─" * 80)
        for r in results:
            print(
                f"{r.model_name:<30} "
                f"{r.rouge1:>8.4f} "
                f"{r.rouge2:>8.4f} "
                f"{r.rougeL:>8.4f} "
                f"{r.bertscore_f1:>10.4f} "
                f"{r.avg_length:>8.1f}"
            )
        print("═" * 80 + "\n")

    @staticmethod
    def save_results(results: list[EvalResult], path: str) -> None:
        """Save results to JSON for your dissertation tables."""
        with open(path, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        print(f"[Evaluator] Results saved to {path}")
