"""
evaluation/validator.py
========================
Validation layer that runs AFTER the LED generates a summary.
Catches bad outputs before they reach the user.

Why validate the output?
------------------------
LED can sometimes produce:
  • Hallucinations (mentions facts not in source chunks)
  • Repetition loops ("The court held... the court held...")
  • Empty or truncated output
  • Factual contradictions (summary says "allowed" when source says "dismissed")
  • Off-topic text (model wandered from legal domain)

The validator runs 5 checks and returns a structured ValidationResult
with a pass/fail per check plus a recommended action.

Checks
------
1. LENGTH CHECK      — is the summary too short or too long?
2. REPETITION CHECK  — does it repeat the same phrases excessively?
3. GROUNDING CHECK   — cosine similarity to source ≥ threshold
4. RELEVANCE CHECK   — cosine similarity to query ≥ threshold (if query given)
5. CONTRADICTION CHECK — detect polarity mismatches (allowed/dismissed)

Integration
-----------
The validator is called automatically in RAGPipeline.run() and
RAGPipeline.summarise_document() — the result is included in the API
response under the key "validation".
"""

import re
from dataclasses import dataclass, field, asdict
from typing import Optional


# ─── Result dataclass ────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    name:    str
    passed:  bool
    detail:  str
    value:   Optional[float] = None    # numeric value when applicable


@dataclass
class ValidationResult:
    checks:           list[CheckResult]
    overall_passed:   bool
    overall_label:    str              # "VALID" | "WARNINGS" | "INVALID"
    warnings:         list[str]        # human-readable warnings
    recommendation:   str              # what to do with this summary
    confidence:       float            # fraction of checks passed (0–1)
    confidence_pct:   float            # confidence × 100

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def summary_line(self) -> str:
        icon = "✓" if self.overall_passed else "⚠"
        return f"{icon} {self.overall_label} — {self.confidence_pct:.0f}% confidence | {self.recommendation}"


# ─── Contradiction word lists ─────────────────────────────────────────────────

# Pairs: if BOTH sides appear in the same short window it may be a contradiction
_VERDICT_PAIRS = [
    ({"allowed", "granted", "upheld", "admitted"},
     {"dismissed", "rejected", "denied", "quashed", "overruled"}),
    ({"guilty", "convicted"},
     {"acquitted", "innocent", "discharged"}),
    ({"liable", "liable to pay"},
     {"not liable", "no liability"}),
]


# ─── Validator ───────────────────────────────────────────────────────────────

class OutputValidator:
    """
    Validates a generated summary against configurable thresholds.

    Parameters
    ----------
    grounding_threshold  : minimum cosine similarity to source chunks
    relevance_threshold  : minimum cosine similarity to query
    min_words            : minimum acceptable summary length
    max_words            : maximum acceptable summary length
    max_repeat_ratio     : max allowed fraction of repeated trigrams
    """

    def __init__(
        self,
        grounding_threshold: float = 0.55,
        relevance_threshold: float = 0.50,
        min_words:           int   = 30,
        max_words:           int   = 600,
        max_repeat_ratio:    float = 0.30,
    ):
        self.grounding_threshold = grounding_threshold
        self.relevance_threshold = relevance_threshold
        self.min_words           = min_words
        self.max_words           = max_words
        self.max_repeat_ratio    = max_repeat_ratio

    def validate(
        self,
        summary:         str,
        cosine_report,                   # CosineSimilarityReport
        query:           Optional[str] = None,
        source_texts:    Optional[list[str]] = None,
    ) -> ValidationResult:
        """
        Run all validation checks.

        Parameters
        ----------
        summary       : generated summary text
        cosine_report : CosineSimilarityReport from CosineScorer.score()
        query         : original user query
        source_texts  : raw texts of retrieved chunks (for contradiction check)
        """
        checks   = []
        warnings = []

        # ── Check 1: Length ──────────────────────────────────────────────────
        word_count = len(summary.split())
        if word_count < self.min_words:
            checks.append(CheckResult(
                name="length",
                passed=False,
                detail=f"Summary too short: {word_count} words (min {self.min_words})",
                value=float(word_count),
            ))
            warnings.append(f"Summary is very short ({word_count} words). May be incomplete.")
        elif word_count > self.max_words:
            checks.append(CheckResult(
                name="length",
                passed=False,
                detail=f"Summary too long: {word_count} words (max {self.max_words})",
                value=float(word_count),
            ))
            warnings.append(f"Summary is unusually long ({word_count} words). May contain filler.")
        else:
            checks.append(CheckResult(
                name="length",
                passed=True,
                detail=f"Acceptable length: {word_count} words",
                value=float(word_count),
            ))

        # ── Check 2: Repetition ──────────────────────────────────────────────
        repeat_ratio = self._repetition_ratio(summary)
        if repeat_ratio > self.max_repeat_ratio:
            checks.append(CheckResult(
                name="repetition",
                passed=False,
                detail=f"High repetition ratio: {repeat_ratio:.2%} of trigrams repeated",
                value=repeat_ratio,
            ))
            warnings.append(f"Summary contains repetitive phrases ({repeat_ratio:.0%} repeat ratio).")
        else:
            checks.append(CheckResult(
                name="repetition",
                passed=True,
                detail=f"Low repetition: {repeat_ratio:.2%}",
                value=repeat_ratio,
            ))

        # ── Check 3: Grounding ───────────────────────────────────────────────
        grounding = cosine_report.mean_chunk_score
        if grounding < self.grounding_threshold:
            checks.append(CheckResult(
                name="grounding",
                passed=False,
                detail=(
                    f"Low source grounding: cosine={grounding:.4f} "
                    f"(threshold {self.grounding_threshold})"
                ),
                value=grounding,
            ))
            warnings.append(
                f"Summary may not be well-grounded in source documents "
                f"(cosine similarity {grounding:.3f} < {self.grounding_threshold})."
            )
        else:
            checks.append(CheckResult(
                name="grounding",
                passed=True,
                detail=f"Good source grounding: cosine={grounding:.4f}",
                value=grounding,
            ))

        # ── Check 4: Query relevance ─────────────────────────────────────────
        if query and query.strip() and cosine_report.query_relevance is not None:
            relevance = cosine_report.query_relevance
            if relevance < self.relevance_threshold:
                checks.append(CheckResult(
                    name="relevance",
                    passed=False,
                    detail=f"Low query relevance: cosine={relevance:.4f}",
                    value=relevance,
                ))
                warnings.append(
                    f"Summary may not directly address the query "
                    f"(relevance score {relevance:.3f} < {self.relevance_threshold})."
                )
            else:
                checks.append(CheckResult(
                    name="relevance",
                    passed=True,
                    detail=f"Good query relevance: cosine={relevance:.4f}",
                    value=relevance,
                ))
        else:
            # No query provided — skip check
            checks.append(CheckResult(
                name="relevance",
                passed=True,
                detail="Skipped (no query provided)",
                value=None,
            ))

        # ── Check 5: Contradiction detection ────────────────────────────────
        contradiction_found, contradiction_detail = self._check_contradiction(
            summary, source_texts or []
        )
        if contradiction_found:
            checks.append(CheckResult(
                name="contradiction",
                passed=False,
                detail=contradiction_detail,
                value=None,
            ))
            warnings.append(f"Possible contradiction: {contradiction_detail}")
        else:
            checks.append(CheckResult(
                name="contradiction",
                passed=True,
                detail="No obvious verdict contradictions detected",
                value=None,
            ))

        # ── Aggregate ────────────────────────────────────────────────────────
        n_passed    = sum(1 for c in checks if c.passed)
        confidence  = n_passed / len(checks)
        overall     = n_passed == len(checks)

        if overall:
            label = "VALID"
            rec   = "Summary is reliable. Present to user."
        elif confidence >= 0.6:
            label = "WARNINGS"
            rec   = "Summary has minor issues. Show with warnings."
        else:
            label = "INVALID"
            rec   = "Summary quality too low. Consider re-generating with different parameters."

        return ValidationResult(
            checks         = checks,
            overall_passed = overall,
            overall_label  = label,
            warnings       = warnings,
            recommendation = rec,
            confidence     = round(confidence, 4),
            confidence_pct = round(confidence * 100, 1),
        )

    # ── Internal checks ───────────────────────────────────────────────────────

    @staticmethod
    def _repetition_ratio(text: str) -> float:
        """Fraction of trigrams that appear more than once."""
        words  = text.lower().split()
        if len(words) < 3:
            return 0.0
        trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
        if not trigrams:
            return 0.0
        unique   = set(trigrams)
        repeated = len(trigrams) - len(unique)
        return repeated / len(trigrams)

    @staticmethod
    def _check_contradiction(summary: str, source_texts: list[str]) -> tuple[bool, str]:
        """
        Check if summary verdict contradicts source text verdict.
        Simple heuristic: look for opposite verdict words.
        """
        summary_lower = summary.lower()

        for positive_set, negative_set in _VERDICT_PAIRS:
            summary_has_pos = any(w in summary_lower for w in positive_set)
            summary_has_neg = any(w in summary_lower for w in negative_set)

            # Both present in summary = contradiction within summary
            if summary_has_pos and summary_has_neg:
                return True, (
                    f"Summary contains conflicting verdict terms: "
                    f"both '{next(w for w in positive_set if w in summary_lower)}' "
                    f"and '{next(w for w in negative_set if w in summary_lower)}'"
                )

            # Check against source: summary says allowed but source says dismissed
            if source_texts:
                combined_source = " ".join(source_texts).lower()
                source_has_pos  = any(w in combined_source for w in positive_set)
                source_has_neg  = any(w in combined_source for w in negative_set)

                if summary_has_pos and source_has_neg and not source_has_pos:
                    return True, (
                        "Summary verdict may contradict source: "
                        f"summary implies positive outcome but source indicates negative"
                    )
                if summary_has_neg and source_has_pos and not source_has_neg:
                    return True, (
                        "Summary verdict may contradict source: "
                        "summary implies negative outcome but source indicates positive"
                    )

        return False, "No contradiction detected"
