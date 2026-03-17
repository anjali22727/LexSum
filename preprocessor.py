"""
data_pipeline/preprocessor.py
==============================
Cleans raw legal text and splits it into overlapping chunks
suitable for Legal-BERT (max 512 tokens) and LED (up to 16 384 tokens).

Stages
------
1. noise_removal   – strip headers/footers, page numbers, watermarks
2. normalisation   – Unicode → ASCII where safe, fix encoding artefacts
3. sentence_split  – SpaCy en_core_web_sm (with custom legal abbrevs)
4. chunking        – sliding window with configurable size + overlap
5. metadata_attach – stamps each chunk with doc-level metadata
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterator

try:
    import spacy
    _nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    _nlp.add_pipe("sentencizer")          # fast sentence boundary detection
except OSError:
    _nlp = None   # graceful fallback — will split on newlines


# ─── Data models ────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """One chunk of text ready for embedding or summarisation."""
    doc_id:    str
    chunk_idx: int
    text:      str
    tokens:    int                         # approximate
    metadata:  dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "doc_id":    self.doc_id,
            "chunk_idx": self.chunk_idx,
            "text":      self.text,
            "tokens":    self.tokens,
            "metadata":  self.metadata,
        }


# ─── Noise removal patterns ─────────────────────────────────────────────────

# Things to strip from Indian legal documents
_NOISE_PATTERNS = [
    r"(?i)page\s+\d+\s+of\s+\d+",        # "Page 3 of 47"
    r"(?i)printed\s+on\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",
    r"(?i)www\.[a-z0-9._-]+\.[a-z]{2,}",  # stray URLs
    r"\f",                                  # form-feed from PDF
    r"_{5,}",                              # long underscores (dividers)
    r"-{5,}",                              # long dashes (dividers)
    r"•\s*",                               # stray bullets
]
_NOISE_RE = re.compile("|".join(_NOISE_PATTERNS))

# Legal-specific abbreviations that spaCy mistakenly sentence-breaks on
_LEGAL_ABBREVS = [
    "Hon'ble", "vs", "v.", "Sec.", "Art.", "Cl.", "Para.", "No.",
    "Col.", "Dt.", "Ref.", "S.C.", "H.C.", "J.", "CJ.", "JJ.",
    "ibid", "supra", "infra", "para", "w.e.f", "r/w",
]


# ─── Core processor ─────────────────────────────────────────────────────────

class LegalPreprocessor:
    """
    Preprocessing pipeline for Indian legal documents.

    Parameters
    ----------
    bert_chunk_size   : target tokens per BERT-sized chunk (default 400)
    bert_overlap      : overlap in tokens between consecutive BERT chunks (default 80)
    led_chunk_size    : larger chunk for LED full-doc summarisation (default 3 000)
    """

    def __init__(
        self,
        bert_chunk_size: int = 400,
        bert_overlap:    int = 80,
        led_chunk_size:  int = 3_000,
    ):
        self.bert_chunk_size = bert_chunk_size
        self.bert_overlap    = bert_overlap
        self.led_chunk_size  = led_chunk_size

    # ── public API ──────────────────────────────────────────────────────────

    def process(self, doc: dict) -> dict:
        """
        Takes a raw document dict (from scraper) and returns:
        {
            "doc_id":        str,
            "clean_text":    str,               # full cleaned text
            "bert_chunks":   list[Chunk],        # small chunks for RAG
            "led_chunks":    list[Chunk],        # large chunks for summariser
            "metadata":      dict,
        }
        """
        raw   = doc["text"]
        clean = self._clean(raw)

        bert_chunks = list(
            self._chunk(doc["id"], clean, self.bert_chunk_size, self.bert_overlap, doc)
        )
        led_chunks = list(
            self._chunk(doc["id"], clean, self.led_chunk_size, 200, doc, prefix="led")
        )

        return {
            "doc_id":      doc["id"],
            "clean_text":  clean,
            "bert_chunks": bert_chunks,
            "led_chunks":  led_chunks,
            "metadata":    doc.get("metadata", {}),
        }

    def process_batch(self, docs: list[dict]) -> list[dict]:
        """Process a list of raw docs."""
        return [self.process(d) for d in docs]

    # ── internals ───────────────────────────────────────────────────────────

    def _clean(self, text: str) -> str:
        # 1. Fix encoding artefacts (curly quotes, en-dashes, etc.)
        text = text.replace("\u2019", "'").replace("\u2018", "'")
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        text = text.replace("\u2013", "-").replace("\u2014", "-")
        text = text.replace("\u00a0", " ")    # non-breaking space

        # 2. Normalise Unicode
        text = unicodedata.normalize("NFKC", text)

        # 3. Remove noise patterns
        text = _NOISE_RE.sub(" ", text)

        # 4. Collapse whitespace
        text = re.sub(r"[ \t]+",  " ",  text)
        text = re.sub(r"\n{3,}",  "\n\n", text)

        return text.strip()

    def _sentences(self, text: str) -> list[str]:
        """
        Returns a list of sentences.
        Falls back to newline splitting if spaCy is unavailable.
        """
        if _nlp is None:
            return [s.strip() for s in text.split("\n") if s.strip()]

        # Add abbreviation exceptions so spaCy doesn't over-split
        for abbrev in _LEGAL_ABBREVS:
            _nlp.tokenizer.add_special_case(abbrev, [{"ORTH": abbrev}])

        doc = _nlp(text[:1_000_000])        # spaCy limit guard
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    def _approx_tokens(self, text: str) -> int:
        """Whitespace-based token count — fast enough for chunking."""
        return len(text.split())

    def _chunk(
        self,
        doc_id:     str,
        text:       str,
        max_tokens: int,
        overlap:    int,
        doc:        dict,
        prefix:     str = "bert",
    ) -> Iterator[Chunk]:
        """
        Sliding-window chunker.

        Algorithm
        ---------
        1. Split text into sentences.
        2. Greedily accumulate sentences until adding the next would exceed max_tokens.
        3. Yield the accumulated window.
        4. Step back `overlap` tokens and start the next window.

        This ensures each chunk:
        - Never cuts mid-sentence (important for legal text)
        - Has semantic continuity via overlap
        """
        sentences = self._sentences(text)
        chunk_sentences: list[str] = []
        chunk_tokens: int = 0
        chunk_idx = 0

        # Shared metadata for every chunk from this document
        base_meta = {
            "title":    doc.get("title", ""),
            "source":   doc.get("source", ""),
            "doc_type": doc.get("doc_type", ""),
            "date":     doc.get("date", ""),
            "url":      doc.get("url", ""),
        }
        base_meta.update(doc.get("metadata", {}))

        i = 0
        while i < len(sentences):
            sent = sentences[i]
            sent_tokens = self._approx_tokens(sent)

            # If a single sentence is longer than max_tokens, hard-split it
            if sent_tokens > max_tokens:
                words = sent.split()
                for start in range(0, len(words), max_tokens - overlap):
                    sub = " ".join(words[start : start + max_tokens])
                    yield Chunk(
                        doc_id    = doc_id,
                        chunk_idx = chunk_idx,
                        text      = sub,
                        tokens    = self._approx_tokens(sub),
                        metadata  = {**base_meta, "chunk_prefix": prefix},
                    )
                    chunk_idx += 1
                i += 1
                continue

            if chunk_tokens + sent_tokens > max_tokens and chunk_sentences:
                # Yield current window
                chunk_text = " ".join(chunk_sentences)
                yield Chunk(
                    doc_id    = doc_id,
                    chunk_idx = chunk_idx,
                    text      = chunk_text,
                    tokens    = chunk_tokens,
                    metadata  = {**base_meta, "chunk_prefix": prefix},
                )
                chunk_idx += 1

                # Slide back: remove sentences from the front until we are
                # below (max_tokens - overlap) so the next window has context
                while chunk_sentences and chunk_tokens > max_tokens - overlap:
                    removed = chunk_sentences.pop(0)
                    chunk_tokens -= self._approx_tokens(removed)

            chunk_sentences.append(sent)
            chunk_tokens += sent_tokens
            i += 1

        # Yield last window
        if chunk_sentences:
            chunk_text = " ".join(chunk_sentences)
            yield Chunk(
                doc_id    = doc_id,
                chunk_idx = chunk_idx,
                text      = chunk_text,
                tokens    = chunk_tokens,
                metadata  = {**base_meta, "chunk_prefix": prefix},
            )
