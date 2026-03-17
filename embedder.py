"""
models/embedder.py
==================
Generates Legal-BERT embeddings and manages a FAISS vector index.

Why Legal-BERT over plain BERT?
-------------------------------
Legal-BERT (nlpaueb/legal-bert-base-uncased) was pre-trained on ~12 GB
of English legal text (EU/US legislation, contracts, court cases).
It understands domain terms like "mens rea", "locus standi", "estoppel"
that general BERT conflates with unrelated contexts.

For Indian law we fine-tune on IN judgments during training — see
models/finetune_legalbert.py for that script.

Index structure
---------------
We store embeddings in FAISS (Facebook AI Similarity Search) for fast
approximate nearest-neighbour (ANN) retrieval.

FAISS index type: IndexFlatIP   (inner-product — equivalent to cosine
similarity after L2 normalisation, which we apply below)

A parallel metadata list `self.chunks` mirrors the FAISS index positions
so we can return full chunk dicts along with scores.
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("[WARNING] faiss-cpu not installed. Install with: pip install faiss-cpu")

from data_pipeline.preprocessor import Chunk

# ─── Constants ───────────────────────────────────────────────────────────────

DEFAULT_MODEL = "nlpaueb/legal-bert-base-uncased"
EMBED_DIM     = 768          # Legal-BERT hidden size
BATCH_SIZE    = 16           # adjust down if OOM on your GPU/CPU


# ─── Embedder ────────────────────────────────────────────────────────────────

class LegalBERTEmbedder:
    """
    Wraps Legal-BERT to produce sentence/chunk embeddings.

    Mean-pooling over the last hidden state is used instead of [CLS]
    because it gives more stable, context-rich representations for
    paragraphs longer than a few words.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "auto"):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[LegalBERTEmbedder] Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("[LegalBERTEmbedder] Ready.")

    @torch.no_grad()
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of strings.
        Returns shape (len(texts), 768) float32 numpy array, L2-normalised.
        """
        all_embeddings = []

        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start : start + BATCH_SIZE]
            enc   = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            output = self.model(**enc)
            # Mean-pool over non-padding tokens
            mask        = enc["attention_mask"].unsqueeze(-1).float()
            summed      = (output.last_hidden_state * mask).sum(1)
            counts      = mask.sum(1).clamp(min=1e-9)
            embeddings  = (summed / counts).cpu().numpy()

            # L2-normalise → cosine similarity becomes inner product
            norms       = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings  = embeddings / np.clip(norms, 1e-9, None)

            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings).astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string. Returns (1, 768) array."""
        return self.embed_texts([query])


# ─── FAISS Vector Store ───────────────────────────────────────────────────────

class FAISSVectorStore:
    """
    Wraps a FAISS IndexFlatIP with a parallel metadata list.

    Typical workflow
    ----------------
    store = FAISSVectorStore(embedder)
    store.add_chunks(processed_docs)    # index all chunks
    store.save("./index")               # persist
    # ─── later ───
    store = FAISSVectorStore.load("./index", embedder)
    results = store.search("breach of contract", top_k=5)
    """

    def __init__(self, embedder: LegalBERTEmbedder):
        self.embedder = embedder
        self.index    = None
        self.chunks: list[dict] = []          # parallel list to FAISS rows

        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(EMBED_DIM)

    def add_chunks(self, processed_docs: list[dict]) -> None:
        """
        processed_docs: output from LegalPreprocessor.process_batch()
        Adds all bert_chunks from each doc.
        """
        all_texts: list[str]  = []
        all_meta:  list[dict] = []

        for doc in processed_docs:
            for chunk in doc["bert_chunks"]:
                all_texts.append(chunk.text)
                all_meta.append(chunk.to_dict())

        if not all_texts:
            return

        print(f"[FAISSVectorStore] Embedding {len(all_texts)} chunks...")
        vectors = self.embedder.embed_texts(all_texts)

        self.index.add(vectors)
        self.chunks.extend(all_meta)
        print(f"[FAISSVectorStore] Index now holds {self.index.ntotal} vectors.")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Retrieve the top_k most similar chunks for a query.

        Returns a list of dicts, each with:
            text, doc_id, chunk_idx, metadata, score
        """
        if self.index is None or self.index.ntotal == 0:
            raise RuntimeError("Index is empty. Call add_chunks() first.")

        q_vec = self.embedder.embed_query(query)
        scores, indices = self.index.search(q_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            results.append(chunk)

        return results

    def save(self, directory: str) -> None:
        """Persist index + metadata to disk."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path / "faiss.index"))
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        print(f"[FAISSVectorStore] Saved to {directory}")

    @classmethod
    def load(cls, directory: str, embedder: LegalBERTEmbedder) -> "FAISSVectorStore":
        """Restore a previously saved index."""
        path  = Path(directory)
        store = cls(embedder)
        store.index  = faiss.read_index(str(path / "faiss.index"))
        with open(path / "chunks.pkl", "rb") as f:
            store.chunks = pickle.load(f)
        print(f"[FAISSVectorStore] Loaded {store.index.ntotal} vectors from {directory}")
        return store
