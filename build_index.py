"""
scripts/build_index.py
=======================
Run this ONCE to:
  1. Scrape legal documents from Indian govt sources
  2. Preprocess and chunk them
  3. Embed with Legal-BERT
  4. Build and save a FAISS index

Usage
-----
    python -m scripts.build_index \
        --queries "Supreme Court contract 2022" "RERA judgment 2023" \
        --acts "Indian Contract Act" "RERA" \
        --max_per_source 30 \
        --index_dir ./faiss_index

After this, start the API:
    uvicorn api.main:app --reload
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s %(levelname)s %(message)s",
    stream  = sys.stdout,
)
logger = logging.getLogger(__name__)


def build(
    queries:        list[str],
    acts:           list[str],
    max_per_source: int,
    index_dir:      str,
    save_corpus:    bool = True,
):
    from data_pipeline.scraper       import collect_corpus
    from data_pipeline.preprocessor  import LegalPreprocessor
    from models.embedder             import LegalBERTEmbedder, FAISSVectorStore

    # ── 1. Collect raw documents ─────────────────────────────────────────────
    logger.info("Collecting corpus from Indian legal sources...")
    corpus = collect_corpus(
        ik_queries     = queries,
        act_titles     = acts,
        max_per_source = max_per_source,
    )
    logger.info(f"Collected {len(corpus)} documents")

    if save_corpus:
        corpus_path = Path(index_dir) / "corpus.json"
        corpus_path.parent.mkdir(parents=True, exist_ok=True)
        with open(corpus_path, "w") as f:
            json.dump(corpus, f, ensure_ascii=False, indent=2)
        logger.info(f"Raw corpus saved to {corpus_path}")

    if not corpus:
        logger.error("No documents collected. Check scraper connectivity.")
        return

    # ── 2. Preprocess ────────────────────────────────────────────────────────
    logger.info("Preprocessing documents...")
    preprocessor   = LegalPreprocessor(bert_chunk_size=400, bert_overlap=80)
    processed_docs = preprocessor.process_batch(corpus)

    total_chunks = sum(len(d["bert_chunks"]) for d in processed_docs)
    logger.info(f"Generated {total_chunks} BERT chunks across {len(processed_docs)} docs")

    # ── 3. Embed & index ─────────────────────────────────────────────────────
    logger.info("Loading Legal-BERT embedder...")
    embedder = LegalBERTEmbedder()

    store = FAISSVectorStore(embedder)
    store.add_chunks(processed_docs)

    # ── 4. Save index ────────────────────────────────────────────────────────
    store.save(index_dir)
    logger.info(f"Done! Index saved to {index_dir}")
    logger.info(f"Total vectors in index: {store.index.ntotal}")


def main():
    parser = argparse.ArgumentParser(description="Build legal NLP index")
    parser.add_argument("--queries",        nargs="*", default=["Supreme Court judgment 2023"])
    parser.add_argument("--acts",           nargs="*", default=["Indian Contract Act"])
    parser.add_argument("--max_per_source", type=int,  default=20)
    parser.add_argument("--index_dir",      type=str,  default="./faiss_index")
    parser.add_argument("--no_save_corpus", action="store_true")
    args = parser.parse_args()

    build(
        queries        = args.queries,
        acts           = args.acts,
        max_per_source = args.max_per_source,
        index_dir      = args.index_dir,
        save_corpus    = not args.no_save_corpus,
    )


if __name__ == "__main__":
    main()
