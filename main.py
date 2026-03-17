"""
api/main.py
===========
FastAPI backend exposing the RAG + LED system as REST endpoints.

Endpoints
---------
POST /api/summarize          — upload text or file, get summary + sources
POST /api/search             — semantic search over the index
GET  /api/health             — liveness check
GET  /api/stats              — index stats
POST /api/evaluate           — run evaluation on a set of (doc, ref) pairs

Run with:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

import os
import io
import time
import logging
from pathlib import Path
from typing import Optional

import fitz                    # PyMuPDF
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Lazy-load heavy models so the import doesn't crash if not installed
from models.embedder   import LegalBERTEmbedder, FAISSVectorStore
from models.summarizer import LEDSummarizer
from rag_engine.pipeline import RAGPipeline, CrossEncoderReranker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── App setup ───────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Legal NLP System — Indian Law",
    description = "RAG + Legal-BERT + LED for Indian legal document summarisation",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],      # tighten in production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ─── Global state (loaded once at startup) ───────────────────────────────────

INDEX_DIR = os.getenv("INDEX_DIR", "./faiss_index")
LED_DIR   = os.getenv("LED_DIR",   "allenai/led-base-16384")    # or your fine-tuned path

_pipeline: Optional[RAGPipeline] = None


@app.on_event("startup")
async def load_models():
    global _pipeline
    try:
        logger.info("Loading Legal-BERT embedder...")
        embedder = LegalBERTEmbedder()

        logger.info("Loading FAISS index...")
        if Path(INDEX_DIR).exists():
            store = FAISSVectorStore.load(INDEX_DIR, embedder)
        else:
            logger.warning(f"No index found at {INDEX_DIR}. Search will be unavailable.")
            store = FAISSVectorStore(embedder)

        logger.info("Loading LED summarizer...")
        summarizer = LEDSummarizer(model_path=LED_DIR)

        try:
            reranker = CrossEncoderReranker()
        except Exception:
            logger.warning("CrossEncoder not available — skipping reranking.")
            reranker = None

        _pipeline = RAGPipeline(
            vector_store = store,
            summarizer   = summarizer,
            reranker     = reranker,
        )
        logger.info("All models loaded. API ready.")

    except Exception as exc:
        logger.error(f"Model load failed: {exc}")
        # Allow app to start even if models not loaded (for UI testing)


# ─── Request / Response models ───────────────────────────────────────────────

class TextSummarizeRequest(BaseModel):
    text:  str
    query: Optional[str] = None        # if provided, use RAG mode

class SearchRequest(BaseModel):
    query:  str
    top_k:  int = 5

class EvalPair(BaseModel):
    document:  str
    reference: str

class EvalRequest(BaseModel):
    pairs:      list[EvalPair]
    model_name: str = "LED+RAG (ours)"


# ─── Helper ──────────────────────────────────────────────────────────────────

def _extract_pdf(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    return "\n\n".join(page.get_text("text") for page in doc)


def _require_pipeline():
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet. Try again shortly.")
    return _pipeline


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {
        "status":  "ok",
        "models_loaded": _pipeline is not None,
        "timestamp": time.time(),
    }


@app.get("/api/stats")
async def stats():
    pipe = _require_pipeline()
    n = pipe.vector_store.index.ntotal if pipe.vector_store.index else 0
    return {
        "index_vectors": n,
        "led_model":     LED_DIR,
    }


@app.post("/api/summarize/text")
async def summarize_text(req: TextSummarizeRequest):
    """Summarise raw text. If query provided, use RAG; else direct LED."""
    pipe = _require_pipeline()

    start = time.time()
    if req.query:
        result = pipe.run(req.query)
    else:
        result = pipe.summarise_document(req.text)

    result["latency_s"] = round(time.time() - start, 2)
    return result


@app.post("/api/summarize/file")
async def summarize_file(file: UploadFile = File(...)):
    """
    Upload a PDF or .txt file and get a summary.
    Supports: .pdf, .txt, .html
    """
    pipe = _require_pipeline()

    data = await file.read()
    ext  = Path(file.filename).suffix.lower()

    if ext == ".pdf":
        text = _extract_pdf(data)
    elif ext in (".txt", ".html", ".htm"):
        text = data.decode("utf-8", errors="replace")
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from file.")

    start  = time.time()
    result = pipe.summarise_document(text)
    result["filename"]  = file.filename
    result["latency_s"] = round(time.time() - start, 2)
    return result


@app.post("/api/search")
async def search(req: SearchRequest):
    """Semantic search over the indexed legal corpus."""
    pipe     = _require_pipeline()
    results  = pipe.vector_store.search(req.query, top_k=req.top_k)
    return {"query": req.query, "results": results}


@app.post("/api/evaluate")
async def evaluate(req: EvalRequest, background_tasks: BackgroundTasks):
    """
    Run evaluation metrics on provided (document, reference) pairs.
    Returns ROUGE + BERTScore for our model vs extractive baseline.
    """
    from evaluation.metrics import Evaluator

    pipe = _require_pipeline()
    ev   = Evaluator()

    documents  = [p.document  for p in req.pairs]
    references = [p.reference for p in req.pairs]

    # Our model predictions
    our_preds = [
        pipe.summarise_document(doc)["summary"] for doc in documents
    ]
    our_result = ev.evaluate(our_preds, references, model_name=req.model_name)

    # Extractive baseline
    ext_preds  = ev.generate_baseline_summaries(documents, model="extractive")
    ext_result = ev.evaluate(ext_preds, references, model_name="Lead-5 (extractive)")

    return {
        "results": [our_result.to_dict(), ext_result.to_dict()]
    }
