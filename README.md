# Legal NLP Dissertation System
## Indian Legal Document Summarisation using RAG + Legal-BERT + LED

---

## Project Structure

```
legal_nlp/
├── data_pipeline/
│   ├── scraper.py           # Scrape IndianKanoon, IndiaCode, Gazette
│   └── preprocessor.py      # Clean text, sliding-window chunking
├── models/
│   ├── embedder.py          # Legal-BERT embeddings + FAISS index
│   └── summarizer.py        # LED summarizer + fine-tuning script
├── rag_engine/
│   └── pipeline.py          # RAG: retrieve → rerank → dedupe → summarise
├── evaluation/
│   └── metrics.py           # ROUGE + BERTScore vs BART/T5/Lead-5 baselines
├── api/
│   └── main.py              # FastAPI REST endpoints
├── frontend/
│   └── index.html           # Complete product UI
├── scripts/
│   └── build_index.py       # One-time index build script
├── notebooks/
│   └── dissertation_experiments.ipynb   # All experiments
└── requirements.txt
```

---

## Quick Start (Step by Step)

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Step 2 — Build the legal corpus & FAISS index
```bash
python -m scripts.build_index \
    --queries "Supreme Court judgment 2023" "RERA dispute" \
    --acts "Indian Contract Act" "RERA" \
    --max_per_source 30 \
    --index_dir ./faiss_index
```
This scrapes IndianKanoon + IndiaCode, chunks the text, embeds it with
Legal-BERT, and saves a FAISS index to `./faiss_index/`.

### Step 3 — Start the API
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Step 4 — Open the UI
Open `frontend/index.html` in a browser.
It connects to `http://localhost:8000` for live inference.

### Step 5 — Run the Jupyter notebook for experiments
```bash
jupyter notebook notebooks/dissertation_experiments.ipynb
```

---

## Running Evaluations

```python
from evaluation.metrics import Evaluator

ev = Evaluator()

# Your predictions from the LED+RAG system
our_preds = [...]        # generated summaries
gold      = [...]        # reference summaries (SCI headnotes)

our_result  = ev.evaluate(our_preds, gold, model_name="LED+RAG (ours)")
bart_preds  = ev.generate_baseline_summaries(docs, model="bart")
bart_result = ev.evaluate(bart_preds, gold, model_name="BART-base")

ev.compare([our_result, bart_result])
ev.save_results([our_result, bart_result], "evaluation/results.json")
```

---

## Fine-tuning LED on Indian Legal Data

1. Collect (document, summary) pairs from SCI headnotes
2. Run the fine-tuner in `models/summarizer.py`:

```python
from models.summarizer import LEDFineTuner

tuner = LEDFineTuner(output_dir="./finetuned_led")
tuner.train(train_data, eval_data, epochs=3)
```

3. Set `LED_DIR=./finetuned_led` in the `.env` before starting the API

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/summarize/text` | Summarise raw text (JSON body) |
| POST | `/api/summarize/file` | Upload PDF/TXT for summarisation |
| POST | `/api/search` | Semantic search over corpus |
| POST | `/api/evaluate` | ROUGE + BERTScore evaluation |
| GET  | `/api/health` | System health check |
| GET  | `/api/stats` | Index statistics |

---

## Research Contribution

**Why this system outperforms BART/T5 baselines:**

1. **Legal-BERT** embeddings understand Indian legal vocabulary
   (mens rea, locus standi, indenture, etc.) better than general BERT

2. **LED (16 384 token input)** processes full judgments without
   truncation, while BART/T5 are capped at 1 024 tokens

3. **RAG retrieval** surfaces the most legally relevant passages
   from multiple documents before summarisation

4. **Cross-encoder reranking** (MiniLM) improves retrieval precision
   from FAISS's approximate nearest-neighbour search

---

## Legal Data Sources (Indian Government)

| Source | URL | Content |
|--------|-----|---------|
| Indian Kanoon | https://indiankanoon.org / https://api.indiankanoon.org | Judgments from all courts |
| Indiacode | https://www.indiacode.nic.in | Central Acts & Amendments |
| Supreme Court | https://sci.gov.in | SCI orders & judgments |
| eGazette | https://egazette.nic.in | Govt notifications |
| e-Courts | https://ecourts.gov.in | District court judgments |

---

## Dissertation Citation / Bibliography

- Chalkidis et al. (2020). *LEGAL-BERT: The Muppets straight out of Law School* — ACL 2020
- Beltagy et al. (2020). *Longformer: The Long-Document Transformer* — arXiv 2020  
- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP* — NeurIPS 2020
- Zhang et al. (2020). *BERTScore: Evaluating Text Generation with BERT* — ICLR 2020
