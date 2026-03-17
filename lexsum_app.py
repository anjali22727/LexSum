"""
LexSum — Indian Legal NLP Summarisation System
================================================
SETUP:
    pip install fastapi uvicorn anthropic httpx python-multipart

CONFIGURE — open this file, find the line below and paste your key:
    ANTHROPIC_API_KEY = ""

RUN:
    python lexsum_app.py
    Open → http://localhost:8000
"""

import os, re, math, json, time
from collections import Counter
from typing import Optional, Generator

try:
    import fastapi, uvicorn, anthropic, httpx
except ImportError:
    print("\nMissing packages. Run:\n")
    print("    pip install fastapi uvicorn anthropic httpx python-multipart\n")
    raise SystemExit(1)

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ══════════════════════════════════════════════════════════════════
#  ▼▼  PASTE YOUR KEY HERE  ▼▼
# ══════════════════════════════════════════════════════════════════
ANTHROPIC_API_KEY = ""
#  Get a free key at:  https://console.anthropic.com → API Keys
# ══════════════════════════════════════════════════════════════════

MODEL = "claude-sonnet-4-20250514"

# ── Pure-Python NLP utilities ─────────────────────────────────────
STOPS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "is","are","was","were","be","been","have","has","had","do","does","did",
    "will","would","shall","should","may","might","can","could","that","this",
    "these","those","it","its","as","by","from","not","no","said","such",
    "also","into","upon","under","over","any","all","court","held","wherein",
    "thereof","herein","appellant","respondent","petitioner","judgment",
    "order","section","hon","ble","mr","justice","chief",
}

def tokenize(text: str) -> list:
    return [w for w in re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()
            if len(w) > 2 and w not in STOPS]

def tfidf_vecs(docs: list) -> list:
    N = len(docs)
    tok = [tokenize(d) for d in docs]
    vocab = list({w for t in tok for w in t})
    df = {t: sum(1 for d in tok if t in d) for t in vocab}
    out = []
    for tokens in tok:
        tf = Counter(tokens)
        n = len(tokens) or 1
        vec = {t: (tf[t]/n) * math.log((N+1)/(df[t]+1)) for t in vocab if tf[t]}
        out.append(vec)
    return out

def cos_sim(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    dot  = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    na   = math.sqrt(sum(v*v for v in a.values()))
    nb   = math.sqrt(sum(v*v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0

def accuracy_score(summary: str, source: str, query: str = "") -> dict:
    docs = [summary, source] + ([query] if query else [])
    v    = tfidf_vecs(docs)
    s    = cos_sim(v[0], v[1])
    q    = cos_sim(v[0], v[2]) if query else None
    acc  = min(1.0, max(0.0, 0.6*s + 0.4*q if q is not None else s))
    lbl  = ("Excellent" if acc >= 0.70 else "Good" if acc >= 0.50
            else "Fair" if acc >= 0.35 else "Poor")
    return {
        "source_cosine":  round(s,   4),
        "query_cosine":   round(q,   4) if q is not None else None,
        "accuracy_score": round(acc, 4),
        "accuracy_pct":   round(acc * 100, 1),
        "accuracy_label": lbl,
    }

def rouge_n(pred: str, ref: str, n: int) -> float:
    def ng(s):
        w = tokenize(s)
        return Counter(tuple(w[i:i+n]) for i in range(len(w)-n+1))
    pg, rg = ng(pred), ng(ref)
    m = sum(min(pg[k], rg[k]) for k in rg)
    return m / sum(rg.values()) if rg else 0.0

def rouge_l(pred: str, ref: str) -> float:
    pw, rw = tokenize(pred), tokenize(ref)
    M, N = len(pw), len(rw)
    dp = [[0]*(N+1) for _ in range(M+1)]
    for i in range(1, M+1):
        for j in range(1, N+1):
            dp[i][j] = (dp[i-1][j-1]+1 if pw[i-1]==rw[j-1]
                        else max(dp[i-1][j], dp[i][j-1]))
    lcs = dp[M][N]
    p = lcs/M if M else 0
    r = lcs/N if N else 0
    return 2*p*r/(p+r) if (p+r) else 0.0

def validate(summary: str, cosine: dict, query: str = "") -> dict:
    words = summary.split()
    checks, warns = [], []

    wc = len(words)
    if   wc < 30:  checks.append(("length", False, f"Too short: {wc} words", wc));  warns.append(f"Too short ({wc} words).")
    elif wc > 600: checks.append(("length", False, f"Too long: {wc} words",   wc));  warns.append(f"Too long ({wc} words).")
    else:          checks.append(("length", True,  f"Good: {wc} words",        wc))

    tg = [" ".join(words[i:i+3]).lower() for i in range(len(words)-2)]
    rr = (len(tg) - len(set(tg))) / len(tg) if tg else 0
    if rr > 0.25: checks.append(("repetition", False, f"High: {rr:.1%}", rr)); warns.append("Repetitive phrases.")
    else:          checks.append(("repetition", True,  f"Low: {rr:.1%}",  rr))

    g = cosine["source_cosine"]
    if g < 0.20: checks.append(("grounding", False, f"Low: {g:.4f}", g)); warns.append("May not reflect source.")
    else:         checks.append(("grounding", True,  f"Good: {g:.4f}", g))

    qr = cosine.get("query_cosine")
    if query and qr is not None:
        if qr < 0.15: checks.append(("relevance", False, f"Low: {qr:.4f}", qr)); warns.append("May not address query.")
        else:          checks.append(("relevance", True,  f"Good: {qr:.4f}", qr))
    else:
        checks.append(("relevance", True, "Skipped (no query)", None))

    sl = summary.lower()
    contra, cdet = False, "No verdict contradictions"
    for pos, neg in [
        (["allowed","granted","upheld"],  ["dismissed","rejected","denied","quashed"]),
        (["guilty","convicted"],          ["acquitted","discharged"]),
    ]:
        if any(w in sl for w in pos) and any(w in sl for w in neg):
            contra = True
            ph = next(w for w in pos if w in sl)
            nh = next(w for w in neg if w in sl)
            cdet = f'Conflict: "{ph}" and "{nh}"'
            warns.append(cdet)
            break
    checks.append(("contradiction", not contra, cdet, None))

    np_ = sum(1 for _,ok,_,_ in checks if ok)
    conf = np_ / len(checks)
    lbl  = "VALID" if conf == 1.0 else "WARNINGS" if conf >= 0.6 else "INVALID"
    return {
        "checks":        [{"name":n,"passed":ok,"detail":d,"value":v} for n,ok,d,v in checks],
        "warnings":      warns,
        "overall_label": lbl,
        "confidence":    round(conf, 4),
        "confidence_pct": round(conf * 100, 1),
        "recommendation": {
            "VALID":    "Summary is reliable.",
            "WARNINGS": "Minor issues — review warnings.",
            "INVALID":  "Low quality — try more source text.",
        }[lbl],
    }

# ── Get API key ───────────────────────────────────────────────────
def get_key() -> str:
    return ANTHROPIC_API_KEY or os.getenv("ANTHROPIC_API_KEY", "")

def check_key():
    k = get_key()
    if not k:
        raise HTTPException(400,
            "API key missing. Open lexsum_app.py, find ANTHROPIC_API_KEY = \"\" "
            "and paste your key. Get one at https://console.anthropic.com")
    if not k.startswith("sk-ant-"):
        raise HTTPException(401,
            f"Invalid key — must start with sk-ant-  "
            f"(yours starts with: {k[:10]}). "
            f"Copy the full key from https://console.anthropic.com")
    return k

# ── Prompts ───────────────────────────────────────────────────────
def make_prompt(text: str, query: str) -> str:
    focus = f'\n\nPay special attention to answering: "{query}"' if query else ""
    if text.strip():
        return (
            "You are an expert Indian legal analyst. "
            "Analyse the document and write a structured summary."
            f"{focus}\n\n"
            "Use these exact bold headings:\n"
            "**Parties** — Appellant vs Respondent\n"
            "**Legal Issue** — Core question of law\n"
            "**Arguments** — Key contentions of both sides\n"
            "**Court's Reasoning** — How the court applied the law\n"
            "**Decision / Held** — Final order and specific reliefs\n"
            "**Legal Principle** — Precedent or rule established\n\n"
            "Write 150-250 words. Use precise legal language.\n\n"
            f"DOCUMENT:\n{text[:8000]}"
        )
    else:
        q = query or "recent important Indian Supreme Court judgment"
        return (
            "You are an expert Indian legal analyst. "
            f'Search the web for this legal topic: "{q}"\n\n'
            "Search indiankanoon.org, sci.gov.in, and legal news sites. "
            "Then write a structured summary with these bold headings:\n"
            "**Case / Topic** — Full name and citation\n"
            "**Parties** — Appellant vs Respondent\n"
            "**Legal Issue** — Core question of law\n"
            "**Court's Reasoning** — Analysis\n"
            "**Decision / Held** — Final order\n"
            "**Legal Principle** — Rule established\n"
            "**Sources** — URLs you found\n\n"
            "Write 200-300 words. Use precise legal language."
        )

# ═══════════════════════════════════════════════════════════════════
# CORE STREAMING FUNCTION — this is the main engine
# ═══════════════════════════════════════════════════════════════════
def stream_summary(text: str, query: str) -> Generator:
    """
    Calls Anthropic API with web_search tool.
    Uses the correct SDK pattern: iterate over stream events using
    event.type attribute (not class name).
    Yields SSE strings.
    """
    key    = check_key()
    client = anthropic.Anthropic(api_key=key)
    prompt = make_prompt(text, query)

    yield f"data: {json.dumps({'type':'status', 'msg':'Connecting to Claude...'})}\n\n"

    full_text   = ""
    all_sources = []
    t0          = time.time()

    try:
        # ── Use the streaming context manager ──────────────────────
        with client.messages.stream(
            model      = MODEL,
            max_tokens = 1500,
            tools      = [{"type": "web_search_20250305", "name": "web_search"}],
            messages   = [{"role": "user", "content": prompt}],
        ) as stream:

            # Iterate raw events from the stream
            for event in stream:

                # Get event type as string — works across all SDK versions
                etype = getattr(event, "type", "")

                # ── Text token arriving ─────────────────────────────
                if etype == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    if delta and getattr(delta, "type", "") == "text_delta":
                        chunk = delta.text
                        full_text += chunk
                        yield f"data: {json.dumps({'type':'chunk', 'text':chunk})}\n\n"

                # ── New content block starting ──────────────────────
                elif etype == "content_block_start":
                    block = getattr(event, "content_block", None)
                    if block:
                        btype = getattr(block, "type", "")
                        if btype == "tool_use":
                            yield f"data: {json.dumps({'type':'status', 'msg':'Searching the web for Indian legal sources...'})}\n\n"
                        elif btype == "text":
                            # New text block starting — nothing to yield yet
                            pass

                # ── Message complete ────────────────────────────────
                elif etype == "message_stop":
                    pass

            # ── After stream ends: get full message for sources ─────
            try:
                final_msg = stream.get_final_message()
                for block in final_msg.content:
                    btype = getattr(block, "type", "")

                    # Collect any text we might have missed
                    if btype == "text" and not full_text:
                        full_text = getattr(block, "text", "")

                    # Extract web search results / sources
                    elif btype == "tool_result":
                        content = getattr(block, "content", [])
                        if isinstance(content, list):
                            for item in content:
                                url   = getattr(item, "url",   None)
                                title = getattr(item, "title", None)
                                if url:
                                    src = {"url": url, "title": title or url}
                                    all_sources.append(src)
                                    yield f"data: {json.dumps({'type':'source', **src})}\n\n"

                    # Also handle server_tool_use blocks (newer SDK format)
                    elif btype == "server_tool_use":
                        pass  # The result comes back separately

            except Exception:
                pass  # final_message extraction is best-effort

    except anthropic.AuthenticationError as e:
        yield f"data: {json.dumps({'type':'error', 'msg':f'Authentication failed: {str(e)}. Check your API key at https://console.anthropic.com'})}\n\n"
        return
    except anthropic.RateLimitError as e:
        yield f"data: {json.dumps({'type':'error', 'msg':f'Rate limit hit. Wait a moment and try again.'})}\n\n"
        return
    except anthropic.APIError as e:
        yield f"data: {json.dumps({'type':'error', 'msg':f'Anthropic API error: {str(e)}'})}\n\n"
        return
    except Exception as e:
        yield f"data: {json.dumps({'type':'error', 'msg':f'Error: {str(e)}'})}\n\n"
        return

    # ── If Claude called the web_search tool, it may need another turn ──
    # The web_search_20250305 tool is handled server-side by Anthropic —
    # Claude's response already includes the search results inline.
    # No manual tool result passing needed.

    if not full_text:
        yield f"data: {json.dumps({'type':'error', 'msg':'No response received. Try again or check your API key.'})}\n\n"
        return

    latency  = round(time.time() - t0, 2)
    src_ref  = text[:3000] if text.strip() else (query or "legal document")
    cosine   = accuracy_score(full_text, src_ref, query or "")
    val      = validate(full_text, cosine, query or "")

    yield f"data: {json.dumps({'type':'done', 'summary':full_text, 'latency_s':latency, 'cosine':cosine, 'validation':val, 'sources':all_sources})}\n\n"


# ═══════════════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════════════
app = FastAPI(title="LexSum", version="3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

class SumReq(BaseModel):
    text:  str = ""
    query: str = ""

class EvalReq(BaseModel):
    document:  str
    reference: str

class SearchReq(BaseModel):
    query:   str
    top_k:   int = 8
    doctype: str = ""

@app.get("/api/health")
def health():
    k = get_key()
    ok = bool(k)
    fmt_ok = k.startswith("sk-ant-") if k else False
    return {
        "status":            "ok",
        "key_present":       ok,
        "key_format_valid":  fmt_ok,
        "model":             MODEL,
        "web_search":        "enabled",
        "note": ("Key looks valid" if fmt_ok
                 else "Key missing — edit ANTHROPIC_API_KEY in lexsum_app.py" if not ok
                 else f"Key format wrong (starts with {k[:10]}) — must start with sk-ant-"),
    }

@app.post("/api/summarise/stream")
def do_stream(req: SumReq):
    # validate key before starting stream so errors are visible
    check_key()
    return StreamingResponse(
        stream_summary(req.text, req.query),
        media_type = "text/event-stream",
        headers    = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.post("/api/summarise/file")
async def do_file(file: UploadFile = File(...), query: str = ""):
    check_key()
    content = await file.read()
    if file.filename.lower().endswith(".txt"):
        text = content.decode("utf-8", errors="ignore")
    elif file.filename.lower().endswith(".pdf"):
        try:
            import pypdf
            from io import BytesIO
            reader = pypdf.PdfReader(BytesIO(content))
            text   = "\n\n".join(p.extract_text() or "" for p in reader.pages)
        except ImportError:
            raise HTTPException(400, "Install pypdf first:  pip install pypdf")
    else:
        raise HTTPException(400, "Only .txt and .pdf files supported")

    # collect stream into final result
    full, cosine_data, val_data, sources = "", None, None, []
    for chunk in stream_summary(text, query):
        if chunk.startswith("data: "):
            try:
                d = json.loads(chunk[6:])
                if d["type"] == "chunk":
                    full += d["text"]
                elif d["type"] == "source":
                    sources.append({"url": d["url"], "title": d.get("title","")})
                elif d["type"] == "done":
                    full       = d["summary"]
                    cosine_data = d["cosine"]
                    val_data    = d["validation"]
                elif d["type"] == "error":
                    raise HTTPException(500, d["msg"])
            except (json.JSONDecodeError, KeyError):
                pass

    return {"summary": full, "cosine": cosine_data, "validation": val_data,
            "sources": sources, "latency_s": 0}

@app.post("/api/search")
async def do_search(req: SearchReq):
    params = {"formInput": req.query}
    if req.doctype:
        params["doctype"] = req.doctype
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(
                "https://api.indiankanoon.org/search/",
                data    = params,
                headers = {"Content-Type": "application/x-www-form-urlencoded"},
            )
            r.raise_for_status()
            data = r.json()
        docs = data.get("docs", [])[:req.top_k]
        results = [{
            "tid":     d.get("tid"),
            "title":   d.get("title", "Untitled"),
            "court":   d.get("docsource", ""),
            "date":    d.get("publishdate", ""),
            "citation":d.get("citation", ""),
            "snippet": re.sub(r"<[^>]+>", "", d.get("headline") or d.get("fragment",""))[:300],
            "url":     f"https://indiankanoon.org/doc/{d.get('tid')}/",
        } for d in docs]
        return {"results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(502, f"Indian Kanoon search failed: {str(e)}")

@app.get("/api/doc/{tid}")
async def get_doc(tid: str):
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post(
                f"https://api.indiankanoon.org/doc/{tid}/",
                headers = {"Content-Type": "application/x-www-form-urlencoded"},
            )
            r.raise_for_status()
            data = r.json()
        html = data.get("doc", "")
        text = re.sub(r"\s{2,}", " ", re.sub(r"<[^>]+>", " ", html)).strip()
        return {"tid": tid, "text": text, "words": len(text.split())}
    except Exception as e:
        raise HTTPException(502, f"Indian Kanoon fetch failed: {str(e)}")

@app.post("/api/evaluate")
def do_eval(req: EvalReq):
    key = check_key()
    client = anthropic.Anthropic(api_key=key)

    # Generate our summary (non-streaming, simpler for eval)
    msg = client.messages.create(
        model      = MODEL,
        max_tokens = 500,
        messages   = [{"role": "user", "content":
            f"Summarise this Indian legal document in 120-180 words. "
            f"Cover: parties, legal issue, court reasoning, final decision.\n\n{req.document}"}],
    )
    # Extract text from response (handle both text blocks and tool_use blocks)
    our_sum = ""
    for block in msg.content:
        if getattr(block, "type", "") == "text":
            our_sum += block.text

    # Lead-5 extractive baseline
    sents   = [s.strip() for s in re.split(r"(?<=[.!?])\s+",
               req.document.replace("\n", " ")) if len(s) > 20]
    lead    = " ".join(sents[:5])

    def sc(pred, ref):
        v = tfidf_vecs([pred, ref])
        return {
            "rouge1": round(rouge_n(pred, ref, 1), 4),
            "rouge2": round(rouge_n(pred, ref, 2), 4),
            "rougeL": round(rouge_l(pred, ref),    4),
            "cosine": round(cos_sim(v[0], v[1]),   4),
            "words":  len(pred.split()),
        }

    return {"models": [
        {"name": "Claude AI + Web Search (ours)", "summary": our_sum,  **sc(our_sum, req.reference)},
        {"name": "Lead-5 (extractive baseline)",  "summary": lead,     **sc(lead,    req.reference)},
    ]}


# ═══════════════════════════════════════════════════════════════════
# FRONTEND HTML
# ═══════════════════════════════════════════════════════════════════
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LexSum — Indian Legal NLP</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{--bg:#0d1117;--s1:#161b22;--s2:#1c2333;--bd:#30363d;--ac:#d4a843;--ac2:#c17f24;--tx:#e6edf3;--mu:#7d8590;--gn:#3fb950;--bl:#58a6ff;--rd:#e15759;--pu:#bc8cff;}
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:'DM Sans',sans-serif;background:var(--bg);color:var(--tx);min-height:100vh;line-height:1.6;}
nav{display:flex;align-items:center;justify-content:space-between;padding:0 1.5rem;height:56px;border-bottom:1px solid var(--bd);background:rgba(13,17,23,.97);position:sticky;top:0;z-index:99;}
.logo{font-family:'DM Serif Display',serif;font-size:1.35rem;color:var(--ac);}.logo em{color:var(--tx);font-style:normal;}
.pill{font-family:'JetBrains Mono',monospace;font-size:11px;padding:3px 10px;border-radius:20px;border:1px solid var(--bd);color:var(--mu);}
.pill.ok{color:var(--gn);border-color:rgba(63,185,80,.35);}
.pill.bad{color:var(--rd);border-color:rgba(225,87,89,.35);}
.pill.warn{color:var(--ac);border-color:rgba(212,168,67,.35);}
.wrap{max-width:1000px;margin:0 auto;padding:1.5rem 1rem 3rem;}
.hero{text-align:center;padding:1.5rem 0 1rem;}
.hero h1{font-family:'DM Serif Display',serif;font-size:2rem;line-height:1.2;margin-bottom:.4rem;}
.hero h1 em{color:var(--ac);font-style:italic;}
.hero p{color:var(--mu);font-size:.88rem;max-width:480px;margin:0 auto .8rem;}
.tags{display:flex;gap:6px;justify-content:center;flex-wrap:wrap;margin-bottom:1.2rem;}
.tag{font-family:'JetBrains Mono',monospace;font-size:.66rem;padding:3px 9px;border-radius:20px;border:1px solid;}
.tabs{display:flex;gap:2px;background:var(--s1);border:1px solid var(--bd);border-radius:10px;padding:3px;width:fit-content;margin-bottom:1.2rem;}
.tab{padding:5px 16px;border-radius:7px;border:1px solid transparent;background:transparent;color:var(--mu);cursor:pointer;font-family:'DM Sans',sans-serif;font-size:.82rem;font-weight:500;}
.tab.on{background:var(--s2);color:var(--tx);border-color:var(--bd);}
.pane{display:none;}.pane.on{display:block;}
.card{background:var(--s1);border:1px solid var(--bd);border-radius:12px;overflow:hidden;margin-bottom:10px;animation:up .25s ease;}
@keyframes up{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:none}}
.chd{background:var(--s2);border-bottom:1px solid var(--bd);padding:7px 14px;display:flex;justify-content:space-between;align-items:center;}
.cht{font-size:11px;font-weight:600;color:var(--mu);text-transform:uppercase;letter-spacing:.7px;}
.cbdy{padding:14px;}
label,.lbl{font-size:11px;font-weight:600;color:var(--mu);text-transform:uppercase;letter-spacing:.7px;display:block;margin-bottom:5px;}
textarea,input[type=text]{width:100%;background:var(--bg);border:1px solid var(--bd);border-radius:8px;color:var(--tx);font-family:'DM Sans',sans-serif;font-size:.88rem;padding:.65rem .9rem;outline:none;transition:border-color .15s;}
textarea{resize:vertical;line-height:1.55;}textarea:focus,input:focus{border-color:var(--ac);}
select{background:var(--bg);border:1px solid var(--bd);border-radius:8px;color:var(--tx);font-family:'DM Sans',sans-serif;font-size:.85rem;padding:.55rem .8rem;outline:none;cursor:pointer;}
.btn{display:inline-flex;align-items:center;gap:5px;padding:.6rem 1.3rem;border-radius:8px;border:none;font-family:'DM Sans',sans-serif;font-size:.85rem;font-weight:600;cursor:pointer;transition:all .15s;}
.bp{background:var(--ac);color:#000;}.bp:hover{background:var(--ac2);}.bp:disabled{opacity:.45;cursor:not-allowed;}
.bg{background:transparent;color:var(--mu);border:1px solid var(--bd);}.bg:hover{color:var(--tx);}
.row{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:10px;}
.dz{border:2px dashed var(--bd);border-radius:10px;padding:1.5rem 1rem;text-align:center;cursor:pointer;transition:all .2s;}
.dz:hover{border-color:var(--ac);background:rgba(212,168,67,.04);}
.dz p{color:var(--mu);font-size:.78rem;}
.bdg{font-family:'JetBrains Mono',monospace;font-size:11px;padding:2px 6px;border-radius:4px;background:var(--s2);color:var(--mu);display:inline-block;}
.sp{width:16px;height:16px;border:2px solid var(--bd);border-top-color:var(--ac);border-radius:50%;animation:spin .7s linear infinite;display:inline-block;flex-shrink:0;}
@keyframes spin{to{transform:rotate(360deg)}}
.sw{display:flex;align-items:center;gap:10px;padding:1.2rem;color:var(--mu);font-size:.85rem;}
.stxt{font-family:'DM Serif Display',serif;font-size:1.02rem;line-height:1.8;color:var(--tx);}
.cur{display:inline-block;width:2px;height:1.1em;background:var(--ac);margin-left:2px;vertical-align:text-bottom;animation:bk .9s step-end infinite;}
@keyframes bk{0%,100%{opacity:1}50%{opacity:0}}
.bar-bg{height:8px;background:var(--bd);border-radius:4px;overflow:hidden;margin-bottom:6px;}
.bar-fg{height:100%;border-radius:4px;transition:width .8s ease;}
.chk{display:flex;gap:8px;align-items:flex-start;padding:4px 0;border-bottom:1px solid rgba(48,54,61,.4);}
.tbl{width:100%;border-collapse:collapse;font-size:.82rem;}
.tbl th{text-align:left;padding:.5rem .9rem;font-size:10px;text-transform:uppercase;letter-spacing:.7px;color:var(--mu);border-bottom:1px solid var(--bd);}
.tbl td{padding:.55rem .9rem;border-bottom:1px solid rgba(48,54,61,.4);font-family:'JetBrains Mono',monospace;}
.tbl tr:last-child td{border:none;}
.tbl .hi td{color:var(--ac);}
.tbl .mn{font-family:'DM Sans',sans-serif;font-weight:500;}
.ik-item{background:var(--bg);border:1px solid var(--bd);border-radius:10px;padding:.9rem 1rem;margin-bottom:8px;transition:border-color .15s;}
.ik-item:hover{border-color:var(--ac);}
.err{background:rgba(225,87,89,.08);border:1px solid rgba(225,87,89,.3);border-radius:8px;padding:.75rem 1rem;font-size:.82rem;color:var(--rd);margin-bottom:10px;}
.info{background:rgba(63,185,80,.06);border:1px solid rgba(63,185,80,.2);border-radius:8px;padding:.65rem 1rem;font-size:.82rem;color:var(--mu);margin-bottom:12px;}
.status-msg{display:flex;align-items:center;gap:8px;padding:.55rem .9rem;font-size:.8rem;color:var(--pu);background:rgba(188,140,255,.06);border:1px solid rgba(188,140,255,.15);border-radius:8px;margin-bottom:8px;}
.warnbox{background:rgba(212,168,67,.07);border:1px solid rgba(212,168,67,.3);border-radius:8px;padding:.7rem .9rem;margin-bottom:10px;}
footer{text-align:center;padding:2rem;color:var(--mu);font-size:.72rem;border-top:1px solid var(--bd);}
code{font-family:'JetBrains Mono',monospace;background:var(--s2);padding:1px 5px;border-radius:3px;font-size:.82rem;}
a{color:var(--bl);text-decoration:none;}a:hover{text-decoration:underline;}
</style>
</head>
<body>
<nav>
  <div class="logo">Lex<em>Sum</em></div>
  <div style="display:flex;gap:8px;align-items:center;">
    <div class="pill" id="kpill">checking...</div>
    <div class="pill ok">🌐 Web Search ON</div>
    <div class="pill">v3.0</div>
  </div>
</nav>
<div class="wrap">
  <div class="hero">
    <h1>Indian Legal NLP<br><em>Summarisation System</em></h1>
    <p>Real-time AI summarisation — Claude searches the web live for every query.</p>
    <div class="tags">
      <span class="tag" style="color:#d4a843;border-color:#c17f24;background:rgba(212,168,67,.08)">Claude Sonnet · Streaming</span>
      <span class="tag" style="color:#3fb950;border-color:rgba(63,185,80,.4);background:rgba(63,185,80,.06)">Live Web Search</span>
      <span class="tag" style="color:#bc8cff;border-color:rgba(188,140,255,.4);background:rgba(188,140,255,.06)">TF-IDF Cosine Scoring</span>
      <span class="tag" style="color:#58a6ff;border-color:rgba(88,166,255,.4);background:rgba(88,166,255,.06)">5-Check Validator</span>
    </div>
  </div>

  <div class="tabs">
    <button class="tab on" onclick="gTab('sum',this)">Summarise</button>
    <button class="tab" onclick="gTab('ik',this)">IK Search</button>
    <button class="tab" onclick="gTab('eval',this)">Evaluation</button>
  </div>

  <!-- SUMMARISE -->
  <div id="pane-sum" class="pane on">
    <div class="info">
      🌐 <strong style="color:var(--gn)">Web Search active.</strong>
      Paste a judgment to summarise it — OR leave text empty, enter a case name/query, and Claude will search the web for it.
    </div>
    <div class="row">
      <div style="flex:2;min-width:240px;">
        <label>Document text <span style="text-transform:none;letter-spacing:0;font-weight:400;">(optional — leave empty to search web)</span></label>
        <textarea id="doc" rows="8" placeholder="Paste Indian court judgment text here.&#10;&#10;OR leave empty and type a case name in the Query field below — Claude will search the web for it."></textarea>
      </div>
      <div style="flex:1;min-width:180px;">
        <label>Upload PDF / TXT</label>
        <div class="dz" onclick="document.getElementById('fu').click()">
          <div style="font-size:2rem;margin-bottom:4px;">📄</div>
          <p><strong style="color:var(--ac)">Click to upload</strong><br>PDF · TXT</p>
        </div>
        <input type="file" id="fu" accept=".pdf,.txt" style="display:none" onchange="uploadFile(this)">
        <div id="fname" style="font-size:11px;color:var(--mu);margin-top:4px;"></div>
      </div>
    </div>
    <label>Query / Case name <span style="text-transform:none;letter-spacing:0;font-weight:400;">(required if text box is empty)</span></label>
    <input type="text" id="qry" style="margin-bottom:10px;"
      placeholder="e.g.  Kesavananda Bharati v State of Kerala  OR  Supreme Court RERA judgment 2023">
    <div style="display:flex;gap:8px;margin-bottom:14px;flex-wrap:wrap;">
      <button class="btn bp" id="sbtn" onclick="doSum()">⚡ Generate Summary</button>
      <button class="btn bg" onclick="loadDemo()">Load Demo</button>
      <button class="btn bg" onclick="clearAll()">Clear</button>
    </div>
    <div id="serr"></div>
    <div id="sout"></div>
  </div>

  <!-- IK SEARCH -->
  <div id="pane-ik" class="pane">
    <div class="card">
      <div class="chd"><span class="cht">Search Indian Kanoon</span><span class="bdg" style="color:var(--gn)">● Live</span></div>
      <div class="cbdy">
        <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px;">
          <input type="text" id="iq" style="flex:1;min-width:200px;"
            placeholder="e.g. breach of contract force majeure COVID Supreme Court 2022"
            onkeydown="if(event.key==='Enter')doSearch()">
          <select id="idt">
            <option value="">All courts</option>
            <option value="supremecourt">Supreme Court</option>
            <option value="highcourt">High Courts</option>
            <option value="tribunal">Tribunals</option>
          </select>
          <button class="btn bp" onclick="doSearch()">Search</button>
        </div>
        <p style="font-size:12px;color:var(--mu);">Live from <strong>indiankanoon.org</strong> — click any result to load into Summarise tab.</p>
      </div>
    </div>
    <div id="ikout"></div>
  </div>

  <!-- EVALUATE -->
  <div id="pane-eval" class="pane">
    <label>Document</label>
    <textarea id="ed" rows="5" style="margin-bottom:10px;" placeholder="Paste the legal document text..."></textarea>
    <label>Reference / Gold Summary</label>
    <textarea id="er" rows="3" style="margin-bottom:10px;" placeholder="Paste a reference summary (e.g. SCI headnote) to score against..."></textarea>
    <div style="display:flex;gap:8px;margin-bottom:14px;flex-wrap:wrap;">
      <button class="btn bp" id="ebtn" onclick="doEval()">📊 Run Evaluation</button>
      <button class="btn bg" onclick="loadEvalDemo()">Load Demo</button>
    </div>
    <div id="eerr"></div>
    <div id="eout"></div>
  </div>
</div>
<footer>LexSum v3.0 · Indian Legal NLP · Claude AI + Live Web Search · Dissertation Research</footer>

<script>
const E = id => document.getElementById(id);
const esc = s => String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
const AC = p => p>=70?"#3fb950":p>=50?"#58a6ff":p>=35?"#d4a843":"#e15759";
const VI = l => l==="VALID"?{i:"✓",c:"#3fb950"}:l==="WARNINGS"?{i:"⚠",c:"#d4a843"}:{i:"✕",c:"#e15759"};

// ── health check ────────────────────────────────────────────────────
fetch("/api/health").then(r=>r.json()).then(d=>{
  const p = E("kpill");
  if (d.key_format_valid) {
    p.textContent = "● Claude ready"; p.className = "pill ok";
  } else if (d.key_present) {
    p.textContent = "⚠ Key format wrong"; p.className = "pill bad";
    E("serr").innerHTML = `<div class="err">⚠ ${esc(d.note)}</div>`;
  } else {
    p.textContent = "⚠ No API key"; p.className = "pill bad";
    E("serr").innerHTML = `<div class="err">⚠ <strong>API key not set.</strong> Open <code>lexsum_app.py</code>, find <code>ANTHROPIC_API_KEY = ""</code> and paste your key. Get one free at <a href="https://console.anthropic.com" target="_blank">console.anthropic.com</a></div>`;
  }
}).catch(()=>{ E("kpill").textContent="● offline"; E("kpill").className="pill bad"; });

// ── tabs ──────────────────────────────────────────────────────────
function gTab(id,btn){
  document.querySelectorAll(".pane").forEach(p=>p.classList.remove("on"));
  document.querySelectorAll(".tab").forEach(b=>b.classList.remove("on"));
  E("pane-"+id).classList.add("on"); btn.classList.add("on");
}

// ── score card ────────────────────────────────────────────────────
function scoreCard(c, v, lat, sources){
  const ap=c.accuracy_pct, ac=AC(ap), {i:vi,c:vc}=VI(v.overall_label);
  const checks=v.checks.map(ch=>`
    <div class="chk">
      <span style="color:${ch.passed?"#3fb950":"#e15759"};flex-shrink:0;margin-top:1px;">${ch.passed?"✓":"✕"}</span>
      <div>
        <span style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.5px;color:var(--mu)">${ch.name}</span>
        <span style="font-size:12px;color:var(--tx);margin-left:6px;">${esc(ch.detail)}</span>
        ${ch.value!=null?`<span style="font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--mu);margin-left:4px;">[${typeof ch.value==="number"?ch.value.toFixed(4):ch.value}]</span>`:""}
      </div>
    </div>`).join("");
  const wb=v.warnings.length?`<div class="warnbox"><div style="font-size:11px;font-weight:600;color:var(--ac);text-transform:uppercase;letter-spacing:.6px;margin-bottom:4px;">Warnings</div>${v.warnings.map(w=>`<div style="font-size:12px;color:var(--tx);">⚠ ${esc(w)}</div>`).join("")}</div>`:"";
  const srcs=sources&&sources.length?`<div class="card" style="margin-top:10px;"><div class="chd"><span class="cht">Web sources used</span><span class="bdg">${sources.length}</span></div><div class="cbdy">${sources.map(s=>`<div style="display:flex;gap:8px;align-items:flex-start;padding:5px 0;border-bottom:1px solid rgba(48,54,61,.3);font-size:12px;"><span style="width:7px;height:7px;border-radius:50%;background:var(--gn);flex-shrink:0;margin-top:4px;display:inline-block;"></span><div><a href="${esc(s.url)}" target="_blank">${esc(s.title||s.url)}</a><div style="font-size:11px;color:var(--mu);word-break:break-all;">${esc(s.url)}</div></div></div>`).join("")}</div></div>`:"";
  return `<div class="card"><div class="chd"><span class="cht">Accuracy score &amp; validation</span></div><div class="cbdy">
    <div style="display:flex;align-items:center;gap:16px;margin-bottom:14px;flex-wrap:wrap;">
      <div style="text-align:center;min-width:80px;">
        <div style="font-family:'DM Serif Display',serif;font-size:2.5rem;line-height:1;color:${ac};">${ap.toFixed(1)}<span style="font-size:1rem;">%</span></div>
        <div style="font-size:10px;color:var(--mu);text-transform:uppercase;letter-spacing:.6px;">Accuracy</div>
        <div style="font-size:12px;font-weight:600;color:${ac};margin-top:2px;">${c.accuracy_label}</div>
      </div>
      <div style="flex:1;min-width:180px;">
        <div class="bar-bg"><div class="bar-fg" style="width:${ap}%;background:${ac};"></div></div>
        <div style="font-size:11px;color:var(--mu);line-height:1.5;">
          TF-IDF cosine: source grounding${c.query_cosine!=null?" + query relevance":""}<br>
          <span style="font-family:'JetBrains Mono',monospace;">src:${c.source_cosine.toFixed(4)}${c.query_cosine!=null?" qry:"+c.query_cosine.toFixed(4):""} · ${lat}s</span>
        </div>
      </div>
      <div style="text-align:center;padding:8px 14px;border:1px solid ${vc};border-radius:10px;">
        <div style="font-size:1.4rem;color:${vc};">${vi}</div>
        <div style="font-size:10px;color:${vc};font-weight:600;text-transform:uppercase;letter-spacing:.6px;">${v.overall_label}</div>
        <div style="font-size:10px;color:var(--mu);">${v.confidence_pct.toFixed(0)}% passed</div>
      </div>
    </div>
    ${wb}
    <div style="font-size:10px;font-weight:600;color:var(--mu);text-transform:uppercase;letter-spacing:.7px;margin-bottom:5px;">Validation checks</div>
    ${checks}
    <div style="margin-top:8px;padding:6px 10px;border-left:2px solid ${vc};font-size:12px;color:var(--tx);">${esc(v.recommendation)}</div>
  </div></div>${srcs}`;
}

// ── SUMMARISE ─────────────────────────────────────────────────────
async function doSum(){
  const text=E("doc").value.trim(), query=E("qry").value.trim();
  if(!text&&!query){ E("serr").innerHTML=`<div class="err">Enter a query or paste a document.</div>`; return; }
  E("serr").innerHTML=""; E("sbtn").disabled=true;
  E("sout").innerHTML=`<div class="card"><div class="chd"><span class="cht">Generated summary</span><span class="bdg" id="lat-lbl">connecting...</span></div><div class="cbdy"><div id="smsg"></div><div class="stxt" id="sstream"><span class="cur"></span></div></div></div>`;

  const resp = await fetch("/api/summarise/stream",{
    method:"POST", headers:{"Content-Type":"application/json"},
    body:JSON.stringify({text, query})
  }).catch(e=>{ E("serr").innerHTML=`<div class="err">${esc(e.message)}</div>`; E("sbtn").disabled=false; return null; });
  if(!resp) return;

  if(!resp.ok){
    const d=await resp.json().catch(()=>({}));
    E("serr").innerHTML=`<div class="err">${esc(d.detail||"HTTP "+resp.status)}</div>`;
    E("sout").innerHTML=""; E("sbtn").disabled=false; return;
  }

  const reader=resp.body.getReader(), dec=new TextDecoder();
  let buf="", full="", sources=[];
  while(true){
    const{done,value}=await reader.read(); if(done) break;
    buf+=dec.decode(value,{stream:true});
    const parts=buf.split("\n\n"); buf=parts.pop();
    for(const part of parts){
      if(!part.startsWith("data: ")) continue;
      let d; try{ d=JSON.parse(part.slice(6)); }catch{ continue; }
      if(d.type==="status"){
        const m=E("smsg"); if(m) m.innerHTML=`<div class="status-msg"><div class="sp"></div>${esc(d.msg)}</div>`;
        const lb=E("lat-lbl"); if(lb) lb.textContent=d.msg.length>35?d.msg.slice(0,35)+"...":d.msg;
      } else if(d.type==="chunk"){
        full+=d.text;
        const el=E("sstream");
        if(el) el.innerHTML=full.replace(/\*\*(.*?)\*\*/g,"<strong>$1</strong>").replace(/\n/g,"<br>")+'<span class="cur"></span>';
      } else if(d.type==="source"){
        sources.push(d);
      } else if(d.type==="error"){
        E("serr").innerHTML=`<div class="err">${esc(d.msg)}</div>`;
        E("sout").innerHTML=""; E("sbtn").disabled=false; return;
      } else if(d.type==="done"){
        const lat=d.latency_s||"—";
        const lb=E("lat-lbl"); if(lb) lb.textContent=lat+"s · Claude";
        const m=E("smsg"); if(m) m.remove();
        const el=E("sstream");
        if(el) el.innerHTML=(d.summary||full).replace(/\*\*(.*?)\*\*/g,"<strong>$1</strong>").replace(/\n/g,"<br>");
        const allSrc=[...sources,...(d.sources||[])];
        E("sout").innerHTML+=scoreCard(d.cosine,d.validation,lat,allSrc);
      }
    }
  }
  E("sbtn").disabled=false;
}

// ── FILE UPLOAD ───────────────────────────────────────────────────
async function uploadFile(input){
  const file=input.files[0]; if(!file) return;
  E("fname").textContent="📎 "+file.name;
  const fd=new FormData(); fd.append("file",file); fd.append("query",E("qry").value.trim());
  E("serr").innerHTML=""; E("sout").innerHTML=`<div class="sw"><div class="sp"></div>Processing ${esc(file.name)}...</div>`;
  try{
    const r=await fetch("/api/summarise/file",{method:"POST",body:fd});
    const d=await r.json(); if(!r.ok) throw new Error(d.detail||"Upload failed");
    E("doc").value=d.text||"";
    E("sout").innerHTML=`<div class="card"><div class="chd"><span class="cht">Generated summary</span></div><div class="cbdy"><div class="stxt">${(d.summary||"").replace(/\*\*(.*?)\*\*/g,"<strong>$1</strong>").replace(/\n/g,"<br>")}</div></div></div>`+(d.cosine&&d.validation?scoreCard(d.cosine,d.validation,"—",d.sources||[]):"");
  }catch(e){ E("serr").innerHTML=`<div class="err">${esc(e.message)}</div>`; E("sout").innerHTML=""; }
}

// ── IK SEARCH ─────────────────────────────────────────────────────
async function doSearch(){
  const q=E("iq").value.trim(), dt=E("idt").value, out=E("ikout");
  if(!q) return;
  out.innerHTML=`<div class="sw"><div class="sp"></div>Searching Indian Kanoon...</div>`;
  try{
    const r=await fetch("/api/search",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({query:q,doctype:dt,top_k:8})});
    const d=await r.json(); if(!r.ok) throw new Error(d.detail||"Search failed");
    const docs=d.results||[];
    if(!docs.length){ out.innerHTML=`<p style="color:var(--mu);padding:1rem;font-size:13px;">No results. Try different keywords.</p>`; return; }
    out.innerHTML=`<p style="font-size:12px;color:var(--mu);margin-bottom:8px;">${docs.length} results — click to load into Summarise</p>`+
      docs.map(d=>`<div class="ik-item">
        <div style="font-weight:600;font-size:.88rem;margin-bottom:4px;">${esc(d.title)}</div>
        <div style="display:flex;gap:4px;flex-wrap:wrap;margin-bottom:5px;">${[d.court,d.date,d.citation].filter(Boolean).map(x=>`<span class="bdg">${esc(x)}</span>`).join("")}</div>
        <div style="font-size:.78rem;color:var(--mu);line-height:1.5;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;">${esc(d.snippet)}</div>
        <div style="margin-top:7px;display:flex;gap:10px;align-items:center;flex-wrap:wrap;">
          <button onclick="loadDoc('${d.tid}','${esc(d.title).replace(/'/g,"&#39;")}')" style="font-size:11px;background:rgba(212,168,67,.1);border:1px solid rgba(212,168,67,.3);color:var(--ac);padding:3px 9px;border-radius:4px;cursor:pointer;font-family:inherit;">Use this document</button>
          <a href="${esc(d.url)}" target="_blank" style="font-size:11px;">View on IK ↗</a>
        </div>
      </div>`).join("");
  }catch(e){ out.innerHTML=`<div class="err">${esc(e.message)}</div>`; }
}

async function loadDoc(tid,title){
  document.querySelectorAll(".tab")[0].click();
  gTab("sum",document.querySelectorAll(".tab")[0]);
  E("doc").value=`Loading "${title}"...`;
  E("sout").innerHTML=`<div class="sw"><div class="sp"></div>Fetching full document from Indian Kanoon...</div>`;
  try{
    const r=await fetch("/api/doc/"+tid), d=await r.json();
    if(!r.ok) throw new Error(d.detail||"Fetch failed");
    E("doc").value=d.text; E("qry").value="";
    E("sout").innerHTML=`<div style="color:var(--gn);font-size:13px;padding:.6rem;">✓ Loaded "${esc(title)}" (${d.words.toLocaleString()} words). Click Generate Summary.</div>`;
  }catch(e){ E("doc").value=""; E("serr").innerHTML=`<div class="err">${esc(e.message)}</div>`; E("sout").innerHTML=""; }
}

// ── EVALUATE ──────────────────────────────────────────────────────
async function doEval(){
  const doc=E("ed").value.trim(), ref=E("er").value.trim();
  if(!doc||!ref){ E("eerr").innerHTML=`<div class="err">Provide both document and reference.</div>`; return; }
  E("eerr").innerHTML=""; E("ebtn").disabled=true;
  E("eout").innerHTML=`<div class="sw"><div class="sp"></div>Generating summary and computing ROUGE + cosine metrics...</div>`;
  try{
    const r=await fetch("/api/evaluate",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({document:doc,reference:ref})});
    const d=await r.json(); if(!r.ok) throw new Error(d.detail||"Evaluation failed");
    const rows=(d.models||[]).map((m,i)=>`<tr${i===0?' class="hi"':""}>
      <td class="mn">${esc(m.name)}</td><td>${m.rouge1.toFixed(4)}</td><td>${m.rouge2.toFixed(4)}</td>
      <td>${m.rougeL.toFixed(4)}</td><td>${m.cosine.toFixed(4)}</td><td>${m.words}</td></tr>`).join("");
    E("eout").innerHTML=`
      <div class="card"><div class="chd"><span class="cht">Evaluation results</span></div><div class="cbdy" style="overflow-x:auto;">
        <table class="tbl"><thead><tr><th>Model</th><th>ROUGE-1</th><th>ROUGE-2</th><th>ROUGE-L</th><th>Cosine</th><th>Words</th></tr></thead>
        <tbody>${rows}</tbody></table>
        <p style="font-size:11px;color:var(--mu);margin-top:8px;">Highlighted = our system. Higher is better.</p>
      </div></div>
      <div class="card"><div class="chd"><span class="cht">Our generated summary</span></div>
        <div class="cbdy"><div class="stxt">${(d.models?.[0]?.summary||"").replace(/\n/g,"<br>")}</div>
      </div></div>`;
  }catch(e){ E("eerr").innerHTML=`<div class="err">${esc(e.message)}</div>`; }
  E("ebtn").disabled=false;
}

// ── demos ─────────────────────────────────────────────────────────
function loadDemo(){
  E("doc").value=`IN THE SUPREME COURT OF INDIA — CIVIL APPEAL NO. 4521 OF 2022\n\nM/s ABC Construction Ltd. ...Appellant  VERSUS  State of Maharashtra & Ors. ...Respondents\n\nJUDGMENT\n\n1. This appeal challenges the High Court's dismissal of a writ petition against MSRDC's cancellation of a construction contract for a flyover bridge on NH-48 worth Rs. 47.62 crores (awarded 12.06.2018, due 11.12.2020).\n\n2. The appellant contends delays due to COVID-19, unprecedented rainfall, and raw material shortage constitute force majeure under Clause 19.\n\n3. The respondent cancelled on 18.01.2021 under Clause 23, forfeiting Rs. 4.76 crores performance security without issuing any show-cause notice.\n\nHELD: (1) Natural justice violated — no show-cause notice before forfeiture. (2) Force majeure applies to pandemic delays. (3) Forfeiture set aside. (4) Dispute referred to arbitration under Clause 27. Appeal allowed with costs.`;
  E("qry").value="What was the court's decision on forfeiture of security deposit?";
}
function loadEvalDemo(){
  E("ed").value="The Supreme Court held that cancellation without show-cause notice violated natural justice. COVID-19 constituted force majeure under Clause 19. Performance security forfeiture of Rs. 4.76 crores set aside. Parties directed to arbitration under Clause 27. Appeal allowed.";
  E("er").value="Court ruled for appellant. Natural justice violated — no notice before forfeiture. Pandemic qualifies as force majeure. Security deposit quashed. Arbitration ordered. Appeal allowed with costs.";
}
function clearAll(){ E("doc").value=""; E("qry").value=""; E("serr").innerHTML=""; E("sout").innerHTML=""; E("fname").textContent=""; }
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
def frontend():
    return HTML


# ═══════════════════════════════════════════════════════════════════
# STARTUP
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Also check env var
    if not ANTHROPIC_API_KEY:
        ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

    print("\n" + "="*55)
    print("  LexSum — Indian Legal NLP  (Web-Search Edition)")
    print("="*55)

    if not ANTHROPIC_API_KEY:
        print("\n  No API key found.")
        print("  Enter it now, or press Enter to skip and add it later in the file.\n")
        try:
            k = input("  Paste your Anthropic API key (sk-ant-...): ").strip()
        except (EOFError, KeyboardInterrupt):
            k = ""
        if k.startswith("sk-ant-"):
            ANTHROPIC_API_KEY = k
            print("  Key accepted.\n")
        elif k:
            print(f"  Key rejected — must start with 'sk-ant-' (you entered: {k[:15]}...)")
            print("  Edit ANTHROPIC_API_KEY = \"\" in this file and paste your key there.\n")
        else:
            print("  Skipped. Edit ANTHROPIC_API_KEY = \"\" in this file.\n")

    if ANTHROPIC_API_KEY:
        print(f"\n  OK  Key: {ANTHROPIC_API_KEY[:12]}...{ANTHROPIC_API_KEY[-4:]}")
        print(f"  OK  Model: {MODEL}")
        print(f"  OK  Web search tool: enabled (no extra key needed)")
    else:
        print("  WARNING: No key set — summarisation will not work")
        print("  Open lexsum_app.py, find ANTHROPIC_API_KEY = \"\"")
        print("  and paste your full key between the quotes, then restart.\n")

    print(f"\n  --> http://localhost:8000\n")
    print("="*55 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")