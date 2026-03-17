"""
data_pipeline/scraper.py
========================
Scrapes Indian government legal databases:
  1. Indian Kanoon (free API + web)
  2. Supreme Court of India (SCI eCourts)
  3. Indiacode.nic.in  (Central Acts)
  4. India Gazette PDFs

Each source returns a list of dicts:
    {
        "id":        unique string,
        "title":     document title,
        "text":      raw extracted text,
        "source":    "indiankanoon" | "sci" | "indiacode" | "gazette",
        "doc_type":  "judgment" | "act" | "notification" | "amendment",
        "date":      "YYYY-MM-DD" or "",
        "url":       original URL,
        "metadata":  {...extra fields}
    }
"""

import re
import time
import logging
from typing import Optional
import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# ─── Shared helpers ─────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121 Safari/537.36"
    )
}


def _get(url: str, params: dict = None, retries: int = 3) -> requests.Response:
    """Robust GET with exponential back-off."""
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=20)
            r.raise_for_status()
            return r
        except Exception as exc:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            logger.warning(f"Retry {attempt+1} for {url} — {exc} — waiting {wait}s")
            time.sleep(wait)


def _clean(text: str) -> str:
    """Basic whitespace normalisation."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# ─── 1. Indian Kanoon ───────────────────────────────────────────────────────

class IndianKanoonScraper:
    """
    Indian Kanoon exposes a free search API (no key needed for small volume).
    Endpoint: https://api.indiankanoon.org/search/?formInput=<query>&pagenum=<n>
    
    Each result has a docid. We then fetch full text from:
    https://api.indiankanoon.org/doc/<docid>/
    """

    BASE_SEARCH = "https://api.indiankanoon.org/search/"
    BASE_DOC    = "https://api.indiankanoon.org/doc/"

    def search(self, query: str, max_docs: int = 50) -> list[dict]:
        """
        query     : e.g. 'contract breach Supreme Court India 2022'
        max_docs  : how many documents to collect
        Returns   : list of document dicts
        """
        results = []
        page = 0

        while len(results) < max_docs:
            params = {"formInput": query, "pagenum": page}
            logger.info(f"IK search page {page} for '{query}'")
            data = _get(self.BASE_SEARCH, params=params).json()

            docs = data.get("docs", [])
            if not docs:
                break

            for doc in docs:
                if len(results) >= max_docs:
                    break
                docid = doc.get("tid")
                if not docid:
                    continue

                # Fetch full document text
                try:
                    full = _get(f"{self.BASE_DOC}{docid}/").json()
                    text = BeautifulSoup(
                        full.get("doc", ""), "html.parser"
                    ).get_text(separator="\n")

                    results.append({
                        "id":       f"ik_{docid}",
                        "title":    full.get("title", doc.get("title", "")),
                        "text":     _clean(text),
                        "source":   "indiankanoon",
                        "doc_type": "judgment",
                        "date":     full.get("publishdate", ""),
                        "url":      f"https://indiankanoon.org/doc/{docid}/",
                        "metadata": {
                            "court":    full.get("court", ""),
                            "citation": full.get("citation", ""),
                        },
                    })
                    time.sleep(0.5)          # respect rate limits
                except Exception as exc:
                    logger.error(f"Failed doc {docid}: {exc}")

            page += 1

        return results


# ─── 2. Indiacode — Central Acts ────────────────────────────────────────────

class IndiaCodeScraper:
    """
    indiacode.nic.in provides PDFs of Central Acts.
    We hit the acts listing page, grab PDF links, then extract text.
    
    Note: A full scrape of all acts is large. Use act_titles filter to be selective.
    """

    BASE = "https://www.indiacode.nic.in"
    LIST = "https://www.indiacode.nic.in/handle/123456789/1362/simple-search"

    def fetch_acts(
        self,
        act_titles: Optional[list[str]] = None,
        max_acts: int = 20,
    ) -> list[dict]:
        """
        act_titles : optional filter list e.g. ['Contract Act', 'RERA', 'IPC']
        max_acts   : cap on how many acts to retrieve
        """
        results = []
        page = 0

        while len(results) < max_acts:
            params = {
                "query": "act",
                "rpp":   20,
                "start": page * 20,
                "sort_by": "score",
                "order":   "desc",
            }
            logger.info(f"IndiaCode page {page}")
            r = _get(self.LIST, params=params)
            soup = BeautifulSoup(r.text, "lxml")

            # Each act is an <a> inside .artifact-title
            links = soup.select(".artifact-title a")
            if not links:
                break

            for link in links:
                if len(results) >= max_acts:
                    break

                title = link.get_text(strip=True)
                if act_titles and not any(
                    kw.lower() in title.lower() for kw in act_titles
                ):
                    continue

                act_url = self.BASE + link["href"]
                try:
                    pdf_text = self._extract_act(act_url, title)
                    if pdf_text:
                        results.append({
                            "id":       f"ic_{re.sub(r'[^a-z0-9]', '_', title.lower()[:40])}",
                            "title":    title,
                            "text":     _clean(pdf_text),
                            "source":   "indiacode",
                            "doc_type": "act",
                            "date":     "",
                            "url":      act_url,
                            "metadata": {},
                        })
                except Exception as exc:
                    logger.error(f"Act '{title}' failed: {exc}")

            page += 1

        return results

    def _extract_act(self, act_url: str, title: str) -> str:
        """Download act page, find PDF link, extract text."""
        r = _get(act_url)
        soup = BeautifulSoup(r.text, "lxml")

        # Look for a PDF download link
        pdf_link = soup.find("a", href=lambda h: h and h.endswith(".pdf"))
        if not pdf_link:
            # Fall back to plain-text rendering
            return soup.get_text(separator="\n")

        pdf_url = pdf_link["href"]
        if not pdf_url.startswith("http"):
            pdf_url = self.BASE + pdf_url

        pdf_bytes = _get(pdf_url).content
        return self._pdf_to_text(pdf_bytes)

    @staticmethod
    def _pdf_to_text(pdf_bytes: bytes) -> str:
        """PyMuPDF-based PDF→text. Handles multi-column layouts better."""
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = [page.get_text("text") for page in doc]
        return "\n\n".join(pages)


# ─── 3. Gazette of India (sample) ───────────────────────────────────────────

class GazetteScraper:
    """
    egazette.nic.in lists notifications as PDFs.
    This scraper fetches recent extraordinary gazette entries.
    """

    LIST_URL = "https://egazette.nic.in/WriteReadData/2024"

    def fetch_recent(self, max_pdfs: int = 10) -> list[dict]:
        """
        Fetches PDF listings from the gazette server.
        NOTE: The gazette site structure changes frequently.
        If the listing page 404s, fall back to direct PDF URLs you know.
        """
        results = []
        # In production you would parse the gazette listing page.
        # Here we demonstrate with a known URL pattern.
        sample_pdfs = [
            "https://egazette.nic.in/WriteReadData/2024/247001.pdf",
        ]

        for url in sample_pdfs[:max_pdfs]:
            try:
                logger.info(f"Gazette PDF: {url}")
                pdf_bytes = _get(url).content
                text = IndiaCodeScraper._pdf_to_text(pdf_bytes)
                fname = url.split("/")[-1].replace(".pdf", "")
                results.append({
                    "id":       f"gz_{fname}",
                    "title":    f"Gazette Notification {fname}",
                    "text":     _clean(text),
                    "source":   "gazette",
                    "doc_type": "notification",
                    "date":     "",
                    "url":      url,
                    "metadata": {},
                })
            except Exception as exc:
                logger.error(f"Gazette {url}: {exc}")

        return results


# ─── Unified entry point ────────────────────────────────────────────────────

def collect_corpus(
    ik_queries: list[str] = None,
    act_titles: list[str] = None,
    max_per_source: int = 30,
) -> list[dict]:
    """
    Runs all scrapers and returns a combined corpus.

    Example
    -------
    corpus = collect_corpus(
        ik_queries=["Supreme Court contract 2022", "RERA judgment"],
        act_titles=["Indian Contract Act", "RERA", "IPC"],
        max_per_source=30,
    )
    """
    corpus = []

    if ik_queries:
        scraper = IndianKanoonScraper()
        for q in ik_queries:
            docs = scraper.search(q, max_docs=max_per_source)
            corpus.extend(docs)
            logger.info(f"IndianKanoon '{q}' → {len(docs)} docs")

    if act_titles is not None:
        ic = IndiaCodeScraper()
        acts = ic.fetch_acts(act_titles=act_titles, max_acts=max_per_source)
        corpus.extend(acts)
        logger.info(f"IndiaCode → {len(acts)} acts")

    logger.info(f"Total corpus size: {len(corpus)} documents")
    return corpus
