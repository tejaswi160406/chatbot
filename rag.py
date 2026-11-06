"""Core RAG logic: extraction, chunking, embeddings, retrieval, generation."""
import hashlib
import io
import json
import os
import re
import time

from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv(override=True)  # .env wins over any stale system variable

EMBED_MODEL = "gemini-embedding-001"
GEN_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")  # free tier: ~500 requests/day
CACHE_DIR = Path(__file__).parent / ".cache"


RETRY_CODES = (429, 500, 503, 504)


def _retry(fn, tries: int = 5):
    """Retry temporary API errors (overload / rate limit) with backoff."""
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:
            code = getattr(e, "code", None)
            msg = str(e).lower()
            if code == 429 and ("per day" in msg or "perday" in msg):
                raise RuntimeError(
                    f"Daily free quota used up for model '{GEN_MODEL}'. Pick another model "
                    "in the sidebar, or wait until the quota resets (midnight Pacific time)."
                ) from e
            if code not in RETRY_CODES or attempt == tries - 1:
                raise
            time.sleep(min(2 ** (attempt + 1), 20))


def get_client():
    key = (os.getenv("GEMINI_API_KEY") or "").strip().strip("\"'")
    if not key or key == "your_key_here":
        raise RuntimeError("GEMINI_API_KEY not set. Put your real key in the .env file, then restart Streamlit.")
    from google import genai
    return genai.Client(api_key=key)


# ---------- Extraction & chunking ----------
def extract_pages(file_bytes: bytes) -> list[str]:
    """Return a list with the text of each page."""
    reader = PdfReader(io.BytesIO(file_bytes))
    return [(p.extract_text() or "").strip() for p in reader.pages]


def ocr_pdf(file_bytes: bytes, client=None) -> list[str]:
    """OCR fallback for scanned/image-only PDFs using Gemini vision."""
    from google.genai import types
    client = client or get_client()
    prompt = ("Transcribe ALL text in this PDF exactly as written. Before each page's text, "
              "write a line '=====PAGE n=====' (n starts at 1). Output only the transcription.")
    res = _retry(lambda: client.models.generate_content(
        model=GEN_MODEL,
        contents=[types.Part.from_bytes(data=file_bytes, mime_type="application/pdf"), prompt],
    ))
    parts = re.split(r"=====PAGE\s+(\d+)=====", res.text or "")
    pages = {int(parts[i]): parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}
    if not pages:
        return [(res.text or "").strip()]
    return [pages.get(n, "") for n in range(1, max(pages) + 1)]


def chunk_pages(pages: list[str], source: str, size: int = 1000, overlap: int = 200):
    """Split each page into overlapping chunks, keeping page metadata."""
    chunks = []
    step = max(size - overlap, 1)
    for page_no, text in enumerate(pages, start=1):
        text = " ".join(text.split())
        for start in range(0, len(text), step):
            piece = text[start:start + size]
            if piece.strip():
                chunks.append({"text": piece, "page": page_no, "source": source})
            if start + size >= len(text):
                break
    return chunks


# ---------- Embeddings & vector store ----------
def embed(texts: list[str], task_type: str, client=None) -> np.ndarray:
    from google.genai import types
    client = client or get_client()
    vectors = []
    for i in range(0, len(texts), 100):  # API batch limit
        batch = texts[i:i + 100]
        res = _retry(lambda: client.models.embed_content(
            model=EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type=task_type),
        ))
        vectors.extend(e.values for e in res.embeddings)
    return np.array(vectors, dtype=np.float32)


class VectorStore:
    """In-memory vector store using cosine similarity."""

    def __init__(self):
        self.chunks: list[dict] = []
        self.matrix: np.ndarray | None = None

    def add(self, chunks: list[dict], vectors: np.ndarray):
        vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10)
        self.chunks.extend(chunks)
        self.matrix = vectors if self.matrix is None else np.vstack([self.matrix, vectors])

    def search(self, query_vec: np.ndarray, k: int = 5):
        if self.matrix is None:
            return []
        q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        scores = self.matrix @ q
        top = np.argsort(scores)[::-1][:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]


def load_pages(data: bytes, client=None) -> list[str]:
    """Extract text (OCR if scanned). Results are cached on disk to save API calls."""
    path = CACHE_DIR / f"{hashlib.sha256(data).hexdigest()}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    pages = extract_pages(data)
    if not any(pages):  # scanned PDF -> OCR with Gemini
        pages = ocr_pdf(data, client)
    if any(pages):
        CACHE_DIR.mkdir(exist_ok=True)
        path.write_text(json.dumps(pages), encoding="utf-8")
    return pages


def build_store(files: dict[str, bytes], client=None):
    """files: {filename: bytes}. Returns (store, pages_by_file)."""
    store, all_pages = VectorStore(), {}
    for name, data in files.items():
        pages = load_pages(data, client)
        all_pages[name] = pages
        chunks = chunk_pages(pages, name)
        if chunks:
            store.add(chunks, embed([c["text"] for c in chunks], "RETRIEVAL_DOCUMENT", client))
    return store, all_pages


# ---------- Generation ----------
def _generate(prompt: str, client=None, json_mode: bool = False) -> str:
    from google.genai import types
    client = client or get_client()
    cfg = types.GenerateContentConfig(
        response_mime_type="application/json" if json_mode else "text/plain"
    )
    return _retry(lambda: client.models.generate_content(
        model=GEN_MODEL, contents=prompt, config=cfg)).text


def answer_question(question: str, store: VectorStore, history=None, k: int = 5, client=None):
    """Return (answer, sources) grounded in retrieved chunks."""
    client = client or get_client()
    qvec = embed([question], "RETRIEVAL_QUERY", client)[0]
    hits = store.search(qvec, k)
    if not hits:
        return "Please upload and process a PDF first.", []
    context = "\n\n".join(f"[{h['source']} p.{h['page']}]\n{h['text']}" for h in hits)
    convo = "\n".join(f"{r}: {t}" for r, t in (history or [])[-6:])
    prompt = f"""You are a document assistant. Answer ONLY using the context below.
If the answer is not in the context, say you could not find it in the document.
Mention page numbers like (p.3) when relevant.

Conversation so far:
{convo}

Context:
{context}

Question: {question}
Answer:"""
    return _generate(prompt, client), hits


def _document_text(pages_by_file: dict, limit: int = 60000) -> str:
    parts = [f"[{n} p.{i}]\n{t}" for n, pages in pages_by_file.items()
             for i, t in enumerate(pages, 1) if t]
    return "\n\n".join(parts)[:limit]


def summarize(pages_by_file: dict, style: str = "concise", client=None) -> str:
    text = _document_text(pages_by_file)
    if not text:
        return "No extractable text found (scanned PDFs need OCR)."
    prompt = (f"Write a {style} summary of the document below. Use short paragraphs "
              f"and a bullet list of key points.\n\n{text}")
    return _generate(prompt, client)


def generate_quiz(pages_by_file: dict, n_mcq: int = 5, n_short: int = 3, client=None) -> dict:
    text = _document_text(pages_by_file)
    if not text:
        return {"mcq": [], "short": []}
    prompt = f"""Create a quiz from the document below. Return ONLY JSON:
{{"mcq":[{{"question":"","options":["A","B","C","D"],"answer_index":0,"explanation":""}}],
"short":[{{"question":"","answer":""}}]}}
Exactly {n_mcq} mcq and {n_short} short questions. Base them strictly on the document.

{text}"""
    raw = _generate(prompt, client, json_mode=True)
    return json.loads(raw.replace("```json", "").replace("```", "").strip())