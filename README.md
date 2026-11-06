# AI-Powered PDF Chatbot (Gemini + RAG)

Upload PDFs, ask questions in natural language, and get answers grounded in the
document with page references. Also generates summaries and quizzes (MCQ + short answer).

## Architecture
PDF -> pypdf text extraction -> overlapping chunks (page metadata) -> Gemini embeddings
(`gemini-embedding-001`) -> in-memory vector store (NumPy cosine similarity) ->
top-k retrieval -> Gemini (`gemini-3.5-flash-lite`, selectable in the sidebar) answer with citations.

## Setup (Python 3.11+)
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then put your key in .env (get one at aistudio.google.com)
streamlit run app.py
```

## Tests
```bash
pytest -v      # offline tests, no API key needed
```

## Files
- `rag.py` - extraction, chunking, embeddings, vector store, Q&A, summary, quiz
- `app.py` - Streamlit UI (Chat, Summary, Quiz tabs)
- `tests/test_rag.py` - unit tests

## Limitations
- Scanned/image-only PDFs are read with Gemini vision OCR (slower, one extra API call per file).
- The vector index is in memory and rebuilt per session; extracted text is cached in `.cache/` (git-ignored).
- The free Gemini tier has daily request limits; temporary errors (429/503) are retried automatically.
- Very long documents are truncated to ~60k characters for summaries and quizzes.

Never commit `.env`. It is listed in `.gitignore`.

To use a different Gemini model, add `GEMINI_MODEL=model-name` to `.env`.