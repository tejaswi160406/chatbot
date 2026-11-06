"""Offline tests (no API calls). Run: pytest -v"""
import numpy as np
import pytest

import rag


def test_chunking_keeps_page_numbers():
    chunks = rag.chunk_pages(["a " * 800, "short page"], "doc.pdf", size=500, overlap=100)
    assert {c["page"] for c in chunks} == {1, 2}
    assert all(len(c["text"]) <= 500 for c in chunks)
    assert all(c["source"] == "doc.pdf" for c in chunks)


def test_chunking_skips_empty_pages():
    assert rag.chunk_pages(["", "   "], "x.pdf") == []


def test_chunk_overlap():
    text = "".join(chr(97 + i % 26) for i in range(1200))
    c = rag.chunk_pages([text], "x.pdf", size=500, overlap=100)
    assert c[0]["text"][-100:] == c[1]["text"][:100]


def test_vector_search_ranks_closest_first():
    store = rag.VectorStore()
    chunks = [{"text": t, "page": i, "source": "d"} for i, t in enumerate("abc", 1)]
    store.add(chunks, np.array([[1, 0], [0, 1], [0.9, 0.1]], dtype=np.float32))
    hits = store.search(np.array([1, 0], dtype=np.float32), k=2)
    assert [h["text"] for h in hits] == ["a", "c"]


def test_empty_store_returns_nothing():
    assert rag.VectorStore().search(np.array([1.0, 0.0]), 3) == []


def test_summarize_handles_no_text():
    assert "No extractable text" in rag.summarize({"a.pdf": ["", ""]})


def test_quiz_handles_no_text():
    assert rag.generate_quiz({"a.pdf": [""]}) == {"mcq": [], "short": []}


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        rag.get_client()


def test_ocr_parses_page_markers():
    pytest.importorskip("google.genai")

    class R: text = "=====PAGE 1=====\nHello\n=====PAGE 2=====\nWorld"
    class M:
        def generate_content(self, **kw): return R()
    class C: models = M()
    assert rag.ocr_pdf(b"%PDF", C()) == ["Hello", "World"]


def test_retry_recovers_from_503(monkeypatch):
    monkeypatch.setattr(rag.time, "sleep", lambda s: None)
    calls = {"n": 0}

    class Overloaded(Exception):
        code = 503

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise Overloaded()
        return "ok"

    assert rag._retry(flaky) == "ok" and calls["n"] == 3


def test_retry_does_not_hide_other_errors():
    class BadKey(Exception):
        code = 400

    def bad():
        raise BadKey()

    with pytest.raises(BadKey):
        rag._retry(bad)


def test_daily_quota_not_retried(monkeypatch):
    monkeypatch.setattr(rag.time, "sleep", lambda s: None)
    calls = {"n": 0}

    class Quota(Exception):
        code = 429
        def __str__(self): return "Generate content requests per day limit exceeded"

    def f():
        calls["n"] += 1
        raise Quota()

    with pytest.raises(RuntimeError, match="Daily free quota"):
        rag._retry(f)
    assert calls["n"] == 1


def test_pages_are_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "CACHE_DIR", tmp_path)
    n = {"c": 0}
    def fake_extract(b):
        n["c"] += 1
        return ["hello"]
    monkeypatch.setattr(rag, "extract_pages", fake_extract)
    assert rag.load_pages(b"pdf") == ["hello"]
    assert rag.load_pages(b"pdf") == ["hello"] and n["c"] == 1