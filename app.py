import os

import streamlit as st

import rag

st.set_page_config(page_title="AI PDF Chatbot", page_icon="📄", layout="wide")
st.title("📄 AI-Powered PDF Chatbot")
st.caption("Upload PDFs, ask questions, get summaries and quizzes (Gemini + RAG)")

for key, default in {"store": None, "pages": {}, "history": [], "quiz": None, "summary": None}.items():
    st.session_state.setdefault(key, default)

# ---------- Sidebar: upload ----------
with st.sidebar:
    st.header("Gemini API key")
    key_in = st.text_input("Paste key (or set it in .env)", type="password", placeholder="AIza...")
    if key_in.strip():
        os.environ["GEMINI_API_KEY"] = key_in.strip()
    st.header("1. Upload")
    uploads = st.file_uploader("PDF files", type="pdf", accept_multiple_files=True)
    if st.button("Process documents", type="primary", disabled=not uploads):
        try:
            with st.spinner("Extracting (OCR is used for scanned PDFs, may take a minute), chunking and embedding..."):
                files = {u.name: u.getvalue() for u in uploads}
                store, pages = rag.build_store(files)
            if store.matrix is None:
                st.error("No readable text found in these PDFs.")
            else:
                st.session_state.update(store=store, pages=pages, history=[], quiz=None, summary=None)
                st.success(f"Indexed {len(store.chunks)} chunks from {len(files)} file(s).")
        except Exception as e:
            st.error(f"Processing failed: {e}")
    top_k = st.slider("Passages to retrieve", 2, 10, 5)

if not st.session_state.store:
    st.info("Upload a PDF in the sidebar and click **Process documents** to begin.")
    st.stop()

chat_tab, sum_tab, quiz_tab = st.tabs(["💬 Chat", "📝 Summary", "❓ Quiz"])

# ---------- Chat ----------
with chat_tab:
    for role, text, sources in st.session_state.history:
        with st.chat_message(role):
            st.markdown(text)
            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.markdown(f"**{s['source']} - page {s['page']}** (score {s['score']:.2f})")
                        st.caption(s["text"][:300] + "...")
    if q := st.chat_input("Ask a question about your documents"):
        hist = [(r, t) for r, t, _ in st.session_state.history]
        st.session_state.history.append(("user", q, None))
        try:
            ans, srcs = rag.answer_question(q, st.session_state.store, hist, k=top_k)
        except Exception as e:
            ans, srcs = f"Error: {e}", []
        st.session_state.history.append(("assistant", ans, srcs))
        st.rerun()

# ---------- Summary ----------
with sum_tab:
    style = st.selectbox("Style", ["concise", "detailed", "executive (very short)"])
    if st.button("Generate summary"):
        try:
            with st.spinner("Summarizing..."):
                st.session_state.summary = rag.summarize(st.session_state.pages, style)
        except Exception as e:
            st.error(f"Failed: {e}")
    if st.session_state.summary:
        st.markdown(st.session_state.summary)

# ---------- Quiz ----------
with quiz_tab:
    c1, c2 = st.columns(2)
    n_mcq = c1.number_input("MCQs", 1, 15, 5)
    n_short = c2.number_input("Short-answer", 0, 10, 3)
    if st.button("Generate quiz"):
        try:
            with st.spinner("Writing questions..."):
                st.session_state.quiz = rag.generate_quiz(st.session_state.pages, n_mcq, n_short)
        except Exception as e:
            st.error(f"Failed: {e}")
    quiz = st.session_state.quiz
    if quiz:
        for i, m in enumerate(quiz.get("mcq", []), 1):
            st.markdown(f"**Q{i}. {m['question']}**")
            choice = st.radio("Select", m["options"], index=None, key=f"mcq{i}", label_visibility="collapsed")
            if choice is not None:
                if m["options"].index(choice) == m["answer_index"]:
                    st.success("Correct! " + m.get("explanation", ""))
                else:
                    st.error(f"Incorrect. Answer: {m['options'][m['answer_index']]}. " + m.get("explanation", ""))
        for i, s in enumerate(quiz.get("short", []), 1):
            st.markdown(f"**Short Q{i}. {s['question']}**")
            with st.expander("Show answer"):
                st.write(s["answer"])