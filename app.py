# app.py — RAG Knowledge Explorer (LangChain + OpenAI Embeddings + FAISS)

import io
import os
from dataclasses import dataclass
from typing import List, Dict, Any

import streamlit as st
import pandas as pd

# -------- Optional raw parsers (we'll keep them to avoid extra loader deps) --------
try:
    import PyPDF2
except Exception:
    PyPDF2 = None

try:
    import docx  # python-docx
except Exception:
    docx = None

# -------- LangChain imports with version-safe fallbacks --------
# Text splitter
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # >=0.0.1
except Exception:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter  # older
    except Exception:
        RecursiveCharacterTextSplitter = None

# OpenAI LLM + embeddings
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # modern packages
except Exception:
    try:
        # very old fallback (not recommended, but keeps code running)
        from langchain.embeddings.openai import OpenAIEmbeddings  # type: ignore
        from langchain.chat_models import ChatOpenAI  # type: ignore
    except Exception:
        ChatOpenAI = None
        OpenAIEmbeddings = None

# Vector store FAISS
try:
    from langchain_community.vectorstores import FAISS  # modern community package
except Exception:
    try:
        from langchain.vectorstores import FAISS  # very old fallback
    except Exception:
        FAISS = None


# ===================== Page & State =====================
st.set_page_config(page_title="RAG Knowledge Explorer", page_icon="📚", layout="wide")
st.title("📚 RAG Knowledge Explorer")
st.caption("Upload documents → Ask a question → Get answers with citations. (Powered by LangChain + OpenAI + FAISS)")

@dataclass
class Chunk:
    doc_name: str
    chunk_id: int
    text: str

def init_state():
    ss = st.session_state
    ss.setdefault("uploader_nonce", 0)
    ss.setdefault("docs", {})          # {doc_name: {"text": str, "meta": {...}}}
    ss.setdefault("chunks", [])        # List[Chunk]
    ss.setdefault("history", [])       # [{q, answer, citations: [{doc, chunk_id, preview}], usage}]
    ss.setdefault("last_query", "")
    ss.setdefault("index_ready", False)
    ss.setdefault("vs", None)          # FAISS index
    ss.setdefault("emb_model_name", "text-embedding-3-small")  # default
    ss.setdefault("llm_model_name", "gpt-4o-mini")             # default
    ss.setdefault("retrieval_k", 4)

init_state()

# ===================== Sidebar: Upload & Settings =====================
st.sidebar.header("Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF / DOCX / TXT (multiple allowed)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.uploader_nonce}",
)

if uploaded_files:
    for f in uploaded_files:
        name = f.name
        if name in st.session_state.docs:
            continue  # idempotent by filename this session
        try:
            raw = f.read()
            ext = os.path.splitext(name)[1].lower()
            if ext == ".pdf":
                if PyPDF2 is None:
                    st.sidebar.error("PyPDF2 not installed. Add `PyPDF2` to requirements.txt.")
                    continue
                reader = PyPDF2.PdfReader(io.BytesIO(raw))
                pages = []
                for i, page in enumerate(reader.pages):
                    try:
                        pages.append(page.extract_text() or "")
                    except Exception:
                        pages.append("")
                text = "\n".join(pages)
            elif ext == ".docx":
                if docx is None:
                    st.sidebar.error("python-docx not installed. Add `python-docx` to requirements.txt.")
                    continue
                d = docx.Document(io.BytesIO(raw))
                text = "\n".join([p.text for p in d.paragraphs])
            else:  # .txt
                text = raw.decode("utf-8", errors="ignore")

            st.session_state.docs[name] = {"text": text, "meta": {"bytes": len(raw)}}
            st.session_state.index_ready = False  # need rebuild

        except Exception as e:
            st.sidebar.error(f"Failed to read {name}: {e}")

st.sidebar.markdown("---")
st.sidebar.subheader("RAG Settings")

chunk_size = st.sidebar.slider("Chunk size (chars)", 300, 2000, 800, 50)
chunk_overlap = st.sidebar.slider("Chunk overlap (chars)", 0, 400, 120, 10)
st.session_state.retrieval_k = st.sidebar.slider("Top-K chunks", 2, 10, st.session_state.retrieval_k, 1)

st.sidebar.markdown("---")

# Reset with confirmation
if "confirm_reset" not in st.session_state:
    st.session_state.confirm_reset = False

if not st.session_state.confirm_reset:
    if st.sidebar.button("🔄 Reset App"):
        st.session_state.confirm_reset = True
        st.rerun()
else:
    st.sidebar.warning("This will clear uploaded docs, vector index, and history.")
    c1, c2 = st.sidebar.columns(2)
    if c1.button("✅ Yes"):
        st.session_state.docs.clear()
        st.session_state.chunks.clear()
        st.session_state.history.clear()
        st.session_state.last_query = ""
        st.session_state.index_ready = False
        st.session_state.vs = None
        st.session_state.uploader_nonce += 1
        st.session_state.confirm_reset = False
        st.rerun()
    if c2.button("❌ No"):
        st.session_state.confirm_reset = False
        st.rerun()

# ===================== Document Table & Chunking =====================
st.subheader("📂 Uploaded documents")
if not st.session_state.docs:
    st.info("Upload one or more documents in the sidebar to begin.")
else:
    df_docs = pd.DataFrame(
        [{"name": k, "size (bytes)": v["meta"]["bytes"], "chars": len(v["text"])} for k, v in st.session_state.docs.items()]
    )
    st.dataframe(df_docs, use_container_width=True, height=180)

    # Build chunks with LangChain splitter (if available), else simple char splitter
    def split_into_chunks(text: str) -> List[str]:
        if RecursiveCharacterTextSplitter is None:
            # fallback: simple char chunks
            chunks = []
            t = text.strip().replace("\r", "")
            start = 0
            n = len(t)
            while start < n:
                end = min(start + chunk_size, n)
                chunk = t[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                if end == n:
                    break
                start = max(0, end - chunk_overlap)
            return chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", " ", ""]
        )
        return splitter.split_text(text or "")

    chunks: List[Chunk] = []
    for doc_name, payload in st.session_state.docs.items():
        raw = payload["text"]
        for i, ch in enumerate(split_into_chunks(raw)):
            if ch.strip():
                chunks.append(Chunk(doc_name=doc_name, chunk_id=i, text=ch))
    st.session_state.chunks = chunks

    with st.expander("🔎 Indexed chunks (preview)"):
        st.caption(f"Total chunks: {len(chunks)}")
        preview = [
            {"doc": c.doc_name, "chunk_id": c.chunk_id, "snippet": (c.text[:140] + "…") if len(c.text) > 150 else c.text}
            for c in chunks[:50]
        ]
        st.dataframe(pd.DataFrame(preview), use_container_width=True, height=220)

# ===================== Build / Rebuild Vector Index =====================
def build_faiss_index(chunks: List[Chunk]):
    if not chunks:
        return None
    if OpenAIEmbeddings is None or FAISS is None:
        st.error("Missing LangChain packages: `langchain-openai` and `langchain-community` (for FAISS).")
        return None

    # OPENAI_API_KEY
    OPENAI_KEY = st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None
    if not OPENAI_KEY:
        OPENAI_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_KEY:
        st.error("OPENAI_API_KEY not configured. Add it in .streamlit/secrets.toml or as an environment variable.")
        return None
    os.environ["OPENAI_API_KEY"] = OPENAI_KEY

    texts = [c.text for c in chunks]
    metadatas = [{"doc": c.doc_name, "chunk_id": c.chunk_id} for c in chunks]

    with st.spinner("Building FAISS index (embeddings)…"):
        emb = OpenAIEmbeddings(model=st.session_state.emb_model_name)
        vs = FAISS.from_texts(texts=texts, embedding=emb, metadatas=metadatas)
    return vs

if st.session_state.docs and not st.session_state.index_ready:
    st.session_state.vs = build_faiss_index(st.session_state.chunks)
    st.session_state.index_ready = st.session_state.vs is not None

# ===================== Q&A with real retrieval =====================
st.subheader("❓ Ask a question")
query = st.text_input(
    "Ask about the uploaded documents (e.g., 'What is the refund policy?')",
    key="q_input",
    value=st.session_state.last_query or "",
)

c_run, c_info = st.columns([1, 4])
with c_run:
    run = st.button("Ask")

with c_info:
    st.caption("Uses OpenAI embeddings + FAISS retrieval + GPT answer with citations.")

def retrieve(vs, q: str, k: int) -> List[Dict[str, Any]]:
    if vs is None or not q.strip():
        return []
    docs = vs.similarity_search(q, k=k)  # returns LangChain Documents with .page_content & .metadata
    out = []
    for d in docs:
        meta = d.metadata or {}
        out.append({
            "text": d.page_content,
            "doc": meta.get("doc", "unknown"),
            "chunk_id": meta.get("chunk_id", -1),
        })
    return out

SYSTEM_PROMPT = """You are a helpful assistant. Answer the user's question using ONLY the provided context.
If the answer is not in the context, say you don't know. Keep answers concise and cite sources as [1], [2], etc.
"""

def answer_with_citations(context_items: List[Dict[str, Any]], question: str) -> Dict[str, Any]:
    if ChatOpenAI is None:
        return {"text": "LLM backend not available (langchain-openai missing).", "usage": {}, "citations": []}

    OPENAI_KEY = os.getenv("OPENAI_API_KEY") or (st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None)
    if not OPENAI_KEY:
        return {"text": "OPENAI_API_KEY not configured.", "usage": {}, "citations": []}

    # Build context string with numeric markers
    ctx_lines = []
    citations = []
    for i, item in enumerate(context_items, start=1):
        ctx_lines.append(f"[{i}] {item['text']}")
        citations.append({"idx": i, "doc": item["doc"], "chunk_id": item["chunk_id"], "preview": item["text"][:200] + ("…" if len(item["text"]) > 200 else "")})

    context_block = "\n\n".join(ctx_lines) if ctx_lines else "No context."

    llm = ChatOpenAI(model=st.session_state.llm_model_name, temperature=0)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question:\n{question}\n\nContext:\n{context_block}\n\nAnswer with citations like [1], [2]."}
    ]

    try:
        with st.spinner("Thinking…"):
            resp = llm.invoke(messages)  # LC 0.2 style; returns BaseMessage
        txt = getattr(resp, "content", str(resp))
        usage = getattr(resp, "response_metadata", {}).get("token_usage", {})
        return {"text": txt, "usage": usage, "citations": citations}
    except Exception as e:
        return {"text": f"LLM error: {e}", "usage": {}, "citations": citations}

if run:
    if not st.session_state.docs:
        st.warning("Please upload at least one document first.")
    elif not st.session_state.index_ready:
        st.warning("Index not ready yet—try again in a moment.")
    else:
        st.session_state.last_query = query
        top = retrieve(st.session_state.vs, query, k=st.session_state.retrieval_k)
        result = answer_with_citations(top, query)
        st.session_state.history.append({
            "q": query,
            "answer": result["text"],
            "citations": [{"doc": c["doc"], "chunk_id": c["chunk_id"], "preview": c["preview"]} for c in result["citations"]],
            "usage": result.get("usage", {}),
        })
        st.success("Answer generated. See below for response & citations.")

# ===================== Results =====================
if st.session_state.history:
    latest = st.session_state.history[-1]
    st.subheader("🧠 Answer")
    st.text_area("Response", value=latest["answer"], height=240)

    with st.expander("📎 Citations (sources)"):
        for i, c in enumerate(latest["citations"], start=1):
            st.markdown(f"**[{i}] {c['doc']} — chunk {c['chunk_id']}**")
            st.write(c["preview"])
            st.markdown("---")

# ===================== History =====================
st.subheader("🗂️ Query history")
if not st.session_state.history:
    st.caption("No queries yet.")
else:
    hist_df = pd.DataFrame([
        {"#": i+1, "Question": h["q"], "Answer (first 80 chars)": (h["answer"][:80] + "…")}
        for i, h in enumerate(st.session_state.history[::-1])
    ])
    st.dataframe(hist_df, use_container_width=True, height=220)

    options = [h["q"] for h in st.session_state.history[::-1]]
    c1, c2 = st.columns([2, 1])
    with c1:
        pick = st.selectbox("Re-run a question", options=options)
    with c2:
        if st.button("▶️ Re-run"):
            if not st.session_state.index_ready:
                st.warning("Index not ready yet—try again.")
            else:
                top = retrieve(st.session_state.vs, pick, k=st.session_state.retrieval_k)
                result = answer_with_citations(top, pick)
                st.session_state.history.append({
                    "q": pick,
                    "answer": result["text"],
                    "citations": [{"doc": c["doc"], "chunk_id": c["chunk_id"], "preview": c["preview"]} for c in result["citations"]],
                    "usage": result.get("usage", {}),
                })
                st.success("Re-ran the question. Scroll up to view the latest answer.")

# ===================== Tips =====================
with st.expander("🛠️ Notes"):
    st.markdown("""
- Set your OpenAI key in `.streamlit/secrets.toml` as `OPENAI_API_KEY="sk-..."` (do **not** commit this file).
- Tweak the sidebar **chunk size/overlap** and **Top-K** to tune retrieval quality.
- For production, consider persistent stores (Chroma, SQLite-backed FAISS) and doc-level metadata (page numbers).
""")
