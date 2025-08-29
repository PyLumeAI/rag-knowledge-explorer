# 📚 RAG-powered Knowledge Explorer

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-red)](https://YOUR_STREAMLIT_URL)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Upload documents (PDF, TXT, DOCX) → Ask questions in natural language → Get answers with **citations**.  
Powered by **RAG (Retrieval Augmented Generation)** with embeddings + vector search.

---

## 🎥 Demo

![Demo](docs/images/rag-knowledge-explorer/demo.gif)

*Above: Uploading a PDF, asking “What is the refund policy?”, and getting an answer with source citations.*

---

## 🚀 Features

- 📂 **Multi-file upload** — PDFs, DOCX, TXT
- 🔖 **Text chunking** — smart split for better retrieval
- 🧠 **Embeddings + Vector DB** — FAISS/Chroma for similarity search
- 🤖 **LLM Q&A** — context-augmented answers with GPT
- 📎 **Citations** — shows which doc + page snippets answer came from
- 🗂 **Query history** — log past questions + answers
- 🔄 **Reset-safe UX** — clears docs and history with confirmation

---

## 🚀 Quickstart

```bash
git clone https://github.com/PyLumeAI/genai-data-explorer.git
cd genai-data-explorer
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```
## 🔑 Configure API Keys
```toml
OPENAI_API_KEY = "sk-..."
# Optional Postgres
PG_URI = "postgresql+psycopg2://user:pass@host:5432/dbname"
```
## 🚀 Run the App
```bash
streamlit run app.py
```

## 📂 Repository Structure
```
rag-knowledge-explorer/
│
├── app.py                  # Main Streamlit app
├── requirements.txt
├── runtime.txt
├── .streamlit/config.toml
│
├── core/
│   ├── loaders.py          # PDF, DOCX, TXT parsing
│   ├── chunker.py          # Text splitting
│   ├── embeddings.py       # Generate/store embeddings
│   ├── retriever.py        # Similarity search logic
│   └── qa_chain.py         # LLM Q&A with context
│
├── docs/images/rag-knowledge-explorer/
│   ├── cover.png
│   ├── demo.gif
│   ├── upload.png
│   ├── qa.png
│   └── history.png
│
└── README.md
```
## 🌐 Live Demo
- TBA

## ✨ About PyLumeAI

PyLumeAI builds data engineering pipelines and AI-powered applications.  
This is the second showcase project, after [PaySim Fraud Analytics](https://github.com/PyLumeAI/paysim-fraud-analytics).  


👉 Visit: [https://pylumeai.com](https://pylumeai.com)  

👉 Contact: contact.pylumeai@gmail.com