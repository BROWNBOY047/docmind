# 🧠 DocMind — RAG System Built from Scratch

> Upload any PDF. Ask anything. Get grounded, cited answers.
> No LangChain. No magic. Just fundamentals.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://docmind-dnbtgpecdwjud8wmrwdfyp.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FAISS](https://img.shields.io/badge/Vector_DB-FAISS-orange)
![Groq](https://img.shields.io/badge/LLM-Groq%20Llama3-purple)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 What is DocMind?

**DocMind** is a fully working **Retrieval-Augmented Generation (RAG)** system — built completely from scratch without LangChain or any abstraction framework.

Upload a PDF document, ask a question, and DocMind will:

- 🔍 **Find** the most relevant chunks from your document using semantic search
- 🧩 **Build** a grounded prompt using only that retrieved context
- 💬 **Generate** an accurate, cited answer using an LLM — with zero hallucination from outside knowledge

> This is exactly how tools like Notion AI, ChatGPT plugins, and enterprise AI search work under the hood.

---

## ✨ Features

- 📄 Upload any PDF and query it instantly
- 🧠 Semantic search powered by vector embeddings
- ⚡ Fast answers via Groq API (Llama 3.1)
- 📚 Every answer cites the exact page it came from
- 🔒 Honest — says "I could not find this" instead of hallucinating
- 🎛️ Adjustable chunk size and top-K retrieval from the sidebar
- 🚀 Fully deployed — no setup needed to try it

---

## 🏗️ How It Works

```
📄 Your PDF
     |
     v
🔪 Chunker          splits text into overlapping pieces (512 chars)
     |
     v
🧬 Embedder         converts each chunk to a 384-dim vector
     |
     v
🗄️ FAISS Index      stores all vectors for fast similarity search
     |
     v
❓ Your Question
     |
     v
🔍 Retriever        embeds question, finds top-K matching chunks
     |
     v
📝 Prompt Builder   wraps chunks + question into a grounded prompt
     |
     v
🤖 Groq LLM         reads ONLY the context, generates the answer
     |
     v
✅ Answer + Source  cited response shown to you
```

---

## 🛠️ Tech Stack

| 🔧 Component | 🛠️ Tool | 💡 Why |
|---|---|---|
| Language | Python 3.10+ | Standard for ML/AI |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) | Free, local, powerful 384-dim vectors |
| Vector DB | FAISS (IndexFlatL2) | Lightweight, no server needed |
| LLM | Groq API — Llama 3.1 8B Instant | Free tier, blazing fast inference |
| PDF Parsing | PyMuPDF (fitz) | Reliable text extraction with page metadata |
| UI | Streamlit | Clean, fast, easy to deploy |

---

## 🚀 Try It Live

👉 **[Open DocMind App](https://docmind-dnbtgpecdwjud8wmrwdfyp.streamlit.app/)**

1. Get a free Groq API key at [console.groq.com](https://console.groq.com)
2. Paste it in the sidebar
3. Upload any PDF
4. Start asking questions!

---

## 💻 Run Locally

**1️⃣ Clone the repo**
```bash
git clone https://github.com/BROWNBOY047/docmind.git
cd docmind
```

**2️⃣ Install dependencies**
```bash
pip install -r requirements.txt
```

**3️⃣ Get your free Groq API key**

Go to [console.groq.com](https://console.groq.com) → API Keys → Create Key

**4️⃣ Run the app**
```bash
streamlit run app.py
```

**5️⃣ Open in your browser**
```
http://localhost:8501
```

---

## 🗂️ Project Structure

```
docmind/
├── app.py            <- Full Streamlit app (all pipeline code inside)
├── requirements.txt  <- All dependencies
└── README.md
```

---

## 🔬 Experiments & Findings

### 📦 Chunk Size Comparison

Tested three chunk sizes on the same document and query:

| Chunk Size | Total Chunks | Best Match Distance | Verdict |
|---|---|---|---|
| 256 chars | 26 | 1.2697 | ❌ Too small — loses context |
| **512 chars** | **14** | **1.1479** | **✅ Best retrieval** |
| 1024 chars | 9 | 1.3236 | ❌ Too large — adds noise |

> 💡 512 characters with 50-char overlap is the sweet spot for this document type.

---

### 🧠 Semantic vs Keyword Search

Proved that embeddings capture **meaning**, not just words:

| Comparison | Cosine Similarity |
|---|---|
| Two sentences with same meaning | **0.6960** ✅ |
| Two completely unrelated sentences | **0.0421** ✅ |

---

### 📊 Evaluation Results (v1)

Tested against 7 questions with known answers:

| Metric | Result |
|---|---|
| ✅ Questions passed | 4 / 7 |
| 🎯 Accuracy | 57.1% |
| 🚫 Hallucination rate | **0%** |

The system never made up an answer. When the context was missing, it correctly said:
> *"I could not find this in the document."*

---

## 💡 Key Concepts Learned

- 🔢 How text is converted into **vector embeddings**
- 📐 How **cosine similarity** and **L2 distance** measure meaning closeness
- ✂️ Why **chunk size and overlap** directly affect retrieval quality
- 📝 How to build a **grounded prompt** that eliminates hallucination
- ⚖️ The difference between **retrieval quality** and **generation quality**
- 🧪 How to **evaluate a RAG system** end to end

---

## 🔜 Future Improvements (v2)

- [ ] Increase `top_k` from 3 to 5 for better recall on buried answers
- [ ] Increase overlap from 50 to 100 chars to fix chunk boundary splits
- [ ] Add **BM25 keyword fallback** for exact term matching
- [ ] Support **multiple documents** simultaneously
- [ ] Add **re-ranking** — retrieve top 10, re-rank to top 3
- [ ] Compare **larger embedding models** (BGE, OpenAI embeddings)

---

## 📄 License

MIT — free to use, modify, and build on.

---

<p align="center">
Built from scratch as Project 1 of an LLM engineering curriculum.<br>
Every line written with full understanding of what it does. No black boxes.
</p>
