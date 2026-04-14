import streamlit as st
import fitz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="DocMind - RAG System",
    page_icon="🧠",
    layout="wide"
)

# ---- STYLING ----
st.markdown("""
<style>
.answer-box {
    background-color: #f0f4ff;
    border-left: 4px solid #4A90D9;
    padding: 16px;
    border-radius: 8px;
    font-size: 16px;
}
.source-box {
    background-color: #f9f9f9;
    border: 1px solid #ddd;
    padding: 10px;
    border-radius: 6px;
    font-size: 13px;
    color: #555;
}
</style>
""", unsafe_allow_html=True)

# ---- HEADER ----
st.title("🧠 DocMind")
st.caption("A Retrieval-Augmented Generation system - built from scratch, no LangChain")
st.divider()

# ---- SIDEBAR ----
with st.sidebar:
    st.header("Settings")
    groq_api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    st.markdown("Get a free API key at [console.groq.com](https://console.groq.com)")
    top_k = st.slider("Chunks to retrieve (top K)", min_value=1, max_value=7, value=3)
    chunk_size = st.select_slider("Chunk size", options=[256, 512, 1024], value=512)
    st.divider()
    st.markdown("**How it works:**")
    st.markdown("1. Upload a PDF")
    st.markdown("2. Doc is chunked + embedded")
    st.markdown("3. Your question is embedded")
    st.markdown("4. FAISS finds matching chunks")
    st.markdown("5. LLM answers using only context")


# ---- HELPER FUNCTIONS ----

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


def extract_text(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    return pages


def chunk_text(pages, chunk_size=512, overlap=50):
    chunks = []
    chunk_id = 0
    for page in pages:
        text = page["text"]
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append({
                "chunk_id": chunk_id,
                "text": text[start:end],
                "page": page["page"]
            })
            chunk_id += 1
            start += chunk_size - overlap
    return chunks


def build_index(chunks, embedder):
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings


def retrieve(query, index, chunks, embedder, top_k=3):
    q_vec = embedder.encode([query]).astype("float32")
    distances, indices = index.search(q_vec, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        chunk = chunks[idx]
        results.append({**chunk, "distance": round(float(dist), 4)})
    return results


def build_prompt(query, retrieved_chunks):
    context_blocks = []
    for chunk in retrieved_chunks:
        block = "[Source: Page " + str(chunk["page"]) + ", Chunk " + str(chunk["chunk_id"]) + "]\n" + chunk["text"]
        context_blocks.append(block)
    context = "\n\n---\n\n".join(context_blocks)

    prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY the context below.\n"
        "If the answer is not in the context, say: I could not find this in the document.\n"
        "Do NOT make up information. Mention the page your answer comes from.\n\n"
        "=== CONTEXT ===\n"
        + context
        + "\n\n=== QUESTION ===\n"
        + query
        + "\n\n=== ANSWER ==="
    )
    return prompt


def ask_llm(prompt, api_key):
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content


# ---- MAIN APP ----

embedder = load_embedder()

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing document..."):
        pages = extract_text(uploaded_file)
        chunks = chunk_text(pages, chunk_size=chunk_size)
        index, _ = build_index(chunks, embedder)

    st.success("Document ready - " + str(len(pages)) + " pages, " + str(len(chunks)) + " chunks created")
    st.divider()

    query = st.text_input(
        "Ask a question about your document",
        placeholder="e.g. What is the main topic of this document?"
    )

    if query:
        if not groq_api_key:
            st.warning("Please enter your Groq API key in the sidebar.")
        else:
            with st.spinner("Retrieving and generating answer..."):
                retrieved = retrieve(query, index, chunks, embedder, top_k=top_k)
                prompt = build_prompt(query, retrieved)
                answer = ask_llm(prompt, groq_api_key)

            st.subheader("Answer")
            st.markdown('<div class="answer-box">' + answer + '</div>', unsafe_allow_html=True)
            st.divider()

            with st.expander("View Retrieved Context (Sources Used)"):
                for i, chunk in enumerate(retrieved):
                    st.markdown(
                        "**Rank " + str(i + 1) +
                        " - Page " + str(chunk["page"]) +
                        " | Chunk " + str(chunk["chunk_id"]) +
                        " | Distance: " + str(chunk["distance"]) + "**"
                    )
                    st.markdown('<div class="source-box">' + chunk["text"] + '</div>', unsafe_allow_html=True)
                    st.write("")

else:
    st.info("Upload a PDF to get started")
    st.markdown("### What is DocMind?")
    st.markdown(
        "DocMind is a Retrieval-Augmented Generation (RAG) system built from scratch. "
        "It lets you upload any document and ask questions - the system finds the most "
        "relevant parts and generates accurate, grounded answers using an LLM."
    )
    st.markdown("**Built with:** Python - FAISS - Sentence Transformers - Groq - Streamlit")
    st.markdown("**No LangChain. No magic. Just fundamentals.**")
