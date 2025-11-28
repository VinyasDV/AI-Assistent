# app.py
import os
import streamlit as st
from dotenv import load_dotenv
import openai
from PyPDF2 import PdfReader
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import time

# ---- Setup ----
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.warning("No OPENAI_API_KEY found. Place it in a .env file or set env var OPENAI_API_KEY.")
openai.api_key = OPENAI_API_KEY

nltk.download("punkt", quiet=True)

st.set_page_config(page_title="Personal AI Chat Assistant", layout="centered")

# ---- Utilities ----
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            # fallback safe continue
            continue
    return "\n".join(texts)

def chunk_text(text, max_chars=1000, overlap=200):
    # chunk by sentences to keep meaning
    sentences = sent_tokenize(text)
    chunks = []
    current = ""
    for s in sentences:
        if len(current) + len(s) + 1 <= max_chars:
            current += " " + s
        else:
            chunks.append(current.strip())
            # start new chunk with overlap
            if overlap > 0 and chunks:
                # take last overlap chars from current chunk to start next
                tail = chunks[-1][-overlap:]
                current = tail + " " + s
            else:
                current = s
    if current.strip():
        chunks.append(current.strip())
    return chunks

def score_and_select_chunks(chunks, user_query, top_k=3):
    # Simple heuristic similarity: count common words. Lightweight — no embeddings required.
    q_tokens = set(user_query.lower().split())
    scored = []
    for c in chunks:
        c_tokens = set(c.lower().split())
        score = len(q_tokens & c_tokens)
        scored.append((score, c))
    scored.sort(reverse=True, key=lambda x: x[0])
    selected = [c for s, c in scored[:top_k] if s > 0]
    return selected

def build_system_prompt(selected_chunks):
    if not selected_chunks:
        return "You are a helpful assistant. Answer the user's question using your general knowledge."
    # include context
    ctx = "\n\n---\n\n".join(selected_chunks)
    return (
        "You are a helpful assistant. Use the following context extracted from uploaded documents to answer the user. "
        "If the context does not contain the answer, be truthful and say you don't know rather than hallucinate.\n\n"
        f"Context:\n{ctx}\n\n"
        "Answer concisely and clearly."
    )

def call_llm(system_prompt, user_prompt, max_tokens=512, temperature=0.2):
    # Calls OpenAI Chat Completion (chat-like). Adjust to provider if different.
    try:
        # Using Chat Completions API via OpenAI library:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # change to your preferred available model; or "gpt-4o" "gpt-4" etc.
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = resp["choices"][0]["message"]["content"].strip()
        return text
    except Exception as e:
        return f"Error calling LLM: {e}"

# ---- Streamlit UI ----
st.title("Personal AI Chat Assistant")
st.markdown("Ask questions and optionally upload PDFs — the assistant will use uploaded docs for context.")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top doc chunks to use (local RAG)", 0, 5, 3)
    temperature = st.slider("LLM creativity (temperature)", 0.0, 1.0, 0.2)
    max_tokens = st.slider("Max LLM tokens (response length)", 100, 2000, 512)
    st.markdown("---")
    st.markdown("**Security & keys**")
    st.markdown("Provide your OpenAI API key in a `.env` file or set env var `OPENAI_API_KEY`.")

# file upload
uploaded_file = st.file_uploader("Upload a PDF to add context (optional)", type=["pdf"])
pdf_text = ""
chunks = []
if uploaded_file:
    with st.spinner("Extracting text from PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_file)
        chunks = chunk_text(pdf_text, max_chars=1200, overlap=200)
    st.success(f"Extracted {len(pdf_text)} characters and created {len(chunks)} chunks.")
    if st.checkbox("Show document chunks (for debugging)"):
        for i, c in enumerate(chunks):
            st.markdown(f"**Chunk {i+1}**")
            st.write(c[:1000] + ("..." if len(c) > 1000 else ""))

# init chat history
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {"role": "user"|"assistant", "text": ...}

def send_message(user_text):
    st.session_state.history.append({"role": "user", "text": user_text})
    # select relevant chunks
    selected_chunks = []
    if chunks:
        selected_chunks = score_and_select_chunks(chunks, user_text, top_k=top_k)
    system_prompt = build_system_prompt(selected_chunks)
    with st.spinner("Calling LLM..."):
        reply = call_llm(system_prompt, user_text, max_tokens=max_tokens, temperature=temperature)
    st.session_state.history.append({"role": "assistant", "text": reply})

# message input
col1, col2 = st.columns([9,1])
with col1:
    user_input = st.text_area("Your question", height=120, key="input_area")
with col2:
    if st.button("Send"):
        if not user_input.strip():
            st.warning("Type a question first.")
        else:
            send_message(user_input.strip())
            st.session_state.input_area = ""

# show history
st.markdown("### Conversation")
for msg in st.session_state.history[::-1]:
    if msg["role"] == "assistant":
        st.markdown(f"**Assistant:** {msg['text']}")
    else:
        st.markdown(f"**You:** {msg['text']}")

if st.button("Clear chat"):
    st.session_state.history = []
    st.success("Chat cleared.")

st.markdown("---")
st.markdown("Tips: Upload short PDFs (<= ~2MB) for best results. This simple app uses a lightweight local chunk-selection heuristic — for large projects use embeddings & a vector DB.")
