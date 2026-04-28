import streamlit as st
import time

from ingest import chunk_text, build_index, get_embedding
from retrieve import hybrid_retrieve
from generate import generate_answer
from pdf_loader import load_pdf
from hybrid import build_bm25 

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="RAG Chatbot", layout="wide")

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
    .main {
        background-color: #0f172a;
        color: white;
    }

    .block-container {
        max-width: 900px;
        margin: auto;
    }

    .stChatMessage {
        border-radius: 12px;
        padding: 10px;
        margin-bottom: 10px;
    }

    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------- CHAT HISTORY ----------
def format_chat_history(messages, max_turns=4):
    history = messages[-max_turns * 2:]

    formatted = ""
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted += f"{role}: {msg['content']}\n"

    return formatted


# ---------- PDF PROCESSING ----------
def process_pdf(file):
    text = load_pdf(file)
    chunks = chunk_text(text)

    index = build_index(chunks)
    chunk_map = dict(enumerate(chunks))
    bm25, texts = build_bm25(chunks)

    return index, chunk_map, bm25, texts


# ---------- SIDEBAR ----------
with st.sidebar:
    st.title("🧠 RAG Chatbot")
    st.markdown("Hybrid Search + PDF")

    uploaded_file = st.file_uploader("📄 Upload a PDF", type="pdf")

    if st.button("🗑️ New Chat"):
        st.session_state.messages = []

    st.markdown("---")
    st.caption("Built with FAISS + BM25")


# ---------- SESSION STATE ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

# ---------- HANDLE PDF UPLOAD ----------
if uploaded_file is not None:
    if st.session_state.get("current_file") != uploaded_file.name:
        with st.spinner("Processing PDF..."):
            pipeline = process_pdf(uploaded_file)

            st.session_state.pipeline = pipeline
            st.session_state.current_file = uploaded_file.name
            st.session_state.messages = []  # reset chat

        st.success(f"Loaded: {uploaded_file.name}")

# ---------- CHECK PIPELINE ----------
if st.session_state.pipeline is None:
    st.warning("Please upload a PDF to start.")
    st.stop()

index, chunk_map, bm25, texts = st.session_state.pipeline


# ---------- DISPLAY CHAT ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ---------- INPUT ----------
query = st.chat_input("Message...")

if query:
    # USER MESSAGE
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    # RETRIEVAL (with memory)
    search_query = format_chat_history(st.session_state.messages) + " " + query

    context_chunks = hybrid_retrieve(
        search_query,
        index,
        bm25,
        texts,
        chunk_map,
        get_embedding
    )

    context = "\n".join(context_chunks)

    # GENERATE ANSWER
    chat_history = format_chat_history(st.session_state.messages)

    answer = generate_answer(query, context, chat_history)

    if not answer:
        answer = "⚠️ No response generated."

    # ASSISTANT MESSAGE
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_text = ""

        for word in answer.split():
            full_text += word + " "
            placeholder.markdown(full_text)
            time.sleep(0.015)

        # CONTEXT
        with st.expander("📄 Retrieved Context"):
            for c in context_chunks:
                st.write(c)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )