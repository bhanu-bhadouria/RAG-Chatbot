import streamlit as st
from ingest import chunk_text, build_index, get_embedding
from retrieve import hybrid_retrieve
from generate import generate_answer
from pdf_loader import load_pdf
from hybrid import build_bm25 

@st.cache_resource
def load_pipeline():
    text = load_pdf("Rahul Sharma TEST Resume PDF.pdf")
    chunks = chunk_text(text)
    index = build_index(chunks)
    chunk_map = dict(enumerate(chunks))
    bm25, texts = build_bm25(chunks)
    return index, chunk_map, bm25, texts


index, chunk_map, bm25, texts = load_pipeline()

st.title("🧠 RAG Chatbot")

query = st.text_input("Ask a question:")

if query:
    context_chunks = hybrid_retrieve(
        query,
        index,
        bm25,
        texts,
        chunk_map,
        get_embedding
    )

    context = "\n".join(context_chunks)

    answer = generate_answer(query, context)

    st.subheader("Answer")
    st.write(answer)

    with st.expander("Retrieved Context"):
        for c in context_chunks:
            st.write(c)