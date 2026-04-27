from openai import OpenAI
import faiss
import numpy as np
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    i = 0

    while start < len(text):
        chunk = text[start:start + chunk_size]
        chunks.append({"text": chunk, "id": i})
        start += chunk_size - overlap
        i += 1

    return chunks


def get_embedding(text):
    if isinstance(text, dict):
        text = text["text"]

    return client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding


def build_index(chunks):
    embeddings = [get_embedding(c["text"]) for c in chunks]

    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)

    index.add(np.array(embeddings).astype("float32"))

    return index