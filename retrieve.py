import numpy as np
from hybrid import bm25_search

def hybrid_retrieve(query, index, bm25, texts, chunk_map, embed_fn, k=3):

    # -------- VECTOR SEARCH --------
    query_embedding = np.array([embed_fn(query)]).astype("float32")
    _, indices = index.search(query_embedding, k)

    vector_results = [chunk_map[i]["text"] for i in indices[0]]

    # -------- KEYWORD SEARCH --------
    keyword_results = bm25_search(query, bm25, texts, k)

    # -------- MERGE --------
    combined = []

    for c in vector_results + keyword_results:
        if c not in combined:
            combined.append(c)

    return combined[:k]