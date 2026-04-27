import numpy as np

def retrieve(query, index, chunk_map, embed_fn, k=3):
    query_embedding = np.array([embed_fn(query)]).astype("float32")
    distances, indices = index.search(query_embedding, k)

    return [chunk_map[i] for i in indices[0]]