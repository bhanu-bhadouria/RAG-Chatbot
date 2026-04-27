from rank_bm25 import BM25Okapi

def build_bm25(chunks):
    texts = [c["text"] for c in chunks]
    tokenized = [t.split() for t in texts]
    return BM25Okapi(tokenized), texts


def bm25_search(query, bm25, texts, k=3):
    scores = bm25.get_scores(query.split())

    top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    return [texts[i] for i in top_k]