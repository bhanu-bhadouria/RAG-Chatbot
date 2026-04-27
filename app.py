from ingest import chunk_text, build_index, get_embedding
from retrieve import retrieve
from generate import generate_answer

with open("sample.txt") as f:
    text = f.read()

chunks = chunk_text(text)
index, _ = build_index(chunks)

chunk_map = {i: chunk for i, chunk in enumerate(chunks)}

while True:
    query = input("Ask: ")
    context_chunks = retrieve(query, index, chunk_map, get_embedding)
    context = "\n".join(context_chunks)

    answer = generate_answer(query, context)
    print("\nAnswer:", answer)