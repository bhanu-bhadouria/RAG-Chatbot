# 🧠 RAG Chatbot from Scratch

A simple yet modular **Retrieval-Augmented Generation (RAG)** chatbot built using OpenAI embeddings, FAISS vector search, and an LLM for answer generation.

---

## 🚀 Features

- 📄 Custom text ingestion and chunking (with overlap)
- 🔢 Embedding generation using OpenAI
- 🗄️ Fast vector search with FAISS
- 🤖 Context-aware response generation using LLM
- 🧩 Modular and extensible codebase

---
## 🏗️ Architecture

- User Query
- ↓
- Embedding Model
- ↓
- Vector Database (FAISS)
- ↓
- Top-K Relevant Chunks
- ↓
- LLM (with context)
- ↓
- Final Answer