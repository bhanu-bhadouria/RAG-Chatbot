# 🧠 RAG Chatbot (ChatGPT-style UI)

A Retrieval-Augmented Generation (RAG) chatbot built from scratch using Python.  
This app allows users to upload any PDF and ask questions about it using a ChatGPT-like interface.

---

## 🚀 Features

- 💬 ChatGPT-style conversational UI (Streamlit)
- 📄 Upload and chat with any PDF
- 🔍 Hybrid search (FAISS + BM25)
- 🧠 Conversational memory (context-aware responses)
- ⚡ Fast semantic retrieval using embeddings
- 📚 Expandable source context display
- 🎯 Improved retrieval using query + chat history

---

## 🧠 How It Works

1. **PDF Upload**
   - User uploads a PDF file
   - Text is extracted using PyMuPDF

2. **Chunking**
   - Text is split into chunks for efficient retrieval

3. **Embedding + Indexing**
   - Chunks are converted into embeddings
   - Stored in FAISS (vector search)
   - BM25 is built for keyword search

4. **Hybrid Retrieval**
   - Combines semantic (FAISS) + keyword (BM25)
   - Uses chat history for better query understanding

5. **Answer Generation**
   - Context + query → LLM → final answer

---

