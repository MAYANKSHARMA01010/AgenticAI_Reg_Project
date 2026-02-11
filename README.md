# Serper Search & Retrieval Agent

A simple, beginner-friendly tool that combines Document Retrieval (FAISS) with live web search (Serper).

## Project Overview

This is a single-file Streamlit application that demonstrates:

- **Document Retrieval (RAG)**: Upload a PDF, extract text, create embeddings, and retrieve relevant chunks using FAISS.
- **Web Search**: Live Google search results using the Serper API.
- **Pure Retrieval**: No LLM required – view raw context from your files and the web side-by-side.

## Architecture

```
User Input (PDF + Query)
    ↓
[1] PDF Text Extraction → [2] Text Chunking → [3] Embedding (sentence-transformers)
    ↓
[4] FAISS Index Storage (in-memory)
    ↓
Query Processing:
├─ [Local] Retrieve relevant chunks from FAISS
└─ [Web] Perform web search via Serper API
    ↓
Display Results in Streamlit (Side-by-Side)
```

## Setup

### 1. Create Virtual Environment & Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

- Copy `.env.example` to `.env`
- Add your Serper API key:
  - `SERPER_API_KEY`: Get from https://serper.dev

### 3. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## How to Use

1. **Upload PDF**: Use the sidebar file uploader to select a PDF document.
2. **Process PDF**: Click "Process PDF" to build the retrieval index.
3. **Search**: Enter a query to search both your document and the live web.
4. **View Results**: Compare document knowledge with live web data in the side-by-side view.

## Technologies Used

- **Streamlit**: Web UI framework
- **Serper API**: Google Search API
- **sentence-transformers**: Embedding model (all-MiniLM-L6-v2)
- **FAISS**: Vector database for similarity search
- **PyPDF2**: PDF text extraction

---

**Created for**: Agentic AI Class Assignment (Beginner Level)
