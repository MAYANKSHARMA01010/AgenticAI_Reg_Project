# RAG + Web Search Agent

A powerful tool that combines Document Retrieval (FAISS), live web search (Serper), and AI answer generation (Gemini).

## Project Overview

This is a single-file Streamlit application that demonstrates:

- **RAG (Retrieval Augmented Generation)**: Upload a PDF and retrieve relevant chunks using FAISS.
- **Web Search**: Live Google search results using the Serper API.
- **AI Answering**: Uses Google Gemini to synthesize document context and web search results into a single comprehensive answer.

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
├─ [Web] Perform web search via Serper API
└─ [AI] Generate Answer using Google Gemini
    ↓
Display Synthesis in Streamlit
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
- Add your API keys:
  - `GEMINI_API_KEY`: Get from https://aistudio.google.com/app/apikey
  - `SERPER_API_KEY`: Get from https://serper.dev

### 3. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## How to Use

1. **Upload PDF**: Select a PDF document in the sidebar.
2. **Process PDF**: Click "Process PDF" to build the index.
3. **Ask**: Enter a question and click "Generate Answer".
4. **View**: Read the AI's synthesized response and check sources if needed.

---

**Created for**: Agentic AI Class Assignment (Beginner Level)
