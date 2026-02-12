import os
import requests
from typing import List
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

import logging
import warnings

# Suppress warnings and logs
logging.getLogger("transformers").setLevel(logging.ERROR)

load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

def extract_text_from_pdf(pdf_file) -> str:
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        if page.extract_text():
            text += page.extract_text()
    return text if text.strip() else None

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

def create_faiss_index(chunks: List[str], embeddings: np.ndarray) -> faiss.IndexFlatL2:
    embeddings = np.asarray(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def search_faiss_index(
    query: str, 
    index: faiss.IndexFlatL2, 
    chunks: List[str], 
    k: int = 3
) -> List[str]:
    if index is None or not chunks:
        return []
    query_embedding = embedding_model.encode([query], convert_to_tensor=False)
    query_embedding = np.asarray(query_embedding).astype('float32')
    distances, indices = index.search(query_embedding, k)
    relevant_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
    return relevant_chunks

def search_web(query: str, num_results: int = 5) -> List[dict]:
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "q": query,
        "num": num_results
    }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        results = response.json()
        return results.get("organic", [])
    except Exception as e:
        return []

def format_results(rag_chunks: List[str], web_results: List[dict]) -> str:
    output = ""
    
    if rag_chunks:
        output += "### 📄 Document Segments\n\n"
        for i, chunk in enumerate(rag_chunks):
            output += f"**Segment {i+1}:**\n> {chunk}\n\n"
    else:
        output += "### 📄 Document Segments\n*No relevant segments found in the uploaded PDF via vector search.*\n\n"

    output += "---\n\n"
    
    if web_results:
        output += "### 🌐 Web Search Results\n\n"
        for res in web_results:
            title = res.get('title', 'No Title')
            link = res.get('link', '#')
            snippet = res.get('snippet', 'No snippet available.')
            output += f"- **[{title}]({link})**\n  {snippet}\n\n"
    else:
        output += "### 🌐 Web Search Results\n*No web search results found.*\n\n"

    return output

st.set_page_config(
    page_title="RAG + Web Search Tool",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 RAG + Web Search Tool")
st.markdown("Upload a PDF document and search for information. This tool retrieves relevant sections from your PDF and performs a live web search.")

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False
if "processed_file_name" not in st.session_state:
    st.session_state.processed_file_name = None

with st.sidebar:
    st.header("📄 Document Management")
    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        help="This will be your RAG knowledge base"
    )
    
    # Reset state if a new file is uploaded
    if uploaded_file is not None and uploaded_file.name != st.session_state.processed_file_name:
        st.session_state.pdf_uploaded = False
        st.session_state.faiss_index = None
        st.session_state.chunks = []
        st.session_state.processed_file_name = None
    
    if uploaded_file is not None:
        if st.button("Process PDF", use_container_width=True):
            with st.spinner("Processing PDF..."):
                pdf_text = extract_text_from_pdf(uploaded_file)
                
                if pdf_text:
                    text_chunks = chunk_text(pdf_text)
                    embeddings = embedding_model.encode(text_chunks, convert_to_tensor=False)
                    faiss_index = create_faiss_index(text_chunks, embeddings)
                    st.session_state.faiss_index = faiss_index
                    st.session_state.chunks = text_chunks
                    st.session_state.pdf_uploaded = True
                    st.session_state.processed_file_name = uploaded_file.name
                    st.success(f"✅ PDF processed!")
                else:
                    st.error("❌ Could not extract text. The PDF might be scanned or empty.")
                    
    elif st.session_state.processed_file_name is not None:
         # File was removed
        st.session_state.pdf_uploaded = False
        st.session_state.faiss_index = None
        st.session_state.chunks = []
        st.session_state.processed_file_name = None

    if st.session_state.pdf_uploaded:
        st.success("✅ Document loaded and ready")

st.header("❓ Search")

user_query = st.text_input("Enter your query:", placeholder="Type clear keywords...")

if st.button("Search", use_container_width=True):
    if user_query.strip():
        with st.spinner("🔄 Searching..."):
            relevant_chunks = []
            
            # Warn if file uploaded but not processed
            if uploaded_file is not None and not st.session_state.pdf_uploaded:
                st.warning("⚠️ You uploaded a file but didn't click 'Process PDF'. Searching web only.")
            
            if st.session_state.pdf_uploaded and st.session_state.faiss_index is not None:
                relevant_chunks = search_faiss_index(user_query, st.session_state.faiss_index, st.session_state.chunks)
            
            web_results = search_web(user_query)
            
            formatted_output = format_results(relevant_chunks, web_results)
            
            st.subheader("📋 Search Results")
            st.markdown(formatted_output)
            
    else:
        st.warning("Please enter a query.")

st.divider()
st.markdown("""
**How it works:**
1. **RAG**: Retrieves relevant sections from your PDF.
2. **Web Search**: Gets top results from Serper.
""")
