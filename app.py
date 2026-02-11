import os
import requests
from typing import List
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

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
        text += page.extract_text()
    return text

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
    query_embedding = embedding_model.encode([query], convert_to_tensor=False)
    query_embedding = np.asarray(query_embedding).astype('float32')
    distances, indices = index.search(query_embedding, k)
    relevant_chunks = [chunks[i] for i in indices[0]]
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
        st.error(f"Web search error: {str(e)}")
        return []

st.set_page_config(
    page_title="Serper Search & Retrieval Agent",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Serper Search & Retrieval Agent")
st.markdown("Upload a PDF document and ask questions. The agent retrieves relevant chunks from your document and performs a live web search using Serper API.")

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False

with st.sidebar:
    st.header("📄 Document Management")
    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        help="This will be your RAG knowledge base"
    )
    if uploaded_file is not None:
        if st.button("Process PDF", use_container_width=True):
            with st.spinner("Processing PDF..."):
                pdf_text = extract_text_from_pdf(uploaded_file)
                text_chunks = chunk_text(pdf_text)
                embeddings = embedding_model.encode(text_chunks, convert_to_tensor=False)
                faiss_index = create_faiss_index(text_chunks, embeddings)
                st.session_state.faiss_index = faiss_index
                st.session_state.chunks = text_chunks
                st.session_state.pdf_uploaded = True
                st.success(f"✅ PDF processed! Created {len(text_chunks)} chunks.")
    if st.session_state.pdf_uploaded:
        st.success("✅ Document loaded and ready for retrieval")

st.header("❓ Search & Retrieve")

user_query = st.text_input(
    "Enter your search query:",
    placeholder="What are the key findings in this document?",
    help="Search within your document and across the web"
)

if st.button("Retrieve Information", use_container_width=True):
    if user_query.strip():
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📚 Document Retrieval (RAG)")
            if st.session_state.pdf_uploaded:
                with st.spinner("Searching document..."):
                    relevant_chunks = search_faiss_index(
                        user_query,
                        st.session_state.faiss_index,
                        st.session_state.chunks,
                        k=3
                    )
                    for i, chunk in enumerate(relevant_chunks, 1):
                        st.info(f"**Chunk {i}:**\n\n{chunk}")
            else:
                st.warning("Please upload a PDF for document retrieval.")
        
        with col2:
            st.subheader("🌐 Web Search Results (Serper)")
            with st.spinner("Searching web..."):
                web_results = search_web(user_query, num_results=5)
                if web_results:
                    for i, result in enumerate(web_results, 1):
                        with st.expander(f"{i}. {result.get('title', 'No Title')}"):
                            st.write(f"**Snippet:** {result.get('snippet', 'No snippet available')}")
                            st.write(f"**URL:** [{result.get('link')}]({result.get('link')})")
                else:
                    st.write("No web results found.")
    else:
        st.warning("Please enter a query.")

st.divider()
st.markdown("""
**How this agent works:**
1. **Document Retrieval**: Finds the most relevant parts of your uploaded PDF using FAISS.
2. **Web Search**: Provides real-time information using the Serper Google Search API.
""")
