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

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

try:
    import google.genai as genai
    client = genai.Client(api_key=GEMINI_API_KEY)
    GEMINI_CLIENT_TYPE = "genai"
except Exception as e:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_CLIENT_TYPE = "generativeai"
    except Exception as e2:
        GEMINI_CLIENT_TYPE = None

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
        return []

def generate_answer(question: str, rag_context: str, web_context: str) -> str:
    if not GEMINI_CLIENT_TYPE or not GEMINI_API_KEY:
        return "⚠️ Gemini API key missing or invalid. Please add `GEMINI_API_KEY` to your `.env` file to enable AI answers."
    
    combined_context = f"""
System: You are a helpful AI assistant. Answer the user question using the provided document context and web search results.
Cite whether information comes from the 'Document' or the 'Web'.

Document Context:
{rag_context}

Web Search Results:
{web_context}

User Question:
{question}
"""
    try:
        if GEMINI_CLIENT_TYPE == "genai":
            response = client.models.generate_content(model="gemini-2.0-flash", contents=combined_context)
            return response.text
        else:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(combined_context)
            return response.text
    except Exception as e:
        return f"Error generating answer: {str(e)}"

st.set_page_config(
    page_title="RAG + Web Search Agent",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 RAG + Web Search Agent")
st.markdown("Upload a PDF document and ask questions. The agent combines document knowledge with live web search using Gemini API.")

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
                st.success(f"✅ PDF processed!")
    if st.session_state.pdf_uploaded:
        st.success("✅ Document loaded and ready")

st.header("❓ Ask a Question")

user_query = st.text_input("Enter your question:", placeholder="What does the document say about...?")

if st.button("Generate Answer", use_container_width=True):
    if user_query.strip():
        with st.spinner("🔄 Searching & Generating..."):
            relevant_chunks = []
            if st.session_state.pdf_uploaded:
                relevant_chunks = search_faiss_index(user_query, st.session_state.faiss_index, st.session_state.chunks)
            
            web_results = search_web(user_query)
            
            rag_context = "\n---\n".join(relevant_chunks)
            web_context_str = "\n".join([f"- {r.get('title')}: {r.get('snippet')}" for r in web_results])
            
            answer = generate_answer(user_query, rag_context, web_context_str)
            
            st.subheader("📋 AI Answer")
            st.write(answer)
            
            with st.expander("📚 View Sources"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Document Chunks:**")
                    for c in relevant_chunks: st.info(c)
                with col2:
                    st.markdown("**Web Results:**")
                    for r in web_results: st.write(f"- [{r.get('title')}]({r.get('link')})")
    else:
        st.warning("Please enter a question.")

st.divider()
st.markdown("""
**How it works:**
1. **RAG**: Finds relevant parts of your PDF.
2. **Search**: Gets live web data via Serper.
3. **AI**: Gemini synthesizes both into a final answer.
""")
