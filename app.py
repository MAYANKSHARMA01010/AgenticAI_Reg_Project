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
        st.error("Could not initialize Gemini API. Please check your API key.")
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


def search_web(query: str, num_results: int = 5) -> str:
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
        
        formatted_results = "=== WEB SEARCH RESULTS ===\n"
        
        if "organic" in results:
            for i, result in enumerate(results["organic"][:num_results], 1):
                formatted_results += f"\n{i}. {result.get('title', 'No Title')}\n"
                formatted_results += f"   URL: {result.get('link', 'No Link')}\n"
                formatted_results += f"   Snippet: {result.get('snippet', 'No snippet available')}\n"
        
        return formatted_results
        
    except Exception as e:
        return f"Web search error: {str(e)}"


def generate_answer(
    question: str, 
    rag_context: str, 
    web_context: str
) -> str:
    if not GEMINI_CLIENT_TYPE:
        return "Gemini API not initialized. Please check your GEMINI_API_KEY in the .env file."
    
    combined_context = f"""
System: You are a helpful AI assistant that answers questions based on a provided document and live web search results.
Always cite your sources and explain if you found the information in the 'Document Context' or 'Web Search Context'.

Document Context (from uploaded PDF):
{rag_context}

Web Search Context:
{web_context}

User Question:
{question}

Instructions:
1. Combine the information from both contexts for a comprehensive answer.
2. If the document doesn't contain the answer, rely on the web results.
3. Explicitly cite whether information comes from the "Document" or the "Web".
4. If neither source has the answer, state that clearly.
"""
    
    try:
        if GEMINI_CLIENT_TYPE == "genai":
            models_to_try = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-pro"]
            
            for model in models_to_try:
                try:
                    response = client.models.generate_content(
                        model=model,
                        contents=combined_context
                    )
                    return response.text
                except Exception as model_error:
                    if "429" in str(model_error) or "quota" in str(model_error).lower():
                        return f"API quota exceeded. Please check your Gemini API billing at console.cloud.google.com"
                    continue
        
        else:
            models_to_try = ["gemini-1.5-flash", "gemini-pro"]
            
            for model_name in models_to_try:
                try:
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(combined_context)
                    return response.text
                except Exception as model_error:
                    if "429" in str(model_error) or "quota" in str(model_error).lower():
                        return f"API quota exceeded. Please check your Gemini API billing at console.cloud.google.com"
                    continue
        
        return "No available Gemini models could generate a response"
        
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            return f"API quota exceeded. Please check your Gemini API billing and usage limits."
        return f"Error generating answer: {error_msg[:200]}"


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
                
                st.success(f"✅ PDF processed! Created {len(text_chunks)} chunks.")
                st.info(f"Total document length: {len(pdf_text)} characters")
    
    if st.session_state.pdf_uploaded:
        st.success("✅ Document loaded and ready for queries")

st.header("❓ Ask a Question")

if st.session_state.pdf_uploaded:
    user_question = st.text_input(
        "Enter your question:",
        placeholder="What is the main topic of this document?",
        help="Ask anything related to your document or general knowledge"
    )
    
    if st.button("Search & Generate Answer", use_container_width=True):
        if user_question.strip():
            with st.spinner("🔄 Searching document and web..."):
                relevant_chunks = search_faiss_index(
                    user_question,
                    st.session_state.faiss_index,
                    st.session_state.chunks,
                    k=3
                )
                
                rag_context = "\n---\n".join(relevant_chunks)
                
                web_results = search_web(user_question, num_results=5)
                
                answer = generate_answer(user_question, rag_context, web_results)
            
            st.subheader("📋 Answer")
            st.write(answer)
            
            with st.expander("📚 View Source Information"):
                st.subheader("Document Chunks Retrieved:")
                for i, chunk in enumerate(relevant_chunks, 1):
                    st.write(f"**Chunk {i}:**")
                    st.write(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                    st.divider()
                
                st.subheader("Web Search Results:")
                st.write(web_results)
        else:
            st.warning("Please enter a question.")

else:
    st.info("👈 Please upload and process a PDF document first using the sidebar.")

st.divider()
st.markdown("""
**How this agent works:**
1. **PDF Processing**: Extract text from uploaded PDF and split into chunks
2. **Embeddings**: Convert chunks into vector embeddings using sentence-transformers
3. **Vector Search**: FAISS stores embeddings and finds relevant chunks based on your question
4. **Web Search**: Serper API performs real-time Google search
5. **LLM Generation**: Gemini combines document + web context to generate comprehensive answers
""")
