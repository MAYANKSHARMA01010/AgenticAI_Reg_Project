import os
import requests
from typing import List
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import docx
import pandas as pd
import json
import logging
import warnings

logging.getLogger("transformers").setLevel(logging.ERROR)

load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

try:
    warnings.filterwarnings("ignore", category=FutureWarning)
    import google.generativeai as genai
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
except ImportError:
    genai = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    import cohere
except ImportError:
    cohere = None




@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

def extract_text_from_file(uploaded_file) -> str:
    """Extract text from various file formats."""
    file_type = uploaded_file.name.split('.')[-1].lower()
    text = ""
    
    try:
        if file_type == 'pdf':
            pdf_reader = PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                if page.extract_text():
                    text += page.extract_text() + "\n"
                    
        elif file_type == 'docx':
            doc = docx.Document(uploaded_file)
            for para in doc.paragraphs:
                text += para.text + "\n"
                
        elif file_type == 'txt':
            text = uploaded_file.read().decode("utf-8")
            
        elif file_type in ['csv', 'xlsx', 'xls']:
            if file_type == 'csv':
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            # Convert dataframe to text representation
            text = df.to_string(index=False)
            
        elif file_type == 'json':
            data = json.load(uploaded_file)
            text = json.dumps(data, indent=2)
            
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None
        
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

def generate_answer(query: str, rag_chunks: List[str], web_results: List[dict], model_type: str = "gemini") -> str:
    """
    Generate a comprehensive answer using RAG chunks and web search results.
    Supports: gemini, grok, huggingface
    """
    context = ""
    
    if rag_chunks:
        context += "**Document Information:**\n"
        for i, chunk in enumerate(rag_chunks, 1):
            context += f"\n[Doc {i}] {chunk[:300]}...\n"
    
    if web_results:
        context += "\n**Web Search Results:**\n"
        for i, res in enumerate(web_results, 1):
            title = res.get('title', 'No Title')
            snippet = res.get('snippet', '')
            context += f"\n[Web {i}] {title}: {snippet}\n"
    
    if not context:
        return "I couldn't find any relevant information to answer your question."
    
    prompt = f"""Based on the following context, provide a comprehensive and accurate answer to the user's question.

Context:
{context}

User Question: {query}

Instructions:
- Synthesize information from both document chunks and web results
- Provide a clear, well-structured answer
- Cite sources when relevant (e.g., "According to the document..." or "Web sources indicate...")
- If information is insufficient, acknowledge limitations

Answer:"""

    try:
        if model_type == "gemini" and genai:
            # Trying older stable model
            model = genai.GenerativeModel('gemini-1.0-pro')
            response = model.generate_content(prompt)
            return response.text
        
        elif model_type == "groq" and Groq and GROQ_API_KEY:
            client = Groq(api_key=GROQ_API_KEY)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that synthesizes information from multiple sources."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1024
            )
            return response.choices[0].message.content
        
        elif model_type == "cohere" and cohere and COHERE_API_KEY:
            co = cohere.Client(COHERE_API_KEY)
            response = co.chat(
                message=prompt,
                model="command-r-08-2024",
                temperature=0.7
            )
            return response.text
        
        elif model_type == "grok" and OpenAI and GROK_API_KEY:
            client = OpenAI(
                api_key=GROK_API_KEY,
                base_url="https://api.x.ai/v1"
            )
            response = client.chat.completions.create(
                model="grok-beta",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that synthesizes information from multiple sources."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content
        
        elif model_type == "huggingface" and InferenceClient and HUGGINGFACE_API_KEY:
            client = InferenceClient(token=HUGGINGFACE_API_KEY)
            response = client.text_generation(
                prompt,
                model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                max_new_tokens=500,
                temperature=0.7
            )
            return response
        
        else:
            return f"❌ {model_type.upper()} is not available. Please check your API key and library installation."
    
    except Exception as e:
        return f"❌ Error generating answer with {model_type}: {str(e)}"

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
st.markdown("Upload a document (PDF, TXT, DOCX, CSV, Excel, JSON) and search for information. This tool retrieves relevant sections from your file and performs a live web search.")

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
        "Upload a document",
        type=["pdf", "txt", "docx", "csv", "xlsx", "json"],
        help="Upload PDF, Text, Word, Excel, CSV, or JSON files"
    )
    
    # Reset state if a new file is uploaded
    if uploaded_file is not None and uploaded_file.name != st.session_state.processed_file_name:
        st.session_state.pdf_uploaded = False
        st.session_state.faiss_index = None
        st.session_state.chunks = []
        st.session_state.processed_file_name = None
    
    if uploaded_file is not None:
        if st.button("Process Document", use_container_width=True):
            with st.spinner("Processing document..."):
                file_text = extract_text_from_file(uploaded_file)
                
                if file_text:
                    text_chunks = chunk_text(file_text)
                    embeddings = embedding_model.encode(text_chunks, convert_to_tensor=False)
                    faiss_index = create_faiss_index(text_chunks, embeddings)
                    st.session_state.faiss_index = faiss_index
                    st.session_state.chunks = text_chunks
                    st.session_state.pdf_uploaded = True
                    st.session_state.processed_file_name = uploaded_file.name
                    st.success(f"✅ Document processed!")
                else:
                    st.error("❌ Could not extract text. The file might be empty or unreadable.")
                    
    elif st.session_state.processed_file_name is not None:
         # File was removed
        st.session_state.pdf_uploaded = False
        st.session_state.faiss_index = None
        st.session_state.chunks = []
        st.session_state.processed_file_name = None

    if st.session_state.pdf_uploaded:
        st.success("✅ Document loaded and ready")
    
    st.divider()
    st.header("🤖 AI Model Selection")
    
    available_models = {}
    if Groq and GROQ_API_KEY:
        available_models["groq"] = "⚡ Groq (Fastest & Free)"
    if cohere and COHERE_API_KEY:
        available_models["cohere"] = "Cohere (Free Tier)"
    if genai and GEMINI_API_KEY:
        available_models["gemini"] = "Google Gemini"
    if OpenAI and GROK_API_KEY:
        available_models["grok"] = "Grok (xAI - Paid)"
    if InferenceClient and HUGGINGFACE_API_KEY:
        available_models["huggingface"] = "Hugging Face"
    
    if not available_models:
        st.error("⚠️ No AI models configured. Please add API keys to your .env file.")
        selected_model = None
    else:
        selected_model = st.selectbox(
            "Choose AI Model:",
            options=list(available_models.keys()),
            format_func=lambda x: available_models[x],
            help="Select which AI model to use for generating answers"
        )

st.header("❓ Search")

user_query = st.text_input("Enter your query:", placeholder="Type clear keywords...")

if st.button("Generate Answer", use_container_width=True, type="primary"):
    if not selected_model:
        st.error("❌ No AI model available. Please configure API keys in .env file.")
    elif user_query.strip():
        with st.spinner(f"🔄 Generating answer using {available_models.get(selected_model, selected_model)}..."):
            relevant_chunks = []
            
            if uploaded_file is not None and not st.session_state.pdf_uploaded:
                st.warning("⚠️ You uploaded a file but didn't click 'Process Document'. Searching web only.")
            
            if st.session_state.pdf_uploaded and st.session_state.faiss_index is not None:
                relevant_chunks = search_faiss_index(user_query, st.session_state.faiss_index, st.session_state.chunks)
            
            web_results = search_web(user_query)
            
            answer = generate_answer(user_query, relevant_chunks, web_results, selected_model)
            
            st.subheader("💡 AI-Generated Answer")
            st.markdown(answer)
            
            with st.expander("📊 View Sources"):
                formatted_output = format_results(relevant_chunks, web_results)
                st.markdown(formatted_output)
            
    else:
        st.warning("Please enter a query.")


st.divider()
st.markdown("""
**How it works:**
1. **RAG**: Retrieves relevant sections from your PDF using FAISS vector search.
2. **Web Search**: Gets top results from Serper API for real-time information.
3. **AI Synthesis**: Combines both sources to generate a comprehensive answer using your selected AI model.
""")

