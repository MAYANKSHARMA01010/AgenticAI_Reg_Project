# RAG + Web Search Agent with AI Answer Generation

A powerful tool that combines Document Retrieval (FAISS), live web search (Serper), and AI answer generation using Google Gemini, Grok, or Hugging Face models.

## Features

- **📄 RAG (Retrieval Augmented Generation)**: Upload documents and retrieve relevant chunks using FAISS vector search
  - **Supported Formats**: PDF, TXT, DOCX, CSV, Excel, JSON
- **🌐 Web Search**: Get real-time information from Google via Serper API
- **🤖 AI Answer Generation**: Synthesize information using your choice of **5 FREE AI models**:
  - **⚡ Groq** (Fastest - Free, uses Llama 3.3 70B)
  - **Google Gemini** (Fast, free tier available)
  - **Cohere** (Good quality, free tier)
  - **Hugging Face** (Open-source Mixtral model, free)
  - **Grok (xAI)** (Premium option - Paid)
- **💡 Smart Synthesis**: AI combines document context and web results into comprehensive answers

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
└─ [AI] Generate comprehensive answer using selected AI model
    ↓
Display Answer in Streamlit
```

## Setup Instructions

### 1. Create Virtual Environment & Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
pip install -r requirements.txt
```

### 2. Set Up API Keys

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

#### Required APIs:
- **GEMINI_API_KEY** (Recommended): Get from [Google AI Studio](https://aistudio.google.com/app/apikey)
- **SERPER_API_KEY**: Get from [Serper.dev](https://serper.dev)

#### Optional APIs (choose at least one AI model):
- **GROK_API_KEY**: Get from [xAI Console](https://console.x.ai/)
- **HUGGINGFACE_API_KEY**: Get from [Hugging Face](https://huggingface.co/settings/tokens)

### 3. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## How to Use

1. **Select AI Model**: Choose your preferred AI model in the sidebar (Gemini, Grok, or Hugging Face)
2. **Upload PDF** (Optional): Select a PDF document to use as your knowledge base
3. **Process PDF**: Click "Process PDF" to build the FAISS index
4. **Ask Questions**: Enter your question and click "Generate Answer"
5. **View Results**: Read the AI-generated answer and expand "View Sources" to see original context

## AI Model Comparison

| Model | Speed | Quality | Cost | Setup Difficulty |
|-------|-------|---------|------|------------------|
| **Gemini** (gemini-1.5-flash) | ⚡ Very Fast | ⭐⭐⭐⭐ | Free tier available | Easy |
| **Grok** (grok-beta) | ⚡ Fast | ⭐⭐⭐⭐⭐ | Pay per use | Medium |
| **Hugging Face** (Mixtral-8x7B) | 🐌 Slower | ⭐⭐⭐ | Free (rate limited) | Easy |

**Recommendation**: Start with **Google Gemini** for the best balance of speed, quality, and ease of setup.

## Example Queries

- "What are the key findings in this document?"
- "What is the latest information about [topic]?" (uses web search)
- "Compare what the document says with current information on [topic]" (uses both)

## Troubleshooting

### No AI models available
- Ensure you've added at least one AI API key to your `.env` file
- Verify the packages are installed: `pip install google-generativeai openai huggingface-hub`

### API Key errors
- Check that your API keys are valid and not expired
- For Gemini: Verify at https://aistudio.google.com/app/apikey
- For Serper: Check your quota at https://serper.dev/dashboard

### PDF not processing
- Ensure the PDF contains extractable text (not scanned images)
- Try a different PDF file
- Check file upload succeeded before clicking "Process PDF"

## Project Structure

```
.
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variable template
├── .env                  # Your API keys (not in git)
└── README.md            # This file
```

## Technologies Used

- **Streamlit**: Web interface
- **FAISS**: Vector similarity search
- **Sentence Transformers**: Document embeddings
- **Google Gemini**: AI answer generation (primary)
- **OpenAI SDK**: For Grok integration
- **Hugging Face**: Alternative AI models
- **Serper API**: Web search

---

**Created for**: Agentic AI Class Assignment
