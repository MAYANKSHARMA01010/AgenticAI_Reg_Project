# 🧠 The Project Explained: Easy & Simple

This project is a **Smart Research Assistant** (RAG + Web Search). It reads your documents, searches the internet, and uses AI to give you a perfect answer.

## 🛠️ What Libraries Are Used & Why?

| Library | What it does? | Why use it? |
| :--- | :--- | :--- |
| **Streamlit** | Builds the website UI | Super easy to make web apps in Python. |
| **LangChain/FAISS** | The "Memory" of the app | Helps the AI find the exact paragraph in your document that answers your question. |
| **Sentence-Transformers** | Text translator | Converts text into numbers (vectors) so the computer can understand and compare meanings. |
| **PyPDF2 / python-docx** | Document Readers | Reads text from PDF and Word files so the AI can understand them. |
| **Pandas** | Data Reader | Reads tables from Excel and CSV files. |
| **Serper API** | Google Searcher | Allows the app to Google search properly and get live results. |

---

## 🤖 AI Models Used (The "Brains")

We support 5 different AI brains. You can choose which one to use:

1. **⚡ Groq (Llama 3)**:
   - **What is it?**: A super-fast AI model.
   - **Why use it?**: It's **FREE** and answers instantly. Best for quick questions.

2. **🌟 Cohere**:
   - **What is it?**: An AI built for business and accurate writing.
   - **Why use it?**: Good balance of quality and speed. Free tier available.

3. **🧠 Google Gemini**:
   - **What is it?**: Google's smart AI.
   - **Why use it?**: Good for general knowledge, but setup can be tricky sometimes.

4. **🤗 Hugging Face**:
   - **What is it?**: Open-source community models.
   - **Why use it?**: Completely free and open, but might be slower.

5. **🚀 Grok (xAI)**:
   - **What is it?**: Elon Musk's AI.
   - **Why use it?**: Very smart and funny, but requires a paid account.

---

## 📂 What Files Can It Read?

The app can now read almost anything you upload:
- **PDF** (`.pdf`) - Books, papers
- **Word** (`.docx`) - Essays, reports
- **Text** (`.txt`) - Notes, code
- **Excel/CSV** (`.xlsx`, `.csv`) - Data tables
- **JSON** (`.json`) - Data files

---

## ⚙️ How It Works (The "Flow")

1. **Upload**: You give it a file (e.g., a PDF about Mars).
2. **Read & Store**: The app reads the text and saves it in its "memory" (FAISS).
3. **Search**: You ask "Is there water on Mars?".
4. **Retrieve**: The app finds the *exact page* in your PDF that talks about water.
5. **Web Check**: It also Googles "Water on Mars latest news" to get fresh info.
6. **AI Answer**: It sends BOTH the PDF info and the Google info to the AI (like Groq) and says:
   > *"Here is what the book says, and here is what Google says. Combine them and answer the user."*
7. **Result**: You get a perfect, up-to-date answer with sources! 
