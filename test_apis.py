
import os
from dotenv import load_dotenv
import time

load_dotenv()

print("--- Testing API Keys ---")

# 1. Testing Groq
print("\n1. Testing Groq...")
try:
    from groq import Groq
    key = os.getenv("GROQ_API_KEY")
    if not key:
        print("❌ GROQ_API_KEY not found in .env")
    else:
        client = Groq(api_key=key)
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello",}],
            model="llama-3.3-70b-versatile",
        )
        print(f"✅ Groq Success! Response: {chat_completion.choices[0].message.content[:50]}...")
except Exception as e:
    print(f"❌ Groq Failed: {e}")

# 2. Testing Cohere
print("\n2. Testing Cohere...")
try:
    import cohere
    key = os.getenv("COHERE_API_KEY")
    if not key:
        print("❌ COHERE_API_KEY not found in .env")
    else:
        co = cohere.Client(key)
        # Using a more recent model
        response = co.chat(message="Hello", model="command-r-08-2024")
        print(f"✅ Cohere Success! Response: {response.text[:50]}...")
except Exception as e:
    print(f"❌ Cohere Failed: {e}")

# 3. Testing Gemini
print("\n3. Testing Gemini...")
try:
    import google.generativeai as genai
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        print("❌ GEMINI_API_KEY not found in .env")
    else:
        genai.configure(api_key=key)
        # Trying older stable model
        model = genai.GenerativeModel('gemini-1.0-pro')
        response = model.generate_content("Hello")
        print(f"✅ Gemini Success! Response: {response.text[:50]}...")
except Exception as e:
    print(f"❌ Gemini Failed: {e}")
