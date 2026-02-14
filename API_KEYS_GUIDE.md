# 🚀 Quick Start Guide - Get FREE AI API Keys

This app supports **5 AI models**. Here's how to get FREE API keys:

## ⚡ Groq (RECOMMENDED - Fastest & Free!)

**Why?** Super fast responses, completely free, uses Llama 3.3 70B

1. Visit: https://console.groq.com/
2. Sign up (free account)
3. Click "API Keys" → "Create API Key"
4. Copy the key
5. Add to `.env`:
   ```
   GROQ_API_KEY=gsk_your_key_here
   ```

## 🌟 Cohere (Free Tier Available)

**Why?** Enterprise-quality AI, good free tier

1. Visit: https://dashboard.cohere.com/
2. Sign up (free)
3. Go to API Keys section
4. Copy your key
5. Add to `.env`:
   ```
   COHERE_API_KEY=your_key_here
   ```

## 🤗 Hugging Face (Free Inference API)

**Why?** Open-source models, completely free

1. Visit: https://huggingface.co/settings/tokens
2. Create account
3. Click "New token"
4. Copy the token
5. Add to `.env`:
   ```
   HUGGINGFACE_API_KEY=hf_your_token_here
   ```

## 🧠 Google Gemini (You already have this!)

**Status:** ✅ Already configured in your `.env`

---

## 🎯 Recommended Setup

**Start with Groq** - it's the fastest and easiest:

```bash
# 1. Get your FREE Groq API key from console.groq.com
# 2. Add it to .env
# 3. Restart your Streamlit app:
streamlit run app.py
```

You should see **"⚡ Groq (Fast & Free)"** in the dropdown!

---

## 💰 Cost Comparison

| Model | Cost | Speed | Quality |
|-------|------|-------|---------|
| Groq | 🆓 FREE | ⚡⚡⚡ | ⭐⭐⭐⭐ |
| Cohere | 🆓 FREE (limited) | ⚡⚡ | ⭐⭐⭐⭐ |
| HuggingFace | 🆓 FREE (rate limits) | ⚡ | ⭐⭐⭐ |
| Gemini | 🆓 FREE tier | ⚡⚡ | ⭐⭐⭐⭐ |
| Grok | 💰 Paid | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ |

**All free options are production-ready!** 🎉
