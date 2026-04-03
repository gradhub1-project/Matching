# Project Matcher API

Semantic similarity API for graduate project ideas.  
Uses Gemini embeddings + FAISS for retrieval, Gemini LLM as judge.

## Endpoint

**POST** `/match`

```json
{
  "title": "My project title",
  "abstract": "My project abstract..."
}
```

Returns similarity score (0-100) against 60 indexed projects.

## Deploy on Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Set **Build Command**: `pip install -r requirements.txt`
5. Set **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120`
6. Add environment variable: `GEMINI_API_KEY` = your key
7. Deploy
