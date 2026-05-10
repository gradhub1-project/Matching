import os
import json
import numpy as np
import faiss
from flask import Flask, request, jsonify
from google import genai
from json import JSONDecoder

# ── Config ──────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
JSONL_PATH     = "project_ideas_60_EN.jsonl"
EMBED_MODEL    = "models/gemini-embedding-001"
JUDGE_MODEL    = "gemini-2.5-flash-lite"          
TOP_K          = 3

client = genai.Client(api_key=GEMINI_API_KEY)

# ── Gemini error helper ─────────────────────────────────────────────
def gemini_error_response(exc, default_code=503):
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None) or default_code
    try:
        code = int(code)
    except Exception:
        code = default_code
    msg = str(exc).strip() or "Gemini request failed"
    return jsonify({"error": {"code": code, "message": msg}}), code

# ── Load DB ─────────────────────────────────────────────────────────
projects = [json.loads(line) for line in open(JSONL_PATH, "r", encoding="utf-8")]

def payload_project(p):
    # include Tags so short keyword queries can match on them too
    return (
        f"Title: {p.get('Project_Title','')}\n"
        f"Domain: {p.get('Domain','')}\n"
        f"Tags: {p.get('Tags','')}\n"
        f"Abstract: {p.get('Abstract','')}"
    )

def embed(texts, batch=16):
    vecs = []
    for i in range(0, len(texts), batch):
        r = client.models.embed_content(model=EMBED_MODEL, contents=texts[i:i+batch])
        vecs.append(np.array([e.values for e in r.embeddings], dtype=np.float32))
    return np.vstack(vecs)

# ── Build FAISS index safely ────────────────────────────────────────
index = None
index_error = None

try:
    texts = [payload_project(p) for p in projects]
    X = embed(texts)
    faiss.normalize_L2(X)
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    print(f"[OK] Indexed {index.ntotal} projects | dim={X.shape[1]}")
except Exception as e:
    index_error = e
    print(f"[ERROR] Failed to build index: {e}")

# ── Retrieval ───────────────────────────────────────────────────────
def retrieve_candidates(query: str, k=TOP_K):
    """
    Accepts any free-form query: keywords, a title, a sentence, or a full abstract.
    Embeds it directly — the embedding model handles short/long inputs well.
    """
    if index is None:
        raise index_error or Exception("FAISS index is not available")
    if index.ntotal == 0 or k <= 0:
        return []

    q = embed([query])
    faiss.normalize_L2(q)
    k_eff = min(k, index.ntotal)
    scores, idxs = index.search(q, k_eff)

    cands = []
    for s, i in zip(scores[0], idxs[0]):
        if int(i) < 0:
            continue
        p = projects[int(i)]
        cands.append({
            "Idea_ID":       p.get("Idea_ID", ""),
            "Project_Title": p.get("Project_Title", ""),
            "Domain":        p.get("Domain", ""),
            "Tags":          p.get("Tags", ""),
            "Abstract":      p.get("Abstract", ""),
            "retrieval":     float(s)
        })
    return cands

# ── Gemini judge ────────────────────────────────────────────────────
def _extract_first_json(text: str):
    text = (text or "").strip()
    start = text.find("{")
    if start == -1:
        return None
    dec = JSONDecoder()
    try:
        obj, _ = dec.raw_decode(text[start:])
        return obj
    except Exception:
        return None

def gemini_decide(query: str, candidates: list):
    """
    Given the user's raw query (anything from keywords to full abstract)
    and the top retrieved candidates, pick the single best match and explain
    how its abstract covers what the user is trying to do.
    """
    if not candidates:
        return {
            "project_title":     "",
            "domain":            "",
            "similarity_gemini": 0,
            "reason":            "No similar projects found."
        }

    cand_text = "\n\n".join([
        f"CANDIDATE {i+1}:\n"
        f"ID: {c['Idea_ID']}\n"
        f"Title: {c['Project_Title']}\n"
        f"Domain: {c['Domain']}\n"
        f"Tags: {c['Tags']}\n"
        f"Abstract: {c['Abstract']}"
        for i, c in enumerate(candidates)
    ])

    prompt = f"""You are a project advisor. A student gave you the following input — it could be
just keywords, a short title, or a full abstract describing what they want to build.

STUDENT INPUT:
\"{query}\"

Below are the most semantically similar existing projects retrieved from a database.
Your job is to:
1. Pick the SINGLE best matching project.
2. Score its relevance to the student's input (0-100).
3. Write a short reason (1-2 sentences) explaining how that project's abstract
   relates to what the student is trying to do — phrase it as helpful advice,
   e.g. "This project covers X which is directly relevant to your idea of Y..."

{cand_text}

Return ONE JSON object ONLY (no markdown, no extra text) with exactly these keys:
project_title, domain, similarity_gemini (integer 0-100), reason.
"""

    r = client.models.generate_content(model=JUDGE_MODEL, contents=prompt)
    txt = (r.text or "").strip()
    obj = _extract_first_json(txt)

    if not obj:
        return {
            "project_title":     "",
            "domain":            "",
            "similarity_gemini": 0,
            "reason":            txt[:400]
        }

    return {
        "project_title":     obj.get("project_title", ""),
        "domain":            obj.get("domain", ""),
        "similarity_gemini": int(obj.get("similarity_gemini", 0) or 0),
        "reason":            obj.get("reason", "")
    }

# ── Flask app ───────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/", methods=["GET"])
def health():
    if index is None:
        code = getattr(index_error, "status_code", None) or getattr(index_error, "code", None) or 503
        try:
            code = int(code)
        except Exception:
            code = 503
        return jsonify({
            "status": "error",
            "projects_indexed": 0,
            "error": {"code": code, "message": str(index_error).strip() or "Failed to build index"}
        }), code

    return jsonify({"status": "ok", "projects_indexed": index.ntotal}), 200

@app.route("/match", methods=["POST"])
def match():
    data = request.get_json(force=True, silent=True) or {}

    # ── Accept single 'query' field (keywords / title / abstract / anything)
    # Also backward-compatible with old 'abstract' or 'title' fields
    query = (
        data.get("query") or
        data.get("abstract") or
        data.get("title") or
        ""
    ).strip()

    if not query:
        return jsonify({"error": "'query' field is required."}), 400

    try:
        cands = retrieve_candidates(query, k=TOP_K)
    except Exception as e:
        return gemini_error_response(e)

    if not cands:
        return jsonify({
            "abstract":          query,
            "project_title":     "",
            "domain":            "",
            "similarity_gemini": 0,
            "reason":            "No similar projects found.",
            "is_similar":        False
        }), 200

    try:
        decision = gemini_decide(query, cands[:2])   # top-2 fed to judge, returns best 1
    except Exception as e:
        return gemini_error_response(e)

    sim = int(decision.get("similarity_gemini", 0) or 0)

    # ── Exact same response shape as before ─────────────────────────
    return jsonify({
        "abstract":          query,
        "project_title":     decision.get("project_title", ""),
        "domain":            decision.get("domain", ""),
        "similarity_gemini": sim,
        "reason":            decision.get("reason", ""),
        "is_similar":        sim >= 50
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
