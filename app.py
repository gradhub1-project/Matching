import os, json
import numpy as np
import faiss
from flask import Flask, request, jsonify
from google import genai
from json import JSONDecoder

# ── Config ──────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]   # set in Render dashboard
JSONL_PATH     = "project_ideas_60_EN.jsonl"
EMBED_MODEL    = "models/gemini-embedding-001"
JUDGE_MODEL    = "gemini-2.5-flash"
TOP_K          = 5

client = genai.Client(api_key=GEMINI_API_KEY)

# ── Load DB + build FAISS index ─────────────────────────────────────
projects = [json.loads(line) for line in open(JSONL_PATH, "r", encoding="utf-8")]

def payload_project(p):
    return f"Title: {p.get('Project_Title','')}\nDomain: {p.get('Domain','')}\nAbstract: {p.get('Abstract','')}"

def payload_query(title, abstract):
    return f"Title: {title}\nAbstract: {abstract}"

def embed(texts, batch=16):
    vecs = []
    for i in range(0, len(texts), batch):
        r = client.models.embed_content(model=EMBED_MODEL, contents=texts[i:i+batch])
        vecs.append(np.array([e.values for e in r.embeddings], dtype=np.float32))
    return np.vstack(vecs)

texts = [payload_project(p) for p in projects]
X = embed(texts)
faiss.normalize_L2(X)

index = faiss.IndexFlatIP(X.shape[1])
index.add(X)
print(f"[OK] Indexed {index.ntotal} projects | dim={X.shape[1]}")

# ── Retrieval ───────────────────────────────────────────────────────
def retrieve_candidates(title, abstract, k=TOP_K):
    if index.ntotal == 0 or k <= 0:
        return []
    q = embed([payload_query(title, abstract)])
    faiss.normalize_L2(q)
    k_eff = min(k, index.ntotal)
    scores, idxs = index.search(q, k_eff)
    cands = []
    for s, i in zip(scores[0], idxs[0]):
        if int(i) < 0:
            continue
        p = projects[int(i)]
        cands.append({
            "Idea_ID": p.get("Idea_ID", ""),
            "Project_Title": p.get("Project_Title", ""),
            "Domain": p.get("Domain", ""),
            "Abstract": p.get("Abstract", ""),
            "retrieval": float(s)
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

def gemini_decide(title, abstract, candidates):
    if not candidates:
        return {"project_title": "", "domain": "", "similarity_gemini": 0,
                "reason": "No any similar Projects"}

    cand_text = "\n\n".join(
        [f"""CANDIDATE {i+1}:
ID: {c['Idea_ID']}
Project_Title: {c['Project_Title']}
Domain: {c['Domain']}
Abstract: {c['Abstract']}
""" for i, c in enumerate(candidates)]
    )

    prompt = f"""You are evaluating semantic similarity between a NEW project and existing projects.
Score independently (0-100) based ONLY on meaning overlap. Be brief.

NEW PROJECT:
Title: {title}
Abstract: {abstract}

{cand_text}

Return ONE JSON object ONLY (no markdown, no extra text) with keys:
project_title, domain, similarity_gemini (0-100), reason (1-2 sentences).
"""
    r = client.models.generate_content(model=JUDGE_MODEL, contents=prompt)
    txt = (r.text or "").strip()
    obj = _extract_first_json(txt)
    if not obj:
        return {"project_title": "", "domain": "", "similarity_gemini": 0, "reason": txt[:400]}
    return {
        "project_title": obj.get("project_title", ""),
        "domain": obj.get("domain", ""),
        "similarity_gemini": int(obj.get("similarity_gemini", 0)),
        "reason": obj.get("reason", "")
    }

# ── Flask app ───────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "projects_indexed": index.ntotal}), 200

@app.route("/match", methods=["POST"])
def match():
    data = request.get_json(force=True, silent=True) or {}
    title = (data.get("title") or "").strip()
    abstract = (data.get("abstract") or "").strip()

    if not abstract:
        return jsonify({"error": "'abstract' is required."}), 400

    cands = retrieve_candidates(title, abstract, k=TOP_K)

    if not cands:
        return jsonify({
            "abstract": abstract,
            "project_title": "",
            "domain": "",
            "similarity_gemini": 0,
            "reason": "No any similar Projects",
            "is_similar": False
        }), 200

    decision = gemini_decide(title, abstract, cands[:2])
    sim = int(decision.get("similarity_gemini", 0) or 0)

    return jsonify({
        "abstract": abstract,
        "project_title": decision.get("project_title", ""),
        "domain": decision.get("domain", ""),
        "similarity_gemini": sim,
        "reason": decision.get("reason", ""),
        "is_similar": sim >= 50
    }), 200
