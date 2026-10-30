"""Small RAG layer: retrieve fraud-policy passages and explain a flagged transaction.

Embeddings: sentence-transformers (all-MiniLM-L6-v2) if installed, else TF-IDF (fallback).
Index: FAISS (inner product on L2-normalised vectors = cosine similarity).
Generator: local Ollama LLM if reachable, else a deterministic template (clearly labelled).
The policy documents in docs/policies are SAMPLE documents written for this demo.
"""
from pathlib import Path

import faiss
import numpy as np

POLICY_DIR = Path(__file__).resolve().parents[1] / "docs" / "policies"


def _chunks():
    out = []
    for f in sorted(POLICY_DIR.glob("*.md")):
        for para in f.read_text().split("\n\n"):
            para = para.strip()
            if para.startswith("#"):   # skip file titles/headings, they are not policy content
                continue
            if len(para) > 40:
                out.append({"source": f.name, "text": para})
    return out


class PolicyRetriever:
    def __init__(self):
        self.chunks = _chunks()
        texts = [c["text"] for c in self.chunks]
        self.backend = "tfidf"
        try:
            from sentence_transformers import SentenceTransformer
            self._st = SentenceTransformer("all-MiniLM-L6-v2")
            vecs = self._st.encode(texts, normalize_embeddings=True).astype("float32")
            self.backend = "sentence-transformers"
        except Exception:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit(texts)
            vecs = self._norm(self._tfidf.transform(texts).toarray().astype("float32"))
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)

    @staticmethod
    def _norm(v):
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.clip(n, 1e-9, None)

    def _embed(self, q):
        if self.backend == "sentence-transformers":
            return self._st.encode([q], normalize_embeddings=True).astype("float32")
        return self._norm(self._tfidf.transform([q]).toarray().astype("float32"))

    def search(self, query: str, k: int = 3):
        scores, idx = self.index.search(self._embed(query), k)
        return [{**self.chunks[i], "score": float(s)} for s, i in zip(scores[0], idx[0]) if i >= 0]


def _llm_answer(prompt: str):
    try:
        import httpx
        r = httpx.post("http://localhost:11434/api/generate",
                       json={"model": "llama3.2", "prompt": prompt, "stream": False}, timeout=60)
        r.raise_for_status()
        return r.json()["response"].strip(), "ollama:llama3.2"
    except Exception:
        return None, "template"


def explain(retriever: PolicyRetriever, prob: float, flagged: bool, drivers: list[dict], amount: float):
    # Plain-language driver description: sign = direction of push, not the feature's raw value
    drv = ", ".join(f"{d['feature']} ({'raises' if d['impact'] > 0 else 'lowers'} risk, {abs(d['impact']):.2f})"
                    for d in drivers)
    # Query depends on the decision, and avoids anonymised feature names (V1..V28) which carry no meaning for retrieval
    query = ("high risk flagged transaction manual review hold customer contact escalation" if flagged
             else "low risk transaction below threshold monitoring no action false alert friction")
    passages = retriever.search(query)
    context = "\n".join(f"[{p['source']}] {p['text']}" for p in passages)
    prompt = (
        "You are a fraud analyst assistant. Use ONLY the information below; do not invent facts.\n"
        f"Decision: {'FLAGGED' if flagged else 'NOT flagged'}. Fraud score: {prob:.3f}. Amount: {amount}.\n"
        f"Top model drivers (each 'raises' or 'lowers' the fraud score; features V1-V28 are anonymised, "
        f"do not guess what they mean): {drv}.\nPolicy excerpts:\n{context}\n\n"
        "In 3 sentences: state the decision and the top drivers using the words 'raises' or 'lowers', "
        "then give the recommended next step from the policy and cite the policy file name."
    )
    text, engine = _llm_answer(prompt)
    if text is None:
        action = ("Route to manual review and consider a temporary hold / customer contact."
                  if flagged else "No action required; continue normal monitoring.")
        text = (f"Fraud probability is {prob:.1%} ({'FLAGGED' if flagged else 'not flagged'}). "
                f"Main model drivers: {drv}. Recommended step: {action} "
                f"Relevant policy: {', '.join(sorted({p['source'] for p in passages}))}.")
    return {"explanation": text, "engine": engine, "retrieval_backend": retriever.backend,
            "sources": [{"source": p["source"], "score": round(p["score"], 3), "text": p["text"]} for p in passages]}