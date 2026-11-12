import json, re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path(__file__).parent / "data" / "knowledge_base.json"

EMERGENCY_PATTERNS = [
    r"\b(chest pain|chest pressure)\b",
    r"\b(can't breathe|cannot breathe|difficulty breathing|severe shortness of breath)\b",
    r"\b(unconscious|not responding|fainted and not waking)\b",
    r"\b(severe bleeding|bleeding heavily)\b",
    r"\b(signs of stroke|face drooping|sudden weakness on one side|sudden trouble speaking)\b",
    r"\b(seizure|convulsion)\b",
    r"\b(overdose|poisoning)\b",
]

def load_knowledge():
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)

class HospitalAssistant:
    def __init__(self, articles):
        self.articles = articles
        self.texts = [a["title"] + " " + a["category"] + " " + a["content"] + " " + " ".join(a.get("keywords", [])) for a in articles]
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words="english", sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(self.texts)
        self.embedder = None
        self.embedding_matrix = None
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            self.embedding_matrix = self.embedder.encode(self.texts, normalize_embeddings=True)
        except Exception:
            pass

    def _emergency(self, query):
        return any(re.search(p, query, re.I) for p in EMERGENCY_PATTERNS)

    def _tfidf(self, query, top_k=3):
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix)[0]
        order = scores.argsort()[::-1][:top_k]
        return [self._item(i, float(scores[i])) for i in order if scores[i] > 0]

    def _embedding(self, query, top_k=3):
        if self.embedder is None:
            return []
        import numpy as np
        q = self.embedder.encode([query], normalize_embeddings=True)[0]
        scores = self.embedding_matrix @ q
        order = np.argsort(scores)[::-1][:top_k]
        return [self._item(int(i), float(scores[i])) for i in order]

    def _item(self, idx, score):
        a = self.articles[idx]
        return {"id": a["id"], "title": a["title"], "snippet": a["content"][:230] + ("…" if len(a["content"]) > 230 else ""), "content": a["content"], "score": score}

    def compare(self, query):
        tf = self._tfidf(query)
        emb = self._embedding(query)
        # Hybrid is a transparent reciprocal-rank blend; no generated medical claims.
        ranks = {}
        for method, items in [("TF-IDF", tf), ("Embeddings", emb)]:
            for rank, item in enumerate(items, 1):
                ranks.setdefault(item["id"], {"item": item, "score": 0.0})
                ranks[item["id"]]["score"] += 1.0 / (60 + rank)
        hybrid = sorted(ranks.values(), key=lambda x: x["score"], reverse=True)
        hy = [{**x["item"], "score": x["score"]} for x in hybrid[:3]]
        return {"Classical · TF-IDF": tf, "Modern · embeddings": emb, "Hybrid · reciprocal-rank fusion": hy}

    def answer(self, query, mode="Hybrid"):
        if self._emergency(query):
            return {
                "text": "**Please seek immediate human help.** The words in your message may describe a medical emergency. Alert nearby hospital staff now or contact your local emergency number. Do not wait for this chatbot or continue using it to assess symptoms. I cannot determine what is happening medically.",
                "sources": [], "method": "Safety hand-off", "score": 1.0
            }
        if mode == "Classical TF-IDF":
            results, method = self._tfidf(query), "TF-IDF"
        elif mode == "Modern embeddings (optional)":
            results, method = self._embedding(query), "Sentence-transformer embeddings"
            if not results:
                results, method = self._tfidf(query), "TF-IDF fallback (embeddings unavailable)"
        else:
            results = self.compare(query)["Hybrid · reciprocal-rank fusion"]
            method = "Hybrid reciprocal-rank fusion"
        if not results or results[0]["score"] < 0.035:
            return {
                "text": "I could not find a sufficiently relevant answer in the approved demo knowledge base, so I don't want to guess. Please contact the hospital information desk or the relevant department. If this is a medical concern, speak directly with a qualified healthcare professional.",
                "sources": [], "method": method + " · low-match fallback", "score": results[0]["score"] if results else 0.0
            }
        best = results[0]
        answer = (
            f"**{best['title']}**\n\n{best['content']}\n\n"
            "**What I used:** This answer is copied from the controlled demo knowledge base, not generated from outside medical sources. "
            "For hospital-specific rules, confirm with hospital staff."
        )
        return {"text": answer, "sources": [best["id"]], "method": method, "score": best["score"]}
