import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


class Retriever:
    def __init__(self, texts, graph=None):
        self.texts = texts
        self.graph = graph
        self.cache = {}

        self.embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True
        )

        dim = self.embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dim)
        self.index.add(np.array(self.embeddings).astype("float32"))

    def query(self, query_text, k=5):

        if query_text in self.cache:
            query_vec = self.cache[query_text]
        else:
            query_vec = model.encode([query_text], normalize_embeddings=True).astype("float32")
            self.cache[query_text] = query_vec

        D, I = self.index.search(query_vec, k)

        results = []
        seen = set()

        for idx, score in zip(I[0], D[0]):
            if idx < len(self.texts):
                results.append({
                    "text": self.texts[idx],
                    "score": float(score)
                })
                seen.add(idx)

        if self.graph is not None:
            for idx, base_score in zip(I[0], D[0]):
                for neighbor in self.graph.neighbors(idx):
                    if neighbor not in seen:
                        weight = self.graph[idx][neighbor].get("weight", 0.5)
                        score = base_score * 0.7 + weight * 0.3

                        results.append({
                            "text": self.texts[neighbor],
                            "score": float(score)
                        })
                        seen.add(neighbor)

        results.sort(key=lambda x: x["score"], reverse=True)

        return [r["text"] for r in results[:k]]