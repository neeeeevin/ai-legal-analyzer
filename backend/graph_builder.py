import networkx as nx
import re
import numpy as np
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")


def extract_section_refs(text):
    pattern = r"(Section|Sec\.?)\s*(\d+[A-Za-z]?)"
    matches = re.findall(pattern, text)

    refs = []
    for m in matches:
        try:
            num = int(re.findall(r"\d+", m[1])[0])
            refs.append(num)
        except:
            continue

    return refs


def build_graph(clauses):

    G = nx.DiGraph()

    if not clauses:
        return G

    clauses = clauses[:120]

    for i, clause in enumerate(clauses):
        G.add_node(i, text=clause)

    for i, clause in enumerate(clauses):
        refs = extract_section_refs(clause)

        for ref in refs:
            idx = ref - 1
            if 0 <= idx < len(clauses):
                G.add_edge(i, idx, type="reference", weight=1.0)

    embeddings = model.encode(
        clauses,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True
    )

    sim_matrix = util.cos_sim(embeddings, embeddings)

    for i in range(len(clauses)):
        top_k = np.argsort(-sim_matrix[i].cpu().numpy())[:8]

        for j in top_k:
            if i == j:
                continue

            sim = float(sim_matrix[i][j])

            if sim > 0.78:
                G.add_edge(i, j, type="semantic", weight=sim)

    return G