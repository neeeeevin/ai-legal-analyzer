import pdfplumber
import re

from backend.compression import compress_clauses
from backend.graph_builder import build_graph
from backend.retriever import Retriever

retriever_instance = None


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_clause(text):

    text = re.sub(r"[-–—]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    # Remove weird broken starts
    text = re.sub(r"^\d+\.\s*", "", text)

    # Fix grammar a bit
    text = text.replace(" ,", ",").replace(" .", ".")

    text = text.strip()

    # Capitalize
    if text:
        text = text[0].upper() + text[1:]

    return text

def split_into_clauses(text):
    sentences = re.split(r'(?<=[.;:])\s+', text)

    clauses = []
    current = ""

    for sent in sentences:
        if len(current) + len(sent) < 300:
            current += " " + sent
        else:
            clauses.append(current.strip())
            current = sent

    if current:
        clauses.append(current.strip())

    return clauses


def process_document(file):
    global retriever_instance

    clauses_with_meta = []

    with pdfplumber.open(file.name) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()

            if not text:
                continue

            text = clean_text(text)
            clauses = split_into_clauses(text)

            for clause in clauses:
                if len(clause) > 40:
                    clauses_with_meta.append({
                        "text": normalize_clause(clause),
                        "page": page_num
                    })

    texts = [c["text"] for c in clauses_with_meta]

    filtered = compress_clauses(texts)

    final_results = []
    for f in filtered:
        for c in clauses_with_meta:
            if f["text"] == c["text"]:
                f["page"] = c["page"]
                break
        final_results.append(f)

    graph = build_graph([r["text"] for r in final_results])
    retriever_instance = Retriever([r["text"] for r in final_results], graph)

    return final_results


def query_system(query):
    global retriever_instance

    if retriever_instance is None:
        return []

    results = retriever_instance.query(query)
    return [{"clause": r} for r in results]