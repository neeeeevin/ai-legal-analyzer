# AI Legal Document Analyzer

AI Legal Document Analyzer is an AI-based system that analyzes legal PDF documents, identifies critical clauses, and explains them in simple language. It also allows users to ask questions about the document.

The project combines machine learning, natural language processing, graph-based retrieval, and large language models to make complex legal documents easier to understand.

---

# Features

- Upload legal PDF documents
- AI-generated plain-English summary
- Automatic detection of critical clauses
- Clause explanations in simple language
- Clause risk level estimation
- Question answering based on the document
- Interactive web interface

---

# System Architecture

```
PDF Upload
   │
   ▼
Text Extraction (pdfplumber)
   │
   ▼
Clause Splitting & Cleaning
   │
   ▼
ML Classification (DistilBERT)
   │
   ▼
Clause Compression (Risk Filtering)
   │
   ▼
Graph Construction
   │
   ▼
Vector Index (FAISS)
   │
   ▼
Query Retrieval
   │
   ▼
LLM Explanation + Summary
   │
   ▼
Gradio Web Interface
```

---

# Tech Stack

## Backend
- Python
- PyTorch
- Transformers

## NLP and AI
- DistilBERT Clause Classifier
- Sentence Embeddings
- Large Language Models

## Libraries
- pdfplumber
- networkx
- faiss
- sentence-transformers
- gradio
- python-dotenv

## AI APIs
- Google Gemini (document summary)
- OpenRouter (clause explanation)

---

# Project Structure

```
AI-Legal-Analyzer/
│
├── app.py                # Gradio web interface
├── config.json
├── requirements.txt
│
├── backend/
│   ├── main.py           # Document processing pipeline
│   ├── compression.py    # Clause filtering and risk scoring
│   ├── ml_model.py       # DistilBERT clause classifier
│   ├── graph_builder.py  # Clause relationship graph
│   ├── retriever.py      # Semantic search
│   ├── summarizer.py     # Gemini document summary
│   ├── explainer.py      # Clause explanation
│   └── train_model.py    # Model training
│
├── models/
│   └── legal_classifier
│
└── tokenizer
    ├── tokenizer.json
    └── tokenizer_config.json
```

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/yourusername/AI-Legal-Analyzer.git
cd AI-Legal-Analyzer
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Create `.env` file

```
GEMINI_API_KEY=your_gemini_key
OPENROUTER_API_KEY=your_openrouter_key
```

## 4. Run the Application

```bash
python app.py
```

The interface will open automatically in your browser.

---

# Usage

1. Upload a legal PDF document  
2. Click **Analyze Document**

The system will:

- Extract clauses
- Identify high-impact clauses
- Generate a simple summary
- Explain important sections

Example questions you can ask:

```
What are the termination conditions?
What are the penalties in this contract?
What are the payment obligations?
```

---

# Example Output

## Summary

- This document describes the agreement between two parties.
- It explains payment obligations and deadlines.
- It includes termination conditions and penalties.
- Liability and dispute resolution clauses are defined.

## Clause Explanation

Clause:
```
The agreement may be terminated by either party with 30 days written notice.
```

Meaning:
```
This clause allows both parties to end the agreement by giving written notice 30 days in advance.
```

Risk Level: **Medium**

---

# Future Improvements

- Multi-document comparison
- Legal clause highlighting in PDF
- Risk heatmap visualization
- Improved legal reasoning models
- GPU optimization
- DOCX and TXT support

---

# Author

Developed as an AI/NLP project for legal document understanding.
