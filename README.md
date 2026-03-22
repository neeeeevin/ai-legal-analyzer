AI Legal Document Analyzer

AI Legal Document Analyzer is an AI-based system that analyzes legal PDF documents, identifies critical clauses, and explains the clauses in simple language. Additionally, the system allows users to pose questions about the legal document. This project utilizes machine learning, natural language processing, graph-based retrieval, and large language models to simplify complex legal documents for users.

Features
Legal PDF document upload
AI-generated simple English summary
Detection of critical clauses
Clause explanation in simple language
Estimation of clause risk level
Question answering based on the legal document
Interactive web interface
System Architecture
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
Tech Stack
Backend
Python
PyTorch
Transformers
NLP and AI
DistilBERT clause classifier
Sentence Embeddings
Large Language Models for explanations
Libraries
pdfplumber
networkx
faiss
sentence-transformers
gradio
python-dotenv
AI APIs
Google Gemini (document summary)
OpenRouter (clause explanation)
Project Structure
AI-Legal-Analyzer/
│
├── app.py                # Gradio web interface
├── config.json
├── requirements.txt
│
├── backend/
│   ├── main.py           # Document processing pipeline
│   ├── models.py         # AI models for classification and explanation
│   ├── utils.py          # Supporting functions
│
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── scripts.js
│
├── data/
│   ├── test_documents/
│   └── test_queries/
│
├── docs/
│   └── Read
│   ├── compression.py    # Clause filtering and risk scoring
│   ├── ml_model.py       # DistilBERT clause classifier
│   ├── graph_builder.py  # Clause relationship graph
│   ├── retriever.py      # Semantic search using FAISS
│   ├── summarizer.py     # Gemini-based document summary
│   ├── explainer.py      # Clause explanation using LLM
│   └── train_model.py    # Model training script
│
├── models/
│   └── legal_classifier  # Fine-tuned DistilBERT model
│
└── tokenizer files
    ├── tokenizer.json
    └── tokenizer_config.json
Installation
1. Clone the repository
git clone https://github.com/yourusername/AI-Legal-Analyzer.git
cd AI-Legal-Analyzer
2. Install dependencies
pip install -r requirements.txt
3. Create a .env file
GEMINI_API_KEY=your_gemini_key
OPENROUTER_API_KEY=your_openrouter_key
4. Run the application
python app.py

The interface will open automatically in your browser.

Usage
Upload a legal PDF document.
Click Analyze Document.

The system will:

Extract clauses
Identify high-impact clauses
Generate a simple summary
Explain important sections
Ask questions such as:
What are the termination conditions?
What are the penalties in this contract?
What are the payment obligations?
Example Output
Summary
• This document describes the agreement between two parties.
• It explains payment obligations and deadlines.
• It includes termination conditions and penalties.
• Liability and dispute resolution clauses are defined.
Clause Explanation
Clause:
The agreement may be terminated by either party with 30 days written notice.

Meaning:
This clause allows both parties to end the agreement by giving written notice 30 days in advance.

Risk Level: Medium
Future Improvements
Multi-document comparison
Legal clause highlighting in PDF
Risk heatmap visualization
Better legal reasoning models
GPU optimization
DOCX and TXT support