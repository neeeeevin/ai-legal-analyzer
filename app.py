import gradio as gr
from backend.main import process_document, query_system
from backend.summarizer import generate_summary
from backend.explainer import explain_clause, get_risk_level

SYSTEM_READY = False


# ---------------- ANALYZE ---------------- #

def analyze(file):
    global SYSTEM_READY

    if file is None:
        return "Upload a PDF file.", "", "Idle"

    try:
        results = process_document(file)

        if not results:
            return "No clauses found.", "", "Error"

        texts = [r["text"] for r in results]

        # ✅ Summary
        summary = generate_summary(texts)

        # ✅ Clause + Explanation (LIMITED for speed)
        clause_blocks = []

        for i, r in enumerate(results[:5]):   # 🔥 limit for CPU/API speed

            explanation = explain_clause(r["text"])
            risk = get_risk_level(r["confidence"])

            block = (
                f"🔹 Clause {i+1}:\n"
                f"{r['text']}\n\n"
                f"💡 Meaning:\n{explanation}\n\n"
                f"⚠️ Risk Level: {risk}\n"
                + (f"📄 Page: {r['page']}\n" if r.get("page") else "")
                + f"⚡ Confidence: {round(r['confidence'], 3)}"
            )

            clause_blocks.append(block)

        clause_text = "\n\n-------------------------\n\n".join(clause_blocks)

        SYSTEM_READY = True

        return summary, clause_text, "Completed ✅"

    except Exception as e:
        return f"Error: {str(e)}", "", "Failed ❌"


# ---------------- QUERY ---------------- #

def ask(query):
    global SYSTEM_READY

    if not SYSTEM_READY:
        return "Analyze a document first."

    if not query:
        return "Enter a question."

    try:
        results = query_system(query)

        if not results:
            return "No relevant clauses found."

        return "\n\n".join([f"• {r['clause']}" for r in results[:6]])

    except Exception as e:
        return f"Query error: {str(e)}"


# ---------------- UI ---------------- #

with gr.Blocks() as app:

    gr.Markdown("# ⚖️ AI Legal Analyzer")

    file = gr.File(label="📄 Upload Legal PDF", file_types=[".pdf"])
    btn = gr.Button("Analyze Document")

    status = gr.Textbox(label="📊 Status", interactive=False)

    summary = gr.Textbox(
        label="🧠 Summary (Easy Explanation)",
        lines=10,
        interactive=False
    )

    clauses = gr.Textbox(
        label="⚡ High Impact Clauses (Explained)",
        lines=18,
        interactive=False
    )

    gr.Markdown("## 💬 Ask Questions")

    query = gr.Textbox(label="Enter your question")
    ask_btn = gr.Button("Get Answer")

    answer = gr.Textbox(
        label="📌 Answer",
        lines=8,
        interactive=False
    )

    # ✅ EVENTS (IMPORTANT — you missed this before)

    btn.click(
        analyze,
        inputs=file,
        outputs=[summary, clauses, status]
    )

    ask_btn.click(
        ask,
        inputs=query,
        outputs=answer
    )


app.launch(inbrowser=True)