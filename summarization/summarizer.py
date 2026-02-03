
@app.post("/summarize")
def summarize(batch_id: str, scope: str, summary_type: str):
    col = Collection(f"batch_{batch_id}")

    if scope == "pdf":
        results = col.query(
            expr="",
            output_fields=["text", "pdf_id"]
        )
        grouped = group_by_pdf(results)
        summaries = {
            pdf: summarize_hierarchically(
                "\n".join(t), summary_type
            )
            for pdf, t in grouped.items()
        }
        return summaries

    # all PDFs
    texts = [r["text"] for r in col.query(expr="", output_fields=["text"])]
    return {
        "summary": summarize_hierarchically(
            "\n".join(texts), summary_type
        )
    }
