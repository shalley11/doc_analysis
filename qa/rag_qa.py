def qa_prompt(q, ctx):
    return f"""
Answer using only the context.
If not found, say "Not found in document".

Question:
{q}

Context:
{ctx}
"""
