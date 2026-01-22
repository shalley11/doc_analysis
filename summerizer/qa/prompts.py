"""Prompt templates for Q&A with citations."""

QA_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided document context.

Instructions:
1. Answer the question using ONLY the information from the provided context
2. If the context doesn't contain enough information to answer, say so clearly
3. Always cite your sources using the format [Source: PDF_NAME, Page X]
4. Be concise but thorough
5. If multiple sources support your answer, cite all of them"""

QA_USER_PROMPT = """Context from documents:
{context}

---

Question: {question}

Please answer the question based on the context above. Include citations for each piece of information using [Source: PDF_NAME, Page X] format."""

SUMMARY_SYSTEM_PROMPT = """You are a helpful assistant that summarizes document content.

Instructions:
1. Create a clear, well-structured summary
2. Include key points and main ideas
3. Cite sources using [Source: PDF_NAME, Page X] format
4. Organize information logically"""

SUMMARY_USER_PROMPT_BRIEF = """Context from documents:
{context}

---

Provide a brief summary (2-3 paragraphs) of the document content. Include citations."""

SUMMARY_USER_PROMPT_DETAILED = """Context from documents:
{context}

---

Provide a detailed summary of the document content. Cover all major topics and key details. Include citations."""

SUMMARY_USER_PROMPT_BULLETS = """Context from documents:
{context}

---

Summarize the document content as bullet points. Group related points together. Include citations."""


def format_context_with_citations(chunks: list) -> str:
    """
    Format retrieved chunks with source information for the prompt.

    Each chunk is formatted with its source (PDF name, page number) for easy citation.
    """
    formatted_parts = []

    for i, chunk in enumerate(chunks, 1):
        pdf_name = chunk.get("pdf_name", "Unknown")
        page_no = chunk.get("page_no", 0) + 1  # Convert to 1-indexed
        content_type = chunk.get("content_type", "text")
        text = chunk.get("text", "")

        # Add source header
        source_info = f"[Source {i}: {pdf_name}, Page {page_no}]"
        if content_type != "text":
            source_info += f" ({content_type})"

        formatted_parts.append(f"{source_info}\n{text}")

    return "\n\n---\n\n".join(formatted_parts)


def build_qa_prompt(question: str, chunks: list) -> tuple:
    """
    Build the full prompt for Q&A.

    Returns:
        tuple: (system_prompt, user_prompt)
    """
    context = format_context_with_citations(chunks)
    user_prompt = QA_USER_PROMPT.format(context=context, question=question)
    return QA_SYSTEM_PROMPT, user_prompt


def build_summary_prompt(chunks: list, summary_type: str = "brief") -> tuple:
    """
    Build the full prompt for summarization.

    Args:
        chunks: List of retrieved chunks
        summary_type: "brief", "detailed", or "bullets"

    Returns:
        tuple: (system_prompt, user_prompt)
    """
    context = format_context_with_citations(chunks)

    if summary_type == "detailed":
        user_prompt = SUMMARY_USER_PROMPT_DETAILED.format(context=context)
    elif summary_type == "bullets":
        user_prompt = SUMMARY_USER_PROMPT_BULLETS.format(context=context)
    else:
        user_prompt = SUMMARY_USER_PROMPT_BRIEF.format(context=context)

    return SUMMARY_SYSTEM_PROMPT, user_prompt
