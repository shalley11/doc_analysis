"""
Prompt templates for summarization.
Gemma-3 optimized with multilingual support and adaptive length.
"""

# =========================
# Batch summarization prompt
# =========================
BATCH_SUMMARY_PROMPT = """
You are summarizing PART {batch_index} of {total_batches} of a larger document.

IMPORTANT RULES:
- If the content is not in English, first translate it internally into English
- Use ONLY the information present in the content
- Do NOT add assumptions, interpretations, or external knowledge
- If information is missing or unclear, state it explicitly
- Preserve factual accuracy (numbers, dates, metrics, names)

CONTENT (text and tables converted to text):
{content}

TASK:
Generate a concise, self-contained English summary of approximately {word_count} words that:
- Captures the main ideas and key findings in this part
- Preserves important facts, figures, and results
- Clearly reflects the meaning of any tables (trends, comparisons, totals)
- Maintains enough context to be merged with summaries from other parts

STYLE GUIDELINES:
- Clear, neutral, and factual tone
- No redundancy or filler
- No references to section numbers or document structure

OUTPUT:
Summary (in English):
"""


# ==================================
# Instructions for different summaries
# ==================================
SUMMARY_TYPE_INSTRUCTIONS = {

    "brief": """
Generate a BRIEF final summary.

RULES:
- If the input is not in English, translate internally before summarizing
- Focus on the single most important idea and outcome
- Keep it as short as possible while remaining complete
- Preserve critical conclusions or facts if present
- Do NOT add background, examples, or assumptions

STYLE:
- Extremely concise
- Clear and factual
""",

    "bulletwise": """
Generate a BULLET-WISE final summary.

RULES:
- If the input is not in English, translate internally before summarizing
- Use bullets to cover all key topics and findings
- Include important facts, figures, or comparisons when relevant
- Avoid redundancy or overlapping bullets
- Exclude minor or repetitive details

STYLE:
- Information-dense bullets
- Start bullets with strong nouns or verbs
""",

    "detailed": """
Generate a DETAILED final summary.

RULES:
- If the input is not in English, translate internally before summarizing
- Cover all major themes and insights across the content
- Integrate information into a coherent and logical narrative
- Preserve important data points, trends, and conclusions
- Avoid unnecessary repetition

STYLE:
- Clear structure and logical flow
- Neutral, professional tone
""",

    "executive": """
You are an experienced executive communications advisor writing for senior leadership
(CXOs, Directors, VPs).

RULES:
- If the input is not in English, translate internally before summarizing
- Use ONLY the information provided in the content
- Do NOT add assumptions, interpretations, or external knowledge
- Focus on outcomes, implications, and decisions rather than operational details

TASK:
Create a concise, high-impact executive summary that:
- Highlights key outcomes, insights, and decisions
- Emphasizes business impact, risks, and opportunities
- Omits technical detail unless essential for understanding

STYLE:
- Professional, neutral, and confident tone
- Clear and structured
- No repetition, no filler, no explanations

OUTPUT FORMAT:
- Short paragraphs or bullet points (whichever best suits the content)
- Executive-ready and easy to scan
"""
}


# =========================
# Final combination prompt
# =========================
FINAL_COMBINE_PROMPT = """
You are combining summaries from different sections of a document into one coherent summary.

IMPORTANT RULES:
- If summaries are not in English, translate internally before combining
- Use ONLY the provided summaries
- Do NOT introduce new information or assumptions

Section Summaries:
{combined_content}

{instruction}

OUTPUT:
Final Summary (in English):
"""


# =========================
# Direct summarization
# =========================
DIRECT_SUMMARY_BASE = """
Analyze the following document content and generate an English summary.

IMPORTANT RULES:
- If the content is not in English, translate internally before summarizing
- Use ONLY the information present in the content
- Do NOT add assumptions or external knowledge

Document Content:
{content}
"""

DIRECT_SUMMARY_INSTRUCTIONS = {

    "brief": """
TASK:
Generate a BRIEF summary focusing on the main idea and key conclusion.

OUTPUT:
Summary (in English):
""",

    "bulletwise": """
TASK:
Generate a BULLET-WISE summary covering all key topics and findings.

OUTPUT:
Bullet Summary (in English):
""",

    "detailed": """
TASK:
Generate a DETAILED summary covering all major themes, findings, and conclusions.

OUTPUT:
Summary (in English):
""",

    "executive": """
You are an experienced executive communications advisor writing for senior leadership.

TASK:
Generate an EXECUTIVE SUMMARY that:
- Focuses on strategic outcomes and implications
- Highlights business impact, risks, and opportunities
- Avoids operational or low-level technical detail

OUTPUT:
Executive Summary (in English):
"""
}


# =========================
# Multi-PDF combination
# =========================
MULTI_PDF_COMBINE_PROMPT = """
You are combining summaries from {total_pdfs} PDF documents.

IMPORTANT RULES:
- If summaries are not in English, translate internally before combining
- Use ONLY the provided summaries
- Do NOT add assumptions or external knowledge

Document Summaries:
{combined_content}

TASK:
Generate a {type_instruction} combined English summary that:
- Synthesizes information across all documents
- Identifies common themes and unique points
- Produces a coherent, unified overview

OUTPUT:
Combined Summary (in English):
"""


# =========================
# Type instructions (adaptive)
# =========================
MULTI_PDF_TYPE_INSTRUCTIONS = {
    "brief": "BRIEF",
    "bulletwise": "BULLET-WISE",
    "detailed": "DETAILED",
    "executive": "EXECUTIVE"
}


# =========================
# Helper functions
# =========================
def get_batch_summary_prompt(content: str, batch_index: int, total_batches: int, word_count: int) -> str:
    """Generate prompt for batch summarization."""
    return BATCH_SUMMARY_PROMPT.format(
        batch_index=batch_index + 1,
        total_batches=total_batches,
        content=content,
        word_count=word_count
    )


def get_final_combine_prompt(summaries: list, summary_type: str) -> str:
    """Generate prompt for combining summaries."""
    combined_content = "\n\n".join(
        f"[Section {i + 1}]: {summary}"
        for i, summary in enumerate(summaries)
    )

    instruction = SUMMARY_TYPE_INSTRUCTIONS.get(
        summary_type, SUMMARY_TYPE_INSTRUCTIONS["detailed"]
    )

    return FINAL_COMBINE_PROMPT.format(
        combined_content=combined_content,
        instruction=instruction
    )


def get_direct_summary_prompt(content: str, summary_type: str) -> str:
    """Generate prompt for direct summarization (small documents)."""
    base = DIRECT_SUMMARY_BASE.format(content=content)
    instruction = DIRECT_SUMMARY_INSTRUCTIONS.get(
        summary_type, DIRECT_SUMMARY_INSTRUCTIONS["detailed"]
    )
    return base + instruction


def get_multi_pdf_combine_prompt(pdf_summaries: dict, summary_type: str) -> str:
    """Generate prompt for combining multiple PDF summaries."""
    combined_content = "\n\n".join(
        f"=== {pdf_name} ===\n{summary}"
        for pdf_name, summary in pdf_summaries.items()
    )

    type_instruction = MULTI_PDF_TYPE_INSTRUCTIONS.get(
        summary_type, "DETAILED"
    )

    return MULTI_PDF_COMBINE_PROMPT.format(
        total_pdfs=len(pdf_summaries),
        combined_content=combined_content,
        type_instruction=type_instruction
    )


# =========================
# Summary Refinement Prompts
# =========================

SIMPLE_REFINEMENT_PROMPT = """
You are refining an existing summary based on user feedback.

ORIGINAL SUMMARY:
{original_summary}

USER FEEDBACK:
{user_feedback}

IMPORTANT RULES:
- Modify the summary according to the user's feedback
- Keep the same general structure unless the feedback requests otherwise
- Do NOT add information that was not implied in the original summary
- Maintain accuracy and factual consistency
- If the user asks for more detail but you don't have the source, acknowledge the limitation

TASK:
Generate a refined summary that addresses the user's feedback while preserving the quality and accuracy of the original.

OUTPUT:
Refined Summary:
"""


CONTEXTUAL_REFINEMENT_PROMPT = """
You are refining an existing summary based on user feedback, with access to relevant source content.

ORIGINAL SUMMARY:
{original_summary}

USER FEEDBACK:
{user_feedback}

RELEVANT SOURCE CONTENT (retrieved based on feedback):
{relevant_context}

IMPORTANT RULES:
- Modify the summary according to the user's feedback
- Use the relevant source content to add missing details requested by the user
- Maintain accuracy - only include information from the provided sources
- Keep content not mentioned in the feedback unchanged (unless contradicted by sources)
- Integrate new information smoothly into the existing summary structure

TASK:
Generate a refined summary that:
1. Addresses the user's specific feedback
2. Incorporates relevant details from the source content
3. Maintains coherence and readability

OUTPUT:
Refined Summary:
"""


def get_simple_refinement_prompt(original_summary: str, user_feedback: str) -> str:
    """Generate prompt for simple (summary-only) refinement."""
    return SIMPLE_REFINEMENT_PROMPT.format(
        original_summary=original_summary,
        user_feedback=user_feedback
    )


def get_contextual_refinement_prompt(
    original_summary: str,
    user_feedback: str,
    relevant_context: str
) -> str:
    """Generate prompt for contextual refinement with source chunks."""
    return CONTEXTUAL_REFINEMENT_PROMPT.format(
        original_summary=original_summary,
        user_feedback=user_feedback,
        relevant_context=relevant_context
    )
