def get_summary_prompt(text: str, summary_type: str | None) -> str:
    summary_type = (summary_type or "brief").lower()

    summary_prompts = {
        "brief": f"""
You are a summarization system.

RULES:
- Use ONLY the information present in the text
- Do NOT assume, infer, or add anything not stated
- Preserve factual accuracy

TASK:
Generate a brief summary that captures the core idea and key outcome.
Exclude minor or repetitive details.

TEXT:
{text}

OUTPUT:
Summary:
""",

        "detailed": f"""
You are a summarization system.

RULES:
- Use ONLY the information present in the text
- Do NOT assume, infer, or add anything not stated
- Preserve important facts, numbers, and conclusions

TASK:
Generate a detailed summary that:
- Covers all major points
- Integrates information logically
- Maintains a clear and coherent flow

TEXT:
{text}

OUTPUT:
Summary:
""",

        "bulletwise": f"""
You are a summarization system.

RULES:
- Use ONLY the information present in the text
- Do NOT assume, infer, or add anything not stated
- Avoid redundancy or overlapping points

TASK:
Generate a bullet-point summary where:
- Each bullet represents one key idea or finding
- Bullets are concise and information-dense
- Important facts or figures are included when present

TEXT:
{text}

OUTPUT:
Bullet Summary:
"""
    }

    return summary_prompts.get(summary_type, summary_prompts["brief"])


def get_prompt(
    task: str,
    text: str,
    summary_type: str | None = None,
    target_language: str | None = "English"
) -> str:
    """
    Generate a prompt for different tasks.
    target_language is used for translation tasks.
    """

    if task == "summary":
        return get_summary_prompt(text, summary_type)

    prompts = {
        "translate": f"""
You are a translation system.

RULES:
- Use ONLY the information present in the text
- Do NOT add, omit, or explain content
- Preserve meaning, tone, and factual accuracy

TASK:
Translate the text into {target_language}.

TEXT:
{text}

OUTPUT:
Translated Text:
""",

        "rephrase": f"""
You are a rewriting system.

RULES:
- Use ONLY the information present in the text
- Do NOT change the meaning
- Do NOT add new information

TASK:
Rephrase the text to improve clarity and readability while preserving meaning.

TEXT:
{text}

OUTPUT:
Rephrased Text:
""",

        "remove_repetitions": f"""
You are a text refinement system.

RULES:
- Use ONLY the information present in the text
- Do NOT add or remove meaning

TASK:
Rewrite the text by removing repetitions and redundancy.
Keep it natural and fluent.

TEXT:
{text}

OUTPUT:
Refined Text:
"""
    }

    if task not in prompts:
        raise ValueError(f"Unsupported task: {task}")

    return prompts[task]


# =========================
# Refinement Prompts
# =========================

def get_refinement_prompt(
    current_result: str,
    user_feedback: str,
    task: str
) -> str:
    """
    Generate a prompt for refining a previous result based on user feedback.

    Args:
        current_result: The current/last result to refine
        user_feedback: User's feedback/instructions for refinement
        task: Original task type (summary, rephrase, etc.)

    Returns:
        Refinement prompt
    """
    task_context = {
        "summary": "summary",
        "rephrase": "rephrased text",
        "translate": "translation",
        "remove_repetitions": "refined text",
        "translate_en": "translation"
    }

    result_type = task_context.get(task, "text")

    return f"""
You are a text refinement assistant.

CURRENT {result_type.upper()}:
{current_result}

USER FEEDBACK:
{user_feedback}

RULES:
- Modify the {result_type} according to the user's feedback
- Maintain the overall meaning and accuracy
- Only change what the user specifically requests
- Keep unchanged parts intact

TASK:
Refine the {result_type} based on the user's feedback.

OUTPUT:
Refined {result_type.title()}:
"""


def get_regenerate_prompt(
    original_text: str,
    user_feedback: str,
    task: str
) -> str:
    """
    Generate a prompt for regenerating output from original text with new instructions.

    Args:
        original_text: The original input text
        user_feedback: User's instructions for regeneration
        task: Original task type (summary, rephrase, etc.)

    Returns:
        Regeneration prompt
    """
    task_instructions = {
        "summary": "Generate a summary of the original text",
        "rephrase": "Rephrase the original text",
        "translate": "Translate the original text",
        "remove_repetitions": "Remove repetitions from the original text",
        "translate_en": "Translate the original text to English"
    }

    base_instruction = task_instructions.get(task, "Process the original text")

    return f"""
You are a text processing assistant.

ORIGINAL TEXT:
{original_text}

USER'S INSTRUCTIONS:
{user_feedback}

TASK:
{base_instruction} while following the user's specific instructions above.

RULES:
- Use ONLY information from the original text
- Apply the user's instructions carefully
- Do NOT add information not present in the original
- Maintain accuracy and coherence

OUTPUT:
Result:
"""


def get_proofreading_prompt(text: str, focus: str = "general") -> str:
    """
    Generate a prompt for proofreading text.

    Args:
        text: Text to proofread
        focus: Focus area (general, grammar, punctuation, clarity)

    Returns:
        Proofreading prompt
    """
    focus_instructions = {
        "general": "Fix all grammar, spelling, punctuation, and clarity issues.",
        "grammar": "Focus on fixing grammatical errors only.",
        "punctuation": "Focus on fixing punctuation errors only.",
        "clarity": "Focus on improving clarity and readability."
    }

    instruction = focus_instructions.get(focus, focus_instructions["general"])

    return f"""
You are a professional proofreader.

RULES:
- Preserve the original meaning
- Do NOT add new content
- Do NOT remove important information
- Make minimal changes necessary

TASK:
{instruction}

TEXT:
{text}

OUTPUT:
Proofread Text:
"""


def get_iterative_refinement_prompt(
    current_result: str,
    user_feedback: str,
    original_text: str = None
) -> str:
    """
    Generate a prompt for iterative refinement with optional original context.

    Args:
        current_result: Current result to refine
        user_feedback: User's refinement instructions
        original_text: Optional original input text for context

    Returns:
        Refinement prompt
    """
    original_context = ""
    if original_text:
        # Include truncated original for context
        truncated = original_text[:500] + "..." if len(original_text) > 500 else original_text
        original_context = f"""
ORIGINAL INPUT (for reference):
{truncated}

"""

    return f"""
You are a text refinement assistant helping to improve content through iteration.
{original_context}
CURRENT VERSION:
{current_result}

USER'S REQUEST:
{user_feedback}

RULES:
- Apply the user's requested changes
- Preserve parts not mentioned in the feedback
- Maintain accuracy and coherence
- Do NOT add information not present in the current version

OUTPUT:
Refined Version:
"""
