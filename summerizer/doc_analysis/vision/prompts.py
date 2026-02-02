def table_prompt(text):
    return f"""
Summarize the table.
Explain meaning, trends, key values.

Table:
{text}
"""


def table_image_prompt():
    return """Task: Create a concise summary of the table for document understanding and question answering.

Rules:
- Focus on key patterns, comparisons, totals, ranges, and extremes
- Mention important numerical relationships (highest, lowest, increase, decrease)
- Do NOT repeat all rows or columns
- Do NOT infer reasons or add external knowledge
- Use only information present in the table
- Keep the summary factual and neutral
- Limit the summary to 3-6 sentences

Extract the following for Q&A:

1. **WHAT**: What is this table about? What type of data does it contain?

2. **WHO/ENTITIES**: List key people, organizations, or entities mentioned.

3. **KEY VALUES**: Important numbers, dates, amounts, or statuses.

4. **RELATIONSHIPS**: How entities relate to each other.

5. **KEY FACTS**: 2-3 important facts someone might ask about.

FORMAT:
- Use bullet points
- Include specific names, numbers, and dates
- Write as complete, searchable sentences
- Be specific, avoid vague descriptions"""


def image_prompt():
    return """Task: Create a concise description of this image for document understanding and question answering.

Rules:
- Describe what the image shows (diagram, chart, photo, illustration, etc.)
- Identify key elements, labels, text, or annotations visible
- Note any data, numbers, or measurements shown
- Do NOT infer meaning beyond what is visible
- Use only information present in the image
- Keep the description factual and neutral
- Limit to 3-6 sentences

Extract the following for Q&A:

1. **TYPE**: What kind of image is this? (chart, diagram, photo, flowchart, etc.)

2. **CONTENT**: What does the image show or represent?

3. **KEY ELEMENTS**: Important labels, text, or components visible.

4. **DATA/VALUES**: Any numbers, percentages, or measurements shown.

5. **KEY FACTS**: 2-3 important facts someone might ask about this image.

FORMAT:
- Use bullet points
- Include specific labels, numbers, and text visible
- Write as complete, searchable sentences
- Be specific about what you see"""
