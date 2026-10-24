from config import RAG_PROMPT_VERSION


def build_prompt_v1(question, context):
    return f"""Use only the document to answer the question.

Instructions:
- Find evidence about the exact plan or entity named in the question.
- Do not transfer facts from one plan or entity to another.
- First quote the relevant sentence from the document.
- Then answer the question using that evidence.
- If the document does not provide the answer, write:
  Evidence: Not found.
  Answer: I don't know based on the provided document.

Document:
{context}

Question:
{question}

Respond in this format:
Evidence: <exact quote from the document, or Not found>
Answer: <answer supported by the evidence>
"""


def build_prompt_v2(question, context):
    return f"""Answer the question using only facts in the document.

Instructions:
- Identify every entity and requested fact in the question.
- Find supporting evidence separately for each entity and requested fact.
- Evidence may come from different sentences or different document sections.
- For a comparison, combine the supported facts into one answer.
- Keep each fact attached to its original entity.
- Preserve numbers, units, limits, and negation exactly.
- Do not require one sentence to contain the entire answer.
- If only part of the answer is supported, answer that part and clearly
  identify the missing information.
- If none of the requested facts are supported, respond:
  Evidence: Not found.
  Answer: I don't know based on the provided document.

Document:
{context}

Question:
{question}

Respond in this format:
Evidence:
- <entity>: "<supporting quote>"
Repeat evidence lines as needed.
Answer: <answer each requested part using the evidence>
"""


def build_prompt(question, context, version=None):
    if version is None:
        version = RAG_PROMPT_VERSION

    if version == "v1":
        return build_prompt_v1(question, context)

    if version == "v2":
        return build_prompt_v2(question, context)

    raise ValueError(f"Unknown prompt version: {version}")
