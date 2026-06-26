# generation/prompt.py

SYSTEM_PROMPT = """You are an expert AI research assistant with deep knowledge of machine learning, NLP, and computer science research papers.

Your job is to answer questions based ONLY on the provided research paper excerpts.

Rules:
- Answer using ONLY information from the provided context chunks
- Always cite your sources using the paper title and section
- If the context doesn't contain enough information, say so clearly
- Be precise and technical — your audience is researchers and engineers
- Never hallucinate facts, numbers, or claims not in the context
- Format citations as: [Paper Title, Section Name]"""


def build_context(chunks: list[dict], max_tokens: int = 3000) -> str:
    """
    Pack retrieved chunks into a context string for the LLM.
    Respects token budget. Most relevant chunks go first (already ranked).
    """
    context_parts = []
    total_chars   = 0
    char_limit    = max_tokens * 4  # rough chars-per-token estimate

    for i, chunk in enumerate(chunks):
        header = (
            f"[Source {i+1}] "
            f"Paper: {chunk['title']} | "
            f"Section: {chunk['heading']} | "
            f"Year: {chunk['year']} | "
            f"Authors: {', '.join(chunk['authors'][:3])}"
        )
        body   = chunk["text"]
        block  = f"{header}\n{body}"

        if total_chars + len(block) > char_limit:
            break

        context_parts.append(block)
        total_chars += len(block)

    return "\n\n---\n\n".join(context_parts)


def build_messages(query: str, chunks: list[dict]) -> list[dict]:
    """
    Build the full message list for the Ollama API call.
    """
    context = build_context(chunks)

    user_message = f"""Based on the following research paper excerpts, please answer the question.

CONTEXT:
{context}

QUESTION:
{query}

Please provide a detailed answer with citations to the specific papers and sections you used."""

    return [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": user_message},
    ]