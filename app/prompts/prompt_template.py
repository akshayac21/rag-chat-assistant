PROMPT_TEMPLATE = """You are a helpful assistant.
Use ONLY the provided context.

Context:
{retrieved_context}

Conversation History:
{history}

Question:
{user_question}

Answer:
"""


def build_prompt(retrieved_context: str, history: str, user_question: str) -> str:
    return PROMPT_TEMPLATE.format(
        retrieved_context=retrieved_context,
        history=history,
        user_question=user_question,
    )
