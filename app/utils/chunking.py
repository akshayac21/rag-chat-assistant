from typing import Any


def chunk_text(text: str, chunk_size: int = 650, overlap: int = 120) -> list[str]:
    cleaned = " ".join(text.split())
    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(cleaned):
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for document in documents:
        questions = document.get("questions", [])
        question_text = " ".join(questions)
        for index, chunk in enumerate(chunk_text(document["content"])):
            base_metadata = {
                "title": document["title"],
                "source": document["source"],
                "chunk_id": index,
                "text": chunk,
            }
            chunks.append(
                base_metadata
                | {
                    "embedding_text": (
                        f"Title: {document['title']}\n"
                        f"Source: {document['source']}\n"
                        f"Common questions: {question_text}\n"
                        f"Content: {chunk}"
                    ),
                    "vector_type": "content",
                }
            )

            for question in questions:
                chunks.append(
                    base_metadata
                    | {
                        "embedding_text": question,
                        "vector_type": "question_anchor",
                    }
                )
    return chunks
