import json
from pathlib import Path
from typing import Any


def load_documents(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Knowledge base file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        documents = json.load(file)

    if not isinstance(documents, list):
        raise ValueError("docs.json must contain a list of documents")

    required_fields = {"title", "source", "content"}
    for document in documents:
        missing = required_fields - set(document)
        if missing:
            raise ValueError(f"Document is missing required fields: {missing}")

    return documents
