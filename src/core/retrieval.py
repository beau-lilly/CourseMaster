"""
Bridge between application-facing questions and the vector database.
"""

from typing import List

from .vector_store import VectorSearchResult, VectorStore


def retrieve_chunks(
    question_text: str,
    k: int = 5,
    vector_store: VectorStore | None = None,
    allowed_doc_ids: list[str] | None = None,
) -> List[VectorSearchResult]:
    """
    Single entrypoint for semantic retrieval.

    Args:
        question_text: Raw question text from the application/user.
        k: Number of chunks to return.
        vector_store: Optional VectorStore instance (helps with testing/injection).

    Returns:
        A list of VectorSearchResult objects ordered by similarity.
    """
    if not isinstance(question_text, str):
        raise TypeError("question_text must be a string")
    if k <= 0:
        raise ValueError("k must be a positive integer")

    query = question_text.strip()
    if not query:
        return []

    store = vector_store or VectorStore()
    return store.search(query, k=k, allowed_doc_ids=allowed_doc_ids)
