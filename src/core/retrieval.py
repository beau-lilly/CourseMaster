"""
Bridge between application-facing questions and the vector database.
"""

from typing import List, Optional

from .database import DatabaseManager
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


def index_problem_context(
    problem_text: str,
    exam_id: str,
    problem_id: str,
    k: int = 5,
    vector_store: Optional[VectorStore] = None,
    db_manager: Optional[DatabaseManager] = None,
    allowed_doc_ids: Optional[list[str]] = None,
) -> List[VectorSearchResult]:
    """
    Precomputes retrieval hits for a problem and logs them to the database.
    No LLM call happens here.
    """
    manager = db_manager or DatabaseManager()
    store = vector_store or VectorStore()

    # Restrict to docs linked to the exam by default
    scoped_doc_ids = allowed_doc_ids or manager.get_document_ids_for_exam(exam_id)
    if not scoped_doc_ids:
        return []

    hits = retrieve_chunks(
        problem_text,
        k=k,
        vector_store=store,
        allowed_doc_ids=scoped_doc_ids,
    )
    for hit in hits:
        manager.log_retrieval(problem_id, hit.chunk.chunk_id, hit.similarity_score)
    return hits
