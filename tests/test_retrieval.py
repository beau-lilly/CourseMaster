"""Tests for the retrieval bridge that wraps VectorStore.search."""

import pytest

from src.core.retrieval import retrieve_chunks
from src.core.types import Chunk
from src.core.vector_store import VectorStore


class DummyStore:
    def search(self, *args, **kwargs):
        raise AssertionError("search should not be called for empty queries")


def test_retrieve_chunks_uses_vector_store(tmp_path):
    """
    Integration: ensure retrieve_chunks proxies to VectorStore.search.
    """
    persist_dir = tmp_path / "retrieval_chroma_db"
    store = VectorStore(
        persist_directory=persist_dir,
        collection_name="retrieval_test",
    )
    store.reset()

    target_text = (
        "Photosynthesis is the process plants use to convert sunlight into chemical energy."
    )
    chunks = [
        Chunk(
            chunk_id="chunk-photosynthesis",
            doc_id="doc-1",
            chunk_text=target_text,
            chunk_index=0,
        ),
        Chunk(
            chunk_id="chunk-astronomy",
            doc_id="doc-2",
            chunk_text="Stars are massive luminous spheres of plasma held together by gravity.",
            chunk_index=0,
        ),
    ]
    store.add_chunks(chunks)

    results = retrieve_chunks(
        "Photosynthesis is how plants turn sunlight into energy through a chemical process.",
        k=1,
        vector_store=store,
    )

    assert results, "Retrieval should surface at least one chunk."
    top_result = results[0]
    assert top_result.chunk.chunk_id == "chunk-photosynthesis"
    assert top_result.chunk.doc_id == "doc-1"
    assert top_result.similarity_score > 0.95


def test_retrieve_chunks_noop_for_empty_question():
    """
    Empty/whitespace-only questions should not hit the vector store.
    """
    dummy_store = DummyStore()
    assert retrieve_chunks("   ", k=3, vector_store=dummy_store) == []


def test_retrieve_chunks_requires_positive_k():
    with pytest.raises(ValueError):
        retrieve_chunks("Valid question", k=0, vector_store=DummyStore())
