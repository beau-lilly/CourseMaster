"""Integration test for the VectorStore (Chroma + SentenceTransformers)."""

from tests.helpers import HashEmbeddings
from src.core.types import Chunk
from src.core.vector_store import VectorStore, VectorSearchResult


def test_vector_store_search_returns_expected_chunk(tmp_path):
    """
    Ensure that a semantically identical query surfaces the correct chunk with a high similarity score.
    """
    persist_dir = tmp_path / "chroma_db"
    store = VectorStore(
        persist_directory=persist_dir,
        collection_name="vector_store_test",
        embedding_function=HashEmbeddings(),
    )
    store.reset()  # Clean slate for the test collection

    target_text = "Photosynthesis is the process plants use to convert sunlight into chemical energy."
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

    results = store.search(target_text, k=2)

    assert results, "Search should return at least one result."
    top_result = results[0]
    assert top_result.chunk.chunk_id == "chunk-photosynthesis"
    assert top_result.chunk.doc_id == "doc-1"
    assert top_result.similarity_score > 0.95


def test_vector_store_deduplicates_hits():
    """
    Duplicate chunk bodies should not be returned multiple times.
    """
    hits = [
        VectorSearchResult(
            chunk=Chunk(
                chunk_id="chunk-1",
                doc_id="doc-1",
                chunk_text="Repeat me",
                chunk_index=0,
            ),
            similarity_score=0.99,
        ),
        VectorSearchResult(
            chunk=Chunk(
                chunk_id="chunk-2",
                doc_id="doc-2",
                chunk_text="Repeat   me",
                chunk_index=1,
            ),
            similarity_score=0.97,
        ),
        VectorSearchResult(
            chunk=Chunk(
                chunk_id="chunk-3",
                doc_id="doc-3",
                chunk_text="Different chunk",
                chunk_index=0,
            ),
            similarity_score=0.5,
        ),
    ]

    deduped = VectorStore._deduplicate_hits(hits, limit=3)
    assert len(deduped) == 2
    assert deduped[0].chunk.chunk_id == "chunk-1"
    assert deduped[1].chunk.chunk_id == "chunk-3"


def test_vector_store_allows_exam_scoping(tmp_path):
    """
    Searches should honor an allowed_doc_ids filter.
    """
    persist_dir = tmp_path / "filtered_chroma_db"
    store = VectorStore(
        persist_directory=persist_dir,
        collection_name="vector_store_filter_test",
        embedding_function=HashEmbeddings(),
    )
    store.reset()

    target_text = "Photosynthesis is the process plants use to convert sunlight into chemical energy."
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

    # Restrict to the astronomy doc; the biology hit should be filtered out.
    results = store.search(
        target_text,
        k=2,
        allowed_doc_ids=["doc-2"],
    )

    assert results, "Search should still return results even when filtered."
    assert all(res.chunk.doc_id == "doc-2" for res in results)
