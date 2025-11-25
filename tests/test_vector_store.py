"""Integration test for the VectorStore (Chroma + SentenceTransformers)."""

from src.core.types import Chunk
from src.core.vector_store import VectorStore


def test_vector_store_search_returns_expected_chunk(tmp_path):
    """
    Ensure that a semantically identical query surfaces the correct chunk with a high similarity score.
    """
    persist_dir = tmp_path / "chroma_db"
    store = VectorStore(persist_directory=persist_dir, collection_name="vector_store_test")
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

    results = store.search(
        "Photosynthesis is how plants turn sunlight into energy through a chemical process.",
        k=2,
    )

    assert results, "Search should return at least one result."
    top_result = results[0]
    assert top_result.chunk.chunk_id == "chunk-photosynthesis"
    assert top_result.chunk.doc_id == "doc-1"
    assert top_result.similarity_score > 0.95
