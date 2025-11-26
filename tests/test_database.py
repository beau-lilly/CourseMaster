# tests/test_database.py

import uuid

from src.core.database import DatabaseManager
from src.core.types import Chunk
from src.core.vector_store import VectorStore


def test_database_metadata_and_vector_store_round_trip(tmp_path):
    db_path = tmp_path / "test_suite.db"
    vector_dir = tmp_path / "vector_store_db"

    db = DatabaseManager(db_path=str(db_path))
    store = VectorStore(persist_directory=vector_dir, collection_name="db_vector_test")
    store.reset()

    doc = db.add_document("test_doc.pdf", "Full document text.")
    problem = db.add_problem("Test problem text.")

    chunk = Chunk(
        chunk_id=f"chunk_{uuid.uuid4()}",
        doc_id=doc.doc_id,
        chunk_text="This is a test chunk.",
        chunk_index=0,
    )

    # VectorStore handles embeddings + vector persistence.
    store.add_chunks([chunk])

    # DatabaseManager stores only metadata/text.
    db.save_chunks([chunk])

    with db._get_connection() as conn:
        saved_doc = conn.execute(
            "SELECT original_filename FROM documents WHERE doc_id = ?",
            (doc.doc_id,),
        ).fetchone()
        saved_problem = conn.execute(
            "SELECT problem_text FROM problems WHERE problem_id = ?",
            (problem.problem_id,),
        ).fetchone()
        saved_chunk = conn.execute(
            "SELECT chunk_text FROM chunks WHERE chunk_id = ?",
            (chunk.chunk_id,),
        ).fetchone()

    assert saved_doc["original_filename"] == "test_doc.pdf"
    assert saved_problem["problem_text"] == "Test problem text."
    assert saved_chunk["chunk_text"] == "This is a test chunk."
    assert db.get_chunk_text(chunk.chunk_id) == "This is a test chunk."

    # Embeddings/search are served from VectorStore.
    results = store.search("This is a test chunk.", k=1)
    assert results, "VectorStore should return at least one result"
    top = results[0]
    assert top.chunk.chunk_id == chunk.chunk_id
    assert top.chunk.doc_id == chunk.doc_id
    assert top.similarity_score > 0.9

    # Database manager does not create its own Chroma sidecar; VectorStore owns vectors.
    chroma_sidecar = db_path.with_name(f"{db_path.stem}_chroma_db")
    assert not chroma_sidecar.exists()
    assert vector_dir.exists()
