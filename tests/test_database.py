# tests/test_database.py

import uuid

from tests.helpers import HashEmbeddings
from src.core.database import DatabaseManager
from src.core.types import Chunk
from src.core.vector_store import VectorStore


def test_add_document_deduplicates_by_content(tmp_path):
    """
    Re-uploading the same content should not create duplicate documents.
    """
    db_path = tmp_path / "dedup.db"
    db = DatabaseManager(db_path=str(db_path))
    course = db.add_course("Course A")

    doc1 = db.add_document("fileA.pdf", "identical content", course_id=course.course_id)
    doc2 = db.add_document("fileB.pdf", "identical content", course_id=course.course_id)

    assert doc1.doc_id == doc2.doc_id
    with db._get_connection() as conn:
        count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    assert count == 1


def test_same_content_allowed_in_different_courses(tmp_path):
    """
    Deduplication is per-course: identical content in another course should be stored separately.
    """
    db_path = tmp_path / "dedup_scope.db"
    db = DatabaseManager(db_path=str(db_path))
    course_a = db.add_course("Course A")
    course_b = db.add_course("Course B")

    doc1 = db.add_document("fileA.pdf", "identical content", course_id=course_a.course_id)
    doc2 = db.add_document("fileB.pdf", "identical content", course_id=course_b.course_id)

    assert doc1.doc_id != doc2.doc_id
    with db._get_connection() as conn:
        count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    assert count == 2


def test_delete_chunks_for_doc(tmp_path):
    """
    Chunks for a document can be removed in bulk.
    """
    db_path = tmp_path / "chunks.db"
    db = DatabaseManager(db_path=str(db_path))
    course = db.add_course("Course A")

    doc = db.add_document("doc.pdf", "content", course_id=course.course_id)
    chunks = [
        Chunk(chunk_id="c1", doc_id=doc.doc_id, chunk_text="a", chunk_index=0),
        Chunk(chunk_id="c2", doc_id=doc.doc_id, chunk_text="b", chunk_index=1),
    ]
    db.save_chunks(chunks)
    assert db.get_chunk_count_for_doc(doc.doc_id) == 2

    db.delete_chunks_for_doc(doc.doc_id)
    assert db.get_chunk_count_for_doc(doc.doc_id) == 0


def test_database_metadata_and_vector_store_round_trip(tmp_path):
    db_path = tmp_path / "test_suite.db"
    vector_dir = tmp_path / "vector_store_db"

    db = DatabaseManager(db_path=str(db_path))
    course = db.add_course("Course A")
    exam = db.add_exam(course.course_id, "Midterm")
    store = VectorStore(
        persist_directory=vector_dir,
        collection_name="db_vector_test",
        embedding_function=HashEmbeddings(),
    )
    store.reset()

    doc = db.add_document("test_doc.pdf", "Full document text.", course_id=course.course_id)
    problem = db.add_problem("Test problem text.", exam_id=exam.exam_id)

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
