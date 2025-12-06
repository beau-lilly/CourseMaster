# tests/test_database.py

import uuid
import pytest

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
    assignment = db.add_assignment(exam.exam_id, "Homework 1")
    problem = db.add_problem("Test problem text.", exam_id=exam.exam_id, assignment_id=assignment.assignment_id, problem_number=1)

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


def test_problem_number_must_be_unique_per_assignment(tmp_path):
    db_path = tmp_path / "assignment_num.db"
    db = DatabaseManager(db_path=str(db_path))
    course = db.add_course("Course A")
    exam = db.add_exam(course.course_id, "Midterm")
    assignment = db.add_assignment(exam.exam_id, "Practice 1")

    db.add_problem("First problem", exam_id=exam.exam_id, assignment_id=assignment.assignment_id, problem_number=1)

    # Duplicate number in same assignment should raise
    with pytest.raises(ValueError):
        db.add_problem("Duplicate number", exam_id=exam.exam_id, assignment_id=assignment.assignment_id, problem_number=1)

    # Null numbers can repeat
    db.add_problem("No number", exam_id=exam.exam_id, assignment_id=assignment.assignment_id)
    db.add_problem("Another null", exam_id=exam.exam_id, assignment_id=assignment.assignment_id)


def test_questions_round_trip(tmp_path):
    db_path = tmp_path / "questions.db"
    db = DatabaseManager(db_path=str(db_path))
    course = db.add_course("Course A")
    exam = db.add_exam(course.course_id, "Final")
    assignment = db.add_assignment(exam.exam_id, "Practice Final")
    problem = db.add_problem("What is photosynthesis?", exam_id=exam.exam_id, assignment_id=assignment.assignment_id)

    question = db.add_question(problem.problem_id, "Explain it", prompt_style="minimal")
    db.update_question_answer(question.question_id, "It converts light to energy.")

    stored = db.get_question(question.question_id)
    assert stored is not None
    assert stored.answer_text == "It converts light to energy."
    questions = db.list_questions_for_problem(problem.problem_id)
    assert len(questions) == 1


def _seed_retrieval_data(tmp_path):
    db_path = tmp_path / "ranking.db"
    db = DatabaseManager(db_path=str(db_path))
    course = db.add_course("Course A")
    exam = db.add_exam(course.course_id, "Final")

    doc1 = db.add_document("doc1.pdf", "Doc one text", course_id=course.course_id)
    doc2 = db.add_document("doc2.pdf", "Doc two text", course_id=course.course_id)
    db.attach_documents_to_exam(exam.exam_id, [doc1.doc_id, doc2.doc_id])

    chunks = [
        Chunk(chunk_id="chunk-a", doc_id=doc1.doc_id, chunk_text="Chunk A", chunk_index=0),
        Chunk(chunk_id="chunk-b", doc_id=doc1.doc_id, chunk_text="Chunk B", chunk_index=1),
        Chunk(chunk_id="chunk-c", doc_id=doc2.doc_id, chunk_text="Chunk C", chunk_index=0),
    ]
    db.save_chunks(chunks)

    prob1 = db.add_problem("Problem one", exam_id=exam.exam_id)
    prob2 = db.add_problem("Problem two", exam_id=exam.exam_id)

    db.log_retrieval(prob1.problem_id, "chunk-a", 0.9)
    db.log_retrieval(prob1.problem_id, "chunk-b", 0.5)
    db.log_retrieval(prob2.problem_id, "chunk-a", 0.8)
    db.log_retrieval(prob2.problem_id, "chunk-c", 0.7)

    return db, exam, doc1, doc2


def test_top_chunks_ranking(tmp_path):
    db, exam, _, _ = _seed_retrieval_data(tmp_path)

    freq = db.get_top_chunks_for_exam(exam.exam_id, "frequency", limit=3)
    assert [row["chunk_id"] for row in freq] == ["chunk-a", "chunk-b", "chunk-c"]
    assert freq[0]["score"] == 2

    weighted = db.get_top_chunks_for_exam(exam.exam_id, "weighted_sum", limit=2)
    assert [row["chunk_id"] for row in weighted] == ["chunk-a", "chunk-c"]
    assert weighted[0]["score"] == pytest.approx(1.7)
    assert weighted[1]["score"] == pytest.approx(0.7)


def test_top_documents_ranking(tmp_path):
    db, exam, doc1, doc2 = _seed_retrieval_data(tmp_path)

    freq = db.get_top_documents_for_exam(exam.exam_id, "frequency", limit=2)
    assert [row["doc_id"] for row in freq] == [doc1.doc_id, doc2.doc_id]
    assert freq[0]["score"] == 2
    assert freq[1]["score"] == 1

    weighted = db.get_top_documents_for_exam(exam.exam_id, "weighted_sum", limit=2)
    assert [row["doc_id"] for row in weighted] == [doc1.doc_id, doc2.doc_id]
    assert weighted[0]["score"] == pytest.approx(2.2)
    assert weighted[1]["score"] == pytest.approx(0.7)
