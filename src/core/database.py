# src/core/database.py

import sqlite3
import uuid
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

# Import our defined types and config
from .config import DB_PATH
from .types import Course, Exam, Document, Chunk, Problem

DEFAULT_COURSE_ID = "course_default"
DEFAULT_EXAM_ID = "exam_default"
DEFAULT_COURSE_NAME = "__default_course__"
DEFAULT_EXAM_NAME = "__default_exam__"


class DatabaseManager:
    """
    Handles SQLite-backed metadata storage for the study tool.
    Vector embeddings and similarity search live in VectorStore.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

        # Ensure the parent directory exists before touching the DB file.
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._create_tables()

    def _get_connection(self) -> sqlite3.Connection:
        """Establishes and returns a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Allows accessing columns by name
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @staticmethod
    def compute_content_hash(text: str) -> str:
        """Consistent hash for a document's raw text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _row_to_document(row: sqlite3.Row) -> Document:
        """Convert a DB row into a Document object."""
        uploaded_value = row["uploaded_at"]
        uploaded_dt = (
            datetime.fromisoformat(uploaded_value)
            if isinstance(uploaded_value, str)
            else uploaded_value
        )
        return Document(
            doc_id=row["doc_id"],
            course_id=row["course_id"],
            original_filename=row["original_filename"],
            extracted_text=row["extracted_text"],
            uploaded_at=uploaded_dt,
            content_hash=row["content_hash"],
        )

    @staticmethod
    def _row_to_course(row: sqlite3.Row) -> Course:
        created_value = row["created_at"]
        created_dt = (
            datetime.fromisoformat(created_value)
            if isinstance(created_value, str)
            else created_value
        )
        return Course(
            course_id=row["course_id"],
            name=row["name"],
            created_at=created_dt,
        )

    @staticmethod
    def _row_to_exam(row: sqlite3.Row) -> Exam:
        created_value = row["created_at"]
        created_dt = (
            datetime.fromisoformat(created_value)
            if isinstance(created_value, str)
            else created_value
        )
        return Exam(
            exam_id=row["exam_id"],
            course_id=row["course_id"],
            name=row["name"],
            created_at=created_dt,
        )

    def _create_tables(self):
        """Creates all necessary tables if they don't already exist."""
        sql_commands = [
            """
            CREATE TABLE IF NOT EXISTS courses (
                course_id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP NOT NULL
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS exams (
                exam_id TEXT PRIMARY KEY,
                course_id TEXT NOT NULL,
                name TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                FOREIGN KEY (course_id) REFERENCES courses (course_id),
                UNIQUE(course_id, name)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                course_id TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                extracted_text TEXT NOT NULL,
                uploaded_at TIMESTAMP NOT NULL,
                content_hash TEXT,
                FOREIGN KEY (course_id) REFERENCES courses (course_id)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS problems (
                problem_id TEXT PRIMARY KEY,
                exam_id TEXT NOT NULL,
                problem_text TEXT NOT NULL,
                uploaded_at TIMESTAMP NOT NULL,
                FOREIGN KEY (exam_id) REFERENCES exams (exam_id)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                chunk_text TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                FOREIGN KEY (doc_id) REFERENCES documents (doc_id)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS exam_documents (
                exam_id TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                PRIMARY KEY (exam_id, doc_id),
                FOREIGN KEY (exam_id) REFERENCES exams (exam_id),
                FOREIGN KEY (doc_id) REFERENCES documents (doc_id)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS retrieval_log (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                problem_id TEXT NOT NULL,
                retrieved_chunk_id TEXT NOT NULL,
                similarity_score REAL NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                FOREIGN KEY (problem_id) REFERENCES problems (problem_id),
                FOREIGN KEY (retrieved_chunk_id) REFERENCES chunks (chunk_id)
            );
            """
        ]

        with self._get_connection() as conn:
            cursor = conn.cursor()
            for command in sql_commands:
                cursor.execute(command)
            conn.commit()

        self._ensure_schema_updates()

    def _ensure_schema_updates(self):
        """Backfill/ensure columns and indexes for older databases."""
        with self._get_connection() as conn:
            self._ensure_column(conn, "documents", "content_hash", "TEXT")
            self._ensure_column(conn, "documents", "course_id", "TEXT")
            self._ensure_column(conn, "problems", "exam_id", "TEXT")

            self._populate_missing_document_hashes(conn)
            self._ensure_default_course_and_exam(conn)
            self._backfill_missing_course_ids(conn)
            self._backfill_missing_exam_ids(conn)

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_course_hash ON documents(course_id, content_hash)"
            )
            conn.commit()

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, definition: str):
        """Add a column if it is missing."""
        cursor = conn.execute(f"PRAGMA table_info({table})")
        columns = {row["name"] for row in cursor.fetchall()}
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _populate_missing_document_hashes(self, conn: sqlite3.Connection):
        """Populate missing content hashes to support deduplication."""
        missing = conn.execute(
            "SELECT doc_id, extracted_text FROM documents WHERE content_hash IS NULL OR content_hash = ''"
        ).fetchall()
        for row in missing:
            content_hash = self.compute_content_hash(row["extracted_text"])
            conn.execute(
                "UPDATE documents SET content_hash = ? WHERE doc_id = ?",
                (content_hash, row["doc_id"]),
            )

    def _ensure_default_course_and_exam(self, conn: sqlite3.Connection):
        """Create default course/exam for legacy rows without scope."""
        course_exists = conn.execute(
            "SELECT 1 FROM courses WHERE course_id = ?", (DEFAULT_COURSE_ID,)
        ).fetchone()
        if not course_exists:
            conn.execute(
                "INSERT OR IGNORE INTO courses (course_id, name, created_at) VALUES (?, ?, ?)",
                (DEFAULT_COURSE_ID, DEFAULT_COURSE_NAME, datetime.now().isoformat()),
            )

        exam_exists = conn.execute(
            "SELECT 1 FROM exams WHERE exam_id = ?", (DEFAULT_EXAM_ID,)
        ).fetchone()
        if not exam_exists:
            conn.execute(
                "INSERT OR IGNORE INTO exams (exam_id, course_id, name, created_at) VALUES (?, ?, ?, ?)",
                (DEFAULT_EXAM_ID, DEFAULT_COURSE_ID, DEFAULT_EXAM_NAME, datetime.now().isoformat()),
            )

    def _backfill_missing_course_ids(self, conn: sqlite3.Connection):
        """Assign legacy documents to the default course if they lack one."""
        conn.execute(
            "UPDATE documents SET course_id = ? WHERE course_id IS NULL OR course_id = ''",
            (DEFAULT_COURSE_ID,),
        )

    def _backfill_missing_exam_ids(self, conn: sqlite3.Connection):
        """Assign legacy problems to the default exam if they lack one."""
        conn.execute(
            "UPDATE problems SET exam_id = ? WHERE exam_id IS NULL OR exam_id = ''",
            (DEFAULT_EXAM_ID,),
        )

    # --- Courses ---

    def add_course(self, name: str) -> Course:
        """Create a course, or return the existing one with the same name."""
        existing = self.get_course_by_name(name)
        if existing:
            return existing

        course = Course(
            course_id=f"course_{uuid.uuid4()}",
            name=name,
            created_at=datetime.now(),
        )
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO courses (course_id, name, created_at) VALUES (?, ?, ?)",
                (course.course_id, course.name, course.created_at.isoformat()),
            )
            conn.commit()
        return course

    def get_course(self, course_id: str) -> Optional[Course]:
        sql = "SELECT * FROM courses WHERE course_id = ?"
        with self._get_connection() as conn:
            row = conn.execute(sql, (course_id,)).fetchone()
            return self._row_to_course(row) if row else None

    def get_course_by_name(self, name: str) -> Optional[Course]:
        sql = "SELECT * FROM courses WHERE name = ?"
        with self._get_connection() as conn:
            row = conn.execute(sql, (name,)).fetchone()
            return self._row_to_course(row) if row else None

    def list_courses(self) -> List[Course]:
        sql = "SELECT * FROM courses ORDER BY created_at DESC"
        with self._get_connection() as conn:
            rows = conn.execute(sql).fetchall()
            return [self._row_to_course(row) for row in rows]

    # --- Exams ---

    def add_exam(self, course_id: str, name: str) -> Exam:
        """Create an exam within a course, or return the existing one with the same name."""
        existing = self.get_exam_by_name(course_id, name)
        if existing:
            return existing

        exam = Exam(
            exam_id=f"exam_{uuid.uuid4()}",
            course_id=course_id,
            name=name,
            created_at=datetime.now(),
        )
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO exams (exam_id, course_id, name, created_at) VALUES (?, ?, ?, ?)",
                (exam.exam_id, exam.course_id, exam.name, exam.created_at.isoformat()),
            )
            conn.commit()
        return exam

    def get_exam(self, exam_id: str) -> Optional[Exam]:
        sql = "SELECT * FROM exams WHERE exam_id = ?"
        with self._get_connection() as conn:
            row = conn.execute(sql, (exam_id,)).fetchone()
            return self._row_to_exam(row) if row else None

    def get_exam_by_name(self, course_id: str, name: str) -> Optional[Exam]:
        sql = "SELECT * FROM exams WHERE course_id = ? AND name = ?"
        with self._get_connection() as conn:
            row = conn.execute(sql, (course_id, name)).fetchone()
            return self._row_to_exam(row) if row else None

    def list_exams_for_course(self, course_id: str) -> List[Exam]:
        sql = "SELECT * FROM exams WHERE course_id = ? ORDER BY created_at DESC"
        with self._get_connection() as conn:
            rows = conn.execute(sql, (course_id,)).fetchall()
            return [self._row_to_exam(row) for row in rows]

    # --- Documents ---

    def add_document(self, filename: str, text: str, course_id: str) -> Document:
        """Adds a new document to the database, scoped to a course."""
        if not course_id:
            raise ValueError("course_id is required to add a document.")

        content_hash = self.compute_content_hash(text)

        # Skip duplicates if we've already ingested identical content for this course
        existing = self.get_document_by_hash(course_id, content_hash)
        if existing:
            return existing

        doc = Document(
            doc_id=f"doc_{uuid.uuid4()}",
            course_id=course_id,
            original_filename=filename,
            extracted_text=text,
            uploaded_at=datetime.now(),
            content_hash=content_hash,
        )

        sql = """
        INSERT INTO documents (doc_id, course_id, original_filename, extracted_text, uploaded_at, content_hash)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        with self._get_connection() as conn:
            conn.execute(
                sql,
                (
                    doc.doc_id,
                    doc.course_id,
                    doc.original_filename,
                    doc.extracted_text,
                    doc.uploaded_at.isoformat(),
                    content_hash,
                ),
            )
            conn.commit()
        return doc

    def get_document(self, doc_id: str) -> Optional[Document]:
        sql = "SELECT * FROM documents WHERE doc_id = ?"
        with self._get_connection() as conn:
            row = conn.execute(sql, (doc_id,)).fetchone()
            return self._row_to_document(row) if row else None

    def get_document_by_hash(self, course_id: str, content_hash: str) -> Optional[Document]:
        """Returns an existing document that matches the given content hash within a course."""
        sql = """
        SELECT doc_id, course_id, original_filename, extracted_text, uploaded_at, content_hash
        FROM documents
        WHERE course_id = ? AND content_hash = ?
        LIMIT 1
        """
        with self._get_connection() as conn:
            row = conn.execute(sql, (course_id, content_hash)).fetchone()
            if not row:
                return None
            return self._row_to_document(row)

    def get_documents_for_course(self, course_id: str) -> List[Document]:
        sql = "SELECT * FROM documents WHERE course_id = ? ORDER BY uploaded_at DESC"
        with self._get_connection() as conn:
            rows = conn.execute(sql, (course_id,)).fetchall()
        return [self._row_to_document(row) for row in rows]

    def get_documents_for_exam(self, exam_id: str) -> List[Document]:
        """Return documents linked to a given exam."""
        sql = """
        SELECT d.*
        FROM documents d
        JOIN exam_documents ed ON d.doc_id = ed.doc_id
        WHERE ed.exam_id = ?
        ORDER BY d.uploaded_at DESC
        """
        with self._get_connection() as conn:
            rows = conn.execute(sql, (exam_id,)).fetchall()
        return [self._row_to_document(row) for row in rows]

    def get_doc_filenames(self, doc_ids: List[str]) -> dict[str, str]:
        """Retrieves filenames for a list of document IDs."""
        if not doc_ids:
            return {}

        placeholders = ",".join("?" * len(doc_ids))
        sql = f"SELECT doc_id, original_filename FROM documents WHERE doc_id IN ({placeholders})"

        with self._get_connection() as conn:
            cursor = conn.execute(sql, doc_ids)
            rows = cursor.fetchall()

        return {row["doc_id"]: row["original_filename"] for row in rows}

    # --- Exam-document links ---

    def attach_document_to_exam(self, exam_id: str, doc_id: str) -> None:
        """Link an existing document to an exam scope."""
        sql = """
        INSERT OR IGNORE INTO exam_documents (exam_id, doc_id)
        VALUES (?, ?)
        """
        with self._get_connection() as conn:
            conn.execute(sql, (exam_id, doc_id))
            conn.commit()

    def attach_documents_to_exam(self, exam_id: str, doc_ids: Sequence[str]) -> None:
        if not doc_ids:
            return
        with self._get_connection() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO exam_documents (exam_id, doc_id) VALUES (?, ?)",
                [(exam_id, doc_id) for doc_id in doc_ids],
            )
            conn.commit()

    def get_document_ids_for_exam(self, exam_id: str) -> List[str]:
        sql = "SELECT doc_id FROM exam_documents WHERE exam_id = ?"
        with self._get_connection() as conn:
            return [row["doc_id"] for row in conn.execute(sql, (exam_id,)).fetchall()]

    # --- Chunks ---

    def save_chunks(self, chunks: List[Chunk]):
        """Saves a list of Chunk metadata/text to SQLite."""
        if not chunks:
            return

        chunk_data = [
            (
                chunk.chunk_id,
                chunk.doc_id,
                chunk.chunk_text,
                chunk.chunk_index,
            )
            for chunk in chunks
        ]

        sql = """
        INSERT INTO chunks (chunk_id, doc_id, chunk_text, chunk_index)
        VALUES (?, ?, ?, ?)
        """
        with self._get_connection() as conn:
            conn.executemany(sql, chunk_data)
            conn.commit()

    def get_chunk_text(self, chunk_id: str) -> str | None:
        """Retrieves the raw text of a single chunk by its ID."""
        sql = "SELECT chunk_text FROM chunks WHERE chunk_id = ?"
        with self._get_connection() as conn:
            cursor = conn.execute(sql, (chunk_id,))
            row = cursor.fetchone()
            return row["chunk_text"] if row else None

    def get_chunk_ids_for_doc(self, doc_id: str) -> List[str]:
        """Returns all chunk IDs belonging to a document."""
        sql = "SELECT chunk_id FROM chunks WHERE doc_id = ?"
        with self._get_connection() as conn:
            return [row["chunk_id"] for row in conn.execute(sql, (doc_id,)).fetchall()]

    def delete_chunks_for_doc(self, doc_id: str) -> None:
        """Deletes all chunk rows for a document."""
        sql = "DELETE FROM chunks WHERE doc_id = ?"
        with self._get_connection() as conn:
            conn.execute(sql, (doc_id,))
            conn.commit()

    def get_chunk_count_for_doc(self, doc_id: str) -> int:
        """Returns how many chunks are stored for a document."""
        sql = "SELECT COUNT(*) as count FROM chunks WHERE doc_id = ?"
        with self._get_connection() as conn:
            row = conn.execute(sql, (doc_id,)).fetchone()
            return row["count"] if row else 0

    def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[Chunk]:
        """Retrieves Chunk objects by their IDs."""
        if not chunk_ids:
            return []

        placeholders = ",".join("?" * len(chunk_ids))
        sql = f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})"

        with self._get_connection() as conn:
            cursor = conn.execute(sql, chunk_ids)
            rows = cursor.fetchall()

        chunks = []
        for row in rows:
            chunks.append(
                Chunk(
                    chunk_id=row["chunk_id"],
                    doc_id=row["doc_id"],
                    chunk_text=row["chunk_text"],
                    chunk_index=row["chunk_index"],
                    embedding=None,
                )
            )
        return chunks

    # --- Problems & retrieval ---

    def add_problem(self, text: str, exam_id: str) -> Problem:
        """Adds a new problem to the database, scoped to an exam."""
        if not exam_id:
            raise ValueError("exam_id is required to add a problem.")

        problem = Problem(
            problem_id=f"prob_{uuid.uuid4()}",
            exam_id=exam_id,
            problem_text=text,
            uploaded_at=datetime.now(),
            embedding=None,
        )

        sql = """
        INSERT INTO problems (problem_id, exam_id, problem_text, uploaded_at)
        VALUES (?, ?, ?, ?)
        """
        with self._get_connection() as conn:
            conn.execute(
                sql,
                (
                    problem.problem_id,
                    problem.exam_id,
                    problem.problem_text,
                    problem.uploaded_at.isoformat(),
                ),
            )
            conn.commit()
        return problem

    def list_problems_for_exam(self, exam_id: str) -> List[Problem]:
        sql = "SELECT * FROM problems WHERE exam_id = ? ORDER BY uploaded_at DESC"
        with self._get_connection() as conn:
            rows = conn.execute(sql, (exam_id,)).fetchall()

        problems = []
        for row in rows:
            uploaded_value = row["uploaded_at"]
            uploaded_dt = (
                datetime.fromisoformat(uploaded_value)
                if isinstance(uploaded_value, str)
                else uploaded_value
            )
            problems.append(
                Problem(
                    problem_id=row["problem_id"],
                    exam_id=row["exam_id"],
                    problem_text=row["problem_text"],
                    uploaded_at=uploaded_dt,
                    embedding=None,
                )
            )
        return problems

    def log_retrieval(self, problem_id: str, chunk_id: str, score: float):
        """Logs a retrieval event."""
        sql = """
        INSERT INTO retrieval_log (problem_id, retrieved_chunk_id, similarity_score, timestamp)
        VALUES (?, ?, ?, ?)
        """
        with self._get_connection() as conn:
            conn.execute(sql, (problem_id, chunk_id, score, datetime.now().isoformat()))
            conn.commit()
