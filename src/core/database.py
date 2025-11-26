# src/core/database.py

import sqlite3
import uuid
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Import our defined types and config
from .config import DB_PATH
from .types import Document, Chunk, Problem

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
        # We can add more configuration here if needed, e.g., for foreign keys
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row # Allows accessing columns by name
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
            original_filename=row["original_filename"],
            extracted_text=row["extracted_text"],
            uploaded_at=uploaded_dt,
            content_hash=row["content_hash"],
        )

    def _create_tables(self):
        """Creates all necessary tables if they don't already exist."""
        sql_commands = [
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                original_filename TEXT NOT NULL,
                extracted_text TEXT NOT NULL,
                uploaded_at TIMESTAMP NOT NULL,
                content_hash TEXT
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS problems (
                problem_id TEXT PRIMARY KEY,
                problem_text TEXT NOT NULL,
                uploaded_at TIMESTAMP NOT NULL
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

        # Backfill/ensure new columns for existing databases
        self._ensure_document_hash_column()

    def _ensure_document_hash_column(self):
        """Add content_hash column and populate it for existing rows."""
        with self._get_connection() as conn:
            cursor = conn.execute("PRAGMA table_info(documents)")
            columns = {row["name"] for row in cursor.fetchall()}

            if "content_hash" not in columns:
                conn.execute("ALTER TABLE documents ADD COLUMN content_hash TEXT")
                conn.commit()

            # Populate any missing hashes so deduplication works on older DBs
            missing = conn.execute(
                "SELECT doc_id, extracted_text FROM documents WHERE content_hash IS NULL OR content_hash = ''"
            ).fetchall()
            for row in missing:
                content_hash = self.compute_content_hash(row["extracted_text"])
                conn.execute(
                    "UPDATE documents SET content_hash = ? WHERE doc_id = ?",
                    (content_hash, row["doc_id"]),
                )

            # Helpful for lookups when many documents are stored
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash)"
            )
            conn.commit()

    def add_document(self, filename: str, text: str) -> Document:
        """Adds a new document to the database."""
        content_hash = self.compute_content_hash(text)

        # Skip duplicates if we've already ingested identical content
        existing = self.get_document_by_hash(content_hash)
        if existing:
            return existing

        doc = Document(
            doc_id=f"doc_{uuid.uuid4()}",
            original_filename=filename,
            extracted_text=text,
            uploaded_at=datetime.now(),
            content_hash=content_hash
        )
        
        sql = """
        INSERT INTO documents (doc_id, original_filename, extracted_text, uploaded_at, content_hash)
        VALUES (?, ?, ?, ?, ?)
        """
        with self._get_connection() as conn:
            # Convert datetime to string explicitly
            conn.execute(
                sql,
                (
                    doc.doc_id,
                    doc.original_filename,
                    doc.extracted_text,
                    doc.uploaded_at.isoformat(),
                    content_hash,
                ),
            )
            conn.commit()
        return doc

    def add_problem(self, text: str) -> Problem:
        """Adds a new problem to the database."""
        problem = Problem(
            problem_id=f"prob_{uuid.uuid4()}",
            problem_text=text,
            uploaded_at=datetime.now(),
            embedding=None 
        )
        
        sql = """
        INSERT INTO problems (problem_id, problem_text, uploaded_at)
        VALUES (?, ?, ?)
        """
        with self._get_connection() as conn:
            # Convert datetime to string explicitly
            conn.execute(sql, (problem.problem_id, problem.problem_text, problem.uploaded_at.isoformat()))
            conn.commit()
        return problem

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
            return row['chunk_text'] if row else None

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

    def log_retrieval(self, problem_id: str, chunk_id: str, score: float):
        """Logs a retrieval event."""
        sql = """
        INSERT INTO retrieval_log (problem_id, retrieved_chunk_id, similarity_score, timestamp)
        VALUES (?, ?, ?, ?)
        """
        with self._get_connection() as conn:
            conn.execute(sql, (problem_id, chunk_id, score, datetime.now().isoformat()))
            conn.commit()

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
            chunks.append(Chunk(
                chunk_id=row['chunk_id'],
                doc_id=row['doc_id'],
                chunk_text=row['chunk_text'],
                chunk_index=row['chunk_index'],
                embedding=None
            ))
        return chunks

    def get_doc_filenames(self, doc_ids: List[str]) -> dict[str, str]:
        """Retrieves filenames for a list of document IDs."""
        if not doc_ids:
            return {}
            
        placeholders = ",".join("?" * len(doc_ids))
        sql = f"SELECT doc_id, original_filename FROM documents WHERE doc_id IN ({placeholders})"
        
        with self._get_connection() as conn:
            cursor = conn.execute(sql, doc_ids)
            rows = cursor.fetchall()
            
        return {row['doc_id']: row['original_filename'] for row in rows}

    def get_document_by_hash(self, content_hash: str) -> Optional[Document]:
        """Returns an existing document that matches the given content hash."""
        sql = """
        SELECT doc_id, original_filename, extracted_text, uploaded_at, content_hash
        FROM documents
        WHERE content_hash = ?
        LIMIT 1
        """
        with self._get_connection() as conn:
            row = conn.execute(sql, (content_hash,)).fetchone()
            if not row:
                return None
            return self._row_to_document(row)
