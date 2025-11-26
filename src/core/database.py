# src/core/database.py

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import List

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

    def _create_tables(self):
        """Creates all necessary tables if they don't already exist."""
        sql_commands = [
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                original_filename TEXT NOT NULL,
                extracted_text TEXT NOT NULL,
                uploaded_at TIMESTAMP NOT NULL
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

    def add_document(self, filename: str, text: str) -> Document:
        """Adds a new document to the database."""
        doc = Document(
            doc_id=f"doc_{uuid.uuid4()}",
            original_filename=filename,
            extracted_text=text,
            uploaded_at=datetime.now()
        )
        
        sql = """
        INSERT INTO documents (doc_id, original_filename, extracted_text, uploaded_at)
        VALUES (?, ?, ?, ?)
        """
        with self._get_connection() as conn:
            # Convert datetime to string explicitly
            conn.execute(sql, (doc.doc_id, doc.original_filename, doc.extracted_text, doc.uploaded_at.isoformat()))
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
