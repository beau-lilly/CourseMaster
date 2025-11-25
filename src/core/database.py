# src/core/database.py

import sqlite3
import uuid
import numpy as np
from datetime import datetime
from typing import List, Tuple

# Import our defined types and config
from .config import DB_PATH
from .types import Document, Chunk, Problem

class DatabaseManager:
    """Handles all database operations for the study tool."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
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
                uploaded_at TIMESTAMP NOT NULL,
                embedding BLOB
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                chunk_text TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                embedding BLOB,
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
            conn.execute(sql, (doc.doc_id, doc.original_filename, doc.extracted_text, doc.uploaded_at))
            conn.commit()
        return doc

    def add_problem(self, text: str) -> Problem:
        """Adds a new problem to the database."""
        problem = Problem(
            problem_id=f"prob_{uuid.uuid4()}",
            problem_text=text,
            uploaded_at=datetime.now(),
            embedding=None # Embedding will be added later
        )
        
        sql = """
        INSERT INTO problems (problem_id, problem_text, uploaded_at)
        VALUES (?, ?, ?)
        """
        with self._get_connection() as conn:
            conn.execute(sql, (problem.problem_id, problem.problem_text, problem.uploaded_at))
            conn.commit()
        return problem

    def save_chunks(self, chunks: List[Chunk]):
        """Saves a list of Chunk objects to the database."""
        chunk_data = []
        for chunk in chunks:
            # Convert numpy embedding to bytes (BLOB) if it exists, else None
            embedding_blob = chunk.embedding.tobytes() if chunk.embedding is not None else None
            chunk_data.append((
                chunk.chunk_id,
                chunk.doc_id,
                chunk.chunk_text,
                chunk.chunk_index,
                embedding_blob
            ))

        sql = """
        INSERT INTO chunks (chunk_id, doc_id, chunk_text, chunk_index, embedding)
        VALUES (?, ?, ?, ?, ?)
        """
        with self._get_connection() as conn:
            conn.executemany(sql, chunk_data)
            conn.commit()

    def get_all_chunks_with_embeddings(self) -> List[Tuple[str, np.ndarray]]:
        """
        Retrieves all chunk IDs and their embeddings.
        This is the "corpus" our retrieval system will search against.
        """
        sql = "SELECT chunk_id, embedding FROM chunks WHERE embedding IS NOT NULL"
        results = []
        with self._get_connection() as conn:
            cursor = conn.execute(sql)
            for row in cursor.fetchall():
                # Convert BLOB back to numpy array
                embedding = np.frombuffer(row['embedding'], dtype=np.float32) # Note: Adjust dtype if your embedding model is different
                results.append((row['chunk_id'], embedding))
        return results

    def get_chunk_text(self, chunk_id: str) -> str | None:
        """Retrieves the raw text of a single chunk by its ID."""
        sql = "SELECT chunk_text FROM chunks WHERE chunk_id = ?"
        with self._get_connection() as conn:
            cursor = conn.execute(sql, (chunk_id,))
            row = cursor.fetchone()
            return row['chunk_text'] if row else None

    # We'll need more methods later, e.g., to save embeddings for problems,
    # log retrievals, and get documents, but this is a solid start.