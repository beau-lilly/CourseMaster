# src/core/database.py

import sqlite3
import uuid
import numpy as np
import chromadb
from datetime import datetime
from typing import List, Tuple, Dict, Any

# Import our defined types and config
from .config import DB_PATH
from .types import Document, Chunk, Problem

class DatabaseManager:
    """Handles all database operations for the study tool."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        
        # Initialize ChromaDB client
        # We create a separate folder for Chroma data based on the DB_PATH
        chroma_path = db_path.replace(".db", "_chroma_db")
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma_client.get_or_create_collection(name="study_chunks")
        
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
        """Saves a list of Chunk objects to the database and ChromaDB."""
        # 1. Save Metadata/Text to SQLite
        chunk_data = []
        for chunk in chunks:
            chunk_data.append((
                chunk.chunk_id,
                chunk.doc_id,
                chunk.chunk_text,
                chunk.chunk_index
            ))

        sql = """
        INSERT INTO chunks (chunk_id, doc_id, chunk_text, chunk_index)
        VALUES (?, ?, ?, ?)
        """
        with self._get_connection() as conn:
            conn.executemany(sql, chunk_data)
            conn.commit()

        # 2. Save Vectors to ChromaDB
        # Filter out chunks that might not have embeddings (if any)
        valid_chunks = [c for c in chunks if c.embedding is not None]
        
        if valid_chunks:
            self.collection.add(
                ids=[c.chunk_id for c in valid_chunks],
                embeddings=[c.embedding.tolist() for c in valid_chunks], # Chroma expects lists
                metadatas=[{"doc_id": c.doc_id, "chunk_index": c.chunk_index} for c in valid_chunks],
                documents=[c.chunk_text for c in valid_chunks]
            )

    def query_similar_chunks(self, query_embedding: np.ndarray, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Searches ChromaDB for chunks similar to the query embedding.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            include=["metadatas", "documents", "distances"]
        )
        
        # Flatten the results (Chroma returns lists of lists)
        formatted_results = []
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    "chunk_id": results['ids'][0][i],
                    "doc_id": results['metadatas'][0][i]['doc_id'],
                    "chunk_text": results['documents'][0][i],
                    "distance": results['distances'][0][i]
                })
                
        return formatted_results

    def get_chunk_text(self, chunk_id: str) -> str | None:
        """Retrieves the raw text of a single chunk by its ID."""
        sql = "SELECT chunk_text FROM chunks WHERE chunk_id = ?"
        with self._get_connection() as conn:
            cursor = conn.execute(sql, (chunk_id,))
            row = cursor.fetchone()
            return row['chunk_text'] if row else None

    # We'll need more methods later, e.g., to save embeddings for problems,
    # log retrievals, and get documents, but this is a solid start.