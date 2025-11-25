"""
Core dataclasses/types (e.g., Document, Chunk, RAGResult).
"""


from dataclasses import dataclass
from datetime import datetime
import numpy as np

@dataclass
class Document:
    """Represents an uploaded document."""
    doc_id: str
    original_filename: str
    extracted_text: str
    uploaded_at: datetime

@dataclass
class Chunk:
    """Represents a single chunk of text derived from a Document."""
    chunk_id: str
    doc_id: str
    chunk_text: str
    chunk_index: int # The sequential order of the chunk (e.g., 0, 1, 2...)
    embedding: np.ndarray | None = None # Will be populated later

@dataclass
class Problem:
    """Represents a single problem input by the user."""
    problem_id: str
    problem_text: str
    uploaded_at: datetime
    embedding: np.ndarray | None = None # Will be populated later