"""
Core dataclasses/types (e.g., Document, Chunk, RAGResult).
"""


from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional
import numpy as np


@dataclass
class Assignment:
    """Represents an assignment label within an exam (e.g., Homework 1)."""
    assignment_id: str
    exam_id: str
    name: str
    created_at: datetime


@dataclass
class Course:
    """Represents a course (top-level scope)."""
    course_id: str
    name: str
    created_at: datetime


@dataclass
class Exam:
    """Represents an exam/scope within a course."""
    exam_id: str
    course_id: str
    name: str
    created_at: datetime

class PromptStyle(Enum):
    MINIMAL = "minimal"
    EXPLANATORY = "explanatory"
    TUTORING = "tutoring"
    SIMILARITY = "similarity"

@dataclass
class Document:
    """Represents an uploaded document."""
    doc_id: str
    course_id: str
    original_filename: str
    extracted_text: str
    uploaded_at: datetime
    content_hash: str | None = None

@dataclass
class Chunk:
    """Represents a single chunk of text derived from a Document."""
    chunk_id: str
    doc_id: str
    chunk_text: str
    chunk_index: int # The sequential order of the chunk (e.g., 0, 1, 2...)
    embedding: np.ndarray | None = None

@dataclass
class Problem:
    """Represents a single problem input by the user."""
    problem_id: str
    exam_id: str
    problem_text: str
    uploaded_at: datetime
    assignment_id: str | None = None
    problem_number: int | None = None
    embedding: np.ndarray | None = None


@dataclass
class Question:
    """Represents a single question asked about a problem."""
    question_id: str
    problem_id: str
    question_text: str
    answer_text: str
    created_at: datetime
    prompt_style: str | None = None

@dataclass
class RAGResult:
    """Encapsulates the full result of a RAG query."""
    question: str
    answer: str
    used_chunks: List[Chunk]
    scores: Optional[List[float]] = None
    question_id: str | None = None
