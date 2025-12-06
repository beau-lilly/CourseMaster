"""
Functions for parsing files (PDF, TXT) and extracting text.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

from src.core.database import DatabaseManager
from src.core.vector_store import VectorStore
from src.core.chunking import chunk_document
from src.core.types import Document
from pypdf import PdfReader

def extract_text_from_file(file_path: str) -> str:
    """
    Reads a file and extracts its text content.
    Currently supports .txt, .md, and .pdf files.
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
        
    suffix = path.suffix.lower()
    
    if suffix in ['.txt', '.md', '.py', '.html', '.css', '.js']:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Fallback to latin-1 if utf-8 fails
            with open(path, 'r', encoding='latin-1') as f:
                return f.read()
    
    if suffix == '.pdf':
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise ValueError(f"Error reading PDF file: {e}")
        
    raise ValueError(f"Unsupported file type: {suffix}")

def process_uploaded_file(
    file_path: str,
    course_id: str,
    exam_ids: Optional[list[str]] = None,
    db_manager: Optional[DatabaseManager] = None,
    vector_store: Optional[VectorStore] = None
) -> Tuple[Document, Optional[str]]:
    """
    Full ingestion pipeline for a single file:
    1. Extract text.
    2. Persist Document metadata to SQLite.
    3. Chunk the text.
    4. Persist Chunk metadata to SQLite.
    5. Embed and store Chunks in VectorStore.
    6. Link the document to any provided exams.
    
    Args:
        file_path: Absolute or relative path to the file on disk.
        course_id: The course scope that owns the document.
        exam_ids: Exams to link this document to.
        db_manager: Optional injected instance.
        vector_store: Optional injected instance.
        
    Returns:
        The created Document object.
    """
    
    # 1. Init dependencies
    if db_manager is None:
        db_manager = DatabaseManager()
        
    if vector_store is None:
        vector_store = VectorStore()
        
    # 2. Extract Text
    print(f"Extracting text from {file_path}...")
    text = extract_text_from_file(file_path)
    filename = os.path.basename(file_path)
    content_hash = db_manager.compute_content_hash(text)

    # Duplicate checks by hash and filename
    existing_doc_by_hash = db_manager.get_document_by_hash(course_id, content_hash)
    if existing_doc_by_hash:
        print(f"Identical document already exists ({existing_doc_by_hash.original_filename}). Skipping re-ingestion.")
        # Still attach to any provided exams
        if exam_ids:
            for exam_id in exam_ids:
                db_manager.attach_document_to_exam(exam_id, existing_doc_by_hash.doc_id)
        return existing_doc_by_hash, f"Identical document already exists ({existing_doc_by_hash.original_filename})"

    existing_doc_by_name = db_manager.get_document_by_name(course_id, filename)
    if existing_doc_by_name:
        print("Document of the same name already uploaded.")
        if exam_ids:
            for exam_id in exam_ids:
                db_manager.attach_document_to_exam(exam_id, existing_doc_by_name.doc_id)
        return existing_doc_by_name, "Document of the same name already uploaded."

    # 3. Save Document (SQLite)
    print("Saving document metadata...")
    doc = db_manager.add_document(filename=filename, text=text, course_id=course_id)

    # 4. Chunking
    print("Chunking document...")
    chunks = chunk_document(doc)
    print(f"Generated {len(chunks)} chunks.")
    
    # 5. Save Chunks (SQLite)
    print("Saving chunk metadata...")
    db_manager.save_chunks(chunks)
    
    # 6. Embed & Store (VectorDB)
    print("Embedding and storing in VectorStore...")
    vector_store.add_chunks(chunks)
    
    # 7. Link document to provided exams (if any)
    if exam_ids:
        for exam_id in exam_ids:
            db_manager.attach_document_to_exam(exam_id, doc.doc_id)
    
    print("Ingestion complete.")
    return doc, None
