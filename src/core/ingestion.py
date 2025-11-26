"""
Functions for parsing files (PDF, TXT) and extracting text.
"""

import os
from pathlib import Path
from typing import Optional

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
    db_manager: Optional[DatabaseManager] = None,
    vector_store: Optional[VectorStore] = None
) -> Document:
    """
    Full ingestion pipeline for a single file:
    1. Extract text.
    2. Persist Document metadata to SQLite.
    3. Chunk the text.
    4. Persist Chunk metadata to SQLite.
    5. Embed and store Chunks in VectorStore.
    
    Args:
        file_path: Absolute or relative path to the file on disk.
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
    
    # 3. Save Document (SQLite)
    # This generates the doc_id and timestamps
    print("Saving document metadata...")
    doc = db_manager.add_document(filename=filename, text=text)
    
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
    
    print("Ingestion complete.")
    return doc
