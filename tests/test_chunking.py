"""
Tests for the chunking logic in src/core/chunking.py
"""
import pytest
from datetime import datetime
from src.core.types import Document
from src.core.chunking import chunk_document

@pytest.fixture
def long_document():
    """Fixture for a document that is long enough to be chunked."""
    text = ("Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 50) # 3450 chars
    text += ("Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. " * 50) # 3600 chars
    text += ("Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris. " * 50) # 3800 chars
    # Total chars approx 10850. Should be ~10-11 chunks.
    return Document(
        doc_id="test-doc-long",
        extracted_text=text,
        original_filename="long_story.txt",
        uploaded_at=datetime.now()
    )

@pytest.fixture
def short_document():
    """Fixture for a document that is shorter than the chunk size."""
    text = "This is a short document. It should not be split into multiple chunks."
    return Document(
        doc_id="test-doc-short",
        extracted_text=text,
        original_filename="short_note.txt",
        uploaded_at=datetime.now()
    )

def test_chunk_document_long(long_document):
    """
    Tests that a long document is split into multiple chunks.
    """
    chunks = chunk_document(long_document)
    
    assert chunks is not None
    assert isinstance(chunks, list)
    assert len(chunks) > 1
    assert all(c.chunk_text for c in chunks) # Ensure no chunk is empty

def test_chunk_document_id_propagation(long_document):
    """
    Tests that the document_id is correctly propagated to all chunks.
    """
    chunks = chunk_document(long_document)
    
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.doc_id == "test-doc-long"

def test_chunk_document_short(short_document):
    """
    Tests that a short document results in a single, unaltered chunk.
    """
    chunks = chunk_document(short_document)
    
    assert chunks is not None
    assert isinstance(chunks, list)
    assert len(chunks) == 1
    assert chunks[0].chunk_text == short_document.extracted_text
    assert chunks[0].doc_id == "test-doc-short"

def test_chunk_overlap(long_document):
    """
    Tests that the chunks have the specified overlap.
    """
    chunks = chunk_document(long_document)
    
    if len(chunks) < 2:
        pytest.skip("Document was not long enough to create overlapping chunks")
        
    # Check the overlap between the first and second chunks
    chunk1_text = chunks[0].chunk_text
    chunk2_text = chunks[1].chunk_text
    
    # The overlap is 200 characters, but the splitter might adjust for boundaries.
    # We verify that the start of the second chunk (first 100 chars) 
    # is present in the end of the first chunk (last 200 chars).
    # This confirms overlap without being brittle to exact split points.
    assert chunk2_text[:100] in chunk1_text[-200:]

def test_chunk_size(long_document):
    """
    Tests that most chunks are at or below the chunk_size.
    (The last chunk might be smaller).
    """
    chunks = chunk_document(long_document)
    
    assert len(chunks) > 0
    
    # Check that all chunks (except potentially the last one) are within reasonable bounds
    for chunk in chunks[:-1]: # All but the last
        assert len(chunk.chunk_text) <= 1000
        # Small lead chunks should be merged to avoid ultra-short results
        assert len(chunk.chunk_text) >= 300

    assert len(chunks[-1].chunk_text) <= 1000 # Last chunk must also be <= 1000
