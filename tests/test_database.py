# tests/test_database.py

import os
import shutil
import numpy as np
import uuid
from numpy.testing import assert_array_equal

# This import will now work thanks to Step 3!
from src.core.database import DatabaseManager
from src.core.types import Chunk

# We'll use a temporary DB file for this test
TEST_DB_PATH = "tests/test_suite.db" # Save the test DB inside the tests folder
TEST_CHROMA_PATH = "tests/test_suite_chroma_db"

def test_database_operations():
    """
    Tests the full add/get cycle for documents, problems, and chunks.
    Pytest will automatically find and run this function.
    """
    print(f"Starting test suite... (using {TEST_DB_PATH})")
    
    # Ensure no old test DB is lying around
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
    if os.path.exists(TEST_CHROMA_PATH):
        shutil.rmtree(TEST_CHROMA_PATH)

    db = None
    try:
        # 1. INITIALIZATION TEST
        db = DatabaseManager(db_path=TEST_DB_PATH)
        assert os.path.exists(TEST_DB_PATH)
        # Chroma folder is created lazily or on init depending on version, check if collection exists
        assert db.collection is not None
        print("  [PASS] 1. Initialization: Database and Chroma collection created.")

        # 2. DOCUMENT ADD/GET TEST
        doc = db.add_document("test_doc.pdf", "Full document text.")
        assert doc.original_filename == "test_doc.pdf"
        print("  [PASS] 2. Documents: add_document() works.")

        # 3. PROBLEM ADD/GET TEST
        prob = db.add_problem("Test problem text.")
        assert prob.problem_text == "Test problem text."
        print("  [PASS] 3. Problems: add_problem() works.")

        # 4. CHUNK & EMBEDDING I/O TEST
        fake_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        test_chunk = Chunk(
            chunk_id=f"chunk_{uuid.uuid4()}",
            doc_id=doc.doc_id,
            chunk_text="This is a test chunk.",
            chunk_index=0,
            embedding=fake_embedding
        )
        db.save_chunks([test_chunk])

        # 4a. Test get_chunk_text (from SQLite)
        retrieved_text = db.get_chunk_text(test_chunk.chunk_id)
        assert retrieved_text == "This is a test chunk."
        print("  [PASS] 4a. Chunks: get_chunk_text() works.")

        # 4b. Test query_similar_chunks (from ChromaDB)
        # We query with the exact same embedding, should find the chunk
        results = db.query_similar_chunks(fake_embedding, n_results=1)
        
        assert len(results) == 1
        assert results[0]['chunk_id'] == test_chunk.chunk_id
        assert results[0]['chunk_text'] == "This is a test chunk."
        print("  [PASS] 4b. Chunks: query_similar_chunks() works.")

        print("\n--- TEST FUNCTION PASSED! ---")

    finally:
        # 5. CLEANUP
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)
        if os.path.exists(TEST_CHROMA_PATH):
            shutil.rmtree(TEST_CHROMA_PATH)
        print(f"Cleanup: Removed test files.")