import pytest
from unittest.mock import MagicMock, patch
from langchain_community.llms import FakeListLLM
from src.core.types import Chunk, PromptStyle, RAGResult
from src.core.rag import format_docs, answer_question, build_llm, PROMPT_TEMPLATES

@pytest.fixture
def sample_chunks():
    return [
        Chunk(chunk_id="c1", doc_id="doc1", chunk_text="Python is a language.", chunk_index=0),
        Chunk(chunk_id="c2", doc_id="doc1", chunk_text="It is readable.", chunk_index=1),
    ]

def test_format_docs(sample_chunks):
    """Test that chunks are serialized correctly with headers."""
    formatted = format_docs(sample_chunks)
    assert "[Chunk 0] (Source: doc1): Python is a language." in formatted
    assert "[Chunk 1] (Source: doc1): It is readable." in formatted

def test_answer_question_stub(sample_chunks):
    """Test the orchestration function using the stub provider."""
    query = "What is Python?"
    
    # Ensure no API key is set for this test to trigger stub
    with patch.dict('os.environ', {}, clear=True):
        # Mock dependencies to avoid real DB/VectorStore calls
        mock_db = MagicMock()
        mock_db.add_problem.return_value = MagicMock(problem_id="prob-1")
        mock_db.get_document_ids_for_exam.return_value = ["doc1"]
        # When searching, return sample_chunks
        mock_vector_store = MagicMock()
        mock_vector_store.search.return_value = [
            MagicMock(chunk=c, similarity_score=0.9) for c in sample_chunks
        ]
        
        result = answer_question(
            question_text=query,
            exam_id="exam-123",
            prompt_style=PromptStyle.MINIMAL,
            vector_store=mock_vector_store,
            db_manager=mock_db
        )
        
        assert isinstance(result, RAGResult)
        assert result.question == query
        assert len(result.used_chunks) == 2
        # The stub response defined in rag.py
        assert "[STUB RESPONSE]" in result.answer
        assert query in result.answer

def test_answer_question_styles(sample_chunks):
    """Test that the function accepts different styles without error."""
    query = "Explain Python"
    mock_db = MagicMock()
    mock_vector_store = MagicMock()
    mock_db.add_problem.return_value = MagicMock(problem_id="prob-1")
    mock_db.get_document_ids_for_exam.return_value = []
    mock_vector_store.search.return_value = []
    
    # Even with no chunks, it should handle the style (though it hits fallback)
    with patch.dict("os.environ", {}, clear=True):
        result = answer_question(
            question_text=query,
            exam_id="exam-123",
            prompt_style=PromptStyle.EXPLANATORY,
            vector_store=mock_vector_store,
            db_manager=mock_db
        )
    assert result.question == query
    # Should prompt the user to attach documents first
    assert "No documents are linked to this exam yet" in result.answer

def test_answer_question_preselected_chunks(sample_chunks):
    """Test bypassing search by providing chunk_ids."""
    query = "Specific context"
    mock_db = MagicMock()
    mock_db.get_chunks_by_ids.return_value = sample_chunks
    mock_db.add_problem.return_value = MagicMock(problem_id="prob-1")
    
    with patch.dict("os.environ", {}, clear=True):
        result = answer_question(
            question_text=query,
            exam_id="exam-123",
            chunk_ids=["c1", "c2"],
            db_manager=mock_db
        )
    
    # Should have used the chunks from get_chunks_by_ids
    assert len(result.used_chunks) == 2
    assert result.used_chunks[0].chunk_id == "c1"
    # Should not have called vector store (it's None by default)


def test_build_llm_prefers_openrouter(monkeypatch):
    """Ensure OpenRouter config is used when the key is present."""

    class DummyLLM:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs
            self.responses = ["ok"]

    # Swap ChatOpenAI for a capturing stub and clear env to only include OpenRouter vars
    monkeypatch.setattr("src.core.rag.ChatOpenAI", DummyLLM)
    with patch.dict(
        "os.environ",
        {"OPENROUTER_API_KEY": "test-key", "OPENROUTER_MODEL": "openai/gpt-4o-mini"},
        clear=True,
    ):
        llm, provider = build_llm("Hello?")

    assert provider == "openrouter"
    assert llm.kwargs["openai_api_key"] == "test-key"
    assert llm.kwargs["base_url"] == "https://openrouter.ai/api/v1"
    assert llm.kwargs["model"] == "openai/gpt-4o-mini"
