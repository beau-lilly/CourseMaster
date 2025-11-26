import pytest
from src.core.types import Chunk, PromptStyle
from src.core.rag import format_context, build_prompt, get_llm_response, PROMPT_REGISTRY

@pytest.fixture
def sample_chunks():
    return [
        Chunk(chunk_id="c1", doc_id="doc1", chunk_text="Python is a language.", chunk_index=0),
        Chunk(chunk_id="c2", doc_id="doc1", chunk_text="It is readable.", chunk_index=1),
    ]

def test_format_context(sample_chunks):
    """Test that chunks are serialized correctly with headers."""
    formatted = format_context(sample_chunks)
    assert "[Chunk 0] (Source: doc1):" in formatted
    assert "Python is a language." in formatted
    assert "[Chunk 1] (Source: doc1):" in formatted

def test_format_context_token_limit(sample_chunks):
    """Test that context formatting respects the token (char) budget."""
    # Increase budget slightly: 15 tokens * 4 chars/token = 60 chars
    # Chunk 1 (approx 47 chars) fits.
    # Chunk 2 would add another ~45 chars, exceeding total.
    formatted = format_context(sample_chunks, max_tokens=15) 
    assert "Python is a language" in formatted
    assert "It is readable" not in formatted

def test_build_prompt_switching(sample_chunks):
    """Test that different styles load different templates."""
    query = "What is Python?"
    
    # Test Minimal
    prompt_minimal = build_prompt(query, sample_chunks, PromptStyle.MINIMAL)
    assert "Keep your answer brief" in prompt_minimal
    assert query in prompt_minimal

    # Test Explanatory
    prompt_explain = build_prompt(query, sample_chunks, PromptStyle.EXPLANATORY)
    assert "You must cite the specific chunk index" in prompt_explain

def test_llm_stub_response(sample_chunks):
    """Test the generate function using the stub provider."""
    query = "Test Query"
    response = get_llm_response(query, sample_chunks, style=PromptStyle.MINIMAL, provider="stub")
    
    # The stub returns a predictable string
    assert "[STUB RESPONSE]" in response
    assert "Processed prompt length" in response
