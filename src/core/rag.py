"""
LLM interaction and prompt engineering.
"""
import os
from typing import List, Optional, Dict, Any, Callable
from src.core.types import Chunk, PromptStyle

# --- 1. Prompt Templates ---

TEMPLATE_MINIMAL = """
You are a helpful assistant. Answer the user's question based ONLY on the provided context.
Keep your answer brief and to the point.

Context:
{context_str}

Question: {query}
Answer:
"""

TEMPLATE_EXPLANATORY = """
You are an expert tutor. Answer the user's question using the provided context.
You must cite the specific chunk index (e.g., [Chunk 1]) that supports each part of your answer.

Context:
{context_str}

Question: {query}
Answer (with citations):
"""

TEMPLATE_TUTORING = """
You are a Socratic tutor. Do not give the answer directly.
Instead, use the context to guide the user toward the answer with a hint or a leading question.

Context:
{context_str}

Question: {query}
Hint:
"""

TEMPLATE_SIMILARITY = """
Analyze why the following chunks were retrieved for the user's question.
Explain the relevance of each chunk to the query.

Context:
{context_str}

Question: {query}
Analysis:
"""

# --- 2. Prompt Registry ---

PROMPT_REGISTRY = {
    PromptStyle.MINIMAL: TEMPLATE_MINIMAL,
    PromptStyle.EXPLANATORY: TEMPLATE_EXPLANATORY,
    PromptStyle.TUTORING: TEMPLATE_TUTORING,
    PromptStyle.SIMILARITY: TEMPLATE_SIMILARITY,
}

# --- 3. Context Formatting ---

def format_context(chunks: List[Chunk], max_tokens: int = 2000) -> str:
    """
    Serializes chunks into a string format for the LLM.
    Format: [Chunk {index}] (Source: {doc_id}): {text}
    """
    formatted_parts = []
    current_length = 0
    
    # Simple character estimation (1 token ~= 4 chars)
    max_chars = max_tokens * 4

    for chunk in chunks:
        # Create the header for the chunk
        header = f"\n[Chunk {chunk.chunk_index}] (Source: {chunk.doc_id}):\n"
        content = chunk.chunk_text
        entry = header + content + "\n"
        
        if current_length + len(entry) > max_chars:
            break
            
        formatted_parts.append(entry)
        current_length += len(entry)
    
    return "".join(formatted_parts).strip()

# --- 4. Prompt Construction ---

def build_prompt(query: str, chunks: List[Chunk], style: PromptStyle) -> str:
    """
    Selects the template from the registry and fills it with context and query.
    """
    template = PROMPT_REGISTRY.get(style, TEMPLATE_MINIMAL)
    context_str = format_context(chunks)
    return template.format(context_str=context_str, query=query)

# --- 5. LLM Abstraction (The Testable Part) ---

class LLMProvider:
    """
    Abstracts the LLM provider. 
    In 'stub' mode, it just returns the prompt (perfect for testing).
    """
    def __init__(self, provider_type: str = "stub", **kwargs):
        self.provider_type = provider_type
        self.kwargs = kwargs

    def generate(self, prompt: str) -> str:
        if self.provider_type == "stub":
            return f"[STUB RESPONSE] Processed prompt length: {len(prompt)}"
        
        if self.provider_type == "openai":
            # Placeholder for future OpenAI implementation
            # return openai.ChatCompletion.create(...)
            pass
        
        if self.provider_type == "huggingface":
            # Placeholder for future HF implementation
            pass

        raise ValueError(f"Unknown provider type: {self.provider_type}")

# Singleton or factory usage
def get_llm_response(query: str, chunks: List[Chunk], style: PromptStyle = PromptStyle.MINIMAL, provider: str = "stub") -> str:
    """
    High-level function to coordinate prompt building and generation.
    """
    prompt = build_prompt(query, chunks, style)
    llm = LLMProvider(provider_type=provider)
    return llm.generate(prompt)