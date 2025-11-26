"""
RAG Orchestration using LangChain.
"""
import os
from typing import List, Optional

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import FakeListLLM
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None  # Fallback if library not present

from src.core.types import Chunk, PromptStyle, RAGResult
from src.core.vector_store import VectorStore
from src.core.database import DatabaseManager

# --- 1. Prompt Templates ---

TEMPLATE_MINIMAL = """You are a helpful assistant. Answer the user's question based ONLY on the provided context.
Keep your answer brief and to the point.

Context:
{context}

Question: {question}
Answer:
"""

TEMPLATE_EXPLANATORY = """You are an expert tutor. Answer the user's question using the provided context.
You must cite the specific chunk index (e.g., [Chunk 1]) that supports each part of your answer.

Context:
{context}

Question: {question}
Answer (with citations):
"""

TEMPLATE_TUTORING = """You are a Socratic tutor. Do not give the answer directly.
Instead, use the context to guide the user toward the answer with a hint or a leading question.

Context:
{context}

Question: {question}
Hint:
"""

TEMPLATE_SIMILARITY = """Analyze why the following chunks were retrieved for the user's question.
Explain the relevance of each chunk to the query.

Context:
{context}

Question: {question}
Analysis:
"""

PROMPT_TEMPLATES = {
    PromptStyle.MINIMAL: PromptTemplate.from_template(TEMPLATE_MINIMAL),
    PromptStyle.EXPLANATORY: PromptTemplate.from_template(TEMPLATE_EXPLANATORY),
    PromptStyle.TUTORING: PromptTemplate.from_template(TEMPLATE_TUTORING),
    PromptStyle.SIMILARITY: PromptTemplate.from_template(TEMPLATE_SIMILARITY),
}

# --- 2. Helpers ---

def format_docs(chunks: List[Chunk]) -> str:
    """Formats chunks for the prompt."""
    return "\n\n".join(
        f"[Chunk {c.chunk_index}] (Source: {c.doc_id}): {c.chunk_text}" 
        for c in chunks
    )

# --- 3. Orchestration ---

def answer_question(
    question_text: str,
    prompt_style: PromptStyle = PromptStyle.MINIMAL,
    k: int = 5,
    chunk_ids: Optional[List[str]] = None,
    vector_store: Optional[VectorStore] = None,
    db_manager: Optional[DatabaseManager] = None
) -> RAGResult:
    """
    Orchestrates the RAG pipeline:
    1. Logs the problem.
    2. Retrieves chunks (via search or ID).
    3. Logs retrieval.
    4. Generates answer using LangChain.
    """
    
    # Init dependencies if not provided
    if vector_store is None:
        # Only initialize if we need to search
        if not chunk_ids:
            vector_store = VectorStore()
            
    if db_manager is None:
        db_manager = DatabaseManager()
        
    # 1. Log Problem
    problem = db_manager.add_problem(question_text)
    
    # 2. Retrieve Chunks
    chunks: List[Chunk] = []
    scores: List[float] = []
    
    if chunk_ids:
        # Preselected path
        chunks = db_manager.get_chunks_by_ids(chunk_ids)
        # Assign dummy scores (1.0) for manual selection
        scores = [1.0] * len(chunks)
    else:
        # Search path
        if vector_store:
            results = vector_store.search(question_text, k=k)
            chunks = [r.chunk for r in results]
            scores = [r.similarity_score for r in results]
            
            # Log retrieval
            for r in results:
                db_manager.log_retrieval(problem.problem_id, r.chunk.chunk_id, r.similarity_score)
        else:
            # Should not happen if logic above is correct, but safe fallback
            chunks = []
            
    # 3. Fallback check
    if not chunks:
        return RAGResult(
            question=question_text,
            answer="No context found for this query.",
            used_chunks=[],
            scores=[]
        )
        
    # 4. Build Chain
    # Select template
    prompt_template = PROMPT_TEMPLATES.get(prompt_style, PROMPT_TEMPLATES[PromptStyle.MINIMAL])
    
    # Initialize LLM
    llm = None
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if api_key and ChatOpenAI:
        # Use real LLM if API key is present
        llm = ChatOpenAI(
            model="gpt-4o-mini",  # Or "gpt-3.5-turbo"
            temperature=0,
            openai_api_key=api_key
        )
    else:
        # Use Stub LLM if no API key or explicitly requested
        llm = FakeListLLM(responses=[f"[STUB RESPONSE] Processed prompt for query: {question_text}"])
    
    # Define chain using LCEL
    chain = (
        prompt_template 
        | llm 
        | StrOutputParser()
    )
    
    # 5. Execute
    context_str = format_docs(chunks)
    answer = chain.invoke({"context": context_str, "question": question_text})
    
    return RAGResult(
        question=question_text,
        answer=answer,
        used_chunks=chunks,
        scores=scores
    )

# Backward compatibility/alias if needed, or can be removed
def get_llm_response(*args, **kwargs):
    raise DeprecationWarning("Use answer_question instead.")
