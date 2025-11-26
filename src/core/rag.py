"""
RAG Orchestration using LangChain.
"""
import os
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import FakeListLLM
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None  # Fallback if library not present

from src.core.types import Chunk, PromptStyle, RAGResult
from src.core.vector_store import VectorStore
from src.core.database import DatabaseManager

# Load environment variables from .env if present so API keys are available.
load_dotenv()

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

def _openrouter_headers() -> Dict[str, str]:
    """Optional headers OpenRouter recommends for attribution."""
    headers: Dict[str, str] = {}
    referer = os.environ.get("OPENROUTER_SITE_URL")
    app_name = os.environ.get("OPENROUTER_APP_NAME")
    if referer:
        headers["HTTP-Referer"] = referer
    if app_name:
        headers["X-Title"] = app_name
    return headers


def build_llm(question_text: str) -> Tuple[object, str]:
    """
    Selects the best available LLM:
    1) OpenRouter if OPENROUTER_API_KEY is set.
    2) Direct OpenAI if OPENAI_API_KEY is set.
    3) A stubbed FakeListLLM for offline/testing.

    Returns:
        (llm_instance, provider_label)
    """
    # Prefer OpenRouter for real responses
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_key and ChatOpenAI:
        model = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        headers = _openrouter_headers()
        llm = ChatOpenAI(
            model=model,
            temperature=0,
            openai_api_key=openrouter_key,
            base_url=base_url,
            default_headers=headers or None,
        )
        return llm, "openrouter"

    # Fall back to vanilla OpenAI if configured
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key and ChatOpenAI:
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        llm = ChatOpenAI(
            model=model,
            temperature=0,
            openai_api_key=openai_key,
        )
        return llm, "openai"

    # Otherwise stay offline with a deterministic stub
    stub = FakeListLLM(responses=[f"[STUB RESPONSE] Processed prompt for query: {question_text}"])
    return stub, "stub"

# --- 3. Orchestration ---

def answer_question(
    question_text: str,
    exam_id: str,
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
    problem = db_manager.add_problem(question_text, exam_id=exam_id)
    
    # 2. Retrieve Chunks
    chunks: List[Chunk] = []
    scores: List[float] = []
    
    if chunk_ids:
        # Preselected path
        chunks = db_manager.get_chunks_by_ids(chunk_ids)
        # Assign dummy scores (1.0) for manual selection
        scores = [1.0] * len(chunks)
    else:
        allowed_doc_ids = db_manager.get_document_ids_for_exam(exam_id)
        if not allowed_doc_ids:
            return RAGResult(
                question=question_text,
                answer="No documents are linked to this exam yet. Upload or attach documents first.",
                used_chunks=[],
                scores=[]
            )
        # Search path
        if vector_store:
            results = vector_store.search(question_text, k=k, allowed_doc_ids=allowed_doc_ids)
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
    
    # Initialize LLM (preferring OpenRouter)
    llm, _provider = build_llm(question_text)
    
    # Define chain using LCEL
    chain = prompt_template | llm | StrOutputParser()
    
    # 5. Execute
    context_str = format_docs(chunks)
    try:
        answer = chain.invoke({"context": context_str, "question": question_text})
    except Exception as exc:
        # Avoid crashing the app if the remote LLM errors (e.g., bad key/network)
        print(f"LLM invocation failed ({exc}); falling back to stub.")
        fallback = FakeListLLM(
            responses=[f"[STUB RESPONSE] Unable to reach LLM ({exc}). Query: {question_text}"]
        )
        fallback_chain = prompt_template | fallback | StrOutputParser()
        answer = fallback_chain.invoke({"context": context_str, "question": question_text})
    
    return RAGResult(
        question=question_text,
        answer=answer,
        used_chunks=chunks,
        scores=scores
    )

# Backward compatibility/alias if needed, or can be removed
def get_llm_response(*args, **kwargs):
    raise DeprecationWarning("Use answer_question instead.")
