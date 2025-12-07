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
from src.core.database import DatabaseManager

# Load environment variables from .env if present so API keys are available.
load_dotenv()

# --- 1. Prompt Templates ---

TEMPLATE_MINIMAL = """You are a helpful assistant. The user is working on the problem below.

Problem:
{problem_text}

Answer the user's question about this problem based on the provided context.
Keep your answer brief and to the point.

Context (retrieved chunks related to the problem):
{context}

Question about the problem: {question}
Answer:
"""

TEMPLATE_EXPLANATORY = """You are an expert tutor. The user is working on the problem below.

Problem:
{problem_text}

Answer the user's question about this problem using the provided context.
You must cite the specific chunk index (e.g., [Chunk 1]) that supports each part of your answer.

Context (retrieved chunks related to the problem):
{context}

Question about the problem: {question}
Answer (with citations):
"""

TEMPLATE_TUTORING = """You are a Socratic tutor helping with the problem below.

Problem:
{problem_text}

Do not give the answer directly. Instead, use the context to guide the user toward the answer with a hint or a leading question.

Context (retrieved chunks related to the problem):
{context}

Question about the problem: {question}
Hint:
"""

TEMPLATE_SIMILARITY = """Analyze why the following chunks were retrieved for the user's question about the problem below.

Problem:
{problem_text}

Explain the relevance of each chunk to the query.

Context (retrieved chunks related to the problem):
{context}

Question about the problem: {question}
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
    problem_id: str,
    prompt_style: PromptStyle = PromptStyle.MINIMAL,
    db_manager: Optional[DatabaseManager] = None
) -> RAGResult:
    """
    Answer a question about a specific problem using precomputed retrievals.
    1. Loads the problem and its logged chunks.
    2. Creates a question row.
    3. Generates an answer using LangChain.
    4. Stores the answer.
    """

    if db_manager is None:
        db_manager = DatabaseManager()

    problem = db_manager.get_problem(problem_id)
    if not problem:
        return RAGResult(
            question=question_text,
            answer="Problem not found.",
            used_chunks=[],
            scores=[],
        )

    # Record the question first so we have an ID to attach the answer to.
    stored_question = db_manager.add_question(
        problem_id=problem_id,
        question_text=question_text,
        prompt_style=prompt_style.value,
        answer_text="",
    )

    # 2. Retrieve precomputed chunks for the problem
    chunk_pairs: List[tuple[Chunk, float]] = db_manager.get_chunks_for_problem(problem_id)
    if not chunk_pairs:
        message = "No context has been logged for this problem yet. Add documents and re-run problem ingestion."
        db_manager.update_question_answer(stored_question.question_id, message)
        return RAGResult(
            question=question_text,
            answer=message,
            used_chunks=[],
            scores=[],
            question_id=stored_question.question_id,
        )

    chunks = [c for c, _ in chunk_pairs]
    scores: List[float] = [score for _, score in chunk_pairs]

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
        answer = chain.invoke(
            {
                "context": context_str,
                "question": question_text,
                "problem_text": problem.problem_text,
            }
        )
    except Exception as exc:
        # Avoid crashing the app if the remote LLM errors (e.g., bad key/network)
        print(f"LLM invocation failed ({exc}); falling back to stub.")
        fallback = FakeListLLM(
            responses=[f"[STUB RESPONSE] Unable to reach LLM ({exc}). Query: {question_text}"]
        )
        fallback_chain = prompt_template | fallback | StrOutputParser()
        answer = fallback_chain.invoke(
            {
                "context": context_str,
                "question": question_text,
                "problem_text": problem.problem_text,
            }
        )

    # Persist answer text for the question record
    db_manager.update_question_answer(stored_question.question_id, answer)
    
    return RAGResult(
        question=question_text,
        answer=answer,
        used_chunks=chunks,
        scores=scores,
        question_id=stored_question.question_id,
    )

# Backward compatibility/alias if needed, or can be removed
def get_llm_response(*args, **kwargs):
    raise DeprecationWarning("Use answer_question instead.")
