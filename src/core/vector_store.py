"""
Vector storage and retrieval using Sentence Transformers + ChromaDB via LangChain.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document as LCDocument

from .types import Chunk

# Suppress fork/parallelism warnings from tokenizers used by sentence-transformers.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

@dataclass
class VectorSearchResult:
    """Container for search results with similarity scores."""
    chunk: Chunk
    similarity_score: float


class VectorStore:
    """
    Encapsulates embedding generation, persistent storage, and similarity search
    using LangChain components.
    """

    def __init__(
        self,
        persist_directory: str | Path = "data/chroma_db",
        collection_name: str = "course_chunks",
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ):
        self.persist_directory = str(persist_directory)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        
        # Initialize Embeddings
        # This will download the model if not present
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        
        # Initialize Chroma
        # collection_metadata={"hnsw:space": "cosine"} ensures cosine similarity
        self.db = Chroma(
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            collection_metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: Sequence[Chunk]) -> None:
        """
        Embed and store a list of chunks.
        """
        if not chunks:
            return

        # Convert Chunks to LangChain Documents
        documents = [
            LCDocument(
                page_content=chunk.chunk_text,
                metadata={
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "chunk_index": chunk.chunk_index
                },
                id=chunk.chunk_id 
            )
            for chunk in chunks
        ]
        
        self.db.add_documents(documents)

    def search(self, query_text: str, k: int = 5) -> List[VectorSearchResult]:
        """
        Perform top-k cosine similarity search for the given query text.
        Returns chunks with cosine SIMILARITY (1 = identical, 0 = opposite).
        Duplicate chunk texts are collapsed so re-ingested files do not flood results.
        """
        fetch_k = max(k * 4, k + 5)  # Grab extra to account for deduplication
        # similarity_search_with_score in LangChain with cosine distance returns DISTANCE (lower is better).
        # Distance = 1 - Similarity (for normalized vectors).
        results = self.db.similarity_search_with_score(query_text, k=fetch_k)
        
        hits = []
        for doc, distance in results:
            # Convert cosine distance to similarity
            similarity = 1.0 - distance
            
            chunk = Chunk(
                chunk_id=doc.metadata.get("chunk_id", "") or (doc.id if hasattr(doc, 'id') else ""),
                doc_id=doc.metadata.get("doc_id", ""),
                chunk_text=doc.page_content,
                chunk_index=doc.metadata.get("chunk_index", 0),
                embedding=None,
            )
            hits.append(VectorSearchResult(chunk=chunk, similarity_score=similarity))

        return self._deduplicate_hits(hits, limit=k)

    def get_retriever(self, k: int = 5):
        """Returns a LangChain retriever interface."""
        return self.db.as_retriever(search_kwargs={"k": k})

    def reset(self) -> None:
        """
        Clear the collection and recreate it (useful for testing).
        """
        try:
            self.db.delete_collection()
        except Exception:
            pass
        
        # Re-initialize
        self.db = Chroma(
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            collection_metadata={"hnsw:space": "cosine"}
        )

    def delete_chunks(self, chunk_ids: Sequence[str]) -> None:
        """
        Remove specific chunk embeddings from the vector store.
        """
        if not chunk_ids:
            return
        try:
            self.db.delete(ids=list(chunk_ids))
        except Exception:
            # Best effort; ignore if ids are missing
            pass

    @staticmethod
    def _normalize_chunk_text(text: str) -> str:
        """Collapse whitespace and case to compare chunk bodies reliably."""
        return " ".join(text.split()).strip().lower()

    @staticmethod
    def _deduplicate_hits(
        hits: Sequence[VectorSearchResult], limit: int
    ) -> List[VectorSearchResult]:
        """
        Remove duplicate chunk bodies while preserving score order.
        """
        seen_texts = set()
        unique_hits: List[VectorSearchResult] = []

        for hit in hits:
            normalized = VectorStore._normalize_chunk_text(hit.chunk.chunk_text)
            if not normalized:
                continue
            if normalized in seen_texts:
                continue
            seen_texts.add(normalized)
            unique_hits.append(hit)
            if len(unique_hits) >= limit:
                break

        return unique_hits
