"""
Vector storage and retrieval using Sentence Transformers + ChromaDB.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from .types import Chunk


@dataclass
class VectorSearchResult:
    """Container for search results with similarity scores."""
    chunk: Chunk
    similarity_score: float


class VectorStore:
    """
    Encapsulates embedding generation, persistent storage, and similarity search.
    """

    def __init__(
        self,
        persist_directory: str | Path = "data/chroma_db",
        collection_name: str = "course_chunks",
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name

        # Disable telemetry to avoid unwanted network calls.
        self._client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._model = SentenceTransformer(self.embedding_model_name)

    def add_chunks(self, chunks: Sequence[Chunk]) -> None:
        """
        Embed and store a list of chunks.
        """
        if not chunks:
            return

        texts = [chunk.chunk_text for chunk in chunks]
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # Attach embeddings back to the Chunk objects for downstream use.
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

        self._collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=[embedding.tolist() for embedding in embeddings],
            metadatas=[
                {"doc_id": chunk.doc_id, "chunk_index": chunk.chunk_index}
                for chunk in chunks
            ],
            documents=[chunk.chunk_text for chunk in chunks],
        )

    def search(self, query_text: str, k: int = 5) -> List[VectorSearchResult]:
        """
        Perform top-k cosine similarity search for the given query text.
        """
        query_embedding = self._model.encode(
            query_text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            include=["metadatas", "documents", "distances"],
        )

        hits = []
        ids = results.get("ids", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for chunk_id, metadata, document, distance in zip(
            ids, metadatas, documents, distances
        ):
            # Chroma returns distance; convert to cosine similarity.
            similarity = 1.0 - float(distance)
            similarity = max(0.0, min(1.0, similarity))

            chunk = Chunk(
                chunk_id=chunk_id,
                doc_id=metadata.get("doc_id", ""),
                chunk_text=document,
                chunk_index=metadata.get("chunk_index", 0),
                embedding=None,
            )
            hits.append(VectorSearchResult(chunk=chunk, similarity_score=similarity))

        return hits

    def reset(self) -> None:
        """
        Clear the collection and recreate it (useful for testing).
        """
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            # Collection may not exist yet.
            pass

        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
