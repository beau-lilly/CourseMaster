import hashlib
from typing import List


class HashEmbeddings:
    """
    Simple deterministic embedding function for tests.
    Produces small vectors derived from a SHA-256 hash so retrieval is stable without network downloads.
    """

    def __init__(self, dim: int = 16):
        self.dim = dim

    def _embed(self, text: str) -> List[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # Use the first dim bytes normalized to [0, 1]
        return [byte / 255 for byte in digest[: self.dim]]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)
