"""Embedding providers.

``onnx`` runs all-MiniLM-L6-v2 locally, so ticket text never leaves the host for
embedding. ``hashing`` is a dependency-free stand-in for tests: it has no notion
of meaning and must not be used for real retrieval.
"""

import hashlib
import math
import re
from typing import Protocol

from app.config import Settings


class Embedder(Protocol):
    """Turns text into unit-length vectors."""

    version: str

    def embed(self, text: str) -> list[float]: ...


class OnnxEmbedder:
    """Local sentence embeddings through Chroma's bundled ONNX model."""

    def __init__(self, model_name: str) -> None:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self._function = DefaultEmbeddingFunction()
        self.version = f"onnx/{model_name}"

    def embed(self, text: str) -> list[float]:
        return [float(value) for value in self._function([text])[0]]


class HashingEmbedder:
    """Deterministic bag-of-words hashing, for tests only."""

    def __init__(self, dimensions: int = 256) -> None:
        self._dimensions = dimensions
        self.version = f"hashing/{dimensions}"

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.sha256(token.encode()).digest()
            vector[int.from_bytes(digest[:4], "big") % self._dimensions] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


def build_embedder(settings: Settings) -> Embedder:
    """Create the configured embedder."""
    if settings.embedding_provider == "onnx":
        return OnnxEmbedder(settings.embedding_model)
    if settings.embedding_provider == "hashing":
        return HashingEmbedder()
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {settings.embedding_provider}")
