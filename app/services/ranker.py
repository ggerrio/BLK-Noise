"""Rank candidate listings by cosine similarity to a query."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Sequence

import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_TOP_N = 5


@lru_cache(maxsize=1)
def get_model() -> Any:
    """Load the sentence-transformers model once (import is lazy)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string as a normalised float32 vector."""
    model = get_model()
    vectors = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)[0]


def decode_embedding(value: Any) -> np.ndarray | None:
    """Turn a stored float32 BLOB (or list) into a 1-D vector."""
    if value is None:
        return None

    if isinstance(value, np.ndarray):
        vector = value.astype(np.float32, copy=False)
    elif isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        if not raw or len(raw) % 4 != 0:
            return None
        vector = np.frombuffer(raw, dtype=np.float32)
    elif isinstance(value, (list, tuple)):
        vector = np.asarray(value, dtype=np.float32)
    else:
        return None

    if vector.ndim != 1 or vector.size == 0:
        return None
    return vector


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity, returning 0.0 when either vector is all zeros."""
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def rank(
    query: str,
    rows: Sequence[dict[str, Any]],
    top_n: int = DEFAULT_TOP_N,
) -> list[dict[str, Any]]:
    """Return the `top_n` rows most similar to `query`.

    Rows without a usable embedding cannot be ranked and are skipped. Each
    returned row is a copy with an added ``score`` key.
    """
    if not rows or top_n <= 0:
        return []

    query_vector = embed_query(query)

    scored: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        vector = decode_embedding(row.get("embedding"))
        if vector is None or vector.shape != query_vector.shape:
            continue
        score = cosine_similarity(query_vector, vector)
        scored.append((score, {**row, "score": round(score, 6)}))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [row for _, row in scored[:top_n]]
