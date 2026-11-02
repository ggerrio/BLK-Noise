"""Tests for app/services/ranker.py with a fake embedding model.

The model is replaced so ranking is checked against hand-made vectors, with no
downloads and deterministic scores.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.services import ranker


class FakeModel:
    """Returns a fixed vector for each known input text."""

    def __init__(self, by_text: dict[str, list[float]]):
        self.by_text = by_text
        self.calls: list[list[str]] = []

    def encode(self, texts, **_kwargs):
        self.calls.append(list(texts))
        return np.array([self.by_text[text] for text in texts], dtype=np.float32)


def blob(values) -> bytes:
    return np.asarray(values, dtype=np.float32).tobytes()


@pytest.fixture
def use_query_vector(monkeypatch):
    """Point the ranker at a fake model that embeds the query as [1, 0, 0]."""

    def install(vector=(1.0, 0.0, 0.0)):
        model = FakeModel({"query": list(vector)})
        monkeypatch.setattr(ranker, "get_model", lambda: model)
        return model

    return install


ROWS = [
    {"id": 1, "town": "BEDOK", "embedding": blob((1.0, 0.0, 0.0))},      # 1.0
    {"id": 2, "town": "TAMPINES", "embedding": blob((0.0, 1.0, 0.0))},  # 0.0
    {"id": 3, "town": "WOODLANDS", "embedding": blob((0.9, 0.1, 0.0))},  # ~0.994
    {"id": 4, "town": "TOA PAYOH", "embedding": blob((-1.0, 0.0, 0.0))},  # -1.0
]


def test_rank_orders_by_cosine_similarity(use_query_vector):
    use_query_vector()

    ranked = ranker.rank("query", ROWS, top_n=4)

    assert [row["id"] for row in ranked] == [1, 3, 2, 4]
    assert ranked[0]["score"] == pytest.approx(1.0)
    assert ranked[1]["score"] == pytest.approx(0.9 / np.linalg.norm([0.9, 0.1]), abs=1e-5)
    assert ranked[2]["score"] == pytest.approx(0.0)
    assert ranked[3]["score"] == pytest.approx(-1.0)


def test_rank_adds_score_and_keeps_original_fields(use_query_vector):
    use_query_vector()

    (best,) = ranker.rank("query", ROWS, top_n=1)

    assert best["town"] == "BEDOK"
    assert best["embedding"] == ROWS[0]["embedding"]
    assert isinstance(best["score"], float)


def test_rank_truncates_to_top_n(use_query_vector):
    use_query_vector()

    assert [row["id"] for row in ranker.rank("query", ROWS, top_n=2)] == [1, 3]


def test_rank_handles_empty_rows(use_query_vector):
    use_query_vector()

    assert ranker.rank("query", [], top_n=5) == []


def test_rank_returns_empty_for_non_positive_top_n(use_query_vector):
    use_query_vector()

    assert ranker.rank("query", ROWS, top_n=0) == []


def test_rank_skips_rows_without_embedding(use_query_vector):
    use_query_vector()
    rows = [
        {"id": 1, "embedding": blob((1.0, 0.0, 0.0))},
        {"id": 2, "embedding": None},
        {"id": 3},
        {"id": 4, "embedding": b""},
        {"id": 5, "embedding": blob((0.0, 1.0, 0.0))},
    ]

    ranked = ranker.rank("query", rows, top_n=10)

    assert [row["id"] for row in ranked] == [1, 5]


def test_rank_skips_vectors_with_wrong_dimension(use_query_vector):
    use_query_vector()
    rows = [
        {"id": 1, "embedding": blob((1.0, 0.0, 0.0, 0.0))},
        {"id": 2, "embedding": blob((0.0, 1.0, 0.0))},
    ]

    assert [row["id"] for row in ranker.rank("query", rows, top_n=5)] == [2]


def test_rank_accepts_list_embeddings(use_query_vector):
    use_query_vector()
    rows = [{"id": 1, "embedding": [0.0, 1.0, 0.0]}, {"id": 2, "embedding": [1, 0, 0]}]

    assert [row["id"] for row in ranker.rank("query", rows, top_n=2)] == [2, 1]


def test_cosine_similarity_edge_cases():
    assert ranker.cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == 1.0
    assert ranker.cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 0.0
    assert ranker.cosine_similarity(np.array([0.0, 0.0]), np.array([1.0, 0.0])) == 0.0


def test_model_is_loaded_once(monkeypatch):
    ranker.get_model.cache_clear()
    loads: list[int] = []

    class Dummy:
        def encode(self, texts, **_kwargs):
            return np.ones((len(texts), 3), dtype=np.float32)

    def fake_sentence_transformers():
        import types

        module = types.ModuleType("sentence_transformers")

        def SentenceTransformer(_name):
            loads.append(1)
            return Dummy()

        module.SentenceTransformer = SentenceTransformer
        return module

    monkeypatch.setitem(
        __import__("sys").modules,
        "sentence_transformers",
        fake_sentence_transformers(),
    )

    first = ranker.get_model()
    second = ranker.get_model()

    assert first is second
    assert len(loads) == 1
    ranker.get_model.cache_clear()
