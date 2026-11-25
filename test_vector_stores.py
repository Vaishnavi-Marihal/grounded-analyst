import numpy as np

from vector_stores import FaissStore, recall_at_k, to_vector_literal


def test_vector_literal():
    assert to_vector_literal([0.5, 1.0]) == "[0.500000,1.000000]"


def test_recall_at_k():
    assert recall_at_k([[1, 2, 3]], [[3, 2, 9]], k=3) == 2 / 3


def test_faiss_matches_numpy():
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(50, 8)).astype("float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    query = rng.normal(size=8).astype("float32")
    query /= np.linalg.norm(query)
    expected = [int(i) for i in np.argsort(-(vectors @ query))[:3]]
    assert FaissStore(vectors).search(query, 3) == expected