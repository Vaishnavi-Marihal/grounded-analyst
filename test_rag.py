import numpy as np

import rag


def fake_embed(texts):
    vocab = ["energy", "forecast", "holiday", "docker"]
    return np.array([[t.lower().count(w) for w in vocab] for t in texts], dtype="float32")


def test_chunk_text_splits_long_text():
    text = "\n\n".join(["para %d " % i + "x" * 300 for i in range(6)])
    chunks = rag.chunk_text(text, size=700, overlap=50)
    assert len(chunks) >= 3
    assert all(len(c) < 1100 for c in chunks)


def test_search_returns_most_similar_chunk():
    chunks = [
        {"source": "a.md", "chunk_id": 0, "text": "holiday holiday forecast"},
        {"source": "b.md", "chunk_id": 0, "text": "docker docker docker"},
    ]
    index = rag.build_index(chunks, embed=fake_embed)
    hits = rag.search(index, "docker", k=1, embed=fake_embed)
    assert hits[0]["source"] == "b.md"


def test_save_and_load_roundtrip(tmp_path):
    chunks = [{"source": "a.md", "chunk_id": 0, "text": "energy forecast"}]
    index = rag.build_index(chunks, embed=fake_embed)
    rag.save_index(index, tmp_path)
    loaded = rag.load_index(tmp_path)
    assert loaded["chunks"] == chunks
    assert loaded["vectors"].shape == (1, 4)