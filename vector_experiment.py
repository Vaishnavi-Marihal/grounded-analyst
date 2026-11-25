import json
import os
import statistics
import time

import numpy as np

import rag
from vector_stores import FaissStore, PgVectorStore, recall_at_k

QUESTION_FILES = ["docs_questions.json", "held_out_questions.json"]
N_SYNTHETIC = 30000
K = 3


def load_queries():
    texts = []
    for name in QUESTION_FILES:
        texts += [q["question"] for q in json.load(open(name)) if q["kind"] == "docs"]
    vectors = rag.normalize(rag.embed_texts(["search_query: " + t for t in texts]))
    return texts, vectors


def numpy_top(vectors, query, k=K):
    scores = vectors @ query
    top = np.argpartition(-scores, k)[:k]
    return [int(i) for i in top[np.argsort(-scores[top])]]


def timed(fn, queries):
    results, times = [], []
    for q in queries:
        start = time.perf_counter()
        results.append(fn(q))
        times.append((time.perf_counter() - start) * 1000)
    return results, times


def summarize(times):
    ordered = sorted(times)
    return {
        "median_ms": round(statistics.median(times), 3),
        "p95_ms": round(ordered[int(0.95 * len(ordered)) - 1], 3),
    }


def correctness(queries, vectors):
    pg = PgVectorStore(table="items_real")
    pg.build(vectors)
    faiss_store = FaissStore(vectors)
    truth = [numpy_top(vectors, q) for q in queries]
    faiss_found = [faiss_store.search(q, K) for q in queries]
    pg_found = [pg.search(q, K) for q in queries]
    return {
        "n_queries": len(queries),
        "faiss_same_order": sum(a == b for a, b in zip(faiss_found, truth)),
        "faiss_same_set": sum(set(a) == set(b) for a, b in zip(faiss_found, truth)),
        "pgvector_same_order": sum(a == b for a, b in zip(pg_found, truth)),
        "pgvector_same_set": sum(set(a) == set(b) for a, b in zip(pg_found, truth)),
    }


def scale(queries, real_vectors):
    rng = np.random.default_rng(0)
    base = real_vectors[rng.integers(0, len(real_vectors), N_SYNTHETIC)]
    vectors = rag.normalize(base + rng.normal(scale=0.01, size=base.shape)).astype("float32")
    timing_queries = np.tile(queries, (5, 1))
    out = {"n_vectors": N_SYNTHETIC, "n_queries": len(timing_queries)}

    truth, times = timed(lambda q: numpy_top(vectors, q), timing_queries)
    out["numpy_exact"] = summarize(times)

    start = time.perf_counter()
    faiss_store = FaissStore(vectors)
    build_s = time.perf_counter() - start
    found, times = timed(lambda q: faiss_store.search(q, K), timing_queries)
    out["faiss_flat"] = {
        "build_s": round(build_s, 2), **summarize(times),
        "memory_mb": round(vectors.nbytes / 1e6, 1),
        "recall_vs_numpy": round(recall_at_k(found, truth, K), 3),
    }

    pg = PgVectorStore(table="items")
    pg.conn.execute("SET max_parallel_maintenance_workers = 0")
    pg.conn.execute("SET maintenance_work_mem = '512MB'")
    start = time.perf_counter()
    pg.build(vectors)
    load_s = time.perf_counter() - start
    pg.conn.execute("ANALYZE items")
    found, times = timed(lambda q: pg.search(q, K), timing_queries)
    size_mb = pg.conn.execute("SELECT pg_total_relation_size('items')").fetchone()[0] / 1e6
    out["pgvector_exact"] = {
        "load_s": round(load_s, 2), **summarize(times),
        "table_mb": round(size_mb, 1),
        "recall_vs_numpy": round(recall_at_k(found, truth, K), 3),
    }

    start = time.perf_counter()
    pg.add_hnsw_index()
    index_s = time.perf_counter() - start
    found, times = timed(lambda q: pg.search(q, K), timing_queries)
    size_mb = pg.conn.execute("SELECT pg_total_relation_size('items')").fetchone()[0] / 1e6
    out["pgvector_hnsw"] = {
        "index_build_s": round(index_s, 2), **summarize(times),
        "table_plus_index_mb": round(size_mb, 1),
        "recall_vs_numpy": round(recall_at_k(found, truth, K), 3),
    }
    return out


if __name__ == "__main__":
    index = rag.load_index()
    texts, queries = load_queries()
    result = {
        "correctness": correctness(queries, index["vectors"]),
        "scale": scale(queries, index["vectors"]),
    }
    os.makedirs("results", exist_ok=True)
    with open("results/vector_store_comparison.json", "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))