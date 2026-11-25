import numpy as np

import rag
from vector_experiment import K, N_SYNTHETIC, load_queries, numpy_top, summarize, timed
from vector_stores import PgVectorStore, recall_at_k

if __name__ == "__main__":
    index = rag.load_index()
    _, queries = load_queries()
    real = index["vectors"]
    rng = np.random.default_rng(0)
    base = real[rng.integers(0, len(real), N_SYNTHETIC)]
    vectors = rag.normalize(base + rng.normal(scale=0.01, size=base.shape)).astype("float32")
    timing_queries = np.tile(queries, (5, 1))
    truth = [numpy_top(vectors, q) for q in timing_queries]
    pg = PgVectorStore(table="items")
    for ef in (40, 100, 200, 400):
        pg.conn.execute(f"SET hnsw.ef_search = {ef}")
        found, times = timed(lambda q: pg.search(q, K), timing_queries)
        gaps = []
        for q, f, t in zip(timing_queries, found, truth):
            exact = sorted((float(vectors[i] @ q) for i in t), reverse=True)
            got = sorted((float(vectors[i] @ q) for i in f), reverse=True)
            gaps.append(np.mean([e - g for e, g in zip(exact, got)]))
        print(
            f"ef_search={ef:4d}  recall@3={recall_at_k(found, truth, K):.3f}  "
            f"mean score gap={np.mean(gaps):.5f}  median_ms={summarize(times)['median_ms']}"
        )