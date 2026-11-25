import numpy as np

DSN = "postgresql://postgres:devpass@localhost:5433/postgres"


def to_vector_literal(vector):
    return "[" + ",".join(f"{x:.6f}" for x in vector) + "]"


def recall_at_k(found, truth, k=3):
    hits = [len(set(f[:k]) & set(t[:k])) / k for f, t in zip(found, truth)]
    return sum(hits) / len(hits)


class FaissStore:
    def __init__(self, vectors):
        import faiss

        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(np.ascontiguousarray(vectors, dtype="float32"))

    def search(self, query, k=3):
        q = np.ascontiguousarray(query[None, :], dtype="float32")
        _, ids = self.index.search(q, k)
        return [int(i) for i in ids[0]]


class PgVectorStore:
    def __init__(self, dsn=DSN, table="items"):
        import psycopg

        self.conn = psycopg.connect(dsn, autocommit=True)
        self.table = table
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    def build(self, vectors):
        dim = vectors.shape[1]
        self.conn.execute(f"DROP TABLE IF EXISTS {self.table}")
        self.conn.execute(
            f"CREATE TABLE {self.table} (id int PRIMARY KEY, embedding vector({dim}))"
        )
        with self.conn.cursor() as cur:
            with cur.copy(f"COPY {self.table} (id, embedding) FROM STDIN") as copy:
                for i, v in enumerate(vectors):
                    copy.write_row((i, to_vector_literal(v)))

    def add_hnsw_index(self):
        self.conn.execute(
            f"CREATE INDEX ON {self.table} USING hnsw (embedding vector_cosine_ops)"
        )

    def search(self, query, k=3):
        rows = self.conn.execute(
            f"SELECT id FROM {self.table} ORDER BY embedding <=> %s::vector LIMIT %s",
            (to_vector_literal(query), k),
        ).fetchall()
        return [r[0] for r in rows]