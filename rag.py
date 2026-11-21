import json
import pathlib
import re
import sys

import numpy as np
import ollama
from rank_bm25 import BM25Okapi

DOCS_DIR = pathlib.Path("docs")
INDEX_DIR = pathlib.Path("index")
EMBED_MODEL = "nomic-embed-text"


def chunk_text(text, size=800, overlap=150):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for p in paragraphs:
        if current and len(current) + len(p) > size:
            chunks.append(current)
            tail = current[-overlap:]
            tail = tail[tail.find(" ") + 1:]
            current = tail + "\n\n" + p
        else:
            current = (current + "\n\n" + p) if current else p
    if current:
        chunks.append(current)
    return chunks


def load_chunks(docs_dir=DOCS_DIR):
    chunks = []
    for path in sorted(pathlib.Path(docs_dir).glob("*.md")):
        for i, text in enumerate(chunk_text(path.read_text())):
            chunks.append({"source": path.name, "chunk_id": i, "text": text})
    return chunks


def embed_texts(texts):
    out = ollama.embed(model=EMBED_MODEL, input=list(texts))
    return np.array(out["embeddings"], dtype="float32")


def normalize(vectors):
    return vectors / (np.linalg.norm(vectors, axis=-1, keepdims=True) + 1e-9)


def build_index(chunks, embed=embed_texts):
    vectors = normalize(embed(["search_document: " + c["text"] for c in chunks]))
    return {"chunks": chunks, "vectors": vectors}


def save_index(index, directory=INDEX_DIR):
    directory = pathlib.Path(directory)
    directory.mkdir(exist_ok=True)
    np.save(directory / "vectors.npy", index["vectors"])
    (directory / "chunks.json").write_text(json.dumps(index["chunks"]))


def load_index(directory=INDEX_DIR):
    directory = pathlib.Path(directory)
    return {
        "chunks": json.loads((directory / "chunks.json").read_text()),
        "vectors": np.load(directory / "vectors.npy"),
    }


def tokenize(text):
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())


def dense_ranking(index, query, embed=embed_texts):
    q = normalize(embed(["search_query: " + query]))[0]
    return list(np.argsort(-(index["vectors"] @ q)))


def bm25_ranking(index, query):
    bm25 = BM25Okapi([tokenize(c["text"]) for c in index["chunks"]])
    return list(np.argsort(-bm25.get_scores(tokenize(query))))


def search(index, query, k=3, embed=embed_texts):
    q = normalize(embed(["search_query: " + query]))[0]
    scores = index["vectors"] @ q
    top = np.argsort(-scores)[:k]
    return [{**index["chunks"][i], "score": round(float(scores[i]), 3)} for i in top]


def hybrid_search(index, query, k=3, embed=embed_texts, rrf_k=60):
    fused = {}
    for ranking in (dense_ranking(index, query, embed), bm25_ranking(index, query)):
        for rank, i in enumerate(ranking, 1):
            fused[i] = fused.get(i, 0.0) + 1.0 / (rrf_k + rank)
    top = sorted(fused, key=fused.get, reverse=True)[:k]
    return [{**index["chunks"][i], "score": round(fused[i], 4)} for i in top]


if __name__ == "__main__":
    if sys.argv[1] == "build":
        chunks = load_chunks()
        save_index(build_index(chunks))
        print(len(chunks), "chunks indexed")
    else:
        for hit in search(load_index(), " ".join(sys.argv[1:])):
            print(hit["score"], hit["source"], "|", hit["text"][:300].replace("\n", " "))