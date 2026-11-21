import json
import sys

import rag

GOLD = {
    "What was the interval coverage on the three flagged days before": "0.77, 0.74 and 0.74",
    "On the three flagged days, what was the median ratio": "0.92, 0.68 and 0.70",
    "What coverage did the three previously flagged holidays reach": "0.83, 0.84 and 0.81",
    "Which US public holidays were": "Labor Day, Thanksgiving, Christmas Day",
    "Why did the monitoring report flag": "had no holiday information",
    "How many monthly folds": "6 monthly folds",
    "Which model serves all buildings": "One global LightGBM model",
    "When is each forecast issued": "Every forecast is issued at 23:00",
    "How many buildings beat last week's value when their site": "48 of 56",
    "Which two sites did not beat": "Lamb (1.22",
    "Did adding weather data": "Weather did not help",
    "How is the demo deployed": "one AWS EC2 server",
}

METHODS = {"dense": rag.search, "hybrid": rag.hybrid_search}


def rank_of_gold(search_fn, index, question, gold, k=10):
    hits = search_fn(index, question, k=k)
    for rank, hit in enumerate(hits, 1):
        if gold in hit["text"]:
            return rank
    return None


if __name__ == "__main__":
    method = sys.argv[1] if len(sys.argv) > 1 else "dense"
    search_fn = METHODS[method]
    index = rag.load_index()
    questions = [q for q in json.load(open("docs_questions.json")) if q["kind"] == "docs"]
    ranks = []
    for q in questions:
        gold = next((g for p, g in GOLD.items() if q["question"].startswith(p)), None)
        if gold is None:
            continue
        r = rank_of_gold(search_fn, index, q["question"], gold)
        ranks.append(r)
        print(r, "|", q["question"][:70])
    n = len(ranks)
    print(f"\nMETHOD {method}")
    for k in (1, 3, 5):
        print(f"recall@{k}: {sum(1 for r in ranks if r and r <= k)}/{n}")
    print("MRR:", round(sum(1 / r for r in ranks if r) / n, 3))