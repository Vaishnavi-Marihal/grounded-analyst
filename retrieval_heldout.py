import json

import rag
from retrieval_eval import METHODS, rank_of_gold

GOLD = {
    "How many tests does the project": "63 tests in total",
    "How many buildings and sites": "56 buildings from 14 sites",
    "Why was the Healthcare": "Healthcare was not an option",
    "How was the weather comparison": "Weather did not help",
    "Which database does the project": "Readings live in PostgreSQL",
    "Which licence covers": "The source code is under the MIT licence",
    "What caveats does the README": "median of only 4 buildings",
    "How is a dead hour defined": "if the reading is zero or missing",
    "When does the monitoring report flag": "limits I chose, not derived",
}

if __name__ == "__main__":
    index = rag.load_index()
    questions = [q for q in json.load(open("held_out_questions.json")) if q["kind"] == "docs"]
    for name, fn in METHODS.items():
        ranks = []
        for q in questions:
            gold = next(g for p, g in GOLD.items() if q["question"].startswith(p))
            ranks.append(rank_of_gold(fn, index, q["question"], gold))
        n = len(ranks)
        print(f"\n{name}: ranks {ranks}")
        for k in (1, 3):
            print(f"  recall@{k}: {sum(1 for r in ranks if r and r <= k)}/{n}")
        print("  MRR:", round(sum(1 / r for r in ranks if r) / n, 3))
