import csv
import json
import os
import random
import re
import sys
from collections import Counter

import agent
import questions as qmod
from checker import extract_numbers, is_grounded

QUESTIONS_FILE = "questions.json"


def load_questions():
    if os.path.exists(QUESTIONS_FILE):
        with open(QUESTIONS_FILE) as f:
            return json.load(f)
    built = qmod.build_questions()
    with open(QUESTIONS_FILE, "w") as f:
        json.dump(built, f, indent=1)
    return built


def score(question, out):
    answer = out["answer"]
    tokens = extract_numbers(answer)
    grounded = not out["ungrounded"]
    correct = all(any(is_grounded(t, [n]) for t in tokens) for n in question["numbers"])
    forbidden = any(re.search(p, answer.lower()) for p in question["forbid"])
    malformed = out["malformed_tool_call"]
    cache_miss = any(
        isinstance(r, dict) and "not in cache" in str(r.get("error", ""))
        for r in out.get("tool_results", [])
    )
    failed = "error" in out
    return {
        "kind": question["kind"],
        "question": question["question"],
        "expected": question["numbers"],
        "answer": answer,
        "tool_results": out.get("tool_results", []),
        "ungrounded": out["ungrounded"],
        "grounded": grounded,
        "correct": correct,
        "forbidden": forbidden,
        "malformed": malformed,
        "cache_miss": cache_miss,
        "passed": grounded and correct and not forbidden and not malformed
        and not failed and not cache_miss,
    }


def run(model):
    rows = []
    for q in load_questions():
        try:
            out = agent.run_agent(q["question"], model=model)
        except Exception as exc:
            out = {"answer": f"CRASH: {exc}", "ungrounded": [],
                   "malformed_tool_call": False, "error": str(exc)}
        row = score(q, out)
        rows.append(row)
        print(("PASS" if row["passed"] else "FAIL"), f"[{row['kind']}]", row["question"], flush=True)
    return rows


def summarize(model, rows):
    n = len(rows)
    print(f"\nMODEL {model}")
    print(f"passed        {sum(r['passed'] for r in rows)}/{n}")
    print(f"ungrounded    {sum(not r['grounded'] for r in rows)}/{n}")
    print(f"wrong number  {sum(not r['correct'] for r in rows)}/{n}")
    print(f"agreed w/ bad {sum(r['forbidden'] for r in rows)}/{n}")
    print(f"malformed     {sum(r['malformed'] for r in rows)}/{n}")
    print(f"cache misses  {sum(r['cache_miss'] for r in rows)}/{n} (not the model's fault)")
    for kind in sorted({r["kind"] for r in rows}):
        group = [r for r in rows if r["kind"] == kind]
        print(f"  {kind:14s} {sum(r['passed'] for r in group)}/{len(group)}")


def write_review(model, rows, name):
    sample = random.Random(0).sample(rows, min(15, len(rows)))
    with open(f"results/{name}_review.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["kind", "question", "answer", "auto_passed", "manual_ok"])
        for r in sample:
            writer.writerow([r["kind"], r["question"], r["answer"], r["passed"], ""])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--build":
        built = load_questions()
        print(len(built), "questions")
        for kind, count in sorted(Counter(x["kind"] for x in built).items()):
            print(f"  {kind:14s} {count}")
        sys.exit()
    model = sys.argv[1] if len(sys.argv) > 1 else agent.MODEL
    rows = run(model)
    summarize(model, rows)
    os.makedirs("results", exist_ok=True)
    name = model.replace(":", "_").replace("/", "_")
    with open(f"results/{name}.json", "w") as f:
        json.dump(rows, f, indent=1)
    write_review(model, rows, name)