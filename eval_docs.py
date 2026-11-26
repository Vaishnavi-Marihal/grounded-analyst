import json
import os
import re
import sys

os.environ.setdefault("DOCS", "1")

import agent
from checker import extract_numbers, is_grounded

YEAR = re.compile(r"(19|20)\d{2}")


def score_doc(q, out):
    answer = out["answer"]
    tokens = extract_numbers(answer)
    asked = extract_numbers(q["question"])
    gold = [float(x) for x in q["numbers"]]
    allowed = gold + [float(x) for x in q["allowed"]]
    must_ok = all(any(is_grounded(t, [n]) for t in tokens) for n in gold)
    checked = [
        t for t in tokens
        if ("." in t or len(t) >= 3) and t not in asked and not YEAR.fullmatch(t)
    ]
    strict_ok = all(any(is_grounded(t, [a]) for a in allowed) for t in checked)
    keywords_ok = all(any(w in answer.lower() for w in group) for group in q["keywords"])
    used_tool = any(isinstance(t, dict) and "passages" in t for t in out["tool_results"])
    needs_tool = q["kind"] == "docs"
    grounded = not out["ungrounded"]
    return {
        "kind": q["kind"],
        "question": q["question"],
        "answer": answer,
        "grounded": grounded,
        "must_ok": must_ok,
        "strict_ok": strict_ok,
        "keywords_ok": keywords_ok,
        "used_tool": used_tool,
        "passed": grounded and must_ok and strict_ok and keywords_ok and (used_tool or not needs_tool),
    }


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else agent.MODEL
    questions = json.load(open(os.environ.get("QUESTIONS_FILE", "docs_questions.json")))
    rows = []
    for q in questions:
        try:
            out = agent.run_agent(q["question"], model=model)
        except Exception as exc:
            out = {"answer": f"CRASH: {exc}", "ungrounded": [], "tool_results": []}
        row = score_doc(q, out)
        rows.append(row)
        print(("PASS" if row["passed"] else "FAIL"), f"[{row['kind']}]", row["question"], flush=True)
    n = len(rows)
    print(f"\nMODEL {model}")
    print(f"passed        {sum(r['passed'] for r in rows)}/{n}")
    print(f"used search   {sum(r['used_tool'] for r in rows)}/{n}")
    print(f"strict fails  {sum(not r['strict_ok'] for r in rows)}/{n} (numbers not on the gold list)")
    print(f"missing gold  {sum(not r['must_ok'] for r in rows)}/{n}")
    print(f"missing words {sum(not r['keywords_ok'] for r in rows)}/{n}")
    os.makedirs("results", exist_ok=True)
    variant = os.environ.get("PROMPT_VARIANT", "v1")
    name = model.replace(":", "_") + ("" if variant == "v1" else "_" + variant)
    if os.environ.get("SEARCH_METHOD") == "hybrid":
        name += "_hybrid"
    if os.environ.get("QUESTIONS_FILE"):
        name += "_heldout"
    with open(f"results/docs_{name}.json", "w") as f:
        json.dump(rows, f, indent=1)
