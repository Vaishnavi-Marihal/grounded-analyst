import json
from collections import Counter

NAMES = ["qwen2.5_7b", "llama3.1_8b", "llama3.2"]
GIVE_UP = ["unable", "cannot", "can't", "unfortunately", "not available"]
NEEDS_CALC = ("gap", "percent", "baseline_gaps")


def label(r):
    results = r["tool_results"]
    used_calc = any(isinstance(t, dict) and "operation" in t for t in results)
    if r["malformed"]:
        return "tool call written as text"
    if not results:
        return "no tool called"
    if not r["grounded"]:
        return "own arithmetic or invented number"
    if r["kind"] in NEEDS_CALC and not used_calc:
        return "skipped the calculator"
    if any(w in r["answer"].lower() for w in GIVE_UP):
        return "gave up or refused"
    return "wrong but grounded"


for name in NAMES:
    rows = json.load(open(f"results/{name}.json"))
    fails = [r for r in rows if not r["passed"]]
    print(f"\n{name}: {len(fails)} failures")
    for mode, count in Counter(label(r) for r in fails).most_common():
        print(f"  {count:3d}  {mode}")

for name in ["llama3.1_8b", "llama3.2"]:
    rows = json.load(open(f"results/{name}.json"))
    for kind in ("gap", "false_premise"):
        r = next(x for x in rows if x["kind"] == kind)
        print(f"\n{name} | {kind} | passed={r['passed']}")
        print("  answer:", r["answer"][:260].replace("\n", " "))