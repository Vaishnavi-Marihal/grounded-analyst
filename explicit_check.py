import json

from checker import corrects_claim

for name in ["qwen2.5_7b", "llama3.1_8b", "llama3.2"]:
    rows = [r for r in json.load(open(f"results/{name}.json")) if r["kind"] == "false_premise"]
    n = sum(corrects_claim(r["question"], r["answer"]) for r in rows)
    print(f"{name}: explicit correction {n}/{len(rows)}")