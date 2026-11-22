import json
import sys

a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))
same_answer = sum(x["answer"] == y["answer"] for x, y in zip(a, b))
same_pass = sum(x["passed"] == y["passed"] for x, y in zip(a, b))
print(len(a), "rows vs", len(b), "rows")
print(f"identical answers {same_answer}/{len(a)}")
print(f"same pass/fail    {same_pass}/{len(a)}")
shown = 0
for x, y in zip(a, b):
    if x["answer"] != y["answer"] and shown < 5:
        shown += 1
        print("\n" + x["question"][:70])
        print("  loop :", x["answer"][:140].replace("\n", " "))
        print("  graph:", y["answer"][:140].replace("\n", " "))