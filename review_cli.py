import csv
import sys

path = sys.argv[1]
rows = list(csv.DictReader(open(path)))
fields = list(rows[0].keys())


def save():
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


for i, r in enumerate(rows, 1):
    if r["manual_ok"] in ("0", "1"):
        continue
    print(f"\n[{i}/{len(rows)}] {r['kind']}")
    print("Q:", r["question"])
    print("A:", r["answer"])
    print("(automatic check:", "PASS)" if r["auto_passed"] == "True" else "FAIL)")
    while True:
        a = input("Does the answer read clearly and answer the question? 1 = yes, 0 = no: ").strip()
        if a in ("0", "1"):
            r["manual_ok"] = a
            save()
            break

disagree = [r for r in rows if (r["manual_ok"] == "1") != (r["auto_passed"] == "True")]
print(f"\nautomatic and manual agree on {len(rows) - len(disagree)}/{len(rows)}")
for r in disagree:
    print("DISAGREE:", r["kind"], "| auto", r["auto_passed"], "| manual", r["manual_ok"], "|", r["question"][:70])
