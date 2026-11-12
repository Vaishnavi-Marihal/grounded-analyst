import json
import os
import sys

import agent
import tools
from checker import extract_numbers, is_grounded

DATES = ["2017-12-25", "2017-09-04", "2017-11-23"]


def build_questions():
    buildings = [b["id"] for b in tools.list_buildings()[:3]]
    questions = []
    for b, d in zip(buildings, DATES):
        s = tools.get_day_forecast(b, d)
        questions.append({
            "kind": "total",
            "question": f"What is the total forecast for {b} on {d}?",
            "expected": s["total_forecast_kwh"],
        })
        questions.append({
            "kind": "peak",
            "question": f"What is the peak hourly forecast for {b} on {d}?",
            "expected": s["peak_forecast_kwh"],
        })
        if "total_actual_kwh" in s:
            gap = round(abs(s["total_forecast_kwh"] - s["total_actual_kwh"]), 1)
            questions.append({
                "kind": "gap",
                "question": f"How far was the total forecast for {b} on {d} from the actual total?",
                "expected": gap,
            })
    metrics = tools.get_building_metrics(buildings[0])
    questions.append({
        "kind": "mae",
        "question": f"What was the model's mean absolute error for {buildings[0]}?",
        "expected": metrics["mae_model"],
    })
    questions.append({
        "kind": "count",
        "question": "How many buildings does the evaluation cover?",
        "expected": tools.get_metrics_summary()["n_buildings"],
    })
    questions.append({
        "kind": "unanswerable",
        "question": f"What is the forecast for {buildings[0]} on 2018-03-01?",
        "expected": None,
    })
    questions.append({
        "kind": "unanswerable",
        "question": "What is the forecast for Nowhere_Hall on 2017-12-25?",
        "expected": None,
    })
    return questions


def score(question, out):
    grounded = not out["ungrounded"]
    if question["expected"] is None:
        correct = True
    else:
        correct = any(
            is_grounded(t, [question["expected"]])
            for t in extract_numbers(out["answer"])
        )
    malformed = out["malformed_tool_call"]
    failed = "error" in out
    return {
        "kind": question["kind"],
        "question": question["question"],
        "expected": question["expected"],
        "answer": out["answer"],
        "tool_results": out["tool_results"],
        "ungrounded": out["ungrounded"],
        "grounded": grounded,
        "correct": correct,
        "malformed": malformed,
        "passed": grounded and correct and not malformed and not failed,
    }


def run(model):
    rows = []
    for q in build_questions():
        try:
            out = agent.run_agent(q["question"], model=model)
        except Exception as exc:
            out = {"answer": f"CRASH: {exc}", "ungrounded": [],
                   "malformed_tool_call": False, "error": str(exc)}
        row = score(q, out)
        rows.append(row)
        mark = "PASS" if row["passed"] else "FAIL"
        print(f"{mark} [{row['kind']}] {row['question']}")
        if not row["passed"]:
            print(f"     grounded={row['grounded']} correct={row['correct']} "
                  f"malformed={row['malformed']} ungrounded={row['ungrounded']}")
    return rows


def summarize(model, rows):
    n = len(rows)
    print(f"\nMODEL {model}")
    print(f"passed      {sum(r['passed'] for r in rows)}/{n}")
    print(f"ungrounded  {sum(not r['grounded'] for r in rows)}/{n}")
    print(f"wrong       {sum(not r['correct'] for r in rows)}/{n}")
    print(f"malformed   {sum(r['malformed'] for r in rows)}/{n}")
    for kind in sorted({r["kind"] for r in rows}):
        group = [r for r in rows if r["kind"] == kind]
        print(f"  {kind:13s} {sum(r['passed'] for r in group)}/{len(group)}")


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else agent.MODEL
    rows = run(model)
    summarize(model, rows)
    os.makedirs("results", exist_ok=True)
    name = model.replace(":", "_").replace("/", "_")
    with open(f"results/{name}.json", "w") as f:
        json.dump(rows, f, indent=1)