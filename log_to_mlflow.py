import glob
import json
import pathlib
import random

import mlflow

KNOWN = {"qwen2.5_7b": "qwen2.5:7b", "llama3.1_8b": "llama3.1:8b", "llama3.2": "llama3.2"}


def parse(stem):
    is_docs = stem.startswith("docs_")
    name = stem[5:] if is_docs else stem
    for key, model in KNOWN.items():
        if name.startswith(key):
            flags = {t for t in name[len(key):].split("_") if t}
            impl = "graph" if "graph" in flags else "langchain" if "langchain" in flags else "loop"
            return {
                "model": model,
                "suite": "docs" if is_docs else "main70",
                "prompt": next((f for f in flags if f in ("v2", "v3")), "v1"),
                "repair": "repair" in flags,
                "forgiving_tools": "t2" in flags,
                "agent_impl": impl,
                "search": "hybrid" if "hybrid" in flags else "dense",
                "heldout": "heldout" in flags,
                "note": ",".join(sorted(flags & {"run1", "before", "gold3"})),
            }
    return None


def bootstrap_ci(passed, n_boot=2000, seed=0):
    rng = random.Random(seed)
    n = len(passed)
    rates = sorted(
        sum(passed[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot)
    )
    return rates[int(0.025 * n_boot)], rates[int(0.975 * n_boot) - 1]


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("grounded-analyst")
    for path in sorted(glob.glob("results/*.json")):
        stem = pathlib.Path(path).stem
        params = parse(stem)
        if params is None:
            continue
        rows = json.load(open(path))
        passed = [bool(r["passed"]) for r in rows]
        low, high = bootstrap_ci(passed)
        with mlflow.start_run(run_name=stem):
            mlflow.log_params(params)
            mlflow.log_metric("n_questions", len(rows))
            mlflow.log_metric("pass_rate", sum(passed) / len(rows))
            mlflow.log_metric("pass_rate_ci_low", low)
            mlflow.log_metric("pass_rate_ci_high", high)
            for kind in sorted({r["kind"] for r in rows}):
                group = [bool(r["passed"]) for r in rows if r["kind"] == kind]
                mlflow.log_metric(f"pass_rate_{kind}", sum(group) / len(group))
            if any("infra_error" in r for r in rows):
                mlflow.log_metric("infra_errors", sum(bool(r.get("infra_error")) for r in rows))
            mlflow.log_artifact(path)
        print(f"{stem:55s} {sum(passed)}/{len(rows)}  95% CI {low:.2f}-{high:.2f}")


if __name__ == "__main__":
    main()