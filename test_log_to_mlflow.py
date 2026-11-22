from log_to_mlflow import bootstrap_ci, parse


def test_parse_flags():
    p = parse("llama3.1_8b_v2_repair_t2")
    assert p["model"] == "llama3.1:8b"
    assert p["prompt"] == "v2"
    assert p["repair"] is True
    assert p["forgiving_tools"] is True


def test_parse_docs_run():
    p = parse("docs_qwen2.5_7b_v3_hybrid_heldout")
    assert p["suite"] == "docs"
    assert p["search"] == "hybrid"
    assert p["heldout"] is True


def test_parse_ignores_other_files():
    assert parse("run_llama31") is None


def test_bootstrap_ci_bounds():
    assert bootstrap_ci([True] * 70) == (1.0, 1.0)
    low, high = bootstrap_ci([True, False] * 35)
    assert low < 0.5 < high