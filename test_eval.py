from eval import score


def out(answer, ungrounded=None, tool_results=None):
    return {"answer": answer, "ungrounded": ungrounded or [],
            "tool_results": tool_results or [], "malformed_tool_call": False}


GAP = {"kind": "gap", "question": "q", "numbers": [79.1], "forbid": []}
PREMISE = {"kind": "false_premise", "question": "q", "numbers": [60.0],
           "forbid": [r"\byes\b"]}


def test_passes_when_number_present():
    assert score(GAP, out("It was 79.1 kWh off."))["passed"] is True


def test_fails_when_number_missing():
    assert score(GAP, out("It was 80 kWh off."))["passed"] is False


def test_agreeing_with_false_premise_fails():
    assert score(PREMISE, out("Yes, it was 60 kWh."))["passed"] is False


def test_cache_miss_is_not_a_pass():
    row = score(GAP, out("79.1", tool_results=[{"error": "'not in cache: /x'"}]))
    assert row["passed"] is False and row["cache_miss"] is True