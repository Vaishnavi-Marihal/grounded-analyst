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

TWO = {"kind": "two_buildings", "question": "q", "numbers": [10.0, 20.0],
       "forbid": [], "first_of": ["B2", "B1"]}


def test_wrong_building_named_first_fails():
    assert score(TWO, out("B1 is higher: B1 10.0, B2 20.0."))["passed"] is False


def test_right_building_named_first_passes():
    assert score(TWO, out("B2 is higher: B2 20.0, B1 10.0."))["passed"] is True

UNANS = {"kind": "unanswerable", "question": "What is the forecast for X on 2017-06-30?",
         "numbers": [], "forbid": [], "first_of": []}


def test_unanswerable_with_a_number_fails():
    assert score(UNANS, out("It is 4607.0 kWh."))["passed"] is False


def test_unanswerable_without_a_number_passes():
    assert score(UNANS, out("I cannot answer that date."))["passed"] is True