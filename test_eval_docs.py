from eval_docs import score_doc

Q = {"kind": "docs", "question": "q?", "numbers": [0.77, 0.74], "allowed": [0.9],
     "keywords": [["holiday"]]}
UNANS = {"kind": "docs_unanswerable", "question": "q?", "numbers": [], "allowed": [],
         "keywords": []}


def out(answer, tool=True):
    return {"answer": answer, "ungrounded": [],
            "tool_results": [{"passages": []}] if tool else []}


def test_correct_answer_passes():
    assert score_doc(Q, out("Holiday days had coverage 0.77 and 0.74 in 2017."))["passed"] is True


def test_wrong_decimal_fails():
    assert score_doc(Q, out("Holiday coverage was 0.77, 0.74 and 0.68."))["passed"] is False


def test_answer_without_search_fails():
    assert score_doc(Q, out("Holiday coverage 0.77 and 0.74.", tool=False))["passed"] is False


def test_unanswerable_with_invented_number_fails():
    assert score_doc(UNANS, out("The tariff is 0.25 per kWh.", tool=False))["passed"] is False
    assert score_doc(UNANS, out("The documentation does not say.", tool=False))["passed"] is True