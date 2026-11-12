from types import SimpleNamespace

import agent


def reply(content=None, calls=None):
    return SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=calls))


CALL = SimpleNamespace(
    function=SimpleNamespace(
        name="get_day_forecast",
        arguments={"building_id": "B", "date": "2017-12-25"},
    )
)


def calc_call(a, b):
    return SimpleNamespace(
        function=SimpleNamespace(
            name="calculate",
            arguments={"operation": "subtract", "a": a, "b": b},
        )
    )


def setup(monkeypatch, replies):
    it = iter(replies)
    monkeypatch.setattr(agent.ollama, "chat", lambda **kwargs: next(it))
    monkeypatch.setitem(
        agent.REGISTRY, "get_day_forecast",
        lambda building_id, date: {"total_forecast_kwh": 60.0, "total_actual_kwh": 50.0},
    )


def test_grounded_answer_passes(monkeypatch):
    setup(monkeypatch, [reply(calls=[CALL]), reply(content="The total forecast is 60 kWh.")])
    out = agent.run_agent("question")
    assert out["ungrounded"] == []


def test_invented_number_is_flagged(monkeypatch):
    setup(monkeypatch, [reply(calls=[CALL]), reply(content="The total forecast is 75 kWh.")])
    out = agent.run_agent("question")
    assert out["ungrounded"] == ["75"]


def test_number_without_any_tool_call_is_flagged(monkeypatch):
    setup(monkeypatch, [reply(content="It is 60 kWh.")])
    out = agent.run_agent("question")
    assert out["ungrounded"] == ["60"]


def test_calculated_difference_is_grounded(monkeypatch):
    setup(monkeypatch, [
        reply(calls=[CALL]),
        reply(calls=[calc_call(60.0, 50.0)]),
        reply(content="The difference is 10 kWh."),
    ])
    assert agent.run_agent("question")["ungrounded"] == []


def test_calculator_rejects_invented_operand(monkeypatch):
    setup(monkeypatch, [
        reply(calls=[CALL]),
        reply(calls=[calc_call(61.0, 50.0)]),
        reply(content="The difference is 11 kWh."),
    ])
    assert agent.run_agent("question")["ungrounded"] == ["11"]


def test_json_in_text_is_marked_malformed(monkeypatch):
    setup(monkeypatch, [reply(content='{"name": "calculate", "parameters": {"a": "1"}}')])
    out = agent.run_agent("question")
    assert out["malformed_tool_call"] is True