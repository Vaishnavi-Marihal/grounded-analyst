from types import SimpleNamespace

import graph_agent


def reply(content=None, calls=None):
    return SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=calls))


CALL = SimpleNamespace(
    function=SimpleNamespace(
        name="get_day_forecast",
        arguments={"building_id": "B", "date": "2017-12-25"},
    )
)


def setup(monkeypatch, replies):
    it = iter(replies)
    monkeypatch.setattr(graph_agent.ollama, "chat", lambda **kwargs: next(it))
    monkeypatch.setitem(
        graph_agent.agent.REGISTRY, "get_day_forecast",
        lambda building_id, date: {"total_forecast_kwh": 60.0, "total_actual_kwh": 50.0},
    )


def test_graph_grounded_answer(monkeypatch):
    setup(monkeypatch, [reply(calls=[CALL]), reply(content="The total forecast is 60 kWh.")])
    out = graph_agent.run_agent("question")
    assert out["ungrounded"] == []
    assert out["repaired_calls"] == 0


def test_graph_invented_number_is_flagged(monkeypatch):
    setup(monkeypatch, [reply(calls=[CALL]), reply(content="The total forecast is 75 kWh.")])
    assert graph_agent.run_agent("question")["ungrounded"] == ["75"]


def test_graph_step_limit_gives_error(monkeypatch):
    setup(monkeypatch, [reply(calls=[CALL])] * 3)
    out = graph_agent.run_agent("question", max_steps=3)
    assert out["answer"] == ""
    assert out["error"] == "max steps reached"


def test_graph_repair_runs_a_text_tool_call(monkeypatch):
    monkeypatch.setenv("REPAIR", "1")
    text = '{"name": "get_day_forecast", "parameters": {"building_id": "B", "date": "2017-12-25"}}'
    setup(monkeypatch, [reply(content=text), reply(content="The total forecast is 60 kWh.")])
    out = graph_agent.run_agent("question")
    assert out["repaired_calls"] == 1
    assert out["ungrounded"] == []