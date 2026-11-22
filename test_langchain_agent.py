from langchain_core.messages import AIMessage

import langchain_agent


class FakeLLM:
    def __init__(self, replies):
        self.replies = iter(replies)

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return next(self.replies)


def tool_call_message():
    return AIMessage(content="", tool_calls=[
        {"name": "get_day_forecast", "args": {"building_id": "B", "date": "2017-12-25"}, "id": "1"}
    ])


def setup(monkeypatch, replies):
    monkeypatch.setattr(langchain_agent, "ChatOllama", lambda **kwargs: FakeLLM(replies))
    monkeypatch.setitem(
        langchain_agent.agent.REGISTRY, "get_day_forecast",
        lambda building_id, date: {"total_forecast_kwh": 60.0, "total_actual_kwh": 50.0},
    )


def test_langchain_grounded_answer(monkeypatch):
    setup(monkeypatch, [tool_call_message(), AIMessage(content="The total forecast is 60 kWh.")])
    out = langchain_agent.run_agent("question")
    assert out["ungrounded"] == []


def test_langchain_invented_number_is_flagged(monkeypatch):
    setup(monkeypatch, [tool_call_message(), AIMessage(content="The total forecast is 75 kWh.")])
    assert langchain_agent.run_agent("question")["ungrounded"] == ["75"]


def test_langchain_step_limit(monkeypatch):
    setup(monkeypatch, [tool_call_message()] * 3)
    out = langchain_agent.run_agent("question", max_steps=3)
    assert out["error"] == "max steps reached"


def test_langchain_repair(monkeypatch):
    monkeypatch.setenv("REPAIR", "1")
    text = '{"name": "get_day_forecast", "parameters": {"building_id": "B", "date": "2017-12-25"}}'
    setup(monkeypatch, [AIMessage(content=text), AIMessage(content="The total forecast is 60 kWh.")])
    out = langchain_agent.run_agent("question")
    assert out["repaired_calls"] == 1