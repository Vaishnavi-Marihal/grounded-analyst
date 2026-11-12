import json
import sys

import ollama

import tools
from checker import collect_numbers, find_ungrounded, is_grounded

MODEL = "llama3.2"

SYSTEM_PROMPT = (
    "You answer questions about building electricity forecasts. "
    "Always call a tool before stating any number. "
    "Only use numbers that appear in tool results. "
    "For any difference or percentage, call the calculate tool with numbers "
    "copied from earlier tool results. Never calculate or round numbers yourself. "
    "If the tools cannot answer the question, say so."
)

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "get_day_forecast",
            "description": (
                "Day-ahead forecast summary for one building on one date between "
                "2017-07-01 and 2017-12-31: total forecast kWh, peak kWh and hour, "
                "mean interval width, last-week baseline total and actual total."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "building_id": {"type": "string", "description": "for example Bear_education_Bob"},
                    "date": {"type": "string", "description": "YYYY-MM-DD"},
                },
                "required": ["building_id", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_building_metrics",
            "description": (
                "Accuracy of the model for one building over July to December 2017: "
                "model error, baseline errors, interval coverage and width."
            ),
            "parameters": {
                "type": "object",
                "properties": {"building_id": {"type": "string"}},
                "required": ["building_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_metrics_summary",
            "description": "Accuracy summary across all 56 buildings.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Difference or percentage difference between two numbers that "
                "came from earlier tool results. subtract gives a minus b. "
                "percent_difference gives (a minus b) divided by b, times 100."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["subtract", "percent_difference"]},
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["operation", "a", "b"],
            },
        },
    },
]

REGISTRY = {
    "get_day_forecast": tools.get_day_forecast,
    "get_building_metrics": tools.get_building_metrics,
    "get_metrics_summary": tools.get_metrics_summary,
}


def operands_grounded(args, allowed):
    try:
        return all(is_grounded(str(args[k]), allowed) for k in ("a", "b"))
    except (KeyError, ValueError):
        return False


def run_agent(question, model=MODEL, max_steps=5):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    tool_results = []
    for _ in range(max_steps):
        response = ollama.chat(model=model, messages=messages, tools=TOOL_SPECS,
                               options={"temperature": 0})
        message = response.message
        messages.append(message)
        if not message.tool_calls:
            answer = message.content or ""
            return {
                "answer": answer,
                "ungrounded": find_ungrounded(answer, collect_numbers(tool_results)),
                "tool_results": tool_results,
                "malformed_tool_call": '"name"' in answer and '"parameters"' in answer,
            }
        for call in message.tool_calls:
            name = call.function.name
            args = dict(call.function.arguments)
            if name == "calculate":
                if operands_grounded(args, collect_numbers(tool_results)):
                    result = tools.calculate(**args)
                else:
                    result = {"error": "a and b must be numbers copied from earlier tool results"}
            else:
                try:
                    result = REGISTRY[name](**args)
                except Exception as exc:
                    result = {"error": str(exc)}
            tool_results.append(result)
            messages.append(
                {"role": "tool", "tool_name": name, "content": json.dumps(result)}
            )
    return {"answer": "", "ungrounded": [], "tool_results": tool_results,
            "malformed_tool_call": False, "error": "max steps reached"}


if __name__ == "__main__":
    out = run_agent(" ".join(sys.argv[1:]))
    print(out["answer"])
    print("NOT GROUNDED:", out["ungrounded"])
    print("MALFORMED TOOL CALL:", out["malformed_tool_call"])