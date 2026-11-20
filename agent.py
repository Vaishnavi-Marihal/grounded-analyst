import json
import os
import sys

import ollama

import tools
from checker import collect_numbers, extract_numbers, find_ungrounded, is_grounded

MODEL = "llama3.2"

SYSTEM_PROMPT = (
    "You answer questions about building electricity forecasts. "
    "Always call a tool before stating any number. "
    "Only use numbers that appear in tool results. "
    "For any difference or percentage, call the calculate tool with numbers "
    "copied from earlier tool results. Never calculate or round numbers yourself. "
    "If the tools cannot answer the question, say so."
)

SYSTEM_PROMPT_V2 = (
    "You answer questions about building electricity forecasts using tools. "
    "Never state a number that is not in a tool result. "
    "For a question about a difference or percentage between the forecast and the actual: "
    "first call get_day_forecast with the building and date and wait for its result, "
    "then call calculate with a and b copied exactly from that result "
    "(total_forecast_kwh and total_actual_kwh, or total_last_week_kwh for the baseline). "
    "Never put one tool call inside another tool's arguments, "
    "and never write a tool call as text."
)

SYSTEM_PROMPT_V3 = (
    SYSTEM_PROMPT
    + " For questions about how the system works, its methods, reported results, "
    "limitations or deployment, call search_docs and answer only from the returned passages. "
    "For numbers about a specific building and date, use the forecast tools."
)

PROMPTS = {"v1": SYSTEM_PROMPT, "v2": SYSTEM_PROMPT_V2, "v3": SYSTEM_PROMPT_V3}

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

DOCS_SPEC = {
    "type": "function",
    "function": {
        "name": "search_docs",
        "description": (
            "Search the project documentation for explanations: how the model works, "
            "why days were flagged by monitoring, how prediction intervals are made, "
            "data, limitations and deployment. Returns text passages."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "what to look up, in plain words"}},
            "required": ["query"],
        },
    },
}

if os.environ.get("DOCS") == "1":
    TOOL_SPECS.append(DOCS_SPEC)
    REGISTRY["search_docs"] = tools.search_docs

KNOWN_TOOLS = set(REGISTRY) | {"calculate"}


def operands_grounded(args, allowed):
    try:
        return all(is_grounded(str(args[k]), allowed) for k in ("a", "b"))
    except (KeyError, ValueError):
        return False


def parse_text_tool_calls(text):
    calls = []
    decoder = json.JSONDecoder()
    i = 0
    while True:
        start = text.find("{", i)
        if start == -1:
            break
        try:
            obj, end = decoder.raw_decode(text[start:])
        except ValueError:
            i = start + 1
            continue
        i = start + end
        if isinstance(obj, dict) and obj.get("name") in KNOWN_TOOLS:
            args = obj.get("parameters", obj.get("arguments", {}))
            if isinstance(args, dict):
                calls.append((obj["name"], args))
    return calls


def execute_call(name, args, tool_results):
    if name == "calculate":
        if operands_grounded(args, collect_numbers(tool_results)):
            try:
                return tools.calculate(**args)
            except TypeError as exc:
                return {"error": str(exc)}
        return {"error": "a and b must be numbers copied from earlier tool results"}
    try:
        return REGISTRY[name](**args)
    except Exception as exc:
        return {"error": str(exc)}


def run_agent(question, model=MODEL, max_steps=None):
    if max_steps is None:
        max_steps = 8 if os.environ.get("TOOLS_V") == "2" else 5
    messages = [
        {"role": "system", "content": PROMPTS[os.environ.get("PROMPT_VARIANT", "v1")]},
        {"role": "user", "content": question},
    ]
    tool_results = []
    repaired = 0
    repair_on = os.environ.get("REPAIR") == "1"
    question_numbers = [float(t) for t in extract_numbers(question)]
    for _ in range(max_steps):
        response = ollama.chat(
            model=model,
            messages=messages,
            tools=TOOL_SPECS,
            options={"temperature": 0},
        )
        message = response.message
        messages.append(message)
        if message.tool_calls:
            for call in message.tool_calls:
                name = call.function.name
                result = execute_call(name, dict(call.function.arguments), tool_results)
                tool_results.append(result)
                messages.append({"role": "tool", "tool_name": name, "content": json.dumps(result)})
            continue
        answer = message.content or ""
        text_calls = parse_text_tool_calls(answer) if repair_on else []
        if text_calls:
            for name, args in text_calls:
                result = execute_call(name, args, tool_results)
                tool_results.append(result)
                repaired += 1
                messages.append({"role": "tool", "tool_name": name, "content": json.dumps(result)})
            continue
        return {
            "answer": answer,
            "ungrounded": find_ungrounded(
                answer, collect_numbers(tool_results) + question_numbers
            ),
            "tool_results": tool_results,
            "malformed_tool_call": '"name"' in answer and '"parameters"' in answer,
            "repaired_calls": repaired,
        }
    return {
        "answer": "",
        "ungrounded": [],
        "tool_results": tool_results,
        "malformed_tool_call": False,
        "repaired_calls": repaired,
        "error": "max steps reached",
    }


if __name__ == "__main__":
    out = run_agent(" ".join(sys.argv[1:]))
    print(out["answer"])
    print("NOT GROUNDED:", out["ungrounded"])
    print("MALFORMED TOOL CALL:", out["malformed_tool_call"])
