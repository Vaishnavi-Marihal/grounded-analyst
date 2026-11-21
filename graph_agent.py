import os
from typing import TypedDict

import ollama
from langgraph.graph import END, START, StateGraph

import agent
from checker import collect_numbers, extract_numbers, find_ungrounded
import json


class AgentState(TypedDict):
    messages: list
    tool_results: list
    pending: list
    repaired: int
    steps: int
    answer: str
    finished: bool


def build_graph(model, max_steps, repair_on):
    def model_node(state):
        response = ollama.chat(
            model=model, messages=state["messages"], tools=agent.TOOL_SPECS,
            options={"temperature": 0},
        )
        message = response.message
        messages = state["messages"] + [message]
        steps = state["steps"] + 1
        if message.tool_calls:
            pending = [(c.function.name, dict(c.function.arguments)) for c in message.tool_calls]
            return {"messages": messages, "pending": pending, "steps": steps}
        answer = message.content or ""
        text_calls = agent.parse_text_tool_calls(answer) if repair_on else []
        if text_calls:
            return {"messages": messages, "pending": text_calls, "steps": steps,
                    "repaired": state["repaired"] + len(text_calls)}
        return {"messages": messages, "pending": [], "steps": steps,
                "answer": answer, "finished": True}

    def tools_node(state):
        results = list(state["tool_results"])
        messages = list(state["messages"])
        for name, args in state["pending"]:
            result = agent.execute_call(name, args, results)
            results.append(result)
            messages.append({"role": "tool", "tool_name": name, "content": json.dumps(result)})
        return {"messages": messages, "tool_results": results, "pending": []}

    def after_model(state):
        return "tools" if state["pending"] else END

    def after_tools(state):
        return END if state["steps"] >= max_steps else "model"

    graph = StateGraph(AgentState)
    graph.add_node("model", model_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", after_model, {"tools": "tools", END: END})
    graph.add_conditional_edges("tools", after_tools, {"model": "model", END: END})
    return graph.compile()


def run_agent(question, model=agent.MODEL, max_steps=None):
    if max_steps is None:
        max_steps = 8 if os.environ.get("TOOLS_V") == "2" else 5
    repair_on = os.environ.get("REPAIR") == "1"
    messages = [
        {"role": "system", "content": agent.PROMPTS[os.environ.get("PROMPT_VARIANT", "v1")]},
        {"role": "user", "content": question},
    ]
    final = build_graph(model, max_steps, repair_on).invoke(
        {"messages": messages, "tool_results": [], "pending": [], "repaired": 0,
         "steps": 0, "answer": "", "finished": False},
        config={"recursion_limit": 100},
    )
    results = final["tool_results"]
    if not final["finished"]:
        return {"answer": "", "ungrounded": [], "tool_results": results,
                "malformed_tool_call": False, "repaired_calls": final["repaired"],
                "error": "max steps reached"}
    answer = final["answer"]
    question_numbers = [float(t) for t in extract_numbers(question)]
    return {
        "answer": answer,
        "ungrounded": find_ungrounded(answer, collect_numbers(results) + question_numbers),
        "tool_results": results,
        "malformed_tool_call": '"name"' in answer and '"parameters"' in answer,
        "repaired_calls": final["repaired"],
    }