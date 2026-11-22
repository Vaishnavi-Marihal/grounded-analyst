import json
import os

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

import agent
from checker import collect_numbers, extract_numbers, find_ungrounded
from graph_agent import AgentState


def build_graph(model, max_steps, repair_on):
    llm = ChatOllama(model=model, temperature=0).bind_tools(agent.TOOL_SPECS)

    def model_node(state):
        message = llm.invoke(state["messages"])
        messages = state["messages"] + [message]
        steps = state["steps"] + 1
        if message.tool_calls:
            pending = [(c["name"], dict(c["args"]), c["id"]) for c in message.tool_calls]
            return {"messages": messages, "pending": pending, "steps": steps}
        answer = message.content or ""
        text_calls = agent.parse_text_tool_calls(answer) if repair_on else []
        if text_calls:
            pending = [(n, a, f"repair-{i}") for i, (n, a) in enumerate(text_calls)]
            return {"messages": messages, "pending": pending, "steps": steps,
                    "repaired": state["repaired"] + len(text_calls)}
        return {"messages": messages, "pending": [], "steps": steps,
                "answer": answer, "finished": True}

    def tools_node(state):
        results = list(state["tool_results"])
        messages = list(state["messages"])
        for name, args, call_id in state["pending"]:
            result = agent.execute_call(name, args, results)
            results.append(result)
            messages.append(ToolMessage(content=json.dumps(result), tool_call_id=call_id, name=name))
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
        SystemMessage(content=agent.PROMPTS[os.environ.get("PROMPT_VARIANT", "v1")]),
        HumanMessage(content=question),
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