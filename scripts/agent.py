"""
LangGraph Support Agent — Instructor version, Session 5.

Replaces the single-shot support_pipeline with an agentic reasoning loop:

    classify → tool_call → evaluate → tool_call (again?) → respond
                                    ↘ escalate

Key concepts taught:
  - StateGraph with a typed state object (everything the agent knows)
  - Conditional edges as routing logic (the agent decides what happens next)
  - Loop guard: never call more than 3 tools on one query
  - Structured escalation — hand off with full context to human agent
  - Per-node LangFuse tracing for full trajectory observability
  - Per-call token cost tracking

Design decisions:
  - evaluate_node() is a GRAPH NODE — it runs the LLM and writes verdict to state
  - _route_from_evaluate() is the ROUTING FUNCTION — reads state["verdict"]
    This separation is cleaner than using one function for both purposes.
  - next_tool is stored in AgentState so tool_node can pick it up on re-entry

Run: python -m scripts.agent
"""
import os
import json
import time
from typing import TypedDict, Annotated
import operator

from openai import OpenAI
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from scripts.support_pipeline import classify_intent, retrieve_policy, generate_response
from scripts.query_classifier import classify_tool
from scripts.mock_tools import lookup_order, lookup_account, format_tool_result

load_dotenv()

client = OpenAI()
langfuse = Langfuse()
console = Console()

EVALUATION_MODEL = "gpt-4o-mini"

# Pricing per 1M tokens (USD) — for per-node cost tracking
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o":      {"input": 2.50, "output": 10.00},
}
USD_TO_INR = 85


def _token_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    p = PRICING.get(model, PRICING["gpt-4o-mini"])
    return (prompt_tokens / 1_000_000) * p["input"] + (completion_tokens / 1_000_000) * p["output"]


# =========================================================================
# STATE — everything the agent carries through the reasoning loop
# =========================================================================

class AgentState(TypedDict):
    query:          str                              # original query, never changes
    intent:         str                              # set by classify_node
    tool_results:   Annotated[list, operator.add]    # accumulates each tool call
    final_answer:   str                              # set by respond_node / escalate_node
    should_escalate: bool                            # True if escalate_node ran
    steps_taken:    int                              # incremented by tool_node (loop guard)
    tools_called:   Annotated[list, operator.add]    # trajectory for observability
    verdict:        str                              # set by evaluate_node (routing key)
    next_tool:      str                              # set by evaluate_node when more info needed


# =========================================================================
# NODE 1 — CLASSIFY
# Determines the intent of the customer query.
# Re-uses the same classify_intent() from support_pipeline.
# =========================================================================

@observe(name="agent_classify")
def classify_node(state: AgentState) -> dict:
    intent = classify_intent(state["query"])
    langfuse_context.update_current_observation(
        output=intent,
        metadata={"node": "classify", "step": 0},
    )
    return {"intent": intent, "verdict": "", "next_tool": ""}


# =========================================================================
# NODE 2 — TOOL CALL
# Selects and executes the right tool based on intent + evaluate verdict.
# On the first call: uses classify_tool() to pick the tool.
# On re-entry (after evaluate said "need: X"): uses state["next_tool"].
# =========================================================================

@observe(name="agent_tool_call")
def tool_node(state: AgentState) -> dict:
    query = state["query"]
    intent = state["intent"]
    already_called = state.get("tools_called", [])

    # Prefer the tool evaluate explicitly requested; fall back to classifier
    requested = state.get("next_tool", "")
    if requested and requested not in already_called:
        tool = requested
    else:
        tool = classify_tool(query, intent)

    # Never call the same non-kb tool twice — fall back to policy knowledge base
    if tool in already_called and tool != "policy_kb":
        tool = "policy_kb"

    # Execute
    if tool == "policy_kb":
        context, _chunks = retrieve_policy(query, intent)
        result_text = context
    elif tool == "order_tracker":
        result = lookup_order("ORD-445521")       # production: extract from query
        result_text = format_tool_result("order_tracker", result)
    elif tool == "account_lookup":
        result = lookup_account("CUST001")         # production: from auth context
        result_text = format_tool_result("account_lookup", result)
    else:
        context, _chunks = retrieve_policy(query, intent)
        result_text = context

    langfuse_context.update_current_observation(
        input={"tool": tool, "query": query[:100]},
        output=result_text[:300],
        metadata={"node": "tool_call", "tool": tool, "step": state["steps_taken"]},
    )

    return {
        "tool_results": [{"tool": tool, "result": result_text}],
        "tools_called": [tool],
        "steps_taken": state["steps_taken"] + 1,
    }


# =========================================================================
# NODE 3 — EVALUATE
# The agent's self-assessment step. Decides whether it has enough info.
# Stores its decision (verdict) in state so _route_from_evaluate() can read it.
# =========================================================================

EVALUATE_PROMPT = """You are evaluating whether a customer support agent has enough
information to fully answer a customer query.

Customer query: {query}
Intent: {intent}
Tools called so far: {tools_called}
Information gathered:
{tool_results}

Rules:
- If the information is sufficient for a complete, accurate answer → "sufficient"
- If important information is missing AND another tool can provide it AND
  fewer than 3 tool calls have been made → "need: <tool_name>"
- If the query involves a billing dispute, account security issue, or requires
  human judgment an AI shouldn't make → "escalate"
- If 3+ tool calls have already been made → "sufficient" (respond with what you have)

Available tools: order_tracker, account_lookup, policy_kb

Respond with ONLY ONE of:
  "sufficient"
  "need: order_tracker"
  "need: account_lookup"
  "need: policy_kb"
  "escalate"
"""


@observe(name="agent_evaluate")
def evaluate_node(state: AgentState) -> dict:
    """
    Runs the LLM evaluation and writes verdict + next_tool into state.
    The actual routing decision is made by _route_from_evaluate() below.
    """
    # Hard loop guard — never loop more than 3 tool calls
    if state["steps_taken"] >= 3:
        langfuse_context.update_current_observation(
            output="sufficient",
            metadata={"node": "evaluate", "reason": "loop_guard", "steps": state["steps_taken"]},
        )
        return {"verdict": "sufficient", "next_tool": ""}

    tool_results_text = "\n\n".join([
        f"[{r['tool']}]: {r['result'][:300]}"
        for r in state["tool_results"]
    ])

    response = client.chat.completions.create(
        model=EVALUATION_MODEL,
        temperature=0,
        messages=[{
            "role": "user",
            "content": EVALUATE_PROMPT.format(
                query=state["query"],
                intent=state["intent"],
                tools_called=state.get("tools_called", []),
                tool_results=tool_results_text,
            ),
        }],
    )

    verdict = response.choices[0].message.content.strip().lower()
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    cost_usd = _token_cost(EVALUATION_MODEL, prompt_tokens, completion_tokens)

    # Extract tool name when verdict is "need: <tool>"
    next_tool = ""
    if verdict.startswith("need:"):
        next_tool = verdict.split("need:")[-1].strip()

    langfuse_context.update_current_observation(
        input={
            "steps_taken": state["steps_taken"],
            "tools_called": state.get("tools_called", []),
        },
        output=verdict,
        metadata={
            "node": "evaluate",
            "decision": verdict,
            "cost_usd": round(cost_usd, 6),
            "cost_inr": round(cost_usd * USD_TO_INR, 4),
        },
        usage={
            "input": prompt_tokens,
            "output": completion_tokens,
            "total": prompt_tokens + completion_tokens,
            "unit": "TOKENS",
        },
    )

    return {"verdict": verdict, "next_tool": next_tool}


def _route_from_evaluate(state: AgentState) -> str:
    """
    Conditional edge routing function — reads verdict from state.
    Called by LangGraph to determine which node to visit next.
    """
    verdict = state.get("verdict", "sufficient")
    if verdict == "sufficient":
        return "respond"
    elif verdict == "escalate":
        return "escalate"
    elif verdict.startswith("need:"):
        return "tool_call"
    return "respond"


# =========================================================================
# NODE 4 — RESPOND
# Generates the final customer-facing answer from all gathered context.
# =========================================================================

@observe(name="agent_respond")
def respond_node(state: AgentState) -> dict:
    # Combine all tool results into a single context block
    context = "\n\n".join([
        f"[{r['tool']}]:\n{r['result']}"
        for r in state["tool_results"]
    ])

    answer = generate_response(state["query"], context, state["intent"])

    langfuse_context.update_current_observation(
        output=answer[:200],
        metadata={"node": "respond", "total_steps": state["steps_taken"]},
    )

    return {"final_answer": answer}


# =========================================================================
# NODE 5 — ESCALATE
# Packages everything the agent found into a structured handoff for a human.
# The human gets: what the customer asked, what was found, why escalating.
# =========================================================================

@observe(name="agent_escalate")
def escalate_node(state: AgentState) -> dict:
    packet = {
        "query": state["query"],
        "intent": state["intent"],
        "tools_called": state.get("tools_called", []),
        "information_gathered": [
            {"tool": r["tool"], "summary": r["result"][:200]}
            for r in state["tool_results"]
        ],
        "escalation_reason": "Requires human judgment",
        "steps_taken": state["steps_taken"],
    }

    langfuse_context.update_current_observation(
        output=json.dumps(packet)[:300],
        metadata={"node": "escalate"},
    )

    return {
        "final_answer": f"[ESCALATED]\n{json.dumps(packet, indent=2)}",
        "should_escalate": True,
    }


# =========================================================================
# BUILD THE GRAPH
# =========================================================================

graph = StateGraph(AgentState)

graph.add_node("classify",  classify_node)
graph.add_node("tool_call", tool_node)
graph.add_node("evaluate",  evaluate_node)
graph.add_node("respond",   respond_node)
graph.add_node("escalate",  escalate_node)

graph.set_entry_point("classify")
graph.add_edge("classify",  "tool_call")   # always call a tool after classifying
graph.add_edge("tool_call", "evaluate")    # always evaluate after every tool call

graph.add_conditional_edges(
    "evaluate",            # from this node
    _route_from_evaluate,  # call this function to get the routing key
    {
        "respond":   "respond",
        "tool_call": "tool_call",
        "escalate":  "escalate",
    },
)

graph.add_edge("respond",  END)
graph.add_edge("escalate", END)

agent = graph.compile()


# =========================================================================
# ENTRYPOINT
# =========================================================================

@observe(name="run_agent")
def run_agent(query: str) -> dict:
    start_time = time.time()
    langfuse_context.update_current_trace(
        input=query,
        metadata={"pipeline": "langgraph_agent_v1"},
    )

    initial_state = AgentState(
        query=query,
        intent="",
        tool_results=[],
        final_answer="",
        should_escalate=False,
        steps_taken=0,
        tools_called=[],
        verdict="",
        next_tool="",
    )

    result = agent.invoke(initial_state)
    elapsed = round(time.time() - start_time, 2)

    langfuse_context.update_current_trace(
        output=result["final_answer"][:200],
        metadata={
            "intent": result["intent"],
            "tools_called": result["tools_called"],
            "steps_taken": result["steps_taken"],
            "should_escalate": result["should_escalate"],
            "elapsed_seconds": elapsed,
        },
    )
    langfuse.flush()

    return {
        "query":          query,
        "answer":         result["final_answer"],
        "should_escalate": result["should_escalate"],
        "steps_taken":    result["steps_taken"],
        "tools_called":   result["tools_called"],
        "intent":         result["intent"],
        "elapsed_seconds": elapsed,
    }


# =========================================================================
# DEMO
# =========================================================================

if __name__ == "__main__":
    console.print(Panel(
        "[bold]LangGraph Support Agent[/]\n"
        "[dim]classify → tool_call → evaluate → respond / escalate[/]\n"
        "[dim]Full trajectory traced in LangFuse per node[/]",
        title="[bold cyan]Project B — Agent, Session 5[/]",
        border_style="cyan",
    ))

    test_queries = [
        ("What is the return window for electronics?",          "simple policy"),
        ("Where is my order ORD-445521?",                       "order lookup"),
        ("I'm Premium Gold — do I get extended returns?",       "account + policy"),
        ("My card was charged twice. I need a refund now.",     "escalation trigger"),
    ]

    table = Table(title="Agent Test Runs", box=box.SIMPLE, title_style="bold green")
    table.add_column("Query", width=45)
    table.add_column("Intent", width=18)
    table.add_column("Tools called", width=28)
    table.add_column("Steps", justify="center", width=6)
    table.add_column("Escalated", justify="center", width=10)
    table.add_column("Time", justify="right", width=7)

    for query, _ in test_queries:
        result = run_agent(query)
        tools_str = " → ".join(result["tools_called"]) if result["tools_called"] else "none"
        escalated = "[red]YES[/]" if result["should_escalate"] else "[green]no[/]"
        table.add_row(
            query[:45],
            result["intent"],
            tools_str[:28],
            str(result["steps_taken"]),
            escalated,
            f"{result['elapsed_seconds']}s",
        )

    console.print(table)
