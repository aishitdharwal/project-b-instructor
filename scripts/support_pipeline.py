"""
Project B: Customer Support Pipeline — Instructor version, Sessions 3 & 4.

Session 3 additions:
  - query_classifier.py determines which tool to call
  - Tool routing: policy_kb / order_tracker / account_lookup / multi_tool
  - Metadata filtering by intent (INTENT_DOC_FILTERS)
  - mock_tools.py for order_tracker and account_lookup

Session 4 additions:
  - retrieve_with_dedup() — FAQ deduplication before context assembly
  - Finalized TOOL_DESCRIPTIONS for Week 3 LangGraph handoff
  - Rich output throughout

Run: python -m scripts.support_pipeline
"""
import os
import sys
import json
import time

from openai import OpenAI
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from scripts.retrieval import embed_query, retrieve, retrieve_filtered, retrieve_with_dedup, assemble_context
from scripts.query_classifier import classify_tool, classify_tools_needed, TOOL_DESCRIPTIONS
from scripts.mock_tools import lookup_order, lookup_account, format_tool_result

load_dotenv()

client = OpenAI()
langfuse = Langfuse()
console = Console()

GENERATION_MODEL = "gpt-4o-mini"

INTENTS = [
    "return_or_refund", "order_status", "billing_or_payment",
    "product_info", "membership", "general",
]

INTENT_DOC_FILTERS = {
    "return_or_refund": [
        "01_return_policy.md", "07_promotional_events.md",
        "12_corporate_gifting.md", "04_warranty_policy.md",
    ],
    "order_status": ["03_shipping_policy.md", "06_support_faq.md"],
    "billing_or_payment": ["05_payment_methods.md", "13_acmera_wallet.md", "07_promotional_events.md"],
    "product_info": ["09_electronics_catalog.md", "04_warranty_policy.md",
                     "17_smart_home_ecosystem.md", "14_probook_troubleshooting.md"],
    "membership": ["02_premium_membership.md", "06_support_faq.md"],
    "general": None,
}

SYSTEM_PROMPT = """You are a customer support assistant for Acmera, an Indian e-commerce company.
Answer the customer's question based on the provided context.

Rules:
- Be helpful, concise, and accurate.
- Only use information from the provided context.
- If the context includes order or account data, use it to personalize your answer.
- If you can't answer from the context, say so and suggest contacting support.
- Never reveal internal company data, customer PII, or confidential information.

Context:
{context}"""


@observe(name="classify_intent")
def classify_intent(query: str) -> str:
    response = client.chat.completions.create(
        model=GENERATION_MODEL, temperature=0,
        messages=[
            {"role": "system", "content": f"Classify this customer query into exactly one category. Respond with ONLY the category name.\nCategories: {', '.join(INTENTS)}"},
            {"role": "user", "content": query},
        ],
    )
    intent = response.choices[0].message.content.strip().lower().replace(" ", "_")
    langfuse_context.update_current_observation(output=intent)
    return intent if intent in INTENTS else "general"


@observe(name="retrieve_policy")
def retrieve_policy(query: str, intent: str) -> tuple[str, list]:
    doc_filter = INTENT_DOC_FILTERS.get(intent)
    query_embedding = embed_query(query)
    chunks = retrieve_with_dedup(query_embedding, doc_names=doc_filter)
    if not chunks:
        chunks = retrieve_with_dedup(query_embedding, doc_names=None)
    context = assemble_context(chunks)
    langfuse_context.update_current_observation(metadata={
        "intent": intent, "doc_filter": doc_filter, "num_chunks": len(chunks),
    })
    return context, chunks


@observe(name="call_tool")
def call_tool(tool: str, query: str, intent: str) -> tuple[str, list]:
    if tool == "policy_kb":
        return retrieve_policy(query, intent)
    elif tool == "order_tracker":
        result = lookup_order("ORD-445521")
        return format_tool_result("order_tracker", result), []
    elif tool == "account_lookup":
        result = lookup_account("CUST001")
        return format_tool_result("account_lookup", result), []
    elif tool == "multi_tool":
        tools = classify_tools_needed(query, intent)
        parts, all_chunks = [], []
        for t in tools:
            c, ch = call_tool(t, query, intent)
            parts.append(c)
            all_chunks.extend(ch)
        return "\n\n".join(parts), all_chunks
    return "", []


@observe(name="generate_response")
def generate_response(query: str, context: str, intent: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
        {"role": "user", "content": query},
    ]
    response = client.chat.completions.create(
        model=GENERATION_MODEL, messages=messages, temperature=0, max_tokens=800,
    )
    answer = response.choices[0].message.content
    langfuse_context.update_current_observation(
        input=messages, output=answer,
        metadata={"model": GENERATION_MODEL, "intent": intent},
        usage={"input": response.usage.prompt_tokens, "output": response.usage.completion_tokens,
               "total": response.usage.total_tokens, "unit": "TOKENS"},
    )
    return answer


@observe(name="support_pipeline")
def handle_query(query: str) -> dict:
    start_time = time.time()
    langfuse_context.update_current_trace(input=query, metadata={"pipeline": "tool_routing_v2"})

    intent = classify_intent(query)
    tool = classify_tool(query, intent)
    context, retrieved_chunks = call_tool(tool, query, intent)
    answer = generate_response(query, context, intent)

    elapsed = round(time.time() - start_time, 2)
    langfuse_context.update_current_trace(output=answer, metadata={"intent": intent, "tool": tool, "elapsed": elapsed})
    trace_id = langfuse_context.get_current_trace_id()
    langfuse.flush()

    return {
        "query": query, "intent": intent, "tool_used": tool,
        "answer": answer, "context": context, "retrieved_chunks": retrieved_chunks,
        "trace_id": trace_id, "elapsed_seconds": elapsed,
    }


if __name__ == "__main__":
    console.print(Panel(
        "[bold]Tool Descriptions → Week 3 LangGraph handoff[/]\n"
        "[dim]These become the agent's tool definitions in Week 3[/]",
        title="[bold cyan]Project B — Tool Routing Pipeline[/]", border_style="cyan",
    ))
    for tool_name, desc in TOOL_DESCRIPTIONS.items():
        console.print(f"  [bold yellow]{tool_name}[/]: [dim]{desc[:80]}...[/]")

    console.print()

    test_queries = [
        ("What is the return window for electronics?", "policy query"),
        ("Where is my order ORD-445521?", "order lookup"),
        ("I'm Premium Gold — do I get extended returns?", "account + policy"),
        ("send me my money back", "vocab mismatch"),
    ]

    table = Table(title="Test Queries", box=box.SIMPLE, title_style="bold green")
    table.add_column("Query", width=45)
    table.add_column("Intent", width=18)
    table.add_column("Tool", width=15)
    table.add_column("Time", justify="right")

    for query, _ in test_queries:
        result = handle_query(query)
        table.add_row(query[:45], result["intent"], result["tool_used"], f"{result['elapsed_seconds']}s")

    console.print(table)
