"""
Query Classifier — Session 3: Tool Routing

Determines WHICH TOOL the pipeline needs based on the query and intent.
This is the proto-reasoning layer that becomes the LangGraph agent in Week 3.

The distinction from intent classification:
  - Intent: WHAT the customer wants (return, order_status, billing...)
  - Tool:   HOW to get the information needed (which system to query)

Tool routing:
  policy_kb     → RAG over policy/FAQ docs (return policy, shipping, warranty...)
  order_tracker → Look up specific order status, delivery, return eligibility
  account_lookup → Check customer account, tier, wallet balance, history
  multi_tool    → Complex queries needing 2+ tools (resolved by calling each)

Usage:
    from query_classifier import classify_tool, TOOL_DESCRIPTIONS
    tool = classify_tool(query, intent)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

TOOLS = ["policy_kb", "order_tracker", "account_lookup", "multi_tool"]

# Precise descriptions — designed so an LLM can select the right tool.
# Week 3 will pass these directly to the LangGraph agent as tool definitions.
TOOL_DESCRIPTIONS = {
    "policy_kb": (
        "Search the Acmera knowledge base for policy and FAQ information. "
        "Use for: return policies, shipping rules, warranty terms, payment methods, "
        "membership benefits, promotional terms, product specifications. "
        "Does NOT know about specific orders or customer accounts."
    ),
    "order_tracker": (
        "Look up the status of a specific customer order. "
        "Returns: order date, product, price, delivery status, delivery date, "
        "whether it was promotional, and return eligibility window. "
        "Requires an order ID or enough context to identify the order."
    ),
    "account_lookup": (
        "Look up a customer's account details. "
        "Returns: membership tier (standard/silver/gold), account creation date, "
        "annual spend, and associated orders. "
        "Use for: 'am I premium?', wallet balance questions, tier-specific policies."
    ),
    "multi_tool": (
        "The query needs information from more than one tool. "
        "Example: 'I'm a Gold member and I want to return an order I placed last week' "
        "needs both account_lookup (to confirm tier) and order_tracker (to check eligibility). "
        "Resolve by calling each required tool in sequence."
    ),
}

# Rule-based routing as a fast path (before hitting the LLM)
INTENT_TO_TOOL = {
    "order_status": "order_tracker",
    "product_info": "policy_kb",
    "membership": "account_lookup",
    "billing_or_payment": "policy_kb",
    "general": "policy_kb",
}

TOOL_CLASSIFIER_PROMPT = """You are a routing agent for a customer support system.
Given a customer query and its intent category, decide which tool to call.

Available tools:
{tool_descriptions}

Intent: {intent}
Query: {query}

Rules:
- If the query asks about a SPECIFIC order (tracking, damaged delivery, return of a specific item) → order_tracker
- If the query asks about the customer's OWN account, tier, or spend → account_lookup
- If the query asks about policies, terms, or general product info → policy_kb
- If the query needs both order details AND account details → multi_tool
- If the query needs order details AND policy context → multi_tool

Respond with ONLY one of: policy_kb, order_tracker, account_lookup, multi_tool"""


def classify_tool(query: str, intent: str) -> str:
    """
    Determine which tool is needed to answer this query.

    Uses a fast rule-based path for clear-cut cases,
    falls back to LLM classification for ambiguous ones.

    Args:
        query: Customer query text
        intent: Intent from classify_intent() (return_or_refund, order_status, etc.)

    Returns:
        Tool name: one of TOOLS
    """
    # Fast path: clear intent → tool mapping
    if intent in INTENT_TO_TOOL and intent not in ("return_or_refund",):
        return INTENT_TO_TOOL[intent]

    # For return_or_refund, need to decide: is this about policy or a specific order?
    # Use LLM to disambiguate
    tool_desc_text = "\n".join(
        f"  {name}: {desc}" for name, desc in TOOL_DESCRIPTIONS.items()
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": TOOL_CLASSIFIER_PROMPT.format(
                    tool_descriptions=tool_desc_text,
                    intent=intent,
                    query=query,
                ),
            }
        ],
    )

    tool = response.choices[0].message.content.strip().lower()
    return tool if tool in TOOLS else "policy_kb"


def classify_tools_needed(query: str, intent: str) -> list[str]:
    """
    Like classify_tool(), but returns a list for multi_tool queries.
    Single-tool queries return a 1-element list.
    """
    tool = classify_tool(query, intent)
    if tool == "multi_tool":
        # Determine which specific tools are needed
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"This query needs multiple tools: '{query}'\n"
                        "List which tools are needed from: policy_kb, order_tracker, account_lookup\n"
                        "Respond with a comma-separated list, e.g.: order_tracker, account_lookup"
                    ),
                }
            ],
        )
        raw = response.choices[0].message.content.strip().lower()
        tools = [t.strip() for t in raw.split(",") if t.strip() in TOOLS[:-1]]
        return tools if tools else ["policy_kb"]
    return [tool]


if __name__ == "__main__":
    test_cases = [
        ("What is the return window for electronics?", "return_or_refund"),
        ("Where is my order ORD-445521?", "order_status"),
        ("I'm a premium member — do I get extended returns?", "membership"),
        ("I was charged twice for my last order", "billing_or_payment"),
        ("I want to return the laptop I bought last week — am I still in window?", "return_or_refund"),
    ]

    print("Query Classifier — Tool Routing\n")
    for query, intent in test_cases:
        tool = classify_tool(query, intent)
        print(f"  Query:  {query[:60]}")
        print(f"  Intent: {intent}  →  Tool: {tool}\n")
