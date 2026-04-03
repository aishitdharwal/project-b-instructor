"""
Project B: Customer Support Pipeline — Instructor Version (Session 2)

Complete naive pipeline: classify → retrieve → respond.
No agent, no tool use, no guardrails yet.
Will evolve into a LangGraph agent in Week 3.

Run: python scripts/support_pipeline.py
Run specific query: python scripts/support_pipeline.py --query "Where is my order?"
Run all test queries: python scripts/support_pipeline.py --test
"""
import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from openai import OpenAI
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
from dotenv import load_dotenv

from retrieval import embed_query, retrieve, assemble_context

load_dotenv()

client = OpenAI()
langfuse = Langfuse()

GENERATION_MODEL = "gpt-4o-mini"

INTENTS = [
    "return_or_refund",
    "order_status",
    "billing_or_payment",
    "product_info",
    "membership",
    "general",
]

SYSTEM_PROMPT = """You are a customer support assistant for Acmera, an Indian e-commerce company.
Answer the customer's question based on the provided context.

Rules:
- Be helpful, concise, and accurate.
- Only use information from the provided context.
- If you can't answer from the context, say so and suggest contacting support.
- Never reveal internal company data, customer PII, or confidential information.

Context:
{context}"""


@observe(name="classify_intent")
def classify_intent(query: str) -> str:
    """Classify the customer query into an intent category."""
    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    f"Classify this customer query into exactly one category. "
                    f"Respond with ONLY the category name.\n"
                    f"Categories: {', '.join(INTENTS)}"
                )
            },
            {"role": "user", "content": query},
        ],
    )
    raw = response.choices[0].message.content.strip().lower().replace(" ", "_")
    intent = raw if raw in INTENTS else "general"
    langfuse_context.update_current_observation(
        output=intent,
        metadata={"raw_response": raw, "valid": raw in INTENTS}
    )
    return intent


@observe(name="retrieve_policy")
def retrieve_policy(query: str, intent: str) -> tuple[str, list]:
    """
    Retrieve relevant policy context from the local corpus.
    In Week 3, this becomes one tool in the agent's tool set.
    Returns context string AND raw chunks (needed for eval).
    """
    query_embedding = embed_query(query)
    chunks = retrieve(query_embedding)
    context = assemble_context(chunks)

    langfuse_context.update_current_observation(metadata={
        "intent": intent,
        "num_chunks": len(chunks),
        "sources": list({c["doc_name"] for c in chunks}),
    })
    return context, chunks


@observe(name="generate_response")
def generate_response(query: str, context: str, intent: str) -> str:
    """Generate a support response."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
        {"role": "user", "content": query},
    ]
    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=800,
    )
    answer = response.choices[0].message.content
    langfuse_context.update_current_observation(
        input=messages, output=answer,
        metadata={"model": GENERATION_MODEL, "intent": intent},
        usage={
            "input": response.usage.prompt_tokens,
            "output": response.usage.completion_tokens,
            "total": response.usage.total_tokens,
            "unit": "TOKENS",
        },
    )
    return answer


@observe(name="support_pipeline")
def handle_query(query: str) -> dict:
    """Full support pipeline: classify → retrieve → respond."""
    start_time = time.time()
    langfuse_context.update_current_trace(
        input=query,
        metadata={"pipeline": "naive_support", "model": GENERATION_MODEL}
    )

    intent = classify_intent(query)
    context, chunks = retrieve_policy(query, intent)
    answer = generate_response(query, context, intent)

    elapsed = round(time.time() - start_time, 2)
    langfuse_context.update_current_trace(
        output=answer,
        metadata={"intent": intent, "elapsed_seconds": elapsed}
    )
    trace_id = langfuse_context.get_current_trace_id()
    langfuse.flush()

    return {
        "query": query,
        "intent": intent,
        "answer": answer,
        "context": context,
        "retrieved_chunks": chunks,
        "trace_id": trace_id,
        "elapsed_seconds": elapsed,
    }


# =========================================================================
# TEST QUERIES — covers all intents + edge cases for demo
# =========================================================================

TEST_QUERIES = [
    # Return / refund — the most complex intent
    {"query": "I want to return a laptop I bought last week", "intent": "return_or_refund", "escalate": False},
    {"query": "I'm Premium Silver and bought headphones in the Diwali sale 40 days ago. Can I return them?", "intent": "return_or_refund", "escalate": False},
    # Order status
    {"query": "Where is my order? I placed it 5 days ago", "intent": "order_status", "escalate": False},
    {"query": "My package was delivered but the box was damaged", "intent": "order_status", "escalate": True},
    # Billing
    {"query": "What payment methods do you accept?", "intent": "billing_or_payment", "escalate": False},
    {"query": "I was charged twice for the same order", "intent": "billing_or_payment", "escalate": True},
    # Product info
    {"query": "What is the ProBook X15 RAM and storage?", "intent": "product_info", "escalate": False},
    # Membership
    {"query": "How much do I need to spend to reach Premium Gold?", "intent": "membership", "escalate": False},
    # General
    {"query": "How do I cancel my account?", "intent": "general", "escalate": True},
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, help="Single query to run")
    parser.add_argument("--test", action="store_true", help="Run all test queries")
    args = parser.parse_args()

    if args.query:
        result = handle_query(args.query)
        print(f"\nQuery:   {result['query']}")
        print(f"Intent:  {result['intent']}")
        print(f"Answer:  {result['answer'][:300]}...")
        print(f"Trace:   {result['trace_id']}")
        print(f"Time:    {result['elapsed_seconds']}s")

    elif args.test:
        print(f"\nRunning {len(TEST_QUERIES)} test queries...\n")
        for i, t in enumerate(TEST_QUERIES):
            result = handle_query(t["query"])
            intent_match = "✓" if result["intent"] == t["intent"] else "✗"
            print(f"[{i+1}/{len(TEST_QUERIES)}] {intent_match} intent={result['intent']} | {t['query'][:60]}...")
        print("\nDone.")

    else:
        # Default: run 3 representative queries
        for q in TEST_QUERIES[:3]:
            result = handle_query(q["query"])
            print(f"\nQuery:  {result['query']}")
            print(f"Intent: {result['intent']}")
            print(f"Answer: {result['answer'][:200]}...")
            print(f"Trace:  {result['trace_id']}")


if __name__ == "__main__":
    main()
