"""
Model Router — Session 7 update: LiteLLM multi-provider routing.

Session 5: Routes each query to cheapest OpenAI model that can handle it.
  Simple factual lookups → gpt-4o-mini  (fast, ~17x cheaper)
  Complex reasoning      → gpt-4o       (capable, handles multi-step logic)

Session 7: MODEL_SIMPLE and MODEL_COMPLEX are now LiteLLM model strings.
  LiteLLM understands model strings from any provider — swap providers
  without changing any code in generate() or anywhere else:

  OpenAI:     "gpt-4o-mini"             / "gpt-4o"
  Anthropic:  "claude-3-haiku-20240307"  / "claude-sonnet-4-6"
  Google:     "gemini/gemini-1.5-flash"  / "gemini/gemini-1.5-pro"
  Mistral:    "mistral/mistral-small"    / "mistral/mistral-large-latest"

  To switch providers: change MODEL_SIMPLE and MODEL_COMPLEX below.
  Set the matching env var (ANTHROPIC_API_KEY, GEMINI_API_KEY, etc.).
  Nothing else changes.

Key concepts taught:
  - Rule-based classification is free and deterministic — use it first
  - LLM-based fallback adds accuracy for genuinely ambiguous queries
  - The routing decision itself costs tokens — keep the classifier cheap
  - Even a simple 80/20 split (80% mini, 20% 4o) saves ~65% on generation costs
  - LiteLLM: one interface, any provider — no vendor lock-in

Complexity signals (rule-based fast path):
  - Comparison queries    → "compare", "difference", "vs", "versus"
  - Multi-condition logic → "if ... and ..."
  - Membership tiers      → "Gold", "Silver", "Premium"
  - Date arithmetic       → "40 days ago", "expiring"
  - Corporate/bulk        → "corporate", "bulk", "multiple orders"
  - Exception cases       → "exception", "edge case", "special"

Run: python -m scripts.model_router
"""
import os
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

# Session 7: these are LiteLLM model strings — change provider by swapping these values.
# Current: OpenAI. To use Anthropic instead: "claude-3-haiku-20240307" / "claude-sonnet-4-6"
MODEL_SIMPLE = "gpt-4o-mini"
MODEL_COMPLEX = "gpt-4o"

# Regex patterns that signal a query needs GPT-4o
COMPLEXITY_SIGNALS: list[tuple[str, str]] = [
    (r"\bcompare\b|\bvs\b|\bversus\b|\bdifference\b",          "comparison"),
    (r"\bif\b.{0,50}\band\b",                                   "multi_condition"),
    (r"\bgold\b|\bsilver\b|\bpremium\b",                        "tier_reasoning"),
    (r"\b\d+\s*days?\s*(ago|old|since)\b",                      "date_arithmetic"),
    (r"\bexpir\w+\b|\bdowngrad\w+\b|\bupgrad\w+\b",             "membership_change"),
    (r"\bcorporate\b|\bbulk\b|\bmultiple\s+order",              "corporate_bulk"),
    (r"\bexception\b|\bspecial\s+case\b|\bedge\s*case\b",       "exception"),
    (r"\bcalculate\b|\bhow\s+much\b|\bwhat.{0,20}cost\b",       "calculation"),
    (r"\blose\b|\bforfe\w+\b|\bpenalt\w+\b",                    "consequence"),
]

ROUTER_PROMPT = """You are deciding which AI model should answer a customer support query.

Complex queries (need GPT-4o):
- Comparing policies across membership tiers
- Multi-step conditional logic ("if X and Y then Z")
- Date / time arithmetic or eligibility calculations
- Exception handling, edge cases, or policy conflicts

Simple queries (gpt-4o-mini is enough):
- Single factual lookups ("what is the return window?")
- Basic how-to questions ("how do I track my order?")
- Policy summaries with no conditional logic

Query: {query}

Respond with exactly one word: "simple" or "complex"
"""


def classify_complexity(query: str) -> tuple[str, str]:
    """
    Rule-based complexity classification. Fast path — no API call.

    Returns:
        (complexity, reason)  where complexity is "simple" or "complex"
    """
    q_lower = query.lower()

    for pattern, reason in COMPLEXITY_SIGNALS:
        if re.search(pattern, q_lower):
            return "complex", reason

    # Long queries are usually multi-part — lean complex
    if len(query.split()) > 25:
        return "complex", "long_query"

    return "simple", "default"


def classify_complexity_llm(query: str) -> tuple[str, str]:
    """
    LLM-based complexity classifier. More accurate on ambiguous cases.
    Uses gpt-4o-mini to keep the classification itself cheap.
    """
    response = client.chat.completions.create(
        model=MODEL_SIMPLE,
        temperature=0,
        max_tokens=5,
        messages=[{"role": "user", "content": ROUTER_PROMPT.format(query=query)}],
    )
    verdict = response.choices[0].message.content.strip().lower()
    complexity = "complex" if "complex" in verdict else "simple"
    return complexity, "llm_classifier"


def route_model(query: str, use_llm: bool = False) -> str:
    """
    Returns the model name to use for this query.

    use_llm: if True, uses the LLM classifier (more accurate, costs tokens).
             Default is rule-based (faster, free, good enough for ~90% of cases).
    """
    if use_llm:
        complexity, _ = classify_complexity_llm(query)
    else:
        complexity, _ = classify_complexity(query)

    return MODEL_COMPLEX if complexity == "complex" else MODEL_SIMPLE


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box

    console = Console()

    test_queries = [
        # Simple — should route to mini
        ("What is the standard return window?",                              "simple"),
        ("Does Acmera ship internationally?",                                "simple"),
        ("What payment methods are accepted?",                               "simple"),
        ("How do I track my order?",                                         "simple"),
        ("What is the dead pixel policy?",                                   "simple"),
        ("What cities have same-day delivery?",                              "simple"),
        # Complex — should route to GPT-4o
        ("I'm Premium Silver and bought a laptop 40 days ago. Can I return?", "complex"),
        ("Compare the refund policy for Gold vs standard during Diwali sale.", "complex"),
        ("My Gold membership is expiring and I'm below the threshold. What do I lose?", "complex"),
        ("Need to return 3 of 8 laptops from a corporate order. What's the process?", "complex"),
        ("If I'm Silver and bought during a promotion, is my return window different?", "complex"),
    ]

    console.print(Panel(
        "[bold]Model Router — Rule-Based Classification[/]\n"
        "[dim]No API calls — purely regex-based fast path[/]",
        title="[bold cyan]Session 5 — Model Router[/]",
        border_style="cyan",
    ))

    table = Table(title="Routing Decisions", box=box.ROUNDED)
    table.add_column("Query", width=56)
    table.add_column("Expected", width=9)
    table.add_column("Model", width=14)
    table.add_column("Reason", width=18)
    table.add_column("OK?", justify="center", width=5)

    correct = 0
    for query, expected in test_queries:
        complexity, reason = classify_complexity(query)
        model = MODEL_COMPLEX if complexity == "complex" else MODEL_SIMPLE
        match = complexity == expected
        correct += match
        ok = "[green]✓[/]" if match else "[red]✗[/]"
        model_str = f"[dim]{model}[/]" if complexity == "simple" else f"[bold]{model}[/]"
        table.add_row(query[:56], expected, model_str, reason, ok)

    console.print(table)
    console.print(
        f"\n[bold]Accuracy:[/] {correct}/{len(test_queries)} "
        f"([{'green' if correct == len(test_queries) else 'yellow'}]{correct/len(test_queries)*100:.0f}%[/])"
    )

    # Cost savings illustration
    console.print()
    PRICING = {
        MODEL_SIMPLE: {"input": 0.15,  "output": 0.60},
        MODEL_COMPLEX: {"input": 2.50, "output": 10.00},
    }
    simple_count = sum(1 for _, e in test_queries if e == "simple")
    complex_count = len(test_queries) - simple_count
    mini_cost_per_1k = (0.15 * 500 + 0.60 * 300) / 1_000_000 * 1000
    all_4o_cost_per_1k = (2.50 * 500 + 10.00 * 300) / 1_000_000 * 1000
    routed_cost_per_1k = (
        simple_count / len(test_queries) * mini_cost_per_1k +
        complex_count / len(test_queries) * all_4o_cost_per_1k
    )
    savings = (1 - routed_cost_per_1k / all_4o_cost_per_1k) * 100

    console.print(Panel(
        f"[bold]With this query mix ({simple_count} simple / {complex_count} complex)[/]\n\n"
        f"All GPT-4o:   [red]${all_4o_cost_per_1k:.3f}[/] per 1,000 queries\n"
        f"Routed:       [green]${routed_cost_per_1k:.3f}[/] per 1,000 queries\n"
        f"Savings:      [bold green]{savings:.0f}%[/]",
        title="[bold yellow]Cost Impact[/]",
        border_style="yellow",
    ))
