"""
Guardrails — Session 7.

Two-layer safety check wrapping the support agent:

  check_input(query)   — runs BEFORE the agent graph (guardrail_node)
    1. OpenAI Moderation API: violence, hate, harassment, self-harm, sexual content
    2. Instructor + gpt-4o-mini: PII detection and anonymisation in the query

  check_output(answer) — scan agent response before returning
    1. Instructor + gpt-4o-mini: detect PII leakage or sensitive internal data

Fits into the LangGraph agent as a guardrail_node at the front of the graph:
  guardrail → classify → tool_call → evaluate → respond / escalate
      ↓ (if unsafe)
    reject   (zero LLM cost — canned response, no pipeline runs)

No spaCy, no Presidio NLP engine — all detection is LLM-based via Instructor.

Usage:
    from scripts.guardrails import check_input, check_output

    guard = check_input(query)
    if not guard.safe:
        return "reject"   # routes to reject_node in the agent graph

Run:
    python -m scripts.guardrails
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import instructor
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

load_dotenv()

_raw_client = OpenAI()
client = instructor.from_openai(OpenAI())
console = Console()

PII_MODEL = "gpt-4o-mini"


# =============================================================================
# RESPONSE MODELS
# =============================================================================

class InputGuardResult(BaseModel):
    safe: bool
    flagged_categories: list[str] = Field(default_factory=list)
    contains_pii: bool = False
    pii_entities: list[str] = Field(default_factory=list)
    anonymized_query: str
    rejection_reason: str = ""


class OutputGuardResult(BaseModel):
    safe: bool
    pii_leaked: bool = False
    sensitive_data_leaked: bool = False
    entities_found: list[str] = Field(default_factory=list)
    clean_answer: str
    leak_reason: str = ""


# =============================================================================
# INPUT GUARD
# =============================================================================

class _PIIDetection(BaseModel):
    contains_pii: bool
    entities_found: list[str] = Field(
        description="List of PII entity types detected, e.g. ['EMAIL', 'PHONE', 'ORDER_ID']"
    )
    anonymized_text: str = Field(
        description="Query with each PII instance replaced by [REDACTED_TYPE]"
    )


def check_input(query: str) -> InputGuardResult:
    """
    Run input guardrails before the agent processes a query.

    Step 1 — OpenAI Moderation API (free, no tokens consumed):
      Catches violence, hate speech, harassment, self-harm, sexual content.
      Unsafe → reject immediately, agent graph never runs.

    Step 2 — Instructor PII detection (gpt-4o-mini):
      Detects emails, phones, order IDs, account numbers, card numbers.
      Anonymises the query so PII is not passed to tools or logged.
    """
    # -- Step 1: Content moderation -----------------------------------------
    mod = _raw_client.moderations.create(input=query)
    result = mod.results[0]

    if result.flagged:
        flagged = [
            cat for cat, is_flagged in result.categories.model_dump().items()
            if is_flagged
        ]
        return InputGuardResult(
            safe=False,
            flagged_categories=flagged,
            anonymized_query=query,
            rejection_reason=f"Content policy violation: {', '.join(flagged)}",
        )

    # -- Step 2: PII detection + anonymisation --------------------------------
    pii: _PIIDetection = client.chat.completions.create(
        model=PII_MODEL,
        response_model=_PIIDetection,
        messages=[
            {
                "role": "system",
                "content": """Detect and anonymise PII in customer support queries.

PII to detect:
- EMAIL        — any email address
- PHONE        — any phone or mobile number
- ORDER_ID     — order reference numbers (e.g. ORD-12345)
- ACCOUNT_ID   — customer or account IDs
- CARD_NUMBER  — credit/debit card numbers
- NAME         — full person names (first + last)
- ADDRESS      — street addresses or PIN codes

Replace each PII instance with [REDACTED_TYPE] in anonymized_text.
If no PII is found, return the original text unchanged.""",
            },
            {"role": "user", "content": query},
        ],
    )

    return InputGuardResult(
        safe=True,
        flagged_categories=[],
        contains_pii=pii.contains_pii,
        pii_entities=pii.entities_found,
        anonymized_query=pii.anonymized_text,
    )


# =============================================================================
# OUTPUT GUARD
# =============================================================================

class _OutputScan(BaseModel):
    pii_leaked: bool = Field(
        description="True if the response contains specific customer PII"
    )
    sensitive_data_leaked: bool = Field(
        description="True if the response reveals internal data: cost prices, "
                    "margins, discount authority levels, employee names, or "
                    "internal Slack/support ticket content"
    )
    entities_found: list[str] = Field(
        description="List of leaked entity types, e.g. ['CUSTOMER_NAME', 'COST_PRICE']",
        default_factory=list,
    )
    clean_answer: str = Field(
        description="Answer with any leaked PII or sensitive data replaced by [REDACTED]"
    )
    leak_reason: str = Field(
        default="",
        description="Brief explanation if any leak was found, empty string otherwise",
    )


def check_output(answer: str) -> OutputGuardResult:
    """
    Scan the agent's generated answer for PII leakage or sensitive internal data.
    Called before the final answer is returned to the user.
    """
    scan: _OutputScan = client.chat.completions.create(
        model=PII_MODEL,
        response_model=_OutputScan,
        messages=[
            {
                "role": "system",
                "content": """Scan this customer support response for two types of leaks:

1. PII leakage — specific customer data:
   - Real customer names, email addresses, phone numbers
   - Specific order IDs or account details from internal records
   - Payment or card information

2. Sensitive internal data:
   - Internal cost prices or margin percentages
   - Discount authority levels (e.g. "agents can offer up to 10%")
   - Internal Slack messages or employee communications
   - Retention offer scripts or escalation playbooks

General policy information (return windows, shipping rates, membership benefits)
is NOT sensitive — that's the correct content for the response.

Replace any leaked data with [REDACTED] in clean_answer.""",
            },
            {"role": "user", "content": answer},
        ],
    )

    return OutputGuardResult(
        safe=not (scan.pii_leaked or scan.sensitive_data_leaked),
        pii_leaked=scan.pii_leaked,
        sensitive_data_leaked=scan.sensitive_data_leaked,
        entities_found=scan.entities_found,
        clean_answer=scan.clean_answer,
        leak_reason=scan.leak_reason,
    )


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    console.print(Panel(
        "[bold]Guardrails Demo — Input + Output[/]\n"
        "[dim]OpenAI Moderation → Instructor PII detection → Output scan[/]",
        title="[bold cyan]Project B Session 7 — Guardrails[/]",
        border_style="cyan",
    ))

    input_tests = [
        ("What is the return window for electronics?",               "clean query"),
        ("My email is rahul.sharma@gmail.com, where is my order?",  "email in query"),
        ("Call me at 9876543210, order ORD-445521 is missing",      "phone + order ID"),
        ("I will hurt the delivery person if this isn't resolved",  "moderation flag"),
    ]

    console.print("\n[bold]Input Guardrail Tests[/]")
    in_table = Table(box=box.SIMPLE)
    in_table.add_column("Query", width=50)
    in_table.add_column("Type", width=20)
    in_table.add_column("Safe", justify="center", width=6)
    in_table.add_column("PII entities", width=28)

    for query, label in input_tests:
        guard = check_input(query)
        safe_str = "[green]✓[/]" if guard.safe else "[red]✗[/]"
        entities = ", ".join(guard.pii_entities) if guard.pii_entities else (
            guard.rejection_reason[:26] if guard.rejection_reason else "—"
        )
        in_table.add_row(query[:48] + ".." if len(query) > 48 else query,
                         label, safe_str, entities)

    console.print(in_table)

    console.print("\n[bold]Output Guardrail Tests[/]")
    output_tests = [
        ("Standard customers have a 30-day return window.", "clean answer"),
        ("Hi Rahul! Your order ORD-445521 placed by rahul@gmail.com "
         "will be refunded to card ending 4242.", "PII leakage"),
        ("Support agents can offer 10% discount; team leads 20%. "
         "Use code STAY20 for retention.", "internal data leak"),
    ]

    out_table = Table(box=box.SIMPLE)
    out_table.add_column("Answer (preview)", width=52)
    out_table.add_column("Type", width=18)
    out_table.add_column("Safe", justify="center", width=6)
    out_table.add_column("Leak type", width=20)

    for answer, label in output_tests:
        guard = check_output(answer)
        safe_str = "[green]✓[/]" if guard.safe else "[red]✗[/]"
        leak = []
        if guard.pii_leaked:
            leak.append("PII")
        if guard.sensitive_data_leaked:
            leak.append("internal data")
        out_table.add_row(answer[:50] + ".." if len(answer) > 50 else answer,
                          label, safe_str, ", ".join(leak) if leak else "—")

    console.print(out_table)
