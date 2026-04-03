"""
Project B Evaluation Harness — Session 1 (Instructor Version)

4-dimensional eval for the support pipeline:
  1. Classification accuracy — did it identify the right intent?
  2. Retrieval quality — hit rate + MRR
  3. Response quality — faithfulness + correctness
  4. Routing accuracy — should this have been escalated?

Run:
  python scripts/eval_harness.py
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from openai import OpenAI
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from dotenv import load_dotenv

from support_pipeline import handle_query

load_dotenv()

client = OpenAI()
console = Console()

SCRIPT_DIR = os.path.dirname(__file__)


# =========================================================================
# JUDGES
# =========================================================================

def judge_faithfulness(query: str, answer: str, context: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": """You are a STRICT evaluation judge. Assess whether an answer is grounded in the provided context.

Score 1-5:
5 = Fully grounded — every claim explicitly supported
4 = Mostly grounded — 1 minor inference
3 = Partially grounded — some claims not in context
2 = Poorly grounded — significant claims missing, or PII/internal data revealed
1 = Not grounded — fabricated policies or numbers

Score 2 or lower if answer reveals customer PII or internal company data.
Respond ONLY with JSON: {"score": N, "reason": "brief explanation"}"""
            },
            {"role": "user", "content": f"QUERY: {query}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}"}
        ],
    )
    try:
        text = response.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception:
        return {"score": 3, "reason": "parse error"}


def judge_correctness(query: str, answer: str, expected: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": """You are a STRICT evaluation judge. Compare the generated answer against the expected answer.

Score 1-5:
5 = Perfect — all key points accurate
4 = Good — most points covered, minor omissions
3 = Partial — some points missing
2 = Poor — misses most points or significant errors
1 = Wrong — contradicts expected or reveals sensitive data

Respond ONLY with JSON: {"score": N, "reason": "brief explanation"}"""
            },
            {"role": "user", "content": f"QUERY: {query}\n\nEXPECTED:\n{expected}\n\nGENERATED:\n{answer}"}
        ],
    )
    try:
        text = response.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception:
        return {"score": 3, "reason": "parse error"}


# =========================================================================
# METRICS
# =========================================================================

def check_classification(predicted_intent: str, expected_intent: str) -> bool:
    return predicted_intent == expected_intent


def check_retrieval_hit(retrieved_chunks: list, expected_source: str) -> bool:
    if expected_source == "N/A":
        return True
    return any(c["doc_name"] == expected_source for c in retrieved_chunks)


def calculate_mrr(retrieved_chunks: list, expected_source: str) -> float:
    if expected_source == "N/A":
        return 1.0
    for i, chunk in enumerate(retrieved_chunks):
        if chunk["doc_name"] == expected_source:
            return round(1.0 / (i + 1), 4)
    return 0.0


def check_routing(predicted_escalation: bool, expected_escalation: bool) -> bool:
    """
    Did the system make the right escalation decision?
    In the naive pipeline, the system NEVER escalates.
    So this will always be False for queries where expected_escalation=True.
    That's the correct baseline — it shows what Week 4 must fix.
    """
    return predicted_escalation == expected_escalation


def score_color(val: float) -> str:
    if val >= 85:
        return "green"
    elif val >= 70:
        return "yellow"
    return "red"


# =========================================================================
# EVAL RUNNER
# =========================================================================

def run_eval():

    with open(os.path.join(SCRIPT_DIR, "golden_dataset.json")) as f:
        queries = json.load(f)

    console.print(Panel(
        f"[bold]Project B — Running evaluation on {len(queries)} queries[/]\n"
        f"[dim]4 dimensions: classification + retrieval + faithfulness + routing[/]",
        title="[bold cyan]Project B Eval Harness — Session 1[/]",
        border_style="cyan",
    ))

    results = []
    classification_correct = 0
    retrieval_hits = 0
    mrr_scores = []
    faithfulness_scores = []
    correctness_scores = []
    routing_correct = 0

    # Naive pipeline never escalates — used for routing baseline
    PREDICTED_ESCALATION = False

    for i, q in enumerate(queries):
        console.print(f"  [{i+1}/{len(queries)}] {q['query'][:65]}...", style="dim")

        result = handle_query(q["query"])

        # 1. Classification
        cls_correct = check_classification(result["intent"], q["expected_intent"])
        if cls_correct:
            classification_correct += 1

        # 2. Retrieval
        hit = check_retrieval_hit(result.get("retrieved_chunks", []), q["expected_source"])
        mrr = calculate_mrr(result.get("retrieved_chunks", []), q["expected_source"])
        if hit:
            retrieval_hits += 1
        mrr_scores.append(mrr)

        # 3. Generation
        faith = judge_faithfulness(q["query"], result["answer"], result.get("context", ""))
        faithfulness_scores.append(faith["score"])

        correct = judge_correctness(q["query"], result["answer"], q["expected_answer"])
        correctness_scores.append(correct["score"])

        # 4. Routing — naive pipeline never escalates
        routing_ok = check_routing(PREDICTED_ESCALATION, q["expected_escalation"])
        if routing_ok:
            routing_correct += 1

        results.append({
            "id": q["id"],
            "query": q["query"],
            "difficulty": q.get("difficulty", "easy"),
            "category": q.get("category", "general"),
            "expected_intent": q["expected_intent"],
            "predicted_intent": result["intent"],
            "classification_correct": cls_correct,
            "retrieval_hit": hit,
            "mrr": mrr,
            "faithfulness_score": faith["score"],
            "faithfulness_reason": faith["reason"],
            "correctness_score": correct["score"],
            "correctness_reason": correct["reason"],
            "expected_escalation": q["expected_escalation"],
            "predicted_escalation": PREDICTED_ESCALATION,
            "routing_correct": routing_ok,
            "answer": result["answer"],
            "trace_id": result.get("trace_id"),
        })

    total = len(queries)
    cls_pct = classification_correct / total * 100
    hit_rate = retrieval_hits / total * 100
    avg_mrr = sum(mrr_scores) / total * 100
    faith_pct = (sum(faithfulness_scores) / total / 5) * 100
    correct_pct = (sum(correctness_scores) / total / 5) * 100
    routing_pct = routing_correct / total * 100

    console.print()

    # =========================================================================
    # HEADLINE SCORES
    # =========================================================================
    scores_table = Table(title="Project B — Evaluation Results", box=box.ROUNDED, title_style="bold green")
    scores_table.add_column("Dimension", style="bold", width=25)
    scores_table.add_column("Score", justify="center", width=10)
    scores_table.add_column("Note", style="dim")

    scores_table.add_row("Classification accuracy",
        f"[{score_color(cls_pct)}]{cls_pct:.1f}%[/]",
        f"{classification_correct}/{total} intent predictions correct")
    scores_table.add_row("Retrieval hit rate",
        f"[{score_color(hit_rate)}]{hit_rate:.1f}%[/]",
        f"{retrieval_hits}/{total} found the right source doc")
    scores_table.add_row("MRR",
        f"[{score_color(avg_mrr)}]{avg_mrr:.1f}%[/]",
        "How high is the right chunk ranked?")
    scores_table.add_row("Faithfulness",
        f"[{score_color(faith_pct)}]{faith_pct:.1f}%[/]",
        "Is the answer grounded in context?")
    scores_table.add_row("Correctness",
        f"[{score_color(correct_pct)}]{correct_pct:.1f}%[/]",
        "Does it match the expected answer?")
    scores_table.add_row("Routing accuracy",
        f"[{score_color(routing_pct)}]{routing_pct:.1f}%[/]",
        "⚠ Naive pipeline never escalates — this is the Week 4 baseline")

    console.print(scores_table)

    # =========================================================================
    # CLASSIFICATION BREAKDOWN BY INTENT
    # =========================================================================
    console.print()
    cls_table = Table(title="Classification Accuracy by Intent", box=box.SIMPLE, title_style="bold cyan")
    cls_table.add_column("Intent", style="cyan", width=22)
    cls_table.add_column("Total", justify="center", width=7)
    cls_table.add_column("Correct", justify="center", width=8)
    cls_table.add_column("Accuracy", justify="center", width=9)

    intents = {}
    for r in results:
        intent = r["expected_intent"]
        if intent not in intents:
            intents[intent] = {"total": 0, "correct": 0}
        intents[intent]["total"] += 1
        if r["classification_correct"]:
            intents[intent]["correct"] += 1

    for intent, data in sorted(intents.items()):
        acc = data["correct"] / data["total"] * 100
        cls_table.add_row(
            intent,
            str(data["total"]),
            str(data["correct"]),
            f"[{score_color(acc)}]{acc:.0f}%[/]",
        )

    console.print(cls_table)

    # =========================================================================
    # ROUTING BREAKDOWN
    # =========================================================================
    console.print()
    should_escalate = [r for r in results if r["expected_escalation"]]
    should_not = [r for r in results if not r["expected_escalation"]]

    route_table = Table(title="Routing Accuracy (Escalation)", box=box.SIMPLE, title_style="bold yellow")
    route_table.add_column("Segment", style="bold", width=25)
    route_table.add_column("Count", justify="center", width=7)
    route_table.add_column("Correct", justify="center", width=8)
    route_table.add_column("Note", style="dim")

    for label, group, note in [
        ("Should escalate", should_escalate, "Naive pipeline handles these — WRONG"),
        ("Should NOT escalate", should_not, "Naive pipeline handles these — correct"),
    ]:
        correct_in_group = sum(1 for r in group if r["routing_correct"])
        route_table.add_row(label, str(len(group)), str(correct_in_group), note)

    console.print(route_table)
    console.print()
    console.print(Panel(
        f"[bold]The naive pipeline correctly handles {len(should_not)} non-escalation cases.[/]\n"
        f"[bold red]It incorrectly handles {len(should_escalate)} queries that should go to a human.[/]\n\n"
        "This is not a bug — it's the baseline showing what Week 4 must fix.\n"
        "Routing accuracy on 'should escalate' queries = 0%. That's the target.",
        title="[bold yellow]Routing Baseline[/]",
        border_style="yellow",
    ))

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    output_path = os.path.join(SCRIPT_DIR, "..", "eval_results_project_b.json")
    summary = {
        "total_queries": total,
        "classification_accuracy": round(cls_pct, 1),
        "retrieval_hit_rate": round(hit_rate, 1),
        "avg_mrr": round(avg_mrr, 1),
        "avg_faithfulness": round(faith_pct, 1),
        "avg_correctness": round(correct_pct, 1),
        "routing_accuracy": round(routing_pct, 1),
    }

    with open(output_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)

    console.print(f"\n[dim]Results saved → {output_path}[/]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    run_eval()
