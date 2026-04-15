"""
Feedback Pipeline — Session 7 (Project B, Instructor Version)

Closes the agent improvement loop:

  1. run_agent_on_dataset()   — run golden dataset through run_agent()
  2. run_ragas_eval()         — RAGAS faithfulness / answer_relevancy /
                                context_precision / context_recall
                                (only for policy_kb tool calls — agent answers)
  3. find_weak_queries()      — bottom quartile or below 0.6 threshold
  4. analyze_escalation_patterns() — group escalations by intent, find recurring
                                     patterns that suggest missing policy coverage
  5. compare_to_baseline()    — diff against baseline_scores.json

Design decisions:
  - RAGAS runs only on policy_kb results (order_tracker/account_lookup are
    not retrieval quality problems)
  - Escalation pattern analysis: repeated escalations on same intent signal
    gaps in the knowledge base or policy documentation
  - Results saved to feedback_results.json alongside agent trajectory data
  - The golden dataset is shared from project-a (same corpus, same docs)

Run:
  python -m scripts.feedback_pipeline
  python -m scripts.feedback_pipeline --save-baseline
  python -m scripts.feedback_pipeline --intent membership
"""
import os
import sys
import json
import argparse
import statistics
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

load_dotenv()

from scripts.agent import run_agent

console = Console()

SCRIPT_DIR = os.path.dirname(__file__)

# Project B can use project-a's golden dataset (same corpus)
PROJECT_A_GOLDEN = os.path.join(SCRIPT_DIR, "..", "..", "project-a-instructor", "scripts", "golden_dataset.json")
LOCAL_GOLDEN = os.path.join(SCRIPT_DIR, "golden_dataset.json")

BASELINE_PATH = os.path.join(SCRIPT_DIR, "baseline_scores.json")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "..", "feedback_results.json")

WEAK_THRESHOLD = 0.6


# =============================================================================
# LOAD GOLDEN DATASET
# =============================================================================

def load_golden_dataset() -> list[dict]:
    """Load golden dataset — prefer local copy, fall back to project-a."""
    for path in [LOCAL_GOLDEN, PROJECT_A_GOLDEN]:
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            console.print(f"[dim]Loaded {len(data)} queries from {path}[/]")
            return data
    return []


# =============================================================================
# STEP 1 — Run the agent on each golden dataset query
# =============================================================================

def run_agent_on_dataset(queries: list[dict]) -> list[dict]:
    """
    Run each query through run_agent() and collect:
      - final answer
      - tools called (trajectory)
      - intent detected
      - steps taken
      - escalation flag
    """
    results = []
    for i, q in enumerate(queries):
        console.print(f"  [{i+1}/{len(queries)}] {q['query'][:70]}...", style="dim")
        try:
            r = run_agent(q["query"])
            results.append({
                "id": q["id"],
                "query": q["query"],
                "expected_answer": q.get("expected_answer", ""),
                "expected_source": q.get("expected_source", "N/A"),
                "difficulty": q.get("difficulty", "easy"),
                "category": q.get("category", "general"),
                "answer": r["answer"],
                "intent": r["intent"],
                "tools_called": r["tools_called"],
                "steps_taken": r["steps_taken"],
                "should_escalate": r["should_escalate"],
                "elapsed_seconds": r["elapsed_seconds"],
                # context is not directly returned by run_agent; placeholder for RAGAS
                "context": "",
                "retrieved_chunks": [],
            })
        except Exception as e:
            console.print(f"  [red]Error on query {q['id']}: {e}[/]")
            results.append({
                "id": q["id"],
                "query": q["query"],
                "expected_answer": q.get("expected_answer", ""),
                "expected_source": q.get("expected_source", "N/A"),
                "difficulty": q.get("difficulty", "easy"),
                "category": q.get("category", "general"),
                "answer": "",
                "intent": "",
                "tools_called": [],
                "steps_taken": 0,
                "should_escalate": False,
                "elapsed_seconds": 0,
                "context": "",
                "retrieved_chunks": [],
                "error": str(e),
            })
    return results


# =============================================================================
# STEP 2 — RAGAS evaluation (policy_kb queries only)
# =============================================================================

def run_ragas_eval(pipeline_results: list[dict]) -> list[dict]:
    """
    Run RAGAS on results where the agent used policy_kb.

    Queries that only hit order_tracker or account_lookup are not retrieval
    quality problems — they are skipped. Escalated queries are also skipped
    (no final answer to evaluate).

    Each result gets a "ragas_scores" key (or None if not applicable).
    """
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from datasets import Dataset
    from scripts.support_pipeline import retrieve_policy

    # Mark which results are eligible for RAGAS
    ragas_rows = []
    ragas_indices = []

    for i, r in enumerate(pipeline_results):
        # Skip escalated queries and errors
        if r.get("should_escalate") or r.get("error") or not r.get("answer"):
            r["ragas_scores"] = None
            r["ragas_skipped_reason"] = "escalated_or_error"
            continue

        # Skip if agent didn't use policy_kb at all
        tools = r.get("tools_called", [])
        if "policy_kb" not in tools:
            r["ragas_scores"] = None
            r["ragas_skipped_reason"] = "no_policy_kb_call"
            continue

        # Re-retrieve context for the query (we need the actual chunks for RAGAS)
        try:
            context, chunks = retrieve_policy(r["query"], r["intent"])
            r["context"] = context
            r["retrieved_chunks"] = chunks
        except Exception:
            context = r.get("context", "")
            chunks = r.get("retrieved_chunks", [])

        if not context:
            r["ragas_scores"] = None
            r["ragas_skipped_reason"] = "no_context"
            continue

        # Build context list for RAGAS
        if chunks:
            contexts = [c["content"] for c in chunks if c.get("content")]
        else:
            contexts = [seg.strip() for seg in context.split("---") if seg.strip()]

        ragas_rows.append({
            "question": r["query"],
            "answer": r["answer"],
            "contexts": contexts,
            "ground_truth": r["expected_answer"],
            "_id": r["id"],
        })
        ragas_indices.append(i)

    if not ragas_rows:
        console.print("[yellow]No policy_kb queries eligible for RAGAS evaluation.[/]")
        return pipeline_results

    console.print(f"\n[dim]Running RAGAS on {len(ragas_rows)} policy_kb queries...[/]")

    ds = Dataset.from_list(ragas_rows)

    llm_wrapper = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    emb_wrapper = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

    for metric in [faithfulness, answer_relevancy, context_precision, context_recall]:
        metric.llm = llm_wrapper
    for metric in [answer_relevancy, context_precision, context_recall]:
        metric.embeddings = emb_wrapper

    ragas_result = evaluate(
        dataset=ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    ragas_df = ragas_result.to_pandas()

    for j, idx in enumerate(ragas_indices):
        row = ragas_df.iloc[j]
        pipeline_results[idx]["ragas_scores"] = {
            "faithfulness":      round(float(row.get("faithfulness", 0) or 0), 4),
            "answer_relevancy":  round(float(row.get("answer_relevancy", 0) or 0), 4),
            "context_precision": round(float(row.get("context_precision", 0) or 0), 4),
            "context_recall":    round(float(row.get("context_recall", 0) or 0), 4),
        }

    return pipeline_results


# =============================================================================
# STEP 3 — Find weak queries
# =============================================================================

def find_weak_queries(
    results: list[dict],
    threshold: float = WEAK_THRESHOLD,
) -> list[dict]:
    """
    A query is weak if:
      - ANY RAGAS metric is below threshold, OR
      - Its composite score is in the bottom quartile of all RAGAS-scored results
    """
    scored = [r for r in results
              if r.get("ragas_scores") and
              any(v is not None for v in r["ragas_scores"].values())]

    def composite(r: dict) -> float:
        vals = [v for v in r["ragas_scores"].values() if v is not None]
        return sum(vals) / len(vals) if vals else 0.0

    composites = [composite(r) for r in scored]
    q1_threshold = sorted(composites)[len(composites) // 4] if len(composites) >= 4 else threshold

    weak = []
    for r in scored:
        scores = r["ragas_scores"]
        below_threshold = any(v is not None and v < threshold for v in scores.values())
        below_q1 = composite(r) <= q1_threshold
        if below_threshold or below_q1:
            r["composite_score"] = round(composite(r), 4)
            r["weak_reasons"] = [
                f"{k}<{threshold}" for k, v in scores.items()
                if v is not None and v < threshold
            ]
            weak.append(r)

    return weak


# =============================================================================
# STEP 4 — Escalation pattern analysis
# =============================================================================

def analyze_escalation_patterns(results: list[dict]) -> dict:
    """
    Group escalated queries by intent to find recurring patterns.

    High escalation rate on a specific intent = the agent doesn't have good
    policy coverage for that topic, OR the queries are genuinely hard edge cases.

    Returns a summary dict with intent-level escalation rates.
    """
    intent_totals: dict[str, int] = defaultdict(int)
    intent_escalations: dict[str, int] = defaultdict(int)
    escalated_queries: dict[str, list[str]] = defaultdict(list)

    for r in results:
        intent = r.get("intent") or "unknown"
        intent_totals[intent] += 1
        if r.get("should_escalate"):
            intent_escalations[intent] += 1
            escalated_queries[intent].append(r["query"])

    total_escalated = sum(intent_escalations.values())
    total_queries = len(results)

    patterns = {}
    for intent in sorted(intent_totals, key=lambda x: intent_escalations[x], reverse=True):
        total = intent_totals[intent]
        escalated = intent_escalations[intent]
        rate = escalated / total if total > 0 else 0.0
        patterns[intent] = {
            "total": total,
            "escalated": escalated,
            "rate": round(rate, 3),
            "sample_queries": escalated_queries[intent][:3],
        }

    return {
        "total_queries": total_queries,
        "total_escalated": total_escalated,
        "overall_escalation_rate": round(total_escalated / total_queries, 3) if total_queries else 0,
        "intent_patterns": patterns,
    }


# =============================================================================
# STEP 5 — Compare to baseline
# =============================================================================

def compare_to_baseline(results: list[dict]) -> dict:
    """Diff current RAGAS averages against baseline_scores.json."""
    if not os.path.exists(BASELINE_PATH):
        console.print("[dim]No baseline found — skipping comparison.[/]")
        return {}

    with open(BASELINE_PATH) as f:
        baseline = json.load(f)

    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    current = {}
    for metric in metrics:
        vals = [
            r["ragas_scores"][metric]
            for r in results
            if r.get("ragas_scores") and r["ragas_scores"].get(metric) is not None
        ]
        current[metric] = round(statistics.mean(vals), 4) if vals else None

    deltas = {}
    for metric in metrics:
        baseline_val = baseline.get("ragas", {}).get(metric)
        current_val = current.get(metric)
        if baseline_val is not None and current_val is not None:
            deltas[metric] = {
                "baseline": baseline_val,
                "current": current_val,
                "delta": round(current_val - baseline_val, 4),
            }

    return {"current": current, "deltas": deltas}


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def _score_color(val: float | None) -> str:
    if val is None:
        return "dim"
    if val >= 0.75:
        return "green"
    elif val >= 0.55:
        return "yellow"
    return "red"


def display_agent_summary(results: list[dict]):
    total = len(results)
    escalated = sum(1 for r in results if r.get("should_escalate"))
    avg_steps = statistics.mean(r["steps_taken"] for r in results) if results else 0
    avg_time = statistics.mean(r["elapsed_seconds"] for r in results) if results else 0

    table = Table(title="Agent Run Summary", box=box.ROUNDED, title_style="bold cyan")
    table.add_column("Metric", style="bold", width=26)
    table.add_column("Value", justify="center", width=12)

    table.add_row("Total queries", str(total))
    table.add_row("Escalated",
        f"[{'red' if escalated > total * 0.2 else 'green'}]{escalated} ({escalated/total*100:.0f}%)[/]")
    table.add_row("Avg steps per query", f"{avg_steps:.1f}")
    table.add_row("Avg response time", f"{avg_time:.2f}s")

    console.print(table)


def display_ragas_summary(results: list[dict]) -> dict:
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    avgs = {}
    for m in metrics:
        vals = [r["ragas_scores"][m] for r in results
                if r.get("ragas_scores") and r["ragas_scores"].get(m) is not None]
        avgs[m] = round(statistics.mean(vals), 4) if vals else None

    ragas_eligible = sum(1 for r in results if r.get("ragas_scores") is not None)
    table = Table(
        title=f"RAGAS Evaluation (policy_kb queries: {ragas_eligible})",
        box=box.ROUNDED, title_style="bold cyan"
    )
    table.add_column("Metric", style="bold", width=22)
    table.add_column("Score", justify="center", width=10)
    table.add_column("Interpretation", style="dim")

    interpretations = {
        "faithfulness":      "Answer grounded in retrieved policy context",
        "answer_relevancy":  "Answer directly addresses the question",
        "context_precision": "Retrieved policy chunks are relevant",
        "context_recall":    "Context covers what's needed to answer",
    }
    for m in metrics:
        v = avgs[m]
        color = _score_color(v)
        score_str = f"[{color}]{v:.3f}[/]" if v is not None else "[dim]N/A[/]"
        table.add_row(m, score_str, interpretations[m])

    console.print(table)
    return avgs


def display_escalation_patterns(patterns: dict):
    intent_data = patterns.get("intent_patterns", {})
    if not intent_data:
        return

    table = Table(title="Escalation Patterns by Intent", box=box.SIMPLE, title_style="bold yellow")
    table.add_column("Intent", style="cyan", width=22)
    table.add_column("Queries", justify="center", width=9)
    table.add_column("Escalated", justify="center", width=10)
    table.add_column("Rate", justify="center", width=8)
    table.add_column("Sample query", width=38)

    for intent, data in intent_data.items():
        rate = data["rate"]
        rate_color = "red" if rate > 0.4 else "yellow" if rate > 0.2 else "green"
        sample = data["sample_queries"][0][:36] + "..." if data["sample_queries"] else ""
        table.add_row(
            intent,
            str(data["total"]),
            str(data["escalated"]),
            f"[{rate_color}]{rate:.0%}[/]",
            sample,
        )

    overall_rate = patterns.get("overall_escalation_rate", 0)
    console.print(table)
    console.print(
        f"\n[dim]Overall escalation rate: "
        f"[{'red' if overall_rate > 0.3 else 'green'}]{overall_rate:.0%}[/] "
        f"({patterns['total_escalated']}/{patterns['total_queries']} queries)[/]"
    )


def display_weak_queries(weak: list[dict]):
    if not weak:
        console.print("[green]No weak queries found — pipeline looks healthy.[/]")
        return

    table = Table(
        title=f"Weak Queries ({len(weak)} found)", box=box.SIMPLE, title_style="bold red"
    )
    table.add_column("ID", width=6)
    table.add_column("Query", width=44)
    table.add_column("Intent", width=18)
    table.add_column("Composite", justify="center", width=9)
    table.add_column("Weak reasons", width=28)

    for r in sorted(weak, key=lambda x: x.get("composite_score", 0)):
        table.add_row(
            r["id"],
            r["query"][:42] + "..." if len(r["query"]) > 42 else r["query"],
            r.get("intent", "unknown")[:16],
            f"[{_score_color(r.get('composite_score'))}]{r.get('composite_score', 0):.3f}[/]",
            ", ".join(r.get("weak_reasons", [])) or "bottom quartile",
        )

    console.print(table)


def display_baseline_comparison(comparison: dict):
    if not comparison or not comparison.get("deltas"):
        return

    table = Table(title="vs Baseline", box=box.SIMPLE, title_style="bold yellow")
    table.add_column("Metric", style="bold", width=22)
    table.add_column("Baseline", justify="center", width=10)
    table.add_column("Current", justify="center", width=10)
    table.add_column("Delta", justify="center", width=10)

    for metric, data in comparison["deltas"].items():
        delta = data["delta"]
        delta_color = "green" if delta >= 0 else "red"
        delta_sign = "+" if delta >= 0 else ""
        table.add_row(
            metric,
            f"{data['baseline']:.3f}",
            f"{data['current']:.3f}",
            f"[{delta_color}]{delta_sign}{delta:.3f}[/]",
        )

    console.print(table)


# =============================================================================
# SAVE BASELINE
# =============================================================================

def save_ragas_baseline(avgs: dict):
    existing = {}
    if os.path.exists(BASELINE_PATH):
        with open(BASELINE_PATH) as f:
            existing = json.load(f)

    existing["ragas"] = avgs

    with open(BASELINE_PATH, "w") as f:
        json.dump(existing, f, indent=2)

    console.print(Panel(
        f"[bold green]RAGAS baseline saved → {BASELINE_PATH}[/]\n"
        + "\n".join(f"  {k}: {v:.3f}" for k, v in avgs.items() if v is not None),
        title="[bold green]Baseline Locked[/]",
        border_style="green",
    ))


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Agent feedback pipeline: RAGAS + escalation analysis")
    parser.add_argument("--save-baseline", action="store_true",
                        help="Save current RAGAS scores as new baseline")
    parser.add_argument("--intent", type=str,
                        help="Filter golden dataset to a specific intent category")
    parser.add_argument("--threshold", type=float, default=WEAK_THRESHOLD,
                        help=f"RAGAS weak score threshold (default: {WEAK_THRESHOLD})")
    args = parser.parse_args()

    console.print(Panel(
        "[bold]Agent Feedback Pipeline — Session 7[/]\n"
        "[dim]RAGAS eval on policy_kb queries + escalation pattern analysis[/]",
        title="[bold cyan]Project B — Feedback Loop[/]",
        border_style="cyan",
    ))

    # Load golden dataset
    golden = load_golden_dataset()
    if not golden:
        console.print("[red]No golden dataset found.[/]")
        console.print("[dim]Expected at scripts/golden_dataset.json or project-a-instructor/scripts/golden_dataset.json[/]")
        return

    if args.intent:
        golden = [q for q in golden if q.get("category", "").startswith(args.intent)]
        console.print(f"[dim]Filtered to {len(golden)} queries matching intent: {args.intent}[/]")

    console.print(f"\n[bold]Step 1:[/] Running agent on {len(golden)} queries...\n")
    results = run_agent_on_dataset(golden)

    console.print()
    display_agent_summary(results)

    console.print(f"\n[bold]Step 2:[/] RAGAS evaluation (policy_kb queries only)...\n")
    results = run_ragas_eval(results)

    console.print()
    avgs = display_ragas_summary(results)

    console.print(f"\n[bold]Step 3:[/] Finding weak queries (threshold={args.threshold})...\n")
    weak = find_weak_queries(results, threshold=args.threshold)
    display_weak_queries(weak)

    console.print(f"\n[bold]Step 4:[/] Escalation pattern analysis...\n")
    escalation_patterns = analyze_escalation_patterns(results)
    display_escalation_patterns(escalation_patterns)

    console.print(f"\n[bold]Step 5:[/] Comparing to baseline...\n")
    comparison = compare_to_baseline(results)
    display_baseline_comparison(comparison)

    if args.save_baseline:
        save_ragas_baseline(avgs)

    # Save full results
    output = {
        "summary": {
            "total_queries": len(results),
            "weak_queries": len(weak),
            "ragas_averages": avgs,
            "escalation_patterns": escalation_patterns,
        },
        "results": results,
        "weak_queries": weak,
        "baseline_comparison": comparison,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    console.print(f"\n[dim]Results saved → {RESULTS_PATH}[/]")
    console.print()


if __name__ == "__main__":
    main()
