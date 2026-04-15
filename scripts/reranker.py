"""
Cross-encoder re-ranking using the Cohere Rerank API — Session 4.

Why re-rank? The bi-encoder (text-embedding-3-small) scores query-document
similarity independently. The cross-encoder sees the query AND document
together — much more accurate relevance scoring, but too slow to run on
the full corpus (that's why we retrieve with bi-encoder first).

Two-stage pipeline:
  1. Bi-encoder retrieves top-N candidates fast (vector similarity)
  2. Cross-encoder re-ranks those N candidates precisely (Cohere Rerank)

Setup:
  pip install cohere
  Add COHERE_API_KEY=... to your .env file (free tier at cohere.com)

Usage:
    from reranker import rerank
    reranked = rerank(query, chunks, top_n=5)

Run:
    python scripts/reranker.py
"""
import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))
load_dotenv()

import cohere

RERANK_MODEL = "rerank-english-v3.0"
co = cohere.Client(os.getenv("COHERE_API_KEY", ""))


def rerank(query: str, chunks: list, top_n: int = 5) -> list:
    """
    Re-rank retrieved chunks using Cohere's cross-encoder.

    The cross-encoder scores each (query, chunk) pair together — unlike the
    bi-encoder which scores them independently. This catches cases where the
    semantically close chunk isn't the most relevant one.

    Args:
        query: The user question
        chunks: List of chunk dicts from retrieve() or hybrid_retrieve()
        top_n: How many to keep after re-ranking

    Returns:
        Re-ranked list of chunk dicts with 'rerank_score' added (0.0–1.0)
    """
    if not chunks:
        return chunks

    if not os.getenv("COHERE_API_KEY"):
        print("Warning: COHERE_API_KEY not set — skipping reranking, returning original order.")
        return chunks[:top_n]

    documents = [c["content"] for c in chunks]

    response = co.rerank(
        model=RERANK_MODEL,
        query=query,
        documents=documents,
        top_n=min(top_n, len(chunks)),
    )

    reranked = []
    for result in response.results:
        chunk = chunks[result.index].copy()
        chunk["rerank_score"] = round(result.relevance_score, 4)
        reranked.append(chunk)

    return reranked


def rerank_with_comparison(query: str, chunks: list, top_n: int = 5) -> dict:
    """
    Re-rank and show how the order changed vs the original bi-encoder ranking.
    Useful for understanding when and why re-ranking helps.

    Returns:
        dict with 'reranked' chunks and 'order_changes' list
    """
    original_order = {c["id"]: i + 1 for i, c in enumerate(chunks)}
    reranked = rerank(query, chunks, top_n=top_n)

    changes = []
    for new_rank, chunk in enumerate(reranked, 1):
        old_rank = original_order.get(chunk["id"], "?")
        delta = old_rank - new_rank if isinstance(old_rank, int) else 0
        changes.append({
            "doc_name": chunk["doc_name"],
            "old_rank": old_rank,
            "new_rank": new_rank,
            "delta": delta,
            "rerank_score": chunk.get("rerank_score"),
        })

    return {"reranked": reranked, "order_changes": changes}


if __name__ == "__main__":
    from rag import embed_query, hybrid_retrieve

    query = "What is the return window for premium members?"
    print(f"Query: {query}\n")

    query_embedding = embed_query(query)
    candidates = hybrid_retrieve(query, query_embedding, top_k=10)

    print(f"Before reranking ({len(candidates)} candidates):")
    for i, c in enumerate(candidates, 1):
        score = c.get("rrf_score", c.get("similarity", 0))
        print(f"  [{i}] {c['doc_name']} (chunk {c['chunk_index']}) — score: {score:.4f}")

    result = rerank_with_comparison(query, candidates, top_n=5)
    print(f"\nAfter reranking (top 5):")
    for change in result["order_changes"]:
        arrow = "↑" if change["delta"] > 0 else ("↓" if change["delta"] < 0 else "→")
        print(
            f"  [{change['new_rank']}] {change['doc_name']} — "
            f"rerank: {change['rerank_score']:.4f} "
            f"{arrow} (was #{change['old_rank']})"
        )
