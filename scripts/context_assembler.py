"""
Context assembly pipeline — Session 4.

After retrieval (and optional re-ranking), raw chunks still need engineering
before they become an effective LLM prompt. This module handles:

  1. Deduplication — remove near-identical chunks that waste context tokens
  2. Context expansion (multi-hop) — fetch adjacent chunks for boundary coverage
  3. Source ordering — group by document, order by chunk_index within each doc
  4. Compression — enforce a token/char budget so we don't overflow the LLM

Why this matters: retrieval gives you the most relevant chunks, but the
context window is how you present them. Bad assembly = good retrieval wasted.

Usage:
    from context_assembler import assemble_advanced
    context = assemble_advanced(query, chunks, conn_fn=get_connection)
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(__file__))


# =========================================================================
# 1. DEDUPLICATION
# =========================================================================

def deduplicate(chunks: list, similarity_threshold: float = 0.75) -> list:
    """
    Remove near-duplicate chunks using word-level Jaccard similarity.

    Two chunks are considered duplicates if their word overlap exceeds
    the threshold. This happens when the same policy sentence appears in
    multiple documents or when chunk boundaries overlap slightly.

    Args:
        chunks: List of chunk dicts
        similarity_threshold: Jaccard threshold (0.75 = 75% word overlap)

    Returns:
        Deduplicated list preserving original order
    """
    seen_words = []
    unique = []

    for chunk in chunks:
        words = set(chunk["content"].lower().split())
        is_dup = False
        for seen in seen_words:
            if not words or not seen:
                continue
            intersection = len(words & seen)
            union = len(words | seen)
            if union > 0 and intersection / union >= similarity_threshold:
                is_dup = True
                break
        if not is_dup:
            unique.append(chunk)
            seen_words.append(words)

    return unique


# =========================================================================
# 2. CONTEXT EXPANSION (Multi-hop)
# =========================================================================

def expand_context(chunks: list, conn_fn, window: int = 1) -> list:
    """
    Small-to-big retrieval: fetch adjacent chunks for each top result.

    When chunk boundaries cut a sentence or paragraph, the adjacent chunk
    contains the missing context. This fetches chunk_index ± window from
    the same document for each retrieved chunk.

    This is the "small-to-big" or "parent-child" retrieval pattern:
    - Small chunk: retrieved because it's the most relevant sentence
    - Big chunk: the surrounding paragraph that gives it full context

    Args:
        chunks: Top retrieved chunks (the "small" chunks)
        conn_fn: Function that returns a DB connection
        window: How many adjacent chunks to fetch on each side

    Returns:
        Expanded list including original + adjacent chunks
    """
    import psycopg2
    from pgvector.psycopg2 import register_vector

    existing = {(c["doc_name"], c["chunk_index"]) for c in chunks}
    to_fetch = set()

    for chunk in chunks:
        doc = chunk["doc_name"]
        idx = chunk["chunk_index"]
        for delta in range(-window, window + 1):
            if delta != 0:
                to_fetch.add((doc, idx + delta))

    conn = conn_fn()
    cur = conn.cursor()

    expanded = list(chunks)
    for doc, idx in to_fetch:
        if (doc, idx) in existing:
            continue
        cur.execute(
            "SELECT id, doc_name, chunk_index, content, metadata FROM chunks WHERE doc_name=%s AND chunk_index=%s",
            (doc, idx),
        )
        row = cur.fetchone()
        if row:
            expanded.append({
                "id": row[0], "doc_name": row[1], "chunk_index": row[2],
                "content": row[3],
                "metadata": row[4] if isinstance(row[4], dict) else json.loads(row[4]),
                "similarity": 0.0,
                "is_expanded": True,  # Flag: this chunk was added for context, not retrieved
            })
            existing.add((doc, idx))

    cur.close()
    conn.close()
    return expanded


# =========================================================================
# 3. SOURCE ORDERING
# =========================================================================

def order_by_source(chunks: list) -> list:
    """
    Group chunks by document, sort by chunk_index within each doc.
    Put the highest-scoring document first.

    Why: Reading a document in chunk order is more coherent than jumping
    between chunks from different documents. The LLM generates better
    answers when context flows naturally.
    """
    # Find the best score per document
    doc_scores = {}
    for chunk in chunks:
        doc = chunk["doc_name"]
        score = chunk.get("rerank_score", chunk.get("rrf_score", chunk.get("similarity", 0)))
        if doc not in doc_scores or score > doc_scores[doc]:
            doc_scores[doc] = score

    # Group by doc
    by_doc = {}
    for chunk in chunks:
        doc = chunk["doc_name"]
        if doc not in by_doc:
            by_doc[doc] = []
        by_doc[doc].append(chunk)

    # Sort within each doc by chunk_index
    for doc in by_doc:
        by_doc[doc].sort(key=lambda c: c["chunk_index"])

    # Flatten: highest-scoring doc first
    sorted_docs = sorted(doc_scores, key=lambda d: doc_scores[d], reverse=True)
    ordered = []
    for doc in sorted_docs:
        ordered.extend(by_doc[doc])

    return ordered


# =========================================================================
# 4. COMPRESSION (Budget enforcement)
# =========================================================================

def compress(chunks: list, max_chars: int = 4000) -> list:
    """
    Drop chunks that exceed the character budget.

    A soft compression: we keep chunks in order until we hit the budget.
    The alternative (truncating chunk content) loses coherence.

    In production: use tiktoken to count tokens precisely.
    Here: chars ÷ 4 ≈ tokens is good enough for teaching purposes.

    Args:
        chunks: Ordered list of chunks
        max_chars: Context budget in characters

    Returns:
        Subset of chunks that fits within budget
    """
    kept = []
    total = 0
    for chunk in chunks:
        chunk_chars = len(chunk["content"]) + 60  # +60 for the source header
        if total + chunk_chars > max_chars and kept:
            break
        kept.append(chunk)
        total += chunk_chars
    return kept


# =========================================================================
# 5. FINAL ASSEMBLY
# =========================================================================

def format_context(chunks: list) -> str:
    """Format chunks into the final context string passed to the LLM."""
    parts = []
    for chunk in chunks:
        tag = " [expanded]" if chunk.get("is_expanded") else ""
        parts.append(
            f"[Source: {chunk['doc_name']}, Chunk {chunk['chunk_index']}{tag}]\n{chunk['content']}"
        )
    return "\n\n---\n\n".join(parts)


def assemble_advanced(
    chunks: list,
    conn_fn=None,
    max_chars: int = 4000,
    dedup_threshold: float = 0.75,
    expand_window: int = 1,
) -> tuple[str, list]:
    """
    Full context assembly pipeline:
      1. Deduplicate near-identical chunks
      2. Expand context (fetch adjacent chunks)
      3. Order by source document
      4. Compress to fit context budget

    Args:
        chunks: Retrieved (and optionally re-ranked) chunks
        conn_fn: DB connection function (needed for expand_context)
        max_chars: Maximum context length in characters
        dedup_threshold: Jaccard similarity threshold for dedup
        expand_window: Adjacent chunks to fetch on each side

    Returns:
        (context_str, final_chunks) — formatted context and the chunk list used
    """
    # Step 1: Deduplicate
    unique = deduplicate(chunks, similarity_threshold=dedup_threshold)

    # Step 2: Expand context (multi-hop) — only if DB connection available
    if conn_fn is not None and expand_window > 0:
        expanded = expand_context(unique, conn_fn, window=expand_window)
    else:
        expanded = unique

    # Step 3: Order by source
    ordered = order_by_source(expanded)

    # Step 4: Compress to budget
    final = compress(ordered, max_chars=max_chars)

    return format_context(final), final


if __name__ == "__main__":
    from rag import embed_query, hybrid_retrieve, get_connection
    from reranker import rerank

    query = "What is the return policy for premium members who bought during Diwali sale?"
    print(f"Query: {query}\n")

    query_embedding = embed_query(query)
    candidates = hybrid_retrieve(query, query_embedding, top_k=10)
    reranked = rerank(query, candidates, top_n=7)

    context, final_chunks = assemble_advanced(
        reranked, conn_fn=get_connection, max_chars=3500, expand_window=1
    )

    print(f"Pipeline: hybrid({len(candidates)}) → rerank({len(reranked)}) → assemble({len(final_chunks)})")
    print(f"Context length: {len(context)} chars\n")
    for i, c in enumerate(final_chunks, 1):
        tag = " [expanded]" if c.get("is_expanded") else ""
        score = c.get("rerank_score", c.get("rrf_score", c.get("similarity", 0)))
        print(f"  [{i}] {c['doc_name']} chunk {c['chunk_index']}{tag} — score: {score:.4f}")
