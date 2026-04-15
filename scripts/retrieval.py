"""
Retrieval layer for Project B — Session 7 upgrade.

Pipeline progression (all functions preserved for backwards compat):

  Session 1  — retrieve()              dense vector search
  Session 3  — retrieve_filtered()     metadata-filtered dense search
  Session 4  — retrieve_with_dedup()   dense + Jaccard dedup
  Session 7  — hybrid_retrieve()       BM25 + dense + RRF fusion
             — retrieve_advanced()     hybrid → Cohere rerank → context assembly
               (now used by retrieve_policy() in support_pipeline.py)

Session 7 additions mirror Project A's Session 4-5 retrieval stack,
applied to the support agent's policy_kb tool. The agent's other tools
(order_tracker, account_lookup) are unaffected — they don't use retrieval.

Why upgrade now: the LangGraph agent's evaluate_node was looping back for
more context because the first policy_kb call returned insufficient chunks.
With hybrid retrieval + reranking, the agent gets it right in one shot.
"""
import os
import json
import sys

sys.path.insert(0, os.path.dirname(__file__))

from openai import OpenAI
from langfuse.decorators import observe, langfuse_context
import psycopg2
from pgvector.psycopg2 import register_vector
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

from reranker import rerank
from context_assembler import assemble_advanced

load_dotenv()

client = OpenAI()
TOP_K = 5
BM25_CANDIDATES = TOP_K * 3
RERANK_TOP_N = TOP_K + 2


def get_connection():
    conn = psycopg2.connect(
        host=os.getenv("PG_HOST", "localhost"),
        port=os.getenv("PG_PORT", "5434"),
        user=os.getenv("PG_USER", "workshop"),
        password=os.getenv("PG_PASSWORD", "workshop123"),
        dbname=os.getenv("PG_DATABASE", "acmera_kb"),
    )
    register_vector(conn)
    return conn


@observe(name="query_embedding")
def embed_query(query):
    response = client.embeddings.create(model="text-embedding-3-small", input=query)
    return response.data[0].embedding


# =============================================================================
# SESSION 1 — Dense retrieval
# =============================================================================

@observe(name="retrieval")
def retrieve(query_embedding, top_k=TOP_K):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """SELECT id, doc_name, chunk_index, content, metadata,
                  1 - (embedding <=> %s::vector) AS similarity
           FROM chunks ORDER BY embedding <=> %s::vector LIMIT %s""",
        (query_embedding, query_embedding, top_k),
    )
    results = []
    for row in cur.fetchall():
        results.append({
            "id": row[0], "doc_name": row[1], "chunk_index": row[2],
            "content": row[3],
            "metadata": row[4] if isinstance(row[4], dict) else json.loads(row[4]),
            "similarity": round(float(row[5]), 4),
        })
    cur.close()
    conn.close()
    langfuse_context.update_current_observation(metadata={
        "top_k": top_k, "filter": None,
        "results": [{"doc_name": r["doc_name"], "similarity": r["similarity"]} for r in results],
    })
    return results


# =============================================================================
# SESSION 3 — Metadata-filtered dense retrieval
# =============================================================================

@observe(name="retrieval_filtered")
def retrieve_filtered(query_embedding, doc_names: list[str], top_k=TOP_K):
    """Session 3: metadata-filtered retrieval — restrict to intent-relevant docs."""
    conn = get_connection()
    cur = conn.cursor()
    placeholders = ",".join(["%s"] * len(doc_names))
    cur.execute(
        f"""SELECT id, doc_name, chunk_index, content, metadata,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM chunks WHERE doc_name IN ({placeholders})
            ORDER BY embedding <=> %s::vector LIMIT %s""",
        (query_embedding, *doc_names, query_embedding, top_k),
    )
    results = []
    for row in cur.fetchall():
        results.append({
            "id": row[0], "doc_name": row[1], "chunk_index": row[2],
            "content": row[3],
            "metadata": row[4] if isinstance(row[4], dict) else json.loads(row[4]),
            "similarity": round(float(row[5]), 4),
        })
    cur.close()
    conn.close()
    langfuse_context.update_current_observation(metadata={
        "top_k": top_k, "filter": doc_names,
        "results": [{"doc_name": r["doc_name"], "similarity": r["similarity"]} for r in results],
    })
    return results


# =============================================================================
# SESSION 4 — Deduplication
# =============================================================================

def deduplicate_chunks(chunks: list, similarity_threshold: float = 0.75) -> list:
    """Session 4: Remove near-duplicate chunks (Jaccard word overlap)."""
    seen_words = []
    unique = []
    for chunk in chunks:
        words = set(chunk["content"].lower().split())
        is_dup = any(
            len(words & seen) / max(len(words | seen), 1) >= similarity_threshold
            for seen in seen_words if words and seen
        )
        if not is_dup:
            unique.append(chunk)
            seen_words.append(words)
    return unique


def retrieve_with_dedup(query_embedding, doc_names: list[str] | None = None,
                        top_k: int = TOP_K + 3) -> list:
    """Session 4: Retrieve more candidates, deduplicate, return top_k."""
    candidates_needed = top_k + 3
    if doc_names:
        candidates = retrieve_filtered(query_embedding, doc_names, top_k=candidates_needed)
    else:
        candidates = retrieve(query_embedding, top_k=candidates_needed)
    return deduplicate_chunks(candidates)[:top_k]


@observe(name="context_assembly")
def assemble_context(retrieved_chunks):
    """Simple context assembly (Sessions 1–4)."""
    parts = [
        f"[Source: {c['doc_name']}, Chunk {c['chunk_index']}]\n{c['content']}"
        for c in retrieved_chunks
    ]
    context = "\n\n---\n\n".join(parts)
    langfuse_context.update_current_observation(metadata={
        "num_chunks": len(retrieved_chunks), "total_context_chars": len(context),
    })
    return context


# =============================================================================
# SESSION 7 — BM25 helpers
# =============================================================================

def _load_chunks_for_bm25(doc_names: list[str] | None = None) -> list[dict]:
    """Load chunks from DB for BM25 indexing, optionally filtered by doc_names."""
    conn = get_connection()
    cur = conn.cursor()
    if doc_names:
        placeholders = ",".join(["%s"] * len(doc_names))
        cur.execute(
            f"SELECT id, doc_name, chunk_index, content, metadata "
            f"FROM chunks WHERE doc_name IN ({placeholders}) ORDER BY id",
            doc_names,
        )
    else:
        cur.execute(
            "SELECT id, doc_name, chunk_index, content, metadata FROM chunks ORDER BY id"
        )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [{
        "id": row[0], "doc_name": row[1], "chunk_index": row[2], "content": row[3],
        "metadata": row[4] if isinstance(row[4], dict) else json.loads(row[4]),
    } for row in rows]


def _build_bm25_index(doc_names: list[str] | None = None):
    """Build an in-memory BM25 index from the corpus (optionally filtered)."""
    all_chunks = _load_chunks_for_bm25(doc_names)
    bm25 = BM25Okapi([c["content"].lower().split() for c in all_chunks])
    return bm25, all_chunks


def _bm25_retrieve(query: str, bm25, all_chunks: list, top_k: int = BM25_CANDIDATES) -> list:
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    results = []
    for idx, score in ranked:
        if score > 0:
            chunk = all_chunks[idx].copy()
            chunk["bm25_score"] = round(float(score), 4)
            chunk["similarity"] = 0.0
            results.append(chunk)
    return results


def _rrf_fusion(dense_results: list, bm25_results: list,
                top_k: int = TOP_K, k: int = 60) -> list:
    """Reciprocal Rank Fusion — merge dense and BM25 ranked lists."""
    scores, chunk_map = {}, {}
    for rank, chunk in enumerate(dense_results):
        cid = chunk["id"]
        scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
        chunk_map[cid] = chunk
    for rank, chunk in enumerate(bm25_results):
        cid = chunk["id"]
        scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
        if cid not in chunk_map:
            chunk_map[cid] = chunk
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [{**chunk_map[cid], "rrf_score": round(rrf, 6)} for cid, rrf in ranked]


# =============================================================================
# SESSION 7 — Hybrid retrieval
# =============================================================================

@observe(name="retrieval_hybrid")
def hybrid_retrieve(query: str, query_embedding: list,
                    doc_names: list[str] | None = None,
                    top_k: int = TOP_K) -> list:
    """
    Session 7: BM25 + dense vector search fused with Reciprocal Rank Fusion.

    When doc_names is provided (intent-based filter), BM25 is built from
    only those documents — keeps the keyword signal focused on the right corpus.
    """
    bm25, all_chunks = _build_bm25_index(doc_names)

    # Dense: use filtered or full retrieval
    if doc_names:
        dense = retrieve_filtered.__wrapped__(query_embedding, doc_names, top_k=BM25_CANDIDATES)
    else:
        dense = retrieve.__wrapped__(query_embedding, top_k=BM25_CANDIDATES)

    bm25_results = _bm25_retrieve(query, bm25, all_chunks, top_k=BM25_CANDIDATES)
    fused = _rrf_fusion(dense, bm25_results, top_k=top_k)

    langfuse_context.update_current_observation(metadata={
        "mode": "hybrid", "doc_filter": doc_names,
        "dense_candidates": len(dense),
        "bm25_candidates": len(bm25_results),
        "fused": len(fused),
        "results": [{"doc_name": r["doc_name"], "rrf_score": r.get("rrf_score")} for r in fused],
    })
    return fused


# =============================================================================
# SESSION 7 — Full advanced pipeline (used by retrieve_policy)
# =============================================================================

@observe(name="retrieval_advanced")
def retrieve_advanced(query: str, query_embedding: list,
                      doc_names: list[str] | None = None) -> tuple[str, list]:
    """
    Session 7: Full advanced retrieval pipeline for the policy_kb tool.

      hybrid_retrieve()    — BM25 + dense + RRF (2× candidates)
      rerank()             — Cohere cross-encoder reranks candidates
      assemble_advanced()  — dedup + context expansion + source ordering + compression

    Args:
        query:           Raw query text (for BM25 + reranker)
        query_embedding: Pre-computed embedding (for dense retrieval)
        doc_names:       Optional intent-based doc filter

    Returns:
        (context_str, final_chunks)
    """
    # Step 1: Hybrid retrieval — get 2× candidates for reranker input
    candidates = hybrid_retrieve(
        query, query_embedding, doc_names=doc_names, top_k=TOP_K * 2
    )

    # Step 2: Cohere reranker — cross-encoder precision on the candidates
    reranked = rerank(query, candidates, top_n=RERANK_TOP_N)

    # Step 3: Context assembly — dedup, expand, order, compress
    context, final_chunks = assemble_advanced(
        reranked, conn_fn=get_connection, max_chars=4000, expand_window=1
    )

    langfuse_context.update_current_observation(metadata={
        "pipeline": "advanced",
        "candidates": len(candidates),
        "reranked": len(reranked),
        "final_chunks": len(final_chunks),
        "context_chars": len(context),
        "doc_filter": doc_names,
    })

    return context, final_chunks
