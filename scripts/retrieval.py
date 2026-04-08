"""
Self-contained retrieval layer for Project B — Instructor version, Session 4.

Includes:
  retrieve()              — standard dense retrieval (Week 1)
  retrieve_filtered()     — metadata-filtered retrieval (Session 3)
  deduplicate_chunks()    — FAQ dedup by Jaccard similarity (Session 4)
  retrieve_with_dedup()   — retrieve + dedup in one call (Session 4)

In Week 3, this module gets replaced by LangGraph tool-based retrieval.
"""
import os
import json
from openai import OpenAI
from langfuse.decorators import observe, langfuse_context
import psycopg2
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
TOP_K = 5


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


@observe(name="retrieval_filtered")
def retrieve_filtered(query_embedding, doc_names: list[str], top_k=TOP_K):
    """Session 3: metadata-filtered retrieval."""
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
    parts = [
        f"[Source: {c['doc_name']}, Chunk {c['chunk_index']}]\n{c['content']}"
        for c in retrieved_chunks
    ]
    context = "\n\n---\n\n".join(parts)
    langfuse_context.update_current_observation(metadata={
        "num_chunks": len(retrieved_chunks), "total_context_chars": len(context),
    })
    return context
