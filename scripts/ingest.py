"""
Ingest policy documents into pgvector for Project B retrieval.
Run: python scripts/ingest.py
"""
import os
import glob
import json
from openai import OpenAI
import psycopg2
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

CHUNK_SIZE = 500
CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "corpus")


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


def naive_chunk(text, chunk_size=CHUNK_SIZE):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def embed_texts(texts):
    response = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [item.embedding for item in response.data]


def ingest():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM chunks;")

    doc_files = sorted(glob.glob(os.path.join(CORPUS_DIR, "*.md")))
    total_chunks = 0

    for filepath in doc_files:
        doc_name = os.path.basename(filepath)
        with open(filepath, "r") as f:
            content = f.read()

        chunks = naive_chunk(content)
        print(f"  {doc_name}: {len(chunks)} chunks")

        for batch_start in range(0, len(chunks), 20):
            batch = chunks[batch_start:batch_start + 20]
            embeddings = embed_texts(batch)

            for i, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                chunk_index = batch_start + i
                metadata = json.dumps({
                    "doc_name": doc_name,
                    "chunk_index": chunk_index,
                })
                cur.execute(
                    """INSERT INTO chunks (doc_name, chunk_index, content, embedding, metadata)
                       VALUES (%s, %s, %s, %s::vector, %s)""",
                    (doc_name, chunk_index, chunk, embedding, metadata),
                )

        total_chunks += len(chunks)

    conn.commit()
    cur.close()
    conn.close()
    print(f"\nDone: {len(doc_files)} documents, {total_chunks} chunks.")


if __name__ == "__main__":
    ingest()
