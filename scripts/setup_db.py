"""
Step 1: Set up the pgvector database for Project B policy retrieval.
Run: python scripts/setup_db.py
"""
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    return psycopg2.connect(
        host=os.getenv("PG_HOST", "localhost"),
        port=os.getenv("PG_PORT", "5434"),
        user=os.getenv("PG_USER", "workshop"),
        password=os.getenv("PG_PASSWORD", "workshop123"),
        dbname=os.getenv("PG_DATABASE", "acmera_kb"),
    )


def setup():
    conn = get_connection()
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute("DROP TABLE IF EXISTS chunks;")

    cur.execute("""
        CREATE TABLE chunks (
            id SERIAL PRIMARY KEY,
            doc_name TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding vector(1536),
            metadata JSONB DEFAULT '{}'
        );
    """)

    cur.execute("""
        CREATE INDEX ON chunks
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
    """)

    cur.close()
    conn.close()
    print("Database setup complete.")


if __name__ == "__main__":
    setup()
