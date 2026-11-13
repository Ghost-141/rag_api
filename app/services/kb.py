# app/services/kb.py
"""Institution knowledge base index: one table partitioned by tenant, searched hybrid.

Synapse's ingestion worker parses and chunks documents; this module only embeds the chunks
it is given, stores them per tenant, flips which version is searchable, and answers queries.
The tenant always comes from the caller's signed claim (see ``app/routes/kb_routes.py``).
"""

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence

KB_DIMENSIONS = 1536
RRF_K = 60
CANDIDATES = 40
EF_SEARCH = 80

_JOINERS = re.compile("[‌‍]")
_BANGLA_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


class Embedder(Protocol):
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]: ...

    async def aembed_query(self, text: str) -> List[float]: ...


@dataclass(frozen=True)
class ChunkIn:
    chunk_index: int
    kind: str
    section_path: str
    page_start: Optional[int]
    page_end: Optional[int]
    text: str
    embed_text: str


def fold(text: str) -> str:
    """Search-side folding, applied identically to indexed text and to queries."""
    text = unicodedata.normalize("NFC", text)
    text = _JOINERS.sub("", text).translate(_BANGLA_DIGITS)
    return text.lower()


def partition_name(tenant_id: str) -> str:
    return "kb_chunks_" + hashlib.sha1(tenant_id.encode("utf-8")).hexdigest()[:16]


def _vector(values: Sequence[float]) -> str:
    if len(values) != KB_DIMENSIONS:
        raise ValueError(
            f"Embedding has {len(values)} dimensions, expected {KB_DIMENSIONS}; "
            "check that the embeddings provider honours KB_EMBEDDINGS_DIMENSIONS"
        )
    return "[" + ",".join(repr(float(v)) for v in values) + "]"


SCHEMA_SQL = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE TABLE IF NOT EXISTS kb_chunks (
    tenant_id    text        NOT NULL,
    doc_id       text        NOT NULL,
    version      integer     NOT NULL,
    chunk_index  integer     NOT NULL,
    kind         text        NOT NULL,
    title        text        NOT NULL,
    doc_type     text        NOT NULL,
    section_path text        NOT NULL DEFAULT '',
    page_start   integer,
    page_end     integer,
    text         text        NOT NULL,
    search_text  text        NOT NULL,
    embedding    halfvec({KB_DIMENSIONS}) NOT NULL,
    tsv          tsvector    NOT NULL,
    is_current   boolean     NOT NULL DEFAULT false,
    created_at   timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, doc_id, version, chunk_index)
) PARTITION BY LIST (tenant_id);
CREATE INDEX IF NOT EXISTS kb_chunks_embedding_hnsw ON kb_chunks
    USING hnsw (embedding halfvec_cosine_ops) WITH (m = 16, ef_construction = 64)
    WHERE is_current;
CREATE INDEX IF NOT EXISTS kb_chunks_tsv ON kb_chunks USING gin (tsv) WHERE is_current;
CREATE INDEX IF NOT EXISTS kb_chunks_trgm ON kb_chunks
    USING gin (search_text gin_trgm_ops) WHERE is_current;
"""


async def ensure_schema(conn) -> None:
    await conn.execute(SCHEMA_SQL)


async def ensure_partition(conn, tenant_id: str) -> None:
    """Indexes declared on the parent are created on each new partition by Postgres."""
    name = partition_name(tenant_id)
    literal = tenant_id.replace("'", "''")
    await conn.execute(
        f"CREATE TABLE IF NOT EXISTS {name} PARTITION OF kb_chunks FOR VALUES IN ('{literal}')"
    )


async def upsert_chunks(
    pool,
    embedder: Embedder,
    tenant_id: str,
    doc_id: str,
    version: int,
    title: str,
    doc_type: str,
    chunks: List[ChunkIn],
) -> int:
    if not chunks:
        return 0
    vectors = await embedder.aembed_documents([c.embed_text for c in chunks])
    rows = [
        (
            tenant_id,
            doc_id,
            version,
            c.chunk_index,
            c.kind,
            title,
            doc_type,
            c.section_path,
            c.page_start,
            c.page_end,
            c.text,
            fold(c.text),
            _vector(vector),
            fold(f"{title} {c.section_path}"),
        )
        for c, vector in zip(chunks, vectors)
    ]
    async with pool.acquire() as conn:
        await ensure_partition(conn, tenant_id)
        await conn.executemany(
            """
            INSERT INTO kb_chunks (tenant_id, doc_id, version, chunk_index, kind, title,
                doc_type, section_path, page_start, page_end, text, search_text, embedding, tsv)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13::halfvec,
                setweight(to_tsvector('simple', $14), 'A')
                    || setweight(to_tsvector('simple', $12), 'D'))
            ON CONFLICT (tenant_id, doc_id, version, chunk_index) DO UPDATE SET
                kind = EXCLUDED.kind, title = EXCLUDED.title, doc_type = EXCLUDED.doc_type,
                section_path = EXCLUDED.section_path, page_start = EXCLUDED.page_start,
                page_end = EXCLUDED.page_end, text = EXCLUDED.text,
                search_text = EXCLUDED.search_text, embedding = EXCLUDED.embedding,
                tsv = EXCLUDED.tsv
            """,
            rows,
        )
    return len(rows)


async def activate(pool, tenant_id: str, doc_id: str, version: int) -> int:
    """Makes ``version`` the searchable one and drops every other version, atomically."""
    async with pool.acquire() as conn:
        async with conn.transaction():
            count = await conn.fetchval(
                "SELECT count(*) FROM kb_chunks WHERE tenant_id = $1 AND doc_id = $2 AND version = $3",
                tenant_id,
                doc_id,
                version,
            )
            if not count:
                return 0
            await conn.execute(
                "DELETE FROM kb_chunks WHERE tenant_id = $1 AND doc_id = $2 AND version <> $3",
                tenant_id,
                doc_id,
                version,
            )
            await conn.execute(
                "UPDATE kb_chunks SET is_current = true WHERE tenant_id = $1 AND doc_id = $2",
                tenant_id,
                doc_id,
            )
    return count


async def list_document_ids(pool, tenant_id: str) -> List[str]:
    """Every document this tenant has chunks for, current or not, for reconciliation."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT DISTINCT doc_id FROM kb_chunks WHERE tenant_id = $1 ORDER BY doc_id", tenant_id
        )
    return [row["doc_id"] for row in rows]


async def remove_document(pool, tenant_id: str, doc_id: str) -> int:
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM kb_chunks WHERE tenant_id = $1 AND doc_id = $2", tenant_id, doc_id
        )
    return int(result.split()[-1])


async def rename_document(
    pool, tenant_id: str, doc_id: str, title: Optional[str], doc_type: Optional[str]
) -> int:
    """Updates a document's title and type on every stored version and rebuilds the weighted
    keyword index, which carries the title. Embeddings keep the old title until re-indexed."""
    async with pool.acquire() as conn:
        async with conn.transaction():
            rows = await conn.fetch(
                "SELECT version, chunk_index, title, section_path FROM kb_chunks "
                "WHERE tenant_id = $1 AND doc_id = $2",
                tenant_id,
                doc_id,
            )
            updates = [
                (
                    tenant_id,
                    doc_id,
                    row["version"],
                    row["chunk_index"],
                    title or row["title"],
                    doc_type,
                    fold(f"{title or row['title']} {row['section_path']}"),
                )
                for row in rows
            ]
            await conn.executemany(
                """
                UPDATE kb_chunks SET title = $5, doc_type = COALESCE($6, doc_type),
                    tsv = setweight(to_tsvector('simple', $7), 'A')
                        || setweight(to_tsvector('simple', search_text), 'D')
                WHERE tenant_id = $1 AND doc_id = $2 AND version = $3 AND chunk_index = $4
                """,
                updates,
            )
    return len(rows)


SEARCH_SQL = """
WITH vec AS (
    SELECT doc_id, version, chunk_index,
           row_number() OVER (ORDER BY embedding <=> $2::halfvec) AS r
    FROM kb_chunks
    WHERE tenant_id = $1 AND is_current AND ($5::text[] IS NULL OR doc_type = ANY($5))
    ORDER BY embedding <=> $2::halfvec
    LIMIT $4
),
kw AS (
    SELECT doc_id, version, chunk_index,
           row_number() OVER (ORDER BY ts_rank_cd(tsv, q) DESC) AS r
    FROM kb_chunks, websearch_to_tsquery('simple', $3) AS q
    WHERE tenant_id = $1 AND is_current AND tsv @@ q
      AND ($5::text[] IS NULL OR doc_type = ANY($5))
    ORDER BY ts_rank_cd(tsv, q) DESC
    LIMIT $4
),
tri AS (
    SELECT doc_id, version, chunk_index,
           row_number() OVER (ORDER BY word_similarity($3, search_text) DESC) AS r
    FROM kb_chunks
    WHERE tenant_id = $1 AND is_current AND $3 <% search_text
      AND ($5::text[] IS NULL OR doc_type = ANY($5))
    ORDER BY word_similarity($3, search_text) DESC
    LIMIT $4
),
fused AS (
    SELECT doc_id, version, chunk_index, sum(1.0 / ($6 + r)) AS score
    FROM (SELECT * FROM vec UNION ALL SELECT * FROM kw UNION ALL SELECT * FROM tri) ranked
    GROUP BY doc_id, version, chunk_index
)
SELECT c.doc_id, c.version, c.chunk_index, c.kind, c.title, c.doc_type, c.section_path,
       c.page_start, c.page_end, c.text, f.score
FROM fused f
JOIN kb_chunks c ON c.tenant_id = $1 AND c.doc_id = f.doc_id
    AND c.version = f.version AND c.chunk_index = f.chunk_index
ORDER BY f.score DESC
LIMIT $7
"""


async def search(
    pool,
    embedder: Embedder,
    tenant_id: str,
    query: str,
    k: int = 8,
    doc_types: Optional[List[str]] = None,
) -> List[dict]:
    vector = _vector(await embedder.aembed_query(query))
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(f"SET LOCAL hnsw.ef_search = {EF_SEARCH}")
            rows = await conn.fetch(
                SEARCH_SQL,
                tenant_id,
                vector,
                fold(query),
                CANDIDATES,
                doc_types or None,
                RRF_K,
                k,
            )
    return [dict(row) for row in rows]
