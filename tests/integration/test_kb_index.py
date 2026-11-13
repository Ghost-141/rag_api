"""Knowledge base index against a real pgvector Postgres: storage, version flips, tenant
isolation and hybrid search. Embeddings are a deterministic bag of words, so the test needs no
provider; the search SQL, indexes and folding are the real ones."""

import hashlib
import math
import re

import asyncpg
import pytest
from fastapi import HTTPException
from starlette.requests import Request
from testcontainers.postgres import PostgresContainer

from app.routes.kb_routes import kb_tenant
from app.services import kb

pytestmark = pytest.mark.integration

WORD = re.compile(r"\w+")


class BagOfWords:
    """Hashes each word into one of KB_DIMENSIONS buckets and normalises the vector."""

    def _embed(self, text: str):
        vector = [0.0] * kb.KB_DIMENSIONS
        for word in WORD.findall(kb.fold(text)):
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % kb.KB_DIMENSIONS
            vector[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    async def aembed_documents(self, texts):
        return [self._embed(text) for text in texts]

    async def aembed_query(self, text):
        return self._embed(text)


EMBEDDER = BagOfWords()


def chunk(index, text, section="", kind="text", page=1):
    return kb.ChunkIn(
        chunk_index=index,
        kind=kind,
        section_path=section,
        page_start=page,
        page_end=page,
        text=text,
        embed_text=f"{section}\n\n{text}",
    )


DIRECTORY = [
    chunk(
        0,
        "| Team | Email | Phone |\n| --- | --- | --- |\n"
        "| Network Operations Centre (NOC) | noc@bdren.net.bd | +880 1711 000111 |\n"
        "| Service Desk | support@bdren.net.bd | +880 1711 000222 |",
        "1. Team directory",
        kind="table",
    ),
    chunk(1, "একাধিক প্রতিষ্ঠানে বিভ্রাট হলে ১৫ মিনিটের মধ্যে NOC-কে জানাতে হবে।", "2. Escalation"),
    chunk(2, "Annual leave is twenty days for permanent staff.", "3. Leave"),
]


@pytest.fixture(scope="module")
def pg_dsn():
    with PostgresContainer("pgvector/pgvector:pg17", driver=None) as pg:
        yield pg.get_connection_url().replace("postgresql+psycopg2://", "postgresql://")


@pytest.fixture
async def pool(pg_dsn):
    pool = await asyncpg.create_pool(pg_dsn)
    async with pool.acquire() as conn:
        await kb.ensure_schema(conn)
        await conn.execute("TRUNCATE kb_chunks")
    yield pool
    await pool.close()


async def index(pool, tenant, doc_id, version, chunks, activate=True):
    await kb.upsert_chunks(pool, EMBEDDER, tenant, doc_id, version, "BdREN contacts", "directory", chunks)
    if activate:
        await kb.activate(pool, tenant, doc_id, version)


async def test_finds_the_noc_row_by_keyword_and_meaning(pool):
    await index(pool, "bdren", "doc-1", 1, DIRECTORY)
    results = await kb.search(pool, EMBEDDER, "bdren", "NOC team email")
    assert results[0]["chunk_index"] == 0
    assert "noc@bdren.net.bd" in results[0]["text"]
    assert results[0]["section_path"] == "1. Team directory"


async def test_bangla_digits_match_ascii_digits(pool):
    await index(pool, "bdren", "doc-1", 1, DIRECTORY)
    results = await kb.search(pool, EMBEDDER, "bdren", "15 মিনিটের মধ্যে", k=1)
    assert results[0]["chunk_index"] == 1


async def test_an_institution_never_sees_another_institutions_chunks(pool):
    await index(pool, "bdren", "doc-1", 1, DIRECTORY)
    await index(pool, "du", "doc-9", 1, [chunk(0, "Dhaka University library hours")])
    assert await kb.search(pool, EMBEDDER, "du", "NOC team email") == [] or all(
        r["doc_id"] == "doc-9" for r in await kb.search(pool, EMBEDDER, "du", "NOC team email")
    )
    tables = await pool.fetch(
        "SELECT tableoid::regclass::text AS part, count(*) FROM kb_chunks GROUP BY 1 ORDER BY 1"
    )
    assert sorted(row["part"] for row in tables) == sorted(
        [kb.partition_name("bdren"), kb.partition_name("du")]
    )


async def test_a_new_version_is_hidden_until_activated_then_replaces_the_old(pool):
    await index(pool, "bdren", "doc-1", 1, DIRECTORY)
    updated = [chunk(0, "| Team | Email |\n| --- | --- |\n| NOC | noc-new@bdren.net.bd |", "Directory")]
    await index(pool, "bdren", "doc-1", 2, updated, activate=False)

    before = await kb.search(pool, EMBEDDER, "bdren", "NOC email")
    assert {r["version"] for r in before} == {1}

    assert await kb.activate(pool, "bdren", "doc-1", 2) == 1
    after = await kb.search(pool, EMBEDDER, "bdren", "NOC email")
    assert {r["version"] for r in after} == {2}
    assert await pool.fetchval("SELECT count(*) FROM kb_chunks WHERE version = 1") == 0


async def test_activating_a_version_with_no_chunks_changes_nothing(pool):
    await index(pool, "bdren", "doc-1", 1, DIRECTORY)
    assert await kb.activate(pool, "bdren", "doc-1", 7) == 0
    assert await pool.fetchval("SELECT count(*) FROM kb_chunks WHERE is_current") == 3


async def test_retried_uploads_overwrite_instead_of_duplicating(pool):
    await index(pool, "bdren", "doc-1", 1, DIRECTORY)
    await index(pool, "bdren", "doc-1", 1, DIRECTORY)
    assert await pool.fetchval("SELECT count(*) FROM kb_chunks") == 3


async def test_removing_a_document_drops_its_chunks(pool):
    await index(pool, "bdren", "doc-1", 1, DIRECTORY)
    assert await kb.remove_document(pool, "bdren", "doc-1") == 3
    assert await kb.search(pool, EMBEDDER, "bdren", "NOC") == []


async def test_search_uses_the_hnsw_index_at_realistic_size(pool):
    await index(pool, "bdren", "doc-1", 1, DIRECTORY)
    await pool.execute(
        """
        INSERT INTO kb_chunks (tenant_id, doc_id, version, chunk_index, kind, title, doc_type,
            text, search_text, embedding, tsv, is_current)
        SELECT 'bdren', 'bulk', 1, n, 'text', 'Bulk', 'other', 'row ' || n, 'row ' || n,
            (SELECT array_agg(random())::real[] FROM generate_series(1, $1) WHERE n > 0)
                ::vector::halfvec,
            to_tsvector('simple', 'row ' || n), true
        FROM generate_series(1, 2000) AS n
        """,
        kb.KB_DIMENSIONS,
    )
    await pool.execute("ANALYZE kb_chunks")
    vector = kb._vector(await EMBEDDER.aembed_query("noc"))
    rows = await pool.fetch(
        "EXPLAIN SELECT doc_id FROM kb_chunks WHERE tenant_id = 'bdren' AND is_current "
        f"ORDER BY embedding <=> '{vector}'::halfvec LIMIT 40"
    )
    plan = "\n".join(row[0] for row in rows)
    assert "Index Scan using" in plan and "embedding_idx" in plan, plan


def _request(user):
    request = Request({"type": "http", "headers": []})
    request.state.user = user
    return request


def test_kb_routes_need_a_kb_token_naming_the_tenant():
    assert kb_tenant(_request({"id": "kb:bdren", "scope": "kb", "tenant": "bdren"})) == "bdren"
    for user in [None, {"id": "user-1"}, {"scope": "kb"}, {"scope": "other", "tenant": "bdren"}]:
        with pytest.raises(HTTPException) as error:
            kb_tenant(_request(user))
        assert error.value.status_code == 403
