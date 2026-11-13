# app/routes/kb_routes.py
"""Institution knowledge base routes. Each call is scoped to the tenant named in the caller's
signed token (``scope: "kb"``, ``tenant``); nothing in the request body can choose a tenant."""

from functools import lru_cache
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.config import EMBEDDINGS_MODEL, EMBEDDINGS_PROVIDER, init_embeddings, logger
from app.services import kb
from app.services.database import PSQLDatabase

router = APIRouter(prefix="/kb", tags=["knowledge-base"])

KB_SCOPE = "kb"


class KbChunk(BaseModel):
    chunk_index: int = Field(ge=0)
    kind: str = Field(pattern="^(text|table)$")
    section_path: str = ""
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    text: str = Field(min_length=1)
    embed_text: str = Field(min_length=1)


class KbChunksRequest(BaseModel):
    doc_id: str = Field(min_length=1, max_length=64)
    version: int = Field(ge=1)
    title: str = Field(min_length=1)
    doc_type: str = "other"
    chunks: List[KbChunk] = Field(max_length=256)


class KbActivateRequest(BaseModel):
    doc_id: str = Field(min_length=1, max_length=64)
    version: int = Field(ge=1)


class KbRenameRequest(BaseModel):
    title: Optional[str] = Field(default=None, min_length=1, max_length=300)
    doc_type: Optional[str] = Field(default=None, min_length=1, max_length=32)


class KbQueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=8, ge=1, le=30)
    doc_types: Optional[List[str]] = None


def kb_tenant(request: Request) -> str:
    user = getattr(request.state, "user", None) or {}
    tenant = user.get("tenant")
    if user.get("scope") != KB_SCOPE or not isinstance(tenant, str) or not tenant:
        raise HTTPException(status_code=403, detail="Knowledge base access needs a kb token")
    return tenant


@lru_cache(maxsize=1)
def kb_embedder() -> kb.Embedder:
    """Same provider and model as file search, asked for ``KB_DIMENSIONS`` so HNSW can index it."""
    return init_embeddings(EMBEDDINGS_PROVIDER, EMBEDDINGS_MODEL, dimensions=kb.KB_DIMENSIONS)


async def ensure_kb_schema() -> None:
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        await kb.ensure_schema(conn)
    logger.info("Knowledge base schema ready")


@router.post("/chunks")
async def upload_chunks(
    body: KbChunksRequest,
    tenant: str = Depends(kb_tenant),
    embedder: kb.Embedder = Depends(kb_embedder),
):
    pool = await PSQLDatabase.get_pool()
    chunks = [kb.ChunkIn(**chunk.model_dump()) for chunk in body.chunks]
    stored = await kb.upsert_chunks(
        pool, embedder, tenant, body.doc_id, body.version, body.title, body.doc_type, chunks
    )
    return {"stored": stored}


@router.post("/activate")
async def activate_version(body: KbActivateRequest, tenant: str = Depends(kb_tenant)):
    pool = await PSQLDatabase.get_pool()
    count = await kb.activate(pool, tenant, body.doc_id, body.version)
    if count == 0:
        raise HTTPException(status_code=409, detail="No chunks stored for this version")
    return {"chunks": count}


@router.patch("/documents/{doc_id}")
async def rename_document(
    doc_id: str, body: KbRenameRequest, tenant: str = Depends(kb_tenant)
):
    pool = await PSQLDatabase.get_pool()
    updated = await kb.rename_document(pool, tenant, doc_id, body.title, body.doc_type)
    return {"updated": updated}


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, tenant: str = Depends(kb_tenant)):
    pool = await PSQLDatabase.get_pool()
    return {"deleted": await kb.remove_document(pool, tenant, doc_id)}


@router.post("/query")
async def query(
    body: KbQueryRequest,
    tenant: str = Depends(kb_tenant),
    embedder: kb.Embedder = Depends(kb_embedder),
):
    pool = await PSQLDatabase.get_pool()
    results = await kb.search(pool, embedder, tenant, body.query, body.k, body.doc_types)
    return {"results": results}
