#!/bin/sh
# Container entrypoint for the Synapse deployment.
# Run the pgvector startup DDL (extension, tables, indexes) once in a single process before
# forking uvicorn workers: on a fresh database, concurrent workers race on CREATE TABLE /
# CREATE INDEX, one dies with a UniqueViolation, and uvicorn 0.28 does not respawn it.
set -e

python - <<'EOF'
import asyncio

from app.config import VECTOR_DB_TYPE, VectorDBType
from app.services.database import PSQLDatabase, ensure_vector_indexes


async def main():
    if VECTOR_DB_TYPE != VectorDBType.PGVECTOR:
        return
    await ensure_vector_indexes()
    await PSQLDatabase.close_pool()


asyncio.run(main())
EOF

exec uvicorn main:app \
  --host "${RAG_HOST:-0.0.0.0}" \
  --port "${RAG_PORT:-8000}" \
  --workers "${RAG_WORKERS:-2}" \
  --timeout-keep-alive 75 \
  --no-access-log
