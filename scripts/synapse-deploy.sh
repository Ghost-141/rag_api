#!/usr/bin/env bash
# Build and roll out rag_api on the code-interpreter server, rolling back if it fails its health check.
# Usage: scripts/synapse-deploy.sh [git-ref]   (default: origin/bdren-prod)
set -euo pipefail
cd "$(dirname "$0")/.."

REF="${1:-origin/bdren-prod}"
HEALTH_URL="http://127.0.0.1:${RAG_HOST_PORT:-18000}/health"
KEEP_IMAGES=5

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

[ -f .env ] || fail ".env not found; create it from .env.synapse.example"
grep -qE '^COMPOSE_FILE=docker-compose.synapse.yml$' .env ||
  fail ".env must set COMPOSE_FILE=docker-compose.synapse.yml"
grep -qE '^RAG_IMAGE_TAG=' .env || fail ".env must contain a RAG_IMAGE_TAG= line"
docker network inspect "${RAG_EDGE_NETWORK:-rag_edge}" >/dev/null 2>&1 ||
  fail "docker network ${RAG_EDGE_NETWORK:-rag_edge} missing; run: docker network create rag_edge && docker network connect rag_edge caddy"
[ -z "$(git status --porcelain --untracked-files=no)" ] ||
  fail "tracked files have local changes; commit or discard them first"

set_tag() {
  sed -i "s/^RAG_IMAGE_TAG=.*/RAG_IMAGE_TAG=$1/" .env
}

wait_healthy() {
  for _ in $(seq 1 30); do
    if curl -fsS --max-time 5 "$HEALTH_URL" >/dev/null 2>&1; then
      return 0
    fi
    sleep 4
  done
  return 1
}

PREV_TAG=$(grep -E '^RAG_IMAGE_TAG=' .env | cut -d= -f2)

git fetch origin --prune
git checkout --detach "$REF"
TAG=$(git rev-parse --short HEAD)

echo "Deploying $TAG (previous: $PREV_TAG)"
docker compose config --quiet
RAG_IMAGE_TAG="$TAG" docker compose build rag_api
set_tag "$TAG"
docker compose up -d rag_db
docker compose up -d --no-deps rag_api

if wait_healthy; then
  echo "Healthy on $TAG"
  echo "$(date -Is) $PREV_TAG -> $TAG" >>deploy-history.log
  docker image ls synapse-rag-api --format '{{.Tag}}' |
    grep -vxF -e "$TAG" -e latest |
    tail -n +"$KEEP_IMAGES" |
    xargs -r -I{} docker image rm "synapse-rag-api:{}"
  exit 0
fi

echo "Health check failed on $TAG" >&2
docker logs --tail 50 synapse-rag-api >&2 || true

if [ "$PREV_TAG" = "$TAG" ] || ! docker image inspect "synapse-rag-api:$PREV_TAG" >/dev/null 2>&1; then
  fail "no previous image to roll back to (previous tag: $PREV_TAG)"
fi

echo "Rolling back to $PREV_TAG" >&2
set_tag "$PREV_TAG"
docker compose up -d --no-deps rag_api
git checkout --detach "$PREV_TAG" 2>/dev/null || true
wait_healthy && echo "Rolled back to $PREV_TAG (healthy)" >&2
exit 1
