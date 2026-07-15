#!/usr/bin/env bash
# Ship the app to the provisioned EC2 host and bring the stack up (W7 D3).
#
# Separate from provision-ec2.sh on purpose: this is the step you re-run every
# time the code or corpus changes — it does not touch AWS resources.
#
# What it does:
#   1. rsync the repo to the box (minus .venv/.git/raw-docs/keys).
#   2. Make sure the box's .env carries DOMAIN (+ optional ACME_EMAIL) so Caddy
#      knows which hostname to get a certificate for.
#   3. docker compose up -d --build   (opensearch + api + caddy)
#   4. Run ingest ONCE against the fresh index.
#   5. Smoke-test /health from the box.
#
# Usage:
#   HOST=<public-ip> DOMAIN=ragops.example.com \
#     [ACME_EMAIL=you@example.com] [KEY=deploy/ragops-key.pem] \
#     bash deploy/deploy.sh
set -euo pipefail

HOST="${HOST:?set HOST=<ec2 public ip>}"
DOMAIN="${DOMAIN:?set DOMAIN=<the hostname whose A record points at HOST>}"
ACME_EMAIL="${ACME_EMAIL:-}"
KEY="${KEY:-$(dirname "$0")/ragops-key.pem}"
REMOTE_DIR="${REMOTE_DIR:-ragops-copilot}"
SSH_USER="${SSH_USER:-ubuntu}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "$SSH_USER@$HOST")

echo "==> 1/5 rsync $REPO_ROOT -> $SSH_USER@$HOST:$REMOTE_DIR"
# .env and data/chunks.jsonl ARE shipped (they're needed and gitignored, so they
# only exist locally). The private key, the venv, git history, raw docs and
# caches are not.
rsync -az --delete \
  -e "ssh -i $KEY -o StrictHostKeyChecking=accept-new" \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude 'deploy/*.pem' \
  --exclude 'data/vllm_src/' \
  --exclude 'node_modules/' \
  "$REPO_ROOT/" "$SSH_USER@$HOST:$REMOTE_DIR/"

echo "==> 2/5 ensure DOMAIN/ACME_EMAIL in the box's .env"
"${SSH[@]}" "cd $REMOTE_DIR && {
  touch .env
  grep -q '^DOMAIN=' .env || echo 'DOMAIN=$DOMAIN' >> .env
  sed -i 's|^DOMAIN=.*|DOMAIN=$DOMAIN|' .env
  grep -q '^ACME_EMAIL=' .env || echo 'ACME_EMAIL=$ACME_EMAIL' >> .env
  sed -i 's|^ACME_EMAIL=.*|ACME_EMAIL=$ACME_EMAIL|' .env
}"

echo "==> 3/5 docker compose up -d --build (first build pulls torch — a few min)"
"${SSH[@]}" "cd $REMOTE_DIR && docker compose -f docker-compose.prod.yml up -d --build"

echo "==> 4/5 wait for opensearch health, then ingest"
"${SSH[@]}" "cd $REMOTE_DIR && for i in \$(seq 1 40); do
  docker compose -f docker-compose.prod.yml exec -T opensearch \
    curl -fsS http://localhost:9200/_cluster/health >/dev/null 2>&1 && break
  sleep 3
done
docker compose -f docker-compose.prod.yml exec -T api python -m src.ingest
# Warm the k-NN graph. OpenSearch loads the HNSW graph lazily on the FIRST
# search after a fresh ingest, so the very first user query can come back with
# zero hits (-> a false 'not in the docs' refusal) while it loads. A couple of
# throwaway searches here pay that cost before any real visitor does.
docker compose -f docker-compose.prod.yml exec -T api python -c \"
from src.retrieve import search
from src.embeddings import Embedder
from src.opensearch_client import get_client
e=Embedder(); c=get_client()
for q in ('warm up','PagedAttention','continuous batching'):
    print('warm:', q, len(search(q, client=c, embedder=e)), 'hits')
\""

echo "==> 5/5 smoke test from the box"
"${SSH[@]}" "cd $REMOTE_DIR && docker compose -f docker-compose.prod.yml exec -T api \
  curl -fsS http://localhost:8000/health" && echo

cat <<EOF

Deployed. Caddy is now requesting a Let's Encrypt cert for $DOMAIN
(needs the A record pointing at $HOST and ports 80/443 open — both handled).

Verify from your laptop once DNS has propagated:
  curl https://$DOMAIN/health
  open  https://$DOMAIN/
EOF
