# RAGOps Copilot — the FastAPI service as a container (W7 D2).
#
# Build:  docker build -t ragops-copilot:latest .
# Run:    docker compose up -d      (compose also brings up OpenSearch)
#
# The shape of this file is driven by four production concerns. Each is explained
# at the line that implements it; the summary:
#
#   1. MULTI-STAGE — `builder` has uv and does the installing; `runtime` receives
#      the finished venv and nothing else. The uv binary, the build cache and the
#      downloaded wheels never reach the shipped image: less to pull, and a
#      smaller attack surface (no package manager sitting in production).
#   2. LAYER CACHING — dependency manifests are copied and installed BEFORE the
#      source. Editing src/ then rebuilds only the last two cheap layers; the
#      multi-minute torch install stays a cache hit. Copying source first would
#      invalidate the dependency layer on every one-character change.
#   3. NON-ROOT — the process runs as `app` (uid 1000). If the service is ever
#      exploited, the attacker lands as an unprivileged user rather than as root
#      inside the container (a much shorter hop to root on the host).
#   4. NO WEIGHTS IN THE IMAGE — bge-small (~130 MB) is NOT baked in. It is
#      downloaded on first start into HF_HOME, which compose mounts as a named
#      volume, so the download happens once and is reused across restarts while
#      the image stays lean.

# ==============================================================================
# Stage 1 — builder: resolve and install dependencies into a self-contained venv
# ==============================================================================
# This base is python:3.11-slim with uv preinstalled. Using it (rather than
# curl-ing uv into a plain slim image) keeps the toolchain pinned and the build
# hermetic.
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

# UV_COMPILE_BYTECODE — precompile .pyc at build time, so the first request does
#   not pay import-compile latency (we optimize startup, not build time).
# UV_LINK_MODE=copy — uv prefers to hardlink from its cache; that cache lives on a
#   different mount (see below), where hardlinks are impossible. `copy` silences
#   the fallback warning and is what we want regardless: the venv must be
#   self-contained, because stage 2 copies it away from the cache entirely.
# UV_PYTHON_DOWNLOADS=never — force uv to use the image's system python. If uv
#   fetched its own standalone interpreter, the venv would point at a path that
#   does not exist in the runtime stage and the image would die on `python`.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# --- The layer-caching seam ---------------------------------------------------
# ONLY the dependency manifests. They change rarely, so the expensive layer below
# is reused across builds. src/ is copied in the runtime stage instead, so editing
# a .py file cannot invalidate the install.
COPY pyproject.toml uv.lock ./

# --- Install ------------------------------------------------------------------
# --frozen             : install exactly what uv.lock pins; never re-resolve. A
#                        build that silently upgraded a dependency would defeat
#                        the entire point of shipping a container.
# --no-default-groups  : drop the `gpu` group (the host default).
# --group cpu          : ...and take the CPU torch wheel instead — ~200 MB, zero
#                        nvidia-* packages. See the long comment in
#                        pyproject.toml; this single flag is the difference
#                        between a ~5.3 GB venv and a small one. The container has
#                        no GPU, so CUDA kernels here would be dead weight.
#                        THE TRADE-OFF: embedding now runs on CPU, measurably
#                        slower than on the 5080. That is the normal cloud-deploy
#                        bargain (GPU instances cost real money) — it is the
#                        latency we quantify today and revisit on D6.
# --mount=type=cache   : keep uv's wheel cache on a BuildKit cache mount — fast
#                        rebuilds, and not one downloaded wheel is left behind in
#                        an image layer.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --group cpu

# ==============================================================================
# Stage 2 — runtime: python, the venv, and our source. Nothing else.
# ==============================================================================
# Deliberately plain python:3.11-slim (no uv). Same Debian bookworm base and same
# python 3.11 as the builder — that ABI match is what makes the copied venv valid.
# Bumping one stage's base without the other would break it in subtle ways.
FROM python:3.11-slim AS runtime

# PYTHONUNBUFFERED — python buffers stdout when it is not a TTY, so `docker logs`
#   would show nothing until the buffer flushed. For a service whose logs ARE the
#   monitoring signal (D4), silent logs are a real bug.
# PYTHONDONTWRITEBYTECODE — the venv is already compiled (UV_COMPILE_BYTECODE), so
#   there is no reason to litter .pyc files at runtime.
# PATH — venv first, so `python` / `uvicorn` resolve to the venv's copies with no
#   `uv run` and no activate script.
# HF_HOME — where sentence-transformers caches downloaded weights. Pointed at a
#   directory the non-root user owns, and mounted as a named volume by compose so
#   the ~130 MB download happens once rather than on every container start.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/home/app/.cache/huggingface

# --- Non-root user (security basic) -------------------------------------------
# Created BEFORE the copies, so we can hand it ownership of the paths it writes.
# Docker's default is to run as root; a container escape would then begin at uid 0.
# This costs one line and removes that whole class of escalation.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app /home/app/.cache/huggingface \
    && chown -R app:app /app /home/app

WORKDIR /app

# The venv built in stage 1 — the only thing we take from it. uv, the build cache
# and every downloaded wheel stay behind in a stage that is never shipped.
COPY --from=builder --chown=app:app /app/.venv /app/.venv

# Source last: the layer most likely to change sits at the very end, so a code
# edit rebuilds only from here down.
COPY --chown=app:app src ./src

USER app

EXPOSE 8000

# --- HEALTHCHECK --------------------------------------------------------------
# How the orchestrator tells "the process is alive" apart from "the service can
# actually answer a question". compose consumes it via
# `depends_on: condition: service_healthy`; a load balancer would use it to pull a
# broken instance out of rotation.
#
# It hits our own /health, which is stricter than a ping: 503 unless OpenSearch is
# reachable AND the index is non-empty. So a freshly created stack is correctly
# "unhealthy" until `src.ingest` has run — a true report, not a bug.
#
# urllib rather than curl: python is already here, so we skip an apt layer (and
# the extra binaries) just for a probe. urlopen raises on 503, which is exactly
# the non-zero exit the healthcheck wants.
#
# start-period is generous (90s): first boot downloads the bge weights and warms
# the model, and failures inside the start period are not charged to the retries.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=8)" || exit 1

# --host 0.0.0.0, NOT uvicorn's 127.0.0.1 default: a server bound to loopback
# inside a container is reachable only from within that container, so the
# published port would connect to nothing. The classic first-container bug.
# No --reload either: that is a dev-only file-watching supervisor.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
