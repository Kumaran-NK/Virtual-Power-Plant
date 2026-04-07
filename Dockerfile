# Dockerfile
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Multi-stage build — keeps the final image lean.
# Stage 1: install Python deps with uv
# Stage 2: copy only the venv + source into a clean runtime image

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# git is required by some VCS-based pip dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

ARG BUILD_MODE=standalone
ARG ENV_NAME=vpp

# Copy the full project into the build context
COPY . /app/env
WORKDIR /app/env

# Ensure uv is available (base image may already have it)
RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

# Install dependencies.
# Prefer the locked versions from uv.lock for reproducibility;
# fall back to a fresh resolve if the lockfile is absent.
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-install-project --no-editable; \
    else \
        uv sync --no-install-project --no-editable; \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-editable; \
    else \
        uv sync --no-editable; \
    fi

# ── Runtime stage ───────────────────────────────────────────────────────────
FROM ${BASE_IMAGE}

# curl is needed for the HEALTHCHECK probe
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy virtualenv from builder
COPY --from=builder /app/env/.venv /app/.venv

# Copy application source
COPY --from=builder /app/env /app/env

# Activate the venv for all subsequent commands
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# Expose the Hugging Face Spaces default port
EXPOSE 7860

# Docker HEALTHCHECK — polls /health every 30 s
# The /health endpoint is defined in server/app.py
HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=15s \
    --retries=3 \
    CMD curl -sf http://localhost:7860/health || exit 1

# Start the FastAPI server
CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 7860"]