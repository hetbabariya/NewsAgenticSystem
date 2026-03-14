FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install deps first (cached unless pyproject.toml/uv.lock change)
COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Copy source after deps (maximizes layer cache hits)
COPY . .

# Install project itself (separate layer)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ─── Runtime ────────────────────────────────────────────────────────────────
FROM python:3.12-slim-bookworm AS runtime

ARG INSTALL_NODE=0
ARG INSTALL_WEASYPRINT_DEPS=0

WORKDIR /app

RUN set -eux; \
    apt-get update -qq; \
    apt-get install -y --no-install-recommends ca-certificates; \
    if [ "$INSTALL_NODE" = "1" ]; then \
      apt-get install -y --no-install-recommends curl; \
      curl -fsSL https://deb.nodesource.com/setup_20.x | bash -; \
      apt-get install -y --no-install-recommends nodejs; \
      apt-get purge -y --auto-remove curl; \
    fi; \
    if [ "$INSTALL_WEASYPRINT_DEPS" = "1" ]; then \
      apt-get install -y --no-install-recommends \
        python3-cffi python3-brotli libcairo2 libgdk-pixbuf-2.0-0 \
        shared-mime-info fontconfig fonts-dejavu-core \
        libpango-1.0-0 libharfbuzz0b libpangoft2-1.0-0 libpangocairo-1.0-0; \
    fi; \
    rm -rf /var/lib/apt/lists/*

# Non-root user for security + smaller attack surface
RUN groupadd --gid 1001 appuser && \
    useradd --uid 1001 --gid appuser --no-create-home appuser

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --chown=appuser:appuser . .

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1

USER appuser

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${WORKERS:-1}"]