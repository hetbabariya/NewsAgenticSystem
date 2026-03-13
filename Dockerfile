FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Keep builder lightweight; compiled wheels are not required for this project.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies using uv
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project

# Final stage
FROM python:3.12-slim-bookworm

WORKDIR /app

# Render 512Mi optimization: install only minimal runtime system deps.
# Optional: enable Node.js for MCP (npx) and/or WeasyPrint native deps via build args.
ARG INSTALL_NODE=0
ARG INSTALL_WEASYPRINT_DEPS=0

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates; \
    if [ "$INSTALL_NODE" = "1" ]; then \
      apt-get install -y --no-install-recommends curl; \
      curl -fsSL https://deb.nodesource.com/setup_20.x | bash -; \
      apt-get install -y --no-install-recommends nodejs; \
    fi; \
    if [ "$INSTALL_WEASYPRINT_DEPS" = "1" ]; then \
      apt-get install -y --no-install-recommends \
        python3-cffi \
        python3-brotli \
        libcairo2 \
        libgdk-pixbuf-2.0-0 \
        shared-mime-info \
        fontconfig \
        fonts-dejavu-core \
        libpango-1.0-0 \
        libharfbuzz0b \
        libpangoft2-1.0-0 \
        libpangocairo-1.0-0; \
    fi; \
    rm -rf /var/lib/apt/lists/*

# Copy virtualenv from builder
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

COPY . .

ENV HOST=0.0.0.0

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
