FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
# Tell Python to use the virtual environment we are about to copy
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy the entire project, including the host's .venv
COPY . .

CMD ["python", "-m", "cknot.main", "--api", "--plugins", "./plugins/agents"]