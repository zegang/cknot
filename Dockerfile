FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Install dependencies first to leverage Docker layer caching
COPY pyproject.toml ./
RUN uv sync --no-install-project

# Copy the rest of the code and install the project
COPY . .
RUN uv sync

CMD ["uv", "run", "python", "-m", "cknot.main", "--api"]