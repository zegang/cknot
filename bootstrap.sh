#!/bin/bash

# Exit on error
set -e

# Determine the compose command
COMPOSE_ENGINE=${COMPOSE_ENGINE:-podman} # Default to docker if not set

if [ "$COMPOSE_ENGINE" == "podman" ]; then
    COMPOSE_CMD="podman compose"
elif [ "$COMPOSE_ENGINE" == "docker" ]; then
    COMPOSE_CMD="docker compose"
else
    echo >&2 "Error: Invalid COMPOSE_ENGINE value '$COMPOSE_ENGINE'. Must be 'docker' or 'podman'."
    exit 1
fi

# Help message
function show_help {
    echo "Usage: ./bootstrap.sh [command]"
    echo ""
    echo "Commands:"
    echo "  init     Initialize local directories (logs, redis_data)"
    echo "  build    Build the containers using $COMPOSE_ENGINE"
    echo "  start    Start the API server in the background"
    echo "  cli      Start an interactive cknot CLI turn inside the container"
    echo "  test     Run tests inside the container (e.g., ./bootstrap.sh test tests/test_article_writer_agent.py)"
    echo "  stop     Stop and remove containers"
    echo "  shell    Enter the app container's bash shell"
    echo "  redmods  List redis modules"
    echo ""
}

case "$1" in
    init)
        echo "Initializing directories..."
        mkdir -p logs redis_data
        echo "Done."
        ;;
    build)
        echo "Building containers with $COMPOSE_ENGINE..."
        $COMPOSE_CMD build
        ;;
    start)
        echo "Starting containers with $COMPOSE_ENGINE..."
        $COMPOSE_CMD up -d
        ;;
    cli)
        echo "Connecting to cknot CLI with $COMPOSE_ENGINE..."
        $COMPOSE_CMD exec -it cknot-app uv run python -m cknot.main --plugins ./plugins/agents
        ;;
    test)
        echo "Running tests with $COMPOSE_ENGINE..."
        # We ensure PYTHONPATH includes 'plugins' so that dynamically loaded agents are testable
        $COMPOSE_CMD exec -it cknot-app env PYTHONPATH=src:plugins uv run pytest "${@:2}"
        ;;
    stop)
        echo "Stopping containers with $COMPOSE_ENGINE..."
        $COMPOSE_CMD down
        ;;
    shell)
        echo "Entering container shell..."
        $COMPOSE_CMD exec cknot-app /bin/bash
        ;;
    redmods)
        echo "Listing redis modules..."
        $COMPOSE_CMD exec redis redis-cli MODULE LIST
        ;;
    *)
        show_help
        ;;
esac