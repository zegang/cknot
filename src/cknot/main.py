import os
import sys
import argparse
import threading
import time
import logging
import uvicorn
from cknot.utils.llm_manager import LLMManager
import asyncio
from cknot.utils.redis_client import get_redis_client
from dotenv import load_dotenv
from cknot.utils.logging_config import setup_logging, user_id_ctx
from cknot.utils.cleanup import delete_old_threads
from cknot.graphs.orchestrator import create_graph
from cknot.cli.cli import run_cli_loop
from cknot.agents.registry import AgentRegistry
from cknot.config.config import settings
from rich.console import Console

# Initialize global logging
setup_logging()
console = Console()
logger = logging.getLogger(__name__)

load_dotenv()

def run_periodic_cleanup(interval_seconds: int, max_age_seconds: int):
    """Background task to periodically clean up stale threads."""
    # Redis client is thread-safe; retrieving the singleton instance is safe here
    client = get_redis_client()
    while True:
        try:
            delete_old_threads(client, max_age_seconds)
        except Exception as e:
            logger.warning(f"Background cleanup encountered an error: {e}")
        
        time.sleep(interval_seconds)

async def main(): # Make main async
    parser = argparse.ArgumentParser(description="Run the cknot agent.")
    parser.add_argument("--redis-port", type=int, help="Override the Redis port")
    parser.add_argument("--api", action="store_true", help="Start the FastAPI server")
    parser.add_argument(
        "--checkpointer", 
        choices=["redis", "memory"], 
        default=os.getenv("CHECKPOINTER_TYPE", "redis"),
        help="Choose the checkpointer type (default: redis)"
    )
    parser.add_argument(
        "--cleanup-days",
        type=int,
        help="If provided, delete Redis threads idle for longer than this many days."
    )
    parser.add_argument(
        "--auto-cleanup-interval",
        type=int,
        help="If provided, run cleanup periodically every X minutes."
    )
    parser.add_argument(
        "--llms-file",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "llms.json"),
        help="Path to JSON or YAML file containing LLM services to register on startup"
    )
    parser.add_argument(
        "--plugins",
        type=str,
        help="Directory containing third-party custom agents"
    )
    args = parser.parse_args()

    if args.redis_port:
        os.environ["REDIS_PORT"] = str(args.redis_port)
    os.environ["CHECKPOINTER_TYPE"] = args.checkpointer

    if args.checkpointer == "redis" or args.llms_file:
        try:
            # Use the singleton client for health check, with a short timeout for initial connection
            client = get_redis_client(socket_connect_timeout=2)
            client.ping()
            logger.info(f"Redis connection successful: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
        except Exception as e:
            logger.error(f"Could not connect to Redis at {settings.REDIS_HOST}:{settings.REDIS_PORT}. Error: {e}")
            sys.exit(1)

        # Handle cleanup: one-time or background task
        if args.cleanup_days:
            max_age_seconds = args.cleanup_days * 86400
            
            if args.auto_cleanup_interval:
                # Start background thread for periodic cleanup
                cleanup_thread = threading.Thread(
                    target=run_periodic_cleanup,
                    args=(args.auto_cleanup_interval * 60, max_age_seconds),
                    daemon=True
                )
                cleanup_thread.start()
                logger.info(f"Started automated background cleanup (every {args.auto_cleanup_interval} min).")
            else:
                # Manual one-time cleanup
                delete_old_threads(client, max_age_seconds)

        # Handle LLM registration
        if args.llms_file:
            llm_manager = LLMManager(client)
            llm_manager.load_services_from_file(args.llms_file)

    # Load third-party agents before the graph orchestrator builds the workflow
    if args.plugins:
        logger.info(f"Loading custom agents from {args.plugins}...")
        AgentRegistry.load_custom_agents(args.plugins)

    if args.api:
        logger.info("Starting CKnot FastAPI server...")
        config = uvicorn.Config("cknot.api.api:app", host="0.0.0.0", port=9999, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()
        logger.info("Goodbye from CKnot FastAPI server...")
        return

    cknot_compiled_graph = create_graph()
    
    # session_id is used both in state and as the Redis key (thread_id) for isolation
    session_id = f"cli_session_{int(time.time())}"
    config = {"configurable": {"thread_id": session_id}}

    # Initialize checkpointer if necessary (required for AsyncRedisSaver)
    if hasattr(cknot_compiled_graph.checkpointer, "asetup"):
        await cknot_compiled_graph.checkpointer.asetup()
    
    await run_cli_loop(cknot_compiled_graph, config, session_id)

if __name__ == "__main__":
    try:
        asyncio.run(main()) # Run the async main function
    except KeyboardInterrupt:
        console.print("\n[bold red]See you next time![/bold red]")
    
