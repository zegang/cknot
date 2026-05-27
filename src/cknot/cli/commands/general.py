import os
import logging
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph.state import CompiledStateGraph
from rich.rule import Rule
from rich.panel import Panel
from .base import COMMAND_REGISTRY

logger = logging.getLogger(__name__)

@COMMAND_REGISTRY.register("/clear")
async def handle_clear(app: CompiledStateGraph, config, console, args):
    """Clears the terminal and resets the conversation context in Redis."""
    thread_id = config["configurable"].get("thread_id")
    if thread_id and os.getenv("CHECKPOINTER_TYPE", "redis").lower() == "redis":
        from cknot.utils.redis_client import get_redis_client
        try:
            client = get_redis_client()
            keys = list(client.scan_iter(f"*:{thread_id}*"))
            if keys:
                client.delete(*keys)
        except Exception as e:
            logger.error(f"Failed to clear Redis context for {thread_id}: {e}")
    
    console.clear()
    console.print(Rule(style="bold magenta"))
    console.print("[bold green]✔ Context reset and terminal cleared.[/bold green]")

@COMMAND_REGISTRY.register("/history")
async def handle_history(app: CompiledStateGraph, config, console, args):
    """Displays the message history of the current session."""
    state = await app.aget_state(config)
    messages = state.values.get("messages", [])
    
    if not messages:
        console.print("[yellow]No history found for this session.[/yellow]")
        return

    console.print(Rule("Conversation History", style="cyan"))
    for msg in messages:
        role = "User" if isinstance(msg, HumanMessage) else "cknot"
        color = "green" if role == "User" else "bold magenta"
        console.print(f"[{color}]{role}:[/{color}] {msg.content}")
    console.print(Rule(style="cyan"))

@COMMAND_REGISTRY.register("/help")
async def handle_help(app: CompiledStateGraph, config, console, args):
    """Lists all available slash commands."""
    help_text = "\n".join([
        f"[bold cyan]{name}[/bold cyan] - {node.help_text.strip()}" 
        for name, node in COMMAND_REGISTRY.commands.items()
    ])
    console.print(Panel(help_text, title="Available Commands", border_style="magenta"))

@COMMAND_REGISTRY.register("/login")
async def handle_login(app: CompiledStateGraph, config, console, args):
    """Authenticates a user and sets the session context."""
    from rich.prompt import Prompt
    from cknot.utils.user_manager import UserManager
    from cknot.utils.redis_client import get_redis_client
    from cknot.utils.logging_config import user_id_ctx

    username = Prompt.ask("[bold yellow]Username[/bold yellow]")
    password = Prompt.ask("[bold yellow]Password[/bold yellow]", password=True)

    try:
        mgr = UserManager(get_redis_client())
        user_data = mgr.authenticate(username, password)

        if user_data:
            config.setdefault("configurable", {})["user_id"] = username
            # Update the logging context for the current thread/session
            user_id_ctx.set(username)
            console.print(f"[bold green]✔ Welcome back, {username}! Session authenticated.[/bold green]")
        else:
            console.print("[bold red]✘ Authentication failed: Invalid username or password.[/bold red]")
    except Exception as e:
        logger.error(f"Login error for user {username}: {e}")
        console.print(f"[bold red]✘ An error occurred during login: {e}[/bold red]")

@COMMAND_REGISTRY.register("/exit")
async def handle_exit(app: CompiledStateGraph, config, console, args):
    """Exits the interactive CLI session."""
    pass

@COMMAND_REGISTRY.register("/quit")
async def handle_quit(app: CompiledStateGraph, config, console, args):
    """Exits the interactive CLI session."""
    pass