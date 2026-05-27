import logging
from rich.table import Table
from rich.rule import Rule
from rich.panel import Panel
from rich.prompt import Confirm
from cknot.tools.tool_manager import ToolManager
from cknot.utils.redis_client import get_redis_client
from .base import COMMAND_REGISTRY
from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)

@COMMAND_REGISTRY.register("/tools")
async def handle_tools(app: CompiledStateGraph, config, console, args):
    """Manages system tools. Subcommands: list, enable <id>, disable <id>, rm <id>, status"""
    console.print(Panel(handle_tools.get_usage(), title="Tool Management", border_style="green"))

@handle_tools.subcommand("list", is_default=True)
async def handle_tools_list(app: CompiledStateGraph, config, console, args):
    """Lists all registered tools and their status."""
    mgr = ToolManager(get_redis_client())
    tools = mgr.list_tool_configs()
    
    if not tools:
        console.print("[yellow]No tools registered in Redis.[/yellow]")
        return

    table = Table(title="Registered Tools", border_style="green", header_style="bold green")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Status", justify="center")
    table.add_column("Usage", justify="right")
    table.add_column("Description", ratio=1)

    for t in tools:
        status = "[bold green]ENABLED[/bold green]" if t.is_enabled else "[bold red]DISABLED[/bold red]"
        table.add_row(t.id, t.name, status, str(t.usage_count), t.description)

    console.print(table)

@handle_tools.subcommand("enable")
async def handle_tools_enable(app: CompiledStateGraph, config, console, args):
    """Enables a tool by ID. Usage: enable <tool_id>"""
    if not args:
        console.print("[red]Usage: /tools enable <tool_id>[/red]")
        return
    
    mgr = ToolManager(get_redis_client())
    cfg = mgr.get_tool_config(args[0])
    if cfg:
        cfg.is_enabled = True
        mgr.save_tool_config(cfg)
        console.print(f"[bold green]✔ Tool '{args[0]}' enabled. (Requires restart to update agent prompt)[/bold green]")
    else:
        console.print(f"[bold red]Error: Tool '{args[0]}' not found.[/bold red]")

@handle_tools.subcommand("disable")
async def handle_tools_disable(app: CompiledStateGraph, config, console, args):
    """Disables a tool by ID. Usage: disable <tool_id>"""
    if not args:
        console.print("[red]Usage: /tools disable <tool_id>[/red]")
        return
    
    mgr = ToolManager(get_redis_client())
    cfg = mgr.get_tool_config(args[0])
    if cfg:
        cfg.is_enabled = False
        mgr.save_tool_config(cfg)
        console.print(f"[bold red]✘ Tool '{args[0]}' disabled. (Requires restart to update agent prompt)[/bold red]")
    else:
        console.print(f"[bold red]Error: Tool '{args[0]}' not found.[/bold red]")

@handle_tools.subcommand("rm")
async def handle_tools_rm(app: CompiledStateGraph, config, console, args):
    """Removes a tool configuration from Redis. Usage: rm <tool_id>"""
    if not args:
        console.print("[red]Usage: /tools rm <tool_id>[/red]")
        return
    
    if Confirm.ask(f"[bold red]Delete tool config for '{args[0]}'? This only removes metadata, not the code.[/bold red]"):
        mgr = ToolManager(get_redis_client())
        mgr.delete_tool_config(args[0])
        console.print(f"[bold green]✔ Tool configuration for '{args[0]}' removed from Redis.[/bold green]")

@handle_tools.subcommand("status")
async def handle_tools_status(app: CompiledStateGraph, config, console, args):
    """Shows tool usage statistics."""
    # Implementation can be expanded if we track more than just usage_count
    await handle_tools_list(app, config, console, args)