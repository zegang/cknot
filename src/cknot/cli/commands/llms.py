import logging
import os
from rich.table import Table
from rich.rule import Rule
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from cknot.utils.llm_manager import LLMManager
from cknot.utils.redis_client import get_redis_client
from cknot.schemas.llm_service import LLMService, LLMProvider
from .base import COMMAND_REGISTRY
from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)

@COMMAND_REGISTRY.register("/llms")
async def handle_llms(app: CompiledStateGraph, config, console, args):
    """Manages LLM services. Subcommands: list, add, rm <id>, test <id>, enable <id>, disable <id>, load <file>"""
    console.print(Panel(handle_llms.get_usage(), title="LLM Service Management", border_style="cyan"))

@handle_llms.subcommand("list", is_default=True)
async def handle_llms_list(app: CompiledStateGraph, config, console, args):
    """Lists all registered LLM services and their health status."""
    mgr = LLMManager(get_redis_client())
    services = mgr.list_llm_services()
    
    table = Table(title="Registered LLM Services", border_style="cyan", header_style="bold cyan")
    table.add_column("ID", style="cyan")
    table.add_column("Provider")
    table.add_column("Model")
    table.add_column("Status", justify="center")
    table.add_column("Valid", justify="center")
    table.add_column("Usage (In/Out)", justify="right")

    for s in services:
        status = "[bold green]ENABLED[/bold green]" if s.is_enabled else "[bold red]DISABLED[/bold red]"
        valid = "[bold green]✔[/bold green]" if s.is_valid else "[bold red]✘[/bold red]"
        usage = f"{s.total_input_tokens} / {s.total_output_tokens}"
        table.add_row(s.id, s.provider.value, s.model, status, valid, usage)

    console.print(table)

@handle_llms.subcommand("add")
async def handle_llms_add(app: CompiledStateGraph, config, console, args):
    """Interactively registers a new LLM service."""
    service_id = Prompt.ask("[bold cyan]Service ID[/bold cyan] (e.g. gpt4-o)")
    name = Prompt.ask("[bold cyan]Display Name[/bold cyan]", default=service_id)
    provider = Prompt.ask("[bold cyan]Provider[/bold cyan]", choices=[p.value for p in LLMProvider])
    model = Prompt.ask("[bold cyan]Model Name[/bold cyan]")
    api_key = Prompt.ask("[bold cyan]API Key[/bold cyan] (optional)", default="")
    base_url = Prompt.ask("[bold cyan]Base URL[/bold cyan] (optional)", default="")
    
    svc = LLMService(
        id=service_id,
        name=name,
        provider=LLMProvider(provider),
        model=model,
        api_key=api_key if api_key else None,
        base_url=base_url if base_url else None
    )
    
    mgr = LLMManager(get_redis_client())
    mgr.register_llm_service(svc)
    console.print(f"[bold green]✔ LLM Service '{service_id}' registered.[/bold green]")

@handle_llms.subcommand("rm")
async def handle_llms_rm(app: CompiledStateGraph, config, console, args):
    """Removes an LLM service by ID."""
    if not args:
        console.print("[red]Usage: /llms rm <service_id>[/red]")
        return
    
    if Confirm.ask(f"[bold red]Delete LLM service '{args[0]}'? This cannot be undone.[/bold red]"):
        mgr = LLMManager(get_redis_client())
        mgr.delete_llm_service(args[0])
        console.print(f"[bold green]✔ Service '{args[0]}' removed.[/bold green]")

@handle_llms.subcommand("test")
async def handle_llms_test(app: CompiledStateGraph, config, console, args):
    """Runs a connectivity check for an LLM service."""
    if not args:
        console.print("[red]Usage: /llms test <service_id>[/red]")
        return
        
    mgr = LLMManager(get_redis_client())
    with console.status(f"[bold cyan]Testing connection to {args[0]}..."):
        is_valid = await mgr.validate_service(args[0])
    
    if is_valid:
        console.print(f"[bold green]✔ Connection to '{args[0]}' successful![/bold green]")
    else:
        console.print(f"[bold red]✘ Connection to '{args[0]}' failed. Check logs for details.[/bold red]")

@handle_llms.subcommand("load")
async def handle_llms_load(app: CompiledStateGraph, config, console, args):
    """Loads LLM services from a JSON or YAML file."""
    path = args[0] if args else "src/cknot/llms.json"
    if not os.path.exists(path):
        console.print(f"[bold red]Error: File {path} not found.[/bold red]")
        return

    mgr = LLMManager(get_redis_client())
    mgr.load_services_from_file(path)
    console.print(f"[bold green]✔ Loaded LLM configurations from {path}[/bold green]")

@handle_llms.subcommand("enable")
async def handle_llms_enable(app: CompiledStateGraph, config, console, args):
    """Enables an LLM service."""
    if not args: return
    mgr = LLMManager(get_redis_client())
    svc = mgr.get_llm_service(args[0])
    if svc:
        svc.is_enabled = True
        mgr.register_llm_service(svc)
        console.print(f"[bold green]✔ Service '{args[0]}' enabled.[/bold green]")