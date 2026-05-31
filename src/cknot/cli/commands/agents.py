import os
import logging
from rich.rule import Rule
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from cknot.config.config import settings
from cknot.agents.registry import AgentRegistry
from cknot.utils.llm_manager import LLMManager
from .base import COMMAND_REGISTRY
from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)

@COMMAND_REGISTRY.register("/agents")
async def handle_agents(app: CompiledStateGraph, config, console, args):
    """Manages agent configuration and LLM mappings."""
    console.print(Panel(handle_agents.get_usage(), title="Agent Management", border_style="magenta"))

@handle_agents.subcommand("list", is_default=True)
async def handle_agents_list(app: CompiledStateGraph, config, console, args):
    """Lists all registered agents."""
    console.print(Rule("Detailed Agent Dump", style="bold magenta"))
    for name in AgentRegistry.list_agents().keys():
        status = AgentRegistry.get_agent_status(name)
        if not status:
            continue
        content = (
            f"[bold cyan]Name:[/bold cyan] {name}\n"
            f"[bold green]Good at:[/bold green] {', '.join(status['good_at']) if status['good_at'] else 'N/A'}\n"
            f"[bold red]Poor at:[/bold red] {', '.join(status['poor_at']) if status['poor_at'] else 'N/A'}\n"
            f"[bold yellow]Policy:[/bold yellow] {status['llm_select_policy']}\n"
            f"[bold blue]LLMs:[/bold blue] {', '.join(status['llm_services']) if status['llm_services'] else 'None'}\n"
            f"[bold white]Tools:[/bold white] {', '.join(status['tools']) if status['tools'] else 'None'}"
        )
        console.print(Panel(content, title=f"Agent: {name}", border_style="cyan"))
        console.print("")
    console.print(Rule(style="bold magenta"))

@handle_agents.subcommand("llm")
async def handle_agents_llm(app: CompiledStateGraph, config, console, args):
    """Configure LLM mapping for agents."""
    console.print(Panel(handle_agents_llm.get_usage(), title="Agent LLM Configuration", border_style="yellow"))

@handle_agents_llm.subcommand("set")
async def handle_agents_llm_set(app: CompiledStateGraph, config, console, args):
    """Assign an LLM service to a specific agent. Usage: set <agent_id> <llm_id>"""
    if len(args) < 2:
        console.print("[red]Usage: /agents llm set <agent_id> <llm_id>[/red]")
        return
    agent_id, llm_id = args[0], args[1]
    agent = AgentRegistry.get_agent(agent_id)
    
    if not agent:
        console.print(f"[bold red]Error: '{agent_id}' is not a registered agent.[/bold red]")
        return

    mgr = LLMManager()
    svc = mgr.get_llm_service(llm_id)
    agent.add_llm_service(svc)
    console.print(f"[bold green]✔ Assigned {llm_id} to {agent_id}[/bold green]")

@handle_agents_llm.subcommand("remove")
async def handle_agents_llm_remove(app: CompiledStateGraph, config, console, args):
    """Remove a specific LLM mapping for an agent. Usage: remove <agent_id> <llm_id>"""
    if not args:
        console.print("[red]Usage: /agents llm remove <agent_id>[/red]")
        return
    agent_id, llm_id = args[0], args[1]
    agent = AgentRegistry.get_agent(agent_id)
    mgr = LLMManager()
    svc = mgr.get_llm_service(llm_id)
    agent.remove_llm_service(svc)
    console.print(f"[bold green]✔ Removed specific LLM mapping for {agent_id}.[/bold green]")

@handle_agents.subcommand("unregister")
async def handle_agents_unregister(app: CompiledStateGraph, config, console, args):
    """Unregisters an agent from the system. Usage: unregister <agent_id>"""
    if not args:
        console.print("[red]Usage: /agents unregister <agent_id>[/red]")
        return
    agent_id = args[0]
    AgentRegistry.unregister_agent(agent_id)
    console.print(f"[bold green]✔ Agent '{agent_id}' unregistered from the registry.[/bold green]")

@handle_agents.subcommand("info")
async def handle_agents_info(app: CompiledStateGraph, config, console, args):
    """Shows detailed info for a specific agent. Usage: info <agent_id>"""
    if not args:
        console.print("[red]Usage: /agents info <agent_id>[/red]")
        return
    agent_id = args[0]
    status = AgentRegistry.get_agent_status(agent_id)
    if not status:
        console.print(f"[bold red]Error: Agent '{agent_id}' not found.[/bold red]")
        return

    console.print(Panel(
        f"[bold cyan]Name:[/bold cyan] {status['name']}\n"
        f"[bold green]Good at:[/bold green] {', '.join(status['good_at']) if status['good_at'] else 'N/A'}\n"
        f"[bold red]Poor at:[/bold red] {', '.join(status['poor_at']) if status['poor_at'] else 'N/A'}\n"
        f"[bold yellow]Policy:[/bold yellow] {status['llm_select_policy']}\n"
        f"[bold blue]LLMs:[/bold blue] {', '.join(status['llm_services']) if status['llm_services'] else 'None'}\n"
        f"[bold white]Tools:[/bold white] {', '.join(status['tools']) if status['tools'] else 'None'}",
        title=f"Agent: {agent_id}",
        border_style="magenta"
    ))

@handle_agents.subcommand("status")
async def handle_agents_status(app: CompiledStateGraph, config, console, args):
    """
    Shows a summary status of all registered agents.
    """
    agents = AgentRegistry.list_agents()
    if not agents:
        console.print("[yellow]No agents registered in the system.[/yellow]")
        return

    table = Table(title="Agent Registry Summary", border_style="magenta", header_style="bold magenta")
    table.add_column("Agent ID", style="cyan")
    table.add_column("Class Implementation")
    table.add_column("Capabilities", ratio=1)

    for agent in agents:
        info = AgentRegistry.get_agent_status(agent)
        if info:
            caps = ", ".join(info.get("good_at", [])) or "General"
            table.add_row(agent, info.get("class", "Unknown"), caps)

    console.print(table)
