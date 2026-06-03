import logging
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from cknot.utils.llm_manager import LLMManager
from cknot.schemas.llm_service import LLMServiceType
from cknot.utils.redis_client import get_redis_client
from .base import COMMAND_REGISTRY
from langgraph.graph.state import CompiledStateGraph
from cknot.agents.registry import AgentRegistry

logger = logging.getLogger(__name__)

@COMMAND_REGISTRY.register("/agents")
async def handle_agents(app: CompiledStateGraph, config, console, args):
    """Manages agents. Subcommands: list, llm, rm"""
    # Root command for agents management.
    pass

@handle_agents.subcommand("list", is_default=True)
async def handle_agents_list(app: CompiledStateGraph, config, console, args):
    """Lists all registered agents and their specialized capabilities."""
    agents = AgentRegistry.list_agents()
    
    if not agents:
        console.print("[yellow]No agents registered in the AgentRegistry.[/yellow]")
        return

    table = Table(title="Agent Registry", border_style="cyan", header_style="bold cyan")
    table.add_column("Agent ID", style="bold cyan")
    table.add_column("Expert At", ratio=1)
    table.add_column("Avoid For", ratio=1)

    for name, agent in agents.items():
        good = ", ".join(agent.expert_in)
        poor = ", ".join(agent.avoid_for)
        table.add_row(name, good, poor)

    console.print(table)

@handle_agents.subcommand("info")
async def handle_agents_info(app: CompiledStateGraph, config, console, args):
    """Shows detailed metadata for a specific agent. Usage: info <agent_id>"""
    agent_id = args[0] if args else None

    if not agent_id:
        agents = AgentRegistry.list_agents()
        if not agents:
            console.print("[yellow]No agents registered in the AgentRegistry.[/yellow]")
            return
            
        choices = list(agents.keys())
        console.print(f"[bold cyan]Registered Agents:[/bold cyan] {', '.join(choices)}")
        agent_id = Prompt.ask("Select an Agent ID", choices=choices)

    status = AgentRegistry.get_agent_status(agent_id)

    if not status:
        console.print(f"[red]Error: Agent '{agent_id}' not found in registry.[/red]")
        return

    info_text = (
        f"[bold]Name:[/bold] {status['name']}\n"
        f"[bold]Class:[/bold] {status['class']}\n"
        f"[bold]Expert at:[/bold] {', '.join(status['expert_in']) or 'None'}\n"
        f"[bold]Avoid for:[/bold] {', '.join(status['avoid_for']) or 'None'}\n"
        f"[bold]Policy:[/bold] {status['llm_select_policy']}\n"
        f"[bold]LLMs:[/bold] {', '.join(status['llm_services']) or 'None'}\n"
        f"[bold]Tools:[/bold] {', '.join(status['tools']) or 'None'}\n"
        f"\n[bold]System Prompt:[/bold]\n[dim]{status['system_prompt']}[/dim]"
    )

    console.print(Panel(info_text, title=f"Agent: {agent_id}", border_style="cyan"))

@handle_agents.subcommand("rm")
async def handle_agents_rm(app: CompiledStateGraph, config, console, args):
    """Unregisters an agent from the system. Usage: rm <agent_id>"""
    agent_id = args[0] if args else None
    
    if not agent_id:
        agents = AgentRegistry.list_agents()
        if not agents:
            console.print("[yellow]No agents registered in the AgentRegistry.[/yellow]")
            return
            
        choices = list(agents.keys())
        console.print(f"[bold cyan]Registered Agents:[/bold cyan] {', '.join(choices)}")
        agent_id = Prompt.ask("Select an Agent ID to unregister", choices=choices)

    if Confirm.ask(f"[bold red]Unregister agent '{agent_id}' from the system?[/bold red]"):
        AgentRegistry.unregister_agent(agent_id)
        console.print(f"[bold green]✔[/bold green] Agent [cyan]{agent_id}[/cyan] has been unregistered.")

@handle_agents.subcommand("llm")
async def handle_agents_llm(app: CompiledStateGraph, config, console, args):
    """Manage agent LLM assignments. Subcommands: set"""
    pass

@handle_agents_llm.subcommand("set")
async def handle_agents_llm_set(app: CompiledStateGraph, config, console, args):
    """
    Sets the LLM for a specific agent.
    Usage: /agents llm set <agent_id> [service_id]
    """
    agent_id = args[0] if args else None

    if not agent_id:
        agents = AgentRegistry.list_agents()
        if not agents:
            console.print("[yellow]No agents registered in the AgentRegistry.[/yellow]")
            return
            
        choices = list(agents.keys())
        console.print(f"[bold cyan]Registered Agents:[/bold cyan] {', '.join(choices)}")
        agent_id = Prompt.ask("Select an Agent ID", choices=choices)

    agent = AgentRegistry.get_agent(agent_id)

    if not agent:
        console.print(f"[red]Error: Agent '{agent_id}' not found.[/red]")
        return

    # The CLI typically uses the synchronous manager for configuration tasks
    mgr = LLMManager(get_redis_client())

    # If service_id is not provided as an argument, fetch and list available CHAT services
    service_id = args[1] if len(args) > 1 else None
    
    if not service_id:
        # Use the filtered list capability to retrieve only models intended for chat
        chat_services = mgr.list_llm_services(service_type=LLMServiceType.CHAT)
        
        if not chat_services:
            console.print("[yellow]No 'chat' type services registered. Please add one using '/llms add'.[/yellow]")
            return

        table = Table(title=f"Available Chat Models for {agent_id}", border_style="cyan")
        table.add_column("ID", style="bold")
        table.add_column("Name")
        table.add_column("Provider")
        table.add_column("Model")
        
        for s in chat_services:
            table.add_row(s.id, s.name, s.provider.value, s.model_name)
        
        console.print(table)
        service_id = Prompt.ask("Select a Service ID")

    # Validate that the chosen service exists and is suitable for an agent (chat type)
    service = mgr.get_llm_service(service_id)
    if not service or service.service_type != LLMServiceType.CHAT:
        console.print(f"[red]Error: '{service_id}' is not a valid chat service. Agents require models with 'chat' service_type.[/red]")
        return

    # Implementation note: In a full system, you would persist this assignment to Redis/Config
    agent.add_llm_service(service)
    console.print(f"[bold green]✔[/bold green] Assigned chat service [cyan]{service_id}[/cyan] to agent [cyan]{agent_id}[/cyan].")