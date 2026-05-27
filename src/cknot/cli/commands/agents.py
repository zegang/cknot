import os
import logging
from langchain_core.messages import AIMessage
from rich.rule import Rule
from rich.panel import Panel
from rich.markdown import Markdown
from cknot.config.config import settings
from cknot.utils.llm_manager import LLMManager
from .base import COMMAND_REGISTRY
from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)

# Sample pricing per 1M tokens ($)
MODEL_PRICING = {
    "gpt-4o": {"input": 5.0, "output": 15.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6},
    "default": {"input": 0.0, "output": 0.0}
}

@COMMAND_REGISTRY.register("/agents")
async def handle_agents(app: CompiledStateGraph, config, console, args):
    """Manages agent configuration and LLM mappings."""
    console.print(Panel(handle_agents.get_usage(), title="Agent Management", border_style="magenta"))

@handle_agents.subcommand("list", is_default=True)
async def handle_agents_list(app: CompiledStateGraph, config, console, args):
    """Lists all active agents in the graph."""
    nodes = list(app.nodes.keys())
    agent_llms = config["configurable"].get("agent_llms", {})

    console.print(Rule("Active Agents in Graph", style="cyan"))
    for node in nodes:
        if node not in ["__start__", "__end__"]:
            current_llm = agent_llms.get(node, f"{settings.DEFAULT_LLM_SERVICE} [dim](default)[/dim]")
            description = ""
            if node == "cknot": description = "→ [italic]Boss Orchestrator[/italic]"
            elif node == "log_parser": description = "→ [italic]Log Analysis Specialist[/italic]"
            elif node == "code_fixer": description = "→ [italic]Remediation specialist[/italic]"
            elif node == "deep_search": description = "→ [italic]Deep Research & Analysis[/italic]"
            elif node == "tools": description = "→ [italic]Tool Execution Engine[/italic]"
            console.print(f"- [bold magenta]{node: <12}[/bold magenta] [cyan]LLM:[/cyan] {current_llm: <20} {description}")
    console.print(Rule(style="cyan"))

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
    agents = [n for n in app.nodes.keys() if n not in ["__start__", "__end__", "tools"]]
    if agent_id not in agents:
        console.print(f"[bold red]Error: '{agent_id}' is not a valid agent node.[/bold red]")
        return

    from cknot.utils.redis_client import get_redis_client
    mgr = LLMManager(get_redis_client())
    svc = mgr.get_llm_service(llm_id)
    if svc and svc.is_enabled:
        if "agent_llms" not in config["configurable"]:
            config["configurable"]["agent_llms"] = {}
        config["configurable"]["agent_llms"][agent_id] = llm_id
        console.print(f"[bold green]✔ Assigned {llm_id} to {agent_id}[/bold green]")
    else:
        reason = "not found" if not svc else "disabled"
        console.print(f"[bold red]Error: LLM service '{llm_id}' is {reason}.[/bold red]")

@handle_agents_llm.subcommand("remove")
async def handle_agents_llm_remove(app: CompiledStateGraph, config, console, args):
    """Remove a specific LLM mapping for an agent. Usage: remove <agent_id>"""
    if not args:
        console.print("[red]Usage: /agents llm remove <agent_id>[/red]")
        return
    agent_id = args[0]
    agent_llms = config["configurable"].get("agent_llms", {})
    if agent_id in agent_llms:
        del agent_llms[agent_id]
        console.print(f"[bold green]✔ Removed specific LLM mapping for {agent_id}.[/bold green]")
    else:
        console.print(f"[yellow]No specific LLM mapping found for {agent_id}.[/yellow]")

@handle_agents.subcommand("status")
async def handle_agents_status(app: CompiledStateGraph, config, console, args):
    """
    Shows token usage and estimated cost for each agent.
    Analyzes current session messages to calculate totals.
    """
    state = await app.aget_state(config)
    messages = state.values.get("messages", [])
    usage_stats = {}

    for msg in messages:
        if isinstance(msg, AIMessage) and hasattr(msg, "usage_metadata") and msg.usage_metadata:
            agent_name = getattr(msg, "name", "unknown")
            model_name = msg.response_metadata.get("model_name", "unknown")
            
            if agent_name not in usage_stats:
                usage_stats[agent_name] = {"input": 0, "output": 0, "model": model_name}
            
            usage_stats[agent_name]["input"] += msg.usage_metadata.get("input_tokens", 0)
            usage_stats[agent_name]["output"] += msg.usage_metadata.get("output_tokens", 0)

    if not usage_stats:
        console.print("[yellow]No token usage data available for this session yet.[/yellow]")
        return

    console.print(Rule("Usage & Cost Status", style="magenta"))
    total_cost = 0.0
    for agent, data in usage_stats.items():
        pricing = MODEL_PRICING.get(data["model"], MODEL_PRICING["default"])
        cost = (data["input"] / 1_000_000 * pricing["input"]) + (data["output"] / 1_000_000 * pricing["output"])
        total_cost += cost
        console.print(
            f"- [bold magenta]{agent: <12}[/bold magenta] [dim]({data['model']})[/dim]\n"
            f"  Tokens: [cyan]{data['input']} in[/cyan] / [cyan]{data['output']} out[/cyan] "
            f"  Est. Cost: [green]${cost:.4f}[/green]"
        )
    console.print(Rule(style="magenta"))
    console.print(f"[bold]Total Session Cost:[/bold] [bold green]${total_cost:.4f}[/bold green]")


@handle_agents.subcommand("switch")
async def handle_agents_switch(app: CompiledStateGraph, config, console, args):
    """Switches the active LLM service for ALL agents. Usage: /switch"""
    from cknot.utils.redis_client import get_redis_client
    
    mgr = LLMManager(get_redis_client())
    services = mgr.list_llm_services()
    
    if not services:
        console.print("[bold red]No LLM services found in Redis. Please register them via API or llms-file.[/bold red]")
        return

    console.print(Rule("LLM Service Switch", style="magenta"))
    for s in services:
        console.print(f"- [cyan]{s.id}[/cyan] ({s.provider}: {s.model})")
    
    choice = console.input("\n[bold yellow]Enter LLM Service ID to activate for all agents: [/bold yellow]").strip()
    if choice:
        try:
            mgr.get_llm_service_client(choice)  # Validate existence
            agent_llms = config["configurable"].get("agent_llms", {})
            for agent in ["cknot", "log_parser", "code_fixer"]: # Explicitly list agents to update
                agent_llms[agent] = choice
            config["configurable"]["agent_llms"] = agent_llms
            console.print(f"[bold green]✔ Switched all agents to LLM:[/bold green] [cyan]{choice}[/cyan]")
        except Exception as e:
            console.print(f"[bold red]Invalid Service ID: {choice}. Error: {e}[/bold red]")
