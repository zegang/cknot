import logging
from .base import COMMAND_REGISTRY
from langgraph.graph.state import CompiledStateGraph
from cknot.graphs.orchestrator import GraphOrchestrator

logger = logging.getLogger(__name__)

@COMMAND_REGISTRY.register("/graph")
async def handle_graph(app: CompiledStateGraph, config, console, args):
    """Visualizes the agentic workflow graph. Subcommands: ascii, mermaid"""
    # Root command for graph visualization. 
    # Subcommands handle the actual logic.
    pass

@handle_graph.subcommand("ascii", is_default=True)
async def handle_graph_ascii(app: CompiledStateGraph, config, console, args):
    """Displays the agentic graph structure in ASCII format for quick terminal debugging."""
    console.print("\n[bold cyan]Graph Architecture (ASCII View):[/bold cyan]")
    
    # We instantiate the orchestrator to ensure we visualize the current build logic
    orchestrator = GraphOrchestrator()
    orchestrator.visualize_ascii()
    
    console.print("")  # Add a newline for spacing
