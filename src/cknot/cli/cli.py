import asyncio
import os
import logging
from cknot.utils.logging_config import user_id_ctx
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style
from prompt_toolkit.key_binding import KeyBindings
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.state import CompiledStateGraph
from rich.console import Console
from rich.markdown import Markdown
from rich.rule import Rule
from rich.panel import Panel
from rich.live import Live
from rich.prompt import Confirm
from .commands import COMMAND_REGISTRY

logger = logging.getLogger(__name__)
console = Console()

CKNOT_LOGO = r"""
[bold magenta]   ____ _  __ _   _  ___ _____ [/bold magenta]
[bold magenta]  / ___| |/ /| \ | |/ _ \_   _|[/bold magenta]
[bold magenta] | |   | ' / |  \| | | | || |  [/bold magenta]
[bold magenta] | |___| . \ | |\  | |_| || |  [/bold magenta]
[bold magenta]  \____|_|\_\|_| \_|\___/ |_|  [/bold magenta]

[bold red]           _   _   [/bold red]
[bold red]          / \ / \  [/bold red]
[bold red]        _ \_ V _/ _[/bold red]
[bold red]       / \ / \ / \ [/bold red]
[bold red]       \__/ _ \__/ [/bold red]
[bold red]        _ / \ / \_ [/bold red]
[bold red]       / \_/ \_/ \ [/bold red]
[bold red]       \ /     \ / [/bold red]
[bold red]        |       |  [/bold red]
"""
CKNOT_LOGO = r"""
[bold magenta]   ____ _  __ _   _  ___ _____ [/bold magenta]
[bold magenta]  / ___| |/ /| \ | |/ _ \_   _|[/bold magenta]
[bold magenta] | |   | ' / |  \| | | | || |  [/bold magenta]
[bold magenta] | |___| . \ | |\  | |_| || |  [/bold magenta]
[bold magenta]  \____|_|\_\|_| \_|\___/ |_|  [/bold magenta]
version 0.0.1-alpha
"""

class CKnotCompleter(Completer):
    """Custom completer for hierarchical slash commands."""
    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return

        parts = text.split()
        has_space = text.endswith(" ")

        def find_node(path_parts):
            if not path_parts: return None
            node = COMMAND_REGISTRY.commands.get(path_parts[0].lower())
            if not node: return None
            for p in path_parts[1:]:
                if p.lower() in node.subcommands:
                    node = node.subcommands[p.lower()]
                else:
                    return None
            return node

        if not has_space and parts:
            last_part = parts[-1].lower()
            prefix = parts[:-1]
            
            if not prefix:
                # Root level completions (e.g., /llms, /agents)
                for cmd_name, node in COMMAND_REGISTRY.commands.items():
                    if cmd_name.startswith(last_part):
                        yield Completion(cmd_name, start_position=-len(last_part), display_meta=node.help_text.split('\n')[0])
            else:
                # Subcommand completions
                parent = find_node(prefix)
                if parent:
                    for sub_name, node in parent.subcommands.items():
                        if sub_name.startswith(last_part):
                            yield Completion(sub_name, start_position=-len(last_part), display_meta=node.help_text.split('\n')[0])
        elif has_space:
            # Show all subcommands of the current node after a space
            target = find_node(parts)
            if target:
                for sub_name, node in target.subcommands.items():
                    yield Completion(sub_name, start_position=0, display_meta=node.help_text.split('\n')[0])


async def dispatch_command(app: CompiledStateGraph, config, user_input: str):
    """Parses input and traverses the command tree to execute the right handler."""
    parts = user_input.split()
    if not parts:
        return

    cmd_name = parts[0].lower()
    if cmd_name not in COMMAND_REGISTRY.commands:
        console.print(f"[bold red]Unknown command: {cmd_name}. Type /help for assistance.[/bold red]")
        return

    current_node = COMMAND_REGISTRY.commands[cmd_name]
    args = parts[1:]

    # Traverse subcommands
    while args and args[0].lower() in current_node.subcommands:
        current_node = current_node.subcommands[args[0].lower()]
        args = args[1:]

    # Check for default subcommand if no more args provided
    while not args and current_node.default_subcommand:
        target = current_node.default_subcommand.lower()
        if target in current_node.subcommands:
            current_node = current_node.subcommands[target]
        else:
            break

    # Handle automatic help
    if args and args[0].lower() == "help":
        console.print(Panel(current_node.get_usage(), title=f"Help: {current_node.name}", border_style="cyan"))
        return

    if current_node.func:
        await current_node.func(app, config, console, args)
    else:
        # If node has no func, it's a category; show help for subcommands
        console.print(Panel(current_node.get_usage(), title=f"Usage: {current_node.name}", border_style="yellow"))

async def _run_interactive_turn(user_input: str, session_id: str, config: dict, app):
    """Handles a single turn of the interactive CLI, including streaming and interrupts."""
    # Update config with immutable context
    config["configurable"].update({
        "session_id": session_id,
        "user_id": config["configurable"].get("user_id", "cli_user"),
        "current_task": "interactive_session"
    })

    current_input = {
        "messages": [HumanMessage(content=user_input)]
    }

    # Internal loop to handle potential interrupts within a single user turn
    while True:
        response_buffer = ""
        try:
            with Live(
                Markdown(""),
                console=console,
                screen=False,
                refresh_per_second=4
            ) as live_panel:
                status_ctx = console.status("[bold magenta]cknot is thinking...", spinner="aesthetic", spinner_style="bold magenta")
                with status_ctx:
                    async for event in app.astream(current_input, config):
                        for node, state in event.items():
                            logger.info(f"Node Transition: {node}")
                            if node == "cknot" and "messages" in state:
                                msg = state["messages"][-1]
                                if isinstance(msg, AIMessage) and msg.content:
                                    logger.info(f"LangGraph Node: {node}, Message: {msg}")
                                    response_buffer += msg.content
                                    live_panel.update(Markdown(response_buffer))
                                    response_buffer = ""
                                if msg.tool_calls:
                                    logger.info(f"LangGraph Node: {node}, Tool Calls: {msg.tool_calls}")
                                    if response_buffer:
                                        live_panel.update(Markdown(response_buffer))
                                    response_buffer = ""
                            elif "messages" in state:
                                msg = state["messages"][-1]
                                if isinstance(msg, AIMessage) and msg.content:
                                    logger.info(f"LangGraph Node: {node}, AIMessage: {msg}")
                                    response_buffer += f'Agent {node}: {msg.content}'
                                    # Ensure the live panel is cleared or updated if it was showing something
                                    if live_panel.is_started and response_buffer:
                                        live_panel.update(Markdown(response_buffer))
                                    response_buffer = ""
        except (KeyboardInterrupt, asyncio.CancelledError):
            if Confirm.ask("\n[bold red]⚠ Interrupt detected. Stop current work?[/bold red]", default=True):
                console.print("[yellow]Turn aborted.[/yellow]")
                return
            else:
                # Note: Continuing a stream after an exception is complex; 
                # usually, we treat the interrupt as a desire to stop.
                console.print("[yellow]Resuming... (Current step might be lost)[/yellow]")
        except Exception as e:
            console.print(f"\n[bold red]✘ Execution Error:[/bold red] {e}")
            logger.error(f"An error occurred during the agent turn: {e}", exc_info=True)
            return

        snapshot = await app.aget_state(config)
        if snapshot.next:
            next_node = snapshot.next[0]
            execution_info = f"[bold yellow]{next_node}[/bold yellow]"

            # If we are about to enter the tools node, extract the tool calls from the state
            if next_node == "tools":
                messages = snapshot.values.get("messages", [])
                last_msg = messages[-1] if messages else None
                if last_msg and hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    tool_details = []
                    for tc in last_msg.tool_calls:
                        tool_details.append(f"\n  • [cyan]{tc['name']}[/cyan]([italic]{tc['args']}[/italic])")
                    execution_info = f"tools: {''.join(tool_details)}"

            console.print(Panel(
                f"The agent is requesting to execute: {execution_info}",
                title="[bold yellow]Action Required[/bold yellow]",
                border_style="yellow"
            ))
            confirm = console.input("[bold yellow]Authorize execution? (yes/no): [/bold yellow]").strip().lower()
            
            if confirm == "yes":
                current_input = None  # Passing None resumes from the checkpoint
                continue
            else:
                console.print("\n[bold red]✘[/bold red] Execution denied. Returning to chat loop.")
                break
        else:
            break

async def run_cli_loop(app: CompiledStateGraph, config, session_id: str):
    """Starts the interactive CLI conversation loop."""
    console.print(CKNOT_LOGO)
    console.print(Rule(style="bold magenta"))
    console.print(f"[bold magenta]cknot Interactive CLI[/bold magenta] (Session: [cyan]{session_id}[/cyan])")
    console.print("Type [italic]/exit[/italic] or [italic]/quit[/italic] to end the session.\n")

    # Custom bindings to handle ESC as an interrupt
    kb = KeyBindings()

    @kb.add('escape')
    def _(event):
        """Handle ESC as a KeyboardInterrupt."""
        event.app.exit(exception=KeyboardInterrupt)

    session = PromptSession(completer=CKnotCompleter(), complete_while_typing=True, key_bindings=kb)

    while True:
        try:
            console.print(Rule(style="dim magenta"))
            user_input = await session.prompt_async(HTML('<ansigreen><b>You &gt; </b></ansigreen>'))
            user_input = user_input.strip()
            console.print(Rule(style="dim magenta"))
        except KeyboardInterrupt:
            if Confirm.ask("\n[bold red]Exit CKnot?[/bold red]", default=False):
                break
            continue
        except EOFError:
            break

        if user_input.lower() in ["/exit", "/quit"]:
            console.print("Goodbye from CKnot!")
            break
        
        if not user_input:
            continue

        # Dispatch Slash Commands
        if user_input.startswith("/"):
            await dispatch_command(app, config, user_input)
            continue

        token_ctx = user_id_ctx.set("cli_user")
        try:
            await _run_interactive_turn(user_input, session_id, config, app)
        finally:
            user_id_ctx.reset(token_ctx)