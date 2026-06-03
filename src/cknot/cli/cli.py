import asyncio
import sys
import time
import re
import logging
from typing import Optional, List, Any, Dict
from cknot.utils.logging_config import user_id_ctx
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.key_binding import KeyBindings
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.state import CompiledStateGraph
from rich.console import Console, Group
from rich.text import Text
from rich.markdown import Markdown
from rich.table import Table
from rich.rule import Rule
from rich.panel import Panel
from rich.live import Live
from rich.progress_bar import ProgressBar
from rich.spinner import Spinner
from rich.columns import Columns
from .utils import clear_lines
from rich.prompt import Confirm
from .commands import COMMAND_REGISTRY

logger = logging.getLogger(__name__)
# Force terminal capabilities to prevent ANSI mangling through the patch_stdout proxy
console = Console(force_terminal=True, color_system="auto", legacy_windows=False)

CKNOT_LOGO = r"""
[bold magenta]   ____ _  __ _   _  ___ _____ [/bold magenta]
[bold magenta]  / ___| |/ /| \ | |/ _ \_   _|[/bold magenta]
[bold magenta] | |   | ' / |  \| | | | || |  [/bold magenta]
[bold magenta] | |___| . \ | |\  | |_| || |  [/bold magenta]
[bold magenta]  \____|_|\_\|_| \_|\___/ |_|  [/bold magenta]
version 0.0.1-alpha
"""

# Global state for task management
_active_task: Optional[asyncio.Task] = None
_confirmation_event = asyncio.Event()
_confirmation_result: Optional[str] = None

# Shared state for progress tracking
_current_status: str = ""
_progress_total: Optional[int] = None
_progress_completed: int = 0
_current_percentage: float = 0.0
_agent_progress_reports: Dict[str, Dict[str, Any]] = {}
_current_agent: str = "cknot"
_current_node: str = ""
_current_namespace: List[str] = []

def get_progress_renderable(agent_output: str = ""):
    """Generates the animated progress text with simulated interactive tabs for the bottom panel."""
    # Dynamically determine pipeline stages and the active stage from agent progress reports
    pipeline_stages = []
    active_stage = None
    for info in _agent_progress_reports.values():
        step = info.get("step")
        if step:
            if step not in pipeline_stages:
                pipeline_stages.append(step)
            if info.get("status") != "done":
                active_stage = step
    
    if not active_stage and pipeline_stages:
        active_stage = pipeline_stages[-1]

    # Calculate overall percentage
    if _progress_total and _progress_total > 0:
        overall_pct = (_progress_completed / _progress_total) * 100
    else:
        overall_pct = _current_percentage

    tab_elements = []
    for s in pipeline_stages:
        style = "reverse bold magenta" if s == active_stage else "dim"
        tab_elements.append(f"[{style}] {s} [/{style}]")

    # Aggregate tokens and calculate cost estimate ($0.02 per 1k tokens)
    total_tokens = sum(info.get("total_tokens", 0) for info in _agent_progress_reports.values())
    cost_estimate = (total_tokens / 1000) * 0.02
    token_info = f"[dim]Tokens:[/dim] [bold yellow]{total_tokens}[/bold yellow] "
    cost_info = f"[dim]Est. Cost:[/dim] [bold green]${cost_estimate:.4f}[/bold green]"

    progress_str = f" [{_progress_completed}/{_progress_total}]" if _progress_total else ""
    tab_line = " ".join(tab_elements)

    # Build an ASCII tree representation of the current execution path
    hierarchy = ["cknot"] + list(_current_namespace)
    if _current_node and _current_node != hierarchy[-1]:
        hierarchy.append(_current_node)

    tree_lines = []
    for i, step in enumerate(hierarchy):
        indent = "  " * i
        branch = "└── " if i > 0 else ""
        if i == len(hierarchy) - 1:
            status_info = f" [dim]({_current_status})[/dim]" if _current_status and _current_status != "Initializing..." else ""
            tree_lines.append(f"{indent}{branch}[bold magenta]{step}[/bold magenta]{status_info}{progress_str}")
        else:
            tree_lines.append(f"{indent}{branch}[bold cyan]{step}[/bold cyan]")
    detail_text = "\n".join(tree_lines)

    # 2. Process content
    display_content = Markdown(agent_output) if agent_output.strip() else Text("CKnot is processing...", style="dim")

    # Create the progress bar
    progress_bar = ProgressBar(total=100, completed=min(100, overall_pct), width=None)

    return Group(
        Rule(style="dim cyan"),
        Panel(display_content, title="[bold cyan]Agent Output[/bold cyan]", border_style="dim cyan", padding=(1, 2)),
        Columns([
            Text.from_markup(f"{tab_line} "), 
            Spinner("dots", style="bold magenta"), 
            Text.from_markup(f"  {token_info} {cost_info}")
        ], expand=False),
        progress_bar,
        Text.from_markup(detail_text),
        Rule(style="dim magenta")
    )

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
                for cmd_name, node in COMMAND_REGISTRY.commands.items():
                    if cmd_name.startswith(last_part):
                        yield Completion(cmd_name, start_position=-len(last_part), display_meta=node.help_text.split('\n')[0])
            else:
                parent = find_node(prefix)
                if parent:
                    for sub_name, node in parent.subcommands.items():
                        if sub_name.startswith(last_part):
                            yield Completion(sub_name, start_position=-len(last_part), display_meta=node.help_text.split('\n')[0])
        elif has_space:
            target = find_node(parts)
            if target:
                for sub_name, node in target.subcommands.items():
                    yield Completion(sub_name, start_position=0, display_meta=node.help_text.split('\n')[0])

async def async_confirm(prompt: str, default: bool = True) -> bool:
    """Non-blocking confirmation prompt using Rich inside an async context."""
    # We use to_thread to keep the event loop running for background agents
    # while waiting for the user to answer the confirmation.
    return await asyncio.to_thread(Confirm.ask, prompt, default=default, console=console)

async def dispatch_command(app: CompiledStateGraph, config, user_input: str):
    """Parses input and traverses the command tree to execute the right handler."""
    parts = user_input.split()
    if not parts:
        return

    cmd_name = parts[0].lower()
    if cmd_name not in COMMAND_REGISTRY.commands:
        print(f"Unknown command: {cmd_name}. Type /help for assistance.")
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
        print(f"\n--- Help: {current_node.name} ---\n{current_node.get_usage()}\n")
        return

    if current_node.func:
        await current_node.func(app, config, console, args)
    else:
        print(f"\n--- Usage: {current_node.name} ---\n{current_node.get_usage()}\n")

async def _run_interactive_turn(user_input: str, session_id: str, config: dict, app, live: Optional[Live] = None):
    """Handles a single turn of the interactive CLI, including streaming, interrupts, and UI updates."""
    global _current_status, _progress_total, _progress_completed, _agent_progress_reports, _current_agent, _current_node, _current_namespace
    
    # Initialize/Reset status for the new turn
    _current_status = "Initializing..."
    _progress_total = None
    _progress_completed = 0
    _current_percentage = 0.0
    _current_agent = "cknot"
    _current_node = "start"
    _current_namespace = []

    # Update config with immutable context
    config["configurable"].update({
        "session_id": session_id,
        "user_id": config["configurable"].get("user_id", "cli_user"),
        "current_task": "interactive_session"
    })

    current_input = {
        "messages": [HumanMessage(content=user_input)]
    }

    agent_response_buffer = ""

    while True:
        step_start_time = time.perf_counter()
        last_node = None
        try:
            async for namespace, chunk in app.astream(current_input, config, subgraphs=True):
                # Update global agent tracking from namespace
                _current_namespace = list(namespace)
                _current_agent = namespace[-1] if namespace else "cknot"
                
                for node, state in chunk.items():
                    _current_node = node
                    if node != last_node:
                        logger.info(f"Node Transition: {node}")
                        now = time.perf_counter()
                        duration = now - step_start_time

                        # 1. Update legacy tracking
                        _current_status = f"Active node: {node}..."

                        # 2. Update new structured tracking
                        reports = state.get("progress_report") if isinstance(state, dict) else None
                        if reports and isinstance(reports, dict):
                            _agent_progress_reports.update(reports)
                            
                            # Extract status and total from reports if available
                            for info in reports.values():
                                if isinstance(info, dict) and "description" in info:
                                    _current_status = info["description"]
                                if "total" in info:
                                    _progress_total = info["total"]
                                    _progress_completed = 0
                                if "percentage" in info:
                                    _current_percentage = info["percentage"]
                        
                        step_start_time = now
                        last_node = node
                    
                    # Handle progress increments (Legacy and Structured)
                    has_increment = False
                    if isinstance(state, dict):
                        reports = state.get("progress_report")
                        if not has_increment and reports:
                            has_increment = any(info.get("current", 0) > 0 for info in reports.values() if isinstance(info, dict))
                            
                        if has_increment:
                            _progress_completed += 1

                    if isinstance(state, dict) and "messages" in state and state["messages"]:
                        msg = state["messages"][-1]
                        if isinstance(msg, AIMessage) and msg.content:
                            agent_response_buffer += msg.content
                
                if live:
                    live.update(get_progress_renderable(agent_response_buffer))

            if agent_response_buffer:
                console.print(Markdown(agent_response_buffer))

        except asyncio.CancelledError:
            return
        except Exception as e:
            console.print(f"\n[bold red]✘ Execution Error:[/bold red] {e}")
            logger.error(f"An error occurred during the agent turn: {e}", exc_info=True)
            return

        snapshot = await app.aget_state(config)
        if snapshot.next:
            # snapshot.next is a tuple representing the path to the next node(s)
            next_path = snapshot.next
            # The node we are interrupting at is the last one in the path if it's a subgraph
            next_node = next_path[-1]
            execution_info = f"[bold yellow]{next_node}[/bold yellow]"

            # Find the state that contains the tool calls (might be nested in subgraphs)
            current_snapshot = snapshot
            while current_snapshot.tasks and current_snapshot.tasks[0].state:
                current_snapshot = current_snapshot.tasks[0].state

            # If we are about to enter a tools node, extract the tool calls from the state
            if "tools" in next_node:
                messages = current_snapshot.values.get("messages", [])
                last_msg = messages[-1] if messages else None
                if last_msg and hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    tool_details = []
                    for tc in last_msg.tool_calls:
                        # Format arguments as a string: key='value', key2=123
                        args_str = ", ".join([f"{k}={repr(v)}" for k, v in tc.get('args', {}).items()])
                        tool_details.append(f"\n  * [cyan]{tc['name']}[/cyan]({args_str})")
                    execution_info = f"tools: {''.join(tool_details)}"

            # Handle ArticleWriter sub-graph saver interrupt
            elif next_node == "article_writer" and "saver" in next_path:
                path = snapshot.values.get("output_file_path")
                draft = snapshot.values.get("draft")
                mode = "Append" if snapshot.values.get("append_file", False) else "Overwrite"
                if path:
                    execution_info = f"article_writer:saver\n  * File Path: {path}\n  * Mode: {mode}"

            console.print(f"\nACTION REQUIRED: The agent is requesting to execute: {execution_info}")
            
            _current_status = "Awaiting authorization..."
            
            # Stop Live display temporarily so the confirmation prompt doesn't conflict with background refreshes
            if live and live.is_started:
                live.stop()

            if await async_confirm(f"Allow [bold yellow]{next_node}[/bold yellow] to proceed?", default=True):
                if live and not live.is_started:
                    live.start()
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
    console.print(f"cknot Interactive CLI (Session: [cyan]{session_id}[/cyan])")
    console.print("Type /exit or /quit to end the session.\n")

    kb = KeyBindings()
    @kb.add('escape')
    def _(event): event.app.exit(exception=KeyboardInterrupt)

    session = PromptSession(completer=CKnotCompleter(), complete_while_typing=True, key_bindings=kb)

    global _active_task, _confirmation_result
    
    # Reset progress tracking at the start of the CLI loop
    _agent_progress_reports.clear()
    _current_percentage = 0.0

    while True:
        try:
            console.print(Rule(style="dim magenta"))
            with patch_stdout():
                user_input = await session.prompt_async(HTML('<ansigreen><b>&gt; </b></ansigreen>'))
                user_input = (user_input or "").strip()
        except KeyboardInterrupt:
            break
        except EOFError:
            break
        else:
            if user_input:
                # Re-render input with top and bottom lines for clean isolation.
                # Doing this inside patch_stdout prevents background tasks from injecting lines mid-clear.
                clear_lines(2)
                input_bar = Table.grid(expand=True)
                input_bar.add_row(f" [bold green]> [/bold green]{user_input}", style="on black")
                console.print(input_bar)

        if user_input.lower() in ["/exit", "/quit"]:
            print("Goodbye from CKnot!")
            break
        
        if not user_input:
            continue

        # Handle Control Commands while task is running or idle
        cmd = user_input.lower()
        if cmd == "/abort":
            if _active_task and not _active_task.done():
                _active_task.cancel()
                print("Task aborted by user.")
            else:
                print("No active task to abort.")
            continue
        elif cmd in ["/yes", "/no"]:
            if _active_task and not _active_task.done():
                _confirmation_result = "yes" if cmd == "/yes" else "no"
                _confirmation_event.set()
            else:
                print("No pending authorization request.")
            continue

        # Dispatch other Slash Commands
        if user_input.startswith("/") and not user_input.lower() in ["/yes", "/no", "/abort"]:
            await dispatch_command(app, config, user_input)
            continue

        # Handle New Tasks
        if _active_task and not _active_task.done():
            print("! An agent task is already running.")
            print("Use /abort to stop it, or /yes /no if it's awaiting approval.")
            continue

        # Start new turn in the background
        # Using refresh_per_second enables the internal animation thread.
        # Using 4Hz is enough for spinners and reduces flicker with patch_stdout.
        live = Live(get_progress_renderable(""), console=console, refresh_per_second=4, transient=True)
        live.start()

        def _task_done_callback(fut):
            global _active_task
            _active_task = None
            if live and live.is_started:
                live.stop()
            if not fut.cancelled() and fut.exception():
                logger.error(f"Task failed: {fut.exception()}")

        async def _run_with_context():
            token = user_id_ctx.set("cli_user")
            try:
                await _run_interactive_turn(user_input, session_id, config, app, live=live)
            finally:
                if live and live.is_started:
                    live.stop()
                user_id_ctx.reset(token)

        _active_task = asyncio.create_task(_run_with_context())
        _active_task.add_done_callback(_task_done_callback)
        await _active_task