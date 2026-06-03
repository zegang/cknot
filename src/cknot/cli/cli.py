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

def get_progress_renderable(agent_output: str = ""):
    """Generates the animated progress text with simulated interactive tabs for the bottom panel."""
    frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    frame = frames[int(time.time() * 10) % len(frames)]
    
    stages = ["TRIAGE", "PLANNING", "RESEARCH", "WRITING", "SAVING"]
    status_lower = _current_status.lower()
    tab_elements = []
    
    for s in stages:
        is_active = False
        if s == "TRIAGE" and ("cknot" in status_lower or "initializing" in status_lower): is_active = True
        elif s == "PLANNING" and "plann" in status_lower: is_active = True
        elif s == "RESEARCH" and ("search" in status_lower or "research" in status_lower): is_active = True
        elif s == "WRITING" and ("writer" in status_lower or "draft" in status_lower or "refin" in status_lower or "edit" in status_lower): is_active = True
        elif s == "SAVING" and "sav" in status_lower: is_active = True
        
        style = "reverse bold magenta" if is_active else "dim"
        tab_elements.append(f"[{style}] {s} [/{style}]")

    progress_str = f" [{_progress_completed}/{_progress_total}]" if _progress_total else ""
    tab_line = " ".join(tab_elements)

    # Create the accumulating response display
    response_display = Markdown(agent_output) if agent_output.strip() else Text("Agent is processing...", style="dim")

    return Group(
        Rule(style="dim cyan"),
        Panel(response_display, title="[bold cyan]Agent Output[/bold cyan]", border_style="dim cyan", padding=(1, 2)),
        Text.from_markup(f"{tab_line}  {frame} [bold magenta]{_current_status}[/bold magenta]{progress_str}"),
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
    global _current_status, _progress_total, _progress_completed
    
    # Initialize/Reset status for the new turn
    _current_status = "Initializing..."
    _progress_total = None
    _progress_completed = 0

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
            if live:
                live.update(get_progress_renderable(agent_response_buffer), refresh=True)
                
            async for namespace, chunk in app.astream(current_input, config, subgraphs=True):
                for node, state in chunk.items():
                    logger.info(f"Node Transition: {node}")
                
                    now = time.perf_counter()
                    duration = now - step_start_time

                    # Update progress tracking state
                    base_desc = state.get("current_progress") or f"Active node: {node}..."
                    _current_status = f"{base_desc} ({duration:.2f}s)"
                
                    total = state.get("progress_total")
                    if total is not None:
                        _progress_total = total
                        _progress_completed = 0
                    elif state.get("progress_increment"):
                        _progress_completed += 1
                    else:
                        _progress_total = None
                    
                    if "messages" in state and state["messages"]:
                        msg = state["messages"][-1]
                        if isinstance(msg, AIMessage) and msg.content:
                            # Clean output: remove internal triggers and mangled escape hallucinations
                            display_text = re.sub(r'TRIGGER_[A-Z_]+', '', msg.content)
                            # Also clean mangled ANSI if they appear
                            display_text = re.sub(r'(?:\\x1b|\\033|u001b|\?)\[[0-9;]*[mK]', '', display_text)

                            # Accumulate content for the live panel
                            agent_response_buffer += display_text

                            if node != last_node:
                                last_node = node
                        
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            pass
                        
                        if live:
                            live.update(get_progress_renderable(agent_response_buffer), refresh=True)
                    
                    step_start_time = now

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
            next_node = next_path[0]
            execution_info = f"[bold yellow]{next_node}[/bold yellow]"

            # If we are about to enter the tools node, extract the tool calls from the state
            if next_node == "tools":
                messages = snapshot.values.get("messages", [])
                last_msg = messages[-1] if messages else None
                if last_msg and hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    tool_details = []
                    for tc in last_msg.tool_calls:
                        tool_details.append(f"\n  * {tc['name']}({tc['args']})")
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

    while True:
        try:
            console.print(Rule(style="dim magenta"))
            with patch_stdout():
                user_input = await session.prompt_async(HTML('<ansigreen><b>You &gt; </b></ansigreen>'))
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
        # Set auto_refresh=False to disable the background refresh thread. 
        # We will manually update the display in _run_interactive_turn using refresh=True.
        live = Live(get_progress_renderable(""), console=console, auto_refresh=False, transient=True)
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