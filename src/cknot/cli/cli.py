import asyncio
import time
import re
import logging
from typing import Optional
from cknot.utils.logging_config import user_id_ctx
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.state import CompiledStateGraph
from rich.console import Console, Group
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, MofNCompleteColumn
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
_last_output_line: str = ""

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
    """Handles a single turn of the interactive CLI, including streaming, interrupts, and UI updates."""
    global _current_status, _progress_total, _progress_completed, _last_output_line
    
    # Initialize/Reset status for the new turn
    _current_status = "Initializing..."
    _progress_total = None
    _progress_completed = 0
    _last_output_line = ""

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
        response_buffer = "" # Buffer for accumulating LLM responses
        step_start_time = time.perf_counter()
        last_node = None
        streaming_active = False
        # Local console instance to pick up the patched sys.stdout proxy 
        # while the main loop is waiting for prompt input.
        # console = Console(force_terminal=True)
        try:
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
                    
                    step_start_time = now

                    if "messages" in state and state["messages"]:
                        msg = state["messages"][-1]
                        if isinstance(msg, AIMessage) and msg.content:
                            # Clean output: remove internal triggers from the LLM
                            display_text = re.sub(r'TRIGGER_[A-Z_]+', '', msg.content)
                            # Remove mangled ANSI escape sequences often hallucinated by local models
                            # display_text = re.sub(r'(?:\x1b|\\x1b|\?|\\033|u001b)\[[0-9;]*[mK]', '', display_text)
                            
                            # Update the fixed panel's "last output" tracker
                            clean_content = display_text.strip()
                            if clean_content:
                                _last_output_line = clean_content.split('\n')[-1]

                            if node != last_node:
                                if streaming_active:
                                    console.print()
                                    streaming_active = False
                                if node != "cknot":
                                    console.print(Rule(f"Agent {node}", style="bold cyan"))
                                last_node = node
                            
                            if not display_text.strip() and not msg.content.strip():
                                continue

                            # Stream deltas for the Boss (cknot), render Markdown for completed specialist reports
                            if node == "cknot":
                                console.print(display_text, end="")
                                streaming_active = True
                            else:
                                console.print(Markdown(display_text))
                        
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            # Tool calls are logged, but we ensure prompt is restored correctly
                            pass
            if streaming_active:
                console.print()

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
                        tool_details.append(f"\n  • [cyan]{tc['name']}[/cyan]([italic]{tc['args']}[/italic])")
                    execution_info = f"tools: {''.join(tool_details)}"

            # Handle ArticleWriter sub-graph saver interrupt
            elif next_node == "article_writer" and "saver" in next_path:
                path = snapshot.values.get("output_file_path")
                draft = snapshot.values.get("draft")
                append_file = snapshot.values.get("append_file", False)
                mode = "[bold red]Append[/bold red]" if append_file else "[bold green]Overwrite[/bold green]"

                if path:
                    execution_info = f"[bold cyan]article_writer:saver[/bold cyan]"
                    execution_info += f"\n  • [cyan]File Path:[/cyan] [italic]{path}[/italic]"
                    execution_info += f"\n  • [cyan]Mode:[/cyan] {mode}"
                    if draft:
                        execution_info += f"\n  • [cyan]Size:[/cyan] {len(draft.encode('utf-8'))} bytes"
                        execution_info += f"\n  • [cyan]Word Count:[/cyan] {len(draft.split())} words"

            console.print(Panel(
                f"The agent is requesting to execute: {execution_info}",
                title="[bold yellow]Action Required[/bold yellow]",
                border_style="yellow"
            ))
            
            _current_status = "Awaiting authorization..."
            
            # Wait for user to provide /yes or /no via the main loop
            _confirmation_event.clear()
            await _confirmation_event.wait()
            res = _confirmation_result
            
            if res == "yes":
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

    async def toolbar_refresher():
        """Background task to ensure the toolbar spinner animates smoothly."""
        while True:
            if _active_task and not _active_task.done():
                session.app.invalidate()
            await asyncio.sleep(0.1)

    asyncio.create_task(toolbar_refresher())

    def get_toolbar():
        """Renders the current agent status in the CLI toolbar."""
        if not _active_task or _active_task.done():
            return None
        
        # Spinner frames
        frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        frame = frames[int(time.time() * 10) % len(frames)]

        # Build multi-line status panel
        progress_str = f" [{_progress_completed}/{_progress_total}]" if _progress_total else ""

        # Line 1: Activity Spinner & Status
        line1 = f"<b>{frame} {_current_status}{progress_str}</b>"
        # Line 2: The latest output from the agent (Fixed Panel feel)
        line2 = f"<ansigray>   └─ Last: {_last_output_line[:60]}...</ansigray>" if _last_output_line else ""
        
        return HTML(
            f'<ansimagenta>{line1}\n{line2}</ansimagenta>'
        )

    global _active_task, _confirmation_result

    while True:
        try:
            console.print(Rule(style="dim magenta"))
            with patch_stdout():
                user_input = await session.prompt_async(
                    HTML('<ansigreen><b>You &gt; </b></ansigreen>'),
                    bottom_toolbar=get_toolbar
                )
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

        # Handle Control Commands while task is running or idle
        cmd = user_input.lower()
        if cmd == "/abort":
            if _active_task and not _active_task.done():
                _active_task.cancel()
                console.print("[bold red]Task aborted by user.[/bold red]")
            else:
                console.print("[dim]No active task to abort.[/dim]")
            continue
        elif cmd in ["/yes", "/no"]:
            if _active_task and not _active_task.done():
                _confirmation_result = "yes" if cmd == "/yes" else "no"
                _confirmation_event.set()
            else:
                console.print("[dim]No pending authorization request.[/dim]")
            continue

        # Dispatch other Slash Commands
        if user_input.startswith("/") and not user_input.lower() in ["/yes", "/no", "/abort"]:
            await dispatch_command(app, config, user_input)
            continue

        # Handle New Tasks
        if _active_task and not _active_task.done():
            console.print("[bold yellow]⚠ An agent task is already running.[/bold yellow]")
            console.print("Use [bold]/abort[/bold] to stop it, or [bold]/yes[/bold]/[bold]/no[/bold] if it's awaiting approval.")
            continue

        # Start new turn in the background
        def _task_done_callback(fut):
            global _active_task
            _active_task = None
            if not fut.cancelled() and fut.exception():
                logger.error(f"Task failed: {fut.exception()}")

        async def _run_with_context():
            token = user_id_ctx.set("cli_user")
            try:
                await _run_interactive_turn(user_input, session_id, config, app)
            finally:
                user_id_ctx.reset(token)

        _active_task = asyncio.create_task(
            _run_with_context()
        )
        _active_task.add_done_callback(_task_done_callback)