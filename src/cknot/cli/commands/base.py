"""Command registry infrastructure."""
from typing import Callable, Optional, Dict, Any
import inspect

class CommandNode:
    def __init__(self, name: str, func: Optional[Callable] = None, help_text: Optional[str] = None, default_subcommand: Optional[str] = None):
        self.name = name
        self.func = func
        self.help_text = help_text or (inspect.getdoc(func) if func else "") or ""
        self.subcommands: Dict[str, 'CommandNode'] = {}
        self.default_subcommand = default_subcommand

    def subcommand(self, name: str, help_text: Optional[str] = None, is_default: bool = False):
        """Decorator to register a subcommand under this node."""
        def decorator(func):
            node = CommandNode(name, func, help_text)
            self.subcommands[name] = node
            if is_default:
                self.default_subcommand = name
            return node
        return decorator

    def get_usage(self) -> str:
        """Generates a help string for this command and its subcommands."""
        usage = f"[bold cyan]{self.name}[/bold cyan]: {self.help_text.strip()}\n"
        if self.subcommands:
            usage += "\n[bold]Subcommands:[/bold]\n"
            for sub in self.subcommands.values():
                default_mark = " [dim](default)[/dim]" if sub.name == self.default_subcommand else ""
                # Use only the first line of help_text for the subcommand list summary
                summary = sub.help_text.strip().split('\n')[0]
                usage += f"  [magenta]{sub.name:<12}[/magenta] - {summary}{default_mark}\n"
        return usage

class RootRegistry:
    def __init__(self):
        self.commands: Dict[str, CommandNode] = {}

    def register(self, name: str, help_text: Optional[str] = None, default: Optional[str] = None):
        """Top-level decorator to register a command."""
        def decorator(func):
            node = CommandNode(name, func, help_text, default_subcommand=default)
            self.commands[name] = node
            return node
        return decorator

COMMAND_REGISTRY = RootRegistry()