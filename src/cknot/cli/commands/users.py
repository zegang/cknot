import logging
from rich.table import Table
from rich.rule import Rule
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from cknot.schemas.user import UserRegister, UserUpdate
from cknot.utils.user_manager import UserManager
from cknot.utils.redis_client import get_redis_client
from .base import COMMAND_REGISTRY
from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)

@COMMAND_REGISTRY.register("/users")
async def handle_users(app: CompiledStateGraph, config, console, args):
    """Manages user accounts. Subcommands: list, add, edit <user>, del <user>, promote <user>, passwd <user>"""
    console.print(Panel(handle_users.get_usage(), title="User Management", border_style="yellow"))

@handle_users.subcommand("list", is_default=True)
async def handle_users_list(app: CompiledStateGraph, config, console, args):
    """Lists all registered users."""
    mgr = UserManager(get_redis_client())
    users = mgr.list_users()
    
    table = Table(title="Registered Users", border_style="yellow", header_style="bold yellow")
    table.add_column("Username", style="cyan")
    table.add_column("Email")
    table.add_column("Admin", justify="center")

    for u in users:
        admin_status = "[bold green]YES[/bold green]" if u.is_admin else "No"
        table.add_row(u.username, u.email or "", admin_status)

    console.print(table)

@handle_users.subcommand("add")
async def handle_users_add(app: CompiledStateGraph, config, console, args):
    """Interactively adds a new user."""
    username = Prompt.ask("[bold yellow]Username[/bold yellow]")
    password = Prompt.ask("[bold yellow]Password[/bold yellow]", password=True)
    email = Prompt.ask("[bold yellow]Email (optional)[/bold yellow]", default="")
    is_admin = Confirm.ask("[bold yellow]Grant Admin Privileges?[/bold yellow]", default=False)

    mgr = UserManager(get_redis_client())
    user_reg = UserRegister(
        username=username,
        password=password,
        email=email,
        is_admin=is_admin
    )
    
    if mgr.register_user(user_reg):
        console.print(f"[bold green]✔ User '{username}' registered successfully.[/bold green]")
    else:
        console.print(f"[bold red]✘ Error: Username '{username}' already exists.[/bold red]")

@handle_users.subcommand("del")
async def handle_users_del(app: CompiledStateGraph, config, console, args):
    """Deletes a user by username."""
    if not args:
        console.print("[red]Usage: /users del <username>[/red]")
        return
    
    username = args[0]
    if Confirm.ask(f"[bold red]Are you sure you want to delete user '{username}'?[/bold red]"):
        mgr = UserManager(get_redis_client())
        if mgr.delete_user(username):
            console.print(f"[bold green]✔ User '{username}' deleted.[/bold green]")
        else:
            console.print(f"[bold red]✘ User '{username}' not found.[/bold red]")

@handle_users.subcommand("edit")
async def handle_users_edit(app: CompiledStateGraph, config, console, args):
    """Updates a user's email address."""
    if not args:
        console.print("[red]Usage: /users edit <username>[/red]")
        return
    
    username = args[0]
    email = Prompt.ask(f"[bold yellow]New email for {username}[/bold yellow]")
    
    mgr = UserManager(get_redis_client())
    if mgr.update_user(username, UserUpdate(email=email)):
        console.print(f"[bold green]✔ User '{username}' updated successfully.[/bold green]")
    else:
        console.print(f"[bold red]✘ User '{username}' not found.[/bold red]")

@handle_users.subcommand("promote")
async def handle_users_promote(app: CompiledStateGraph, config, console, args):
    """Promotes a user to Admin status."""
    if not args:
        console.print("[red]Usage: /users promote <username>[/red]")
        return
    
    mgr = UserManager(get_redis_client())
    if mgr.update_user(args[0], UserUpdate(is_admin=True)):
        console.print(f"[bold green]✔ User '{args[0]}' promoted to Admin.[/bold green]")
    else:
        console.print(f"[bold red]✘ User '{args[0]}' not found.[/bold red]")

@handle_users.subcommand("passwd")
async def handle_users_passwd(app: CompiledStateGraph, config, console, args):
    """Updates a user's password."""
    if not args:
        console.print("[red]Usage: /users passwd <username>[/red]")
        return
    
    username = args[0]
    new_password = Prompt.ask(f"[bold yellow]New password for {username}[/bold yellow]", password=True)
    
    mgr = UserManager(get_redis_client())
    if mgr.update_user(username, UserUpdate(password=new_password)):
        console.print(f"[bold green]✔ Password for '{username}' updated successfully.[/bold green]")
    else:
        console.print(f"[bold red]✘ User '{username}' not found.[/bold red]")