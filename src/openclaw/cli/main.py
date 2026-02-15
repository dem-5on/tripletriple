"""
OpenClaw CLI — Full Command Interface

Provides commands to start the gateway, manage models,
individual channels, and system diagnostics.
"""

import typer
import uvicorn
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

app = typer.Typer(
    name="openclaw",
    help="OpenClaw AI Gateway — Local-first AI agent for all your messaging channels.",
    add_completion=False,
)

# ─── Models Subcommand ─────────────────────────────────────────────

models_app = typer.Typer(help="Manage LLM models and providers.")
app.add_typer(models_app, name="models")


@models_app.command("list")
def models_list(
    provider: str = typer.Option(None, "--provider", "-p", help="Filter by provider name"),
    all_models: bool = typer.Option(False, "--all", help="Show full catalog"),
):
    """List available models, grouped by provider."""
    from openclaw.agents.model_catalog import ModelSelector

    selector = ModelSelector()
    output = selector.format_model_list(provider=provider)
    typer.echo(output)


@models_app.command("set")
def models_set(
    model_ref: str = typer.Argument(..., help="Model ref (provider/model or alias)"),
):
    """Set the primary model."""
    from openclaw.agents.model_catalog import ModelSelector

    selector = ModelSelector()
    model = selector.set_model(model_ref)
    if model:
        typer.echo(f"✅ Primary model set to: {model.full_id} ({model.name})")
    else:
        typer.echo(f'❌ Model "{model_ref}" not found. Use `openclaw models list` to see available models.')
        raise typer.Exit(1)


@models_app.command("status")
def models_status():
    """Show current model selection, fallbacks, and aliases."""
    from openclaw.agents.model_catalog import ModelSelector

    selector = ModelSelector()
    typer.echo(selector.format_status())


@models_app.command("fallbacks")
def models_fallbacks(
    action: str = typer.Argument("list", help="list | add | remove | clear"),
    model_ref: str = typer.Argument(None, help="Model ref for add/remove"),
):
    """Manage fallback model chain."""
    from openclaw.agents.model_catalog import ModelSelector

    selector = ModelSelector()

    if action == "list":
        chain = selector.get_fallback_chain()
        typer.echo("Fallback chain:")
        for i, m in enumerate(chain):
            prefix = "  ◆ Primary" if i == 0 else f"  {i}."
            typer.echo(f"{prefix} {m.full_id} ({m.name})")
    elif action == "add" and model_ref:
        model = selector.add_fallback(model_ref)
        if model:
            typer.echo(f"✅ Added {model.full_id} to fallback chain")
        else:
            typer.echo(f'❌ Model "{model_ref}" not found.')
    elif action == "remove" and model_ref:
        if selector.remove_fallback(model_ref):
            typer.echo(f"✅ Removed from fallback chain")
        else:
            typer.echo(f'❌ Model "{model_ref}" not in fallback chain.')
    elif action == "clear":
        selector.selection.fallbacks.clear()
        typer.echo("✅ Fallback chain cleared")
    else:
        typer.echo("Usage: openclaw models fallbacks [list|add|remove|clear] [model_ref]")


@models_app.command("aliases")
def models_aliases(
    action: str = typer.Argument("list", help="list | add | remove"),
    alias: str = typer.Argument(None, help="Alias name"),
    model_ref: str = typer.Argument(None, help="Model ref for add"),
):
    """Manage model aliases."""
    from openclaw.agents.model_catalog import ModelSelector

    selector = ModelSelector()

    if action == "list":
        typer.echo("Built-in aliases:")
        for p in selector.catalog.values():
            for m in p.models:
                if m.alias:
                    typer.echo(f"  {m.alias} -> {m.full_id}")
        if selector.selection.aliases:
            typer.echo("\nCustom aliases:")
            for a, r in selector.selection.aliases.items():
                typer.echo(f"  {a} -> {r}")
    elif action == "add" and alias and model_ref:
        model = selector.add_alias(alias, model_ref)
        if model:
            typer.echo(f"✅ Alias '{alias}' -> {model.full_id}")
        else:
            typer.echo(f'❌ Model "{model_ref}" not found.')
    elif action == "remove" and alias:
        if alias in selector.selection.aliases:
            del selector.selection.aliases[alias]
            typer.echo(f"✅ Alias '{alias}' removed")
        else:
            typer.echo(f'❌ Alias "{alias}" not found.')
    else:
        typer.echo("Usage: openclaw models aliases [list|add|remove] [alias] [model_ref]")


# ─── Sessions Subcommand ───────────────────────────────────────────

sessions_app = typer.Typer(help="Manage sessions.")
app.add_typer(sessions_app, name="sessions")


@sessions_app.command("list")
def sessions_list(
    active: int = typer.Option(None, "--active", "-a", help="Only show sessions active in last N minutes"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """List all sessions."""
    from openclaw.gateway.session import SessionManager, SessionConfig

    sm = SessionManager(config=SessionConfig())
    entries = sm.list_sessions(active_minutes=active)

    if json_output:
        import json
        data = [
            {
                "key": e.session_key,
                "id": e.session_id,
                "type": e.chat_type.value,
                "channel": e.channel,
                "tokens": e.tokens.total_tokens,
                "updated_at": e.updated_at,
            }
            for e in entries
        ]
        typer.echo(json.dumps(data, indent=2))
    else:
        typer.echo(sm.format_sessions_list(active_minutes=active))


@sessions_app.command("reset")
def sessions_reset(
    key: str = typer.Argument(..., help="Session key to reset"),
):
    """Reset a specific session."""
    from openclaw.gateway.session import SessionManager, SessionConfig

    sm = SessionManager(config=SessionConfig())
    new = sm.reset_session(key)
    if new:
        typer.echo(f"✅ Session reset. New ID: {new.id[:8]}...")
    else:
        typer.echo(f"❌ Session '{key}' not found.")
        raise typer.Exit(1)


@sessions_app.command("inspect")
def sessions_inspect(
    key: str = typer.Argument(..., help="Session key to inspect"),
):
    """Show detailed info for a session."""
    from openclaw.gateway.session import SessionManager, SessionConfig

    sm = SessionManager(config=SessionConfig())
    typer.echo(sm.format_session_detail(key))


# ─── Workspace Subcommand ──────────────────────────────────────────

workspace_app = typer.Typer(help="Manage workspace identity files (SOUL.md, IDENTITY.md, etc).")
app.add_typer(workspace_app, name="workspace")


@workspace_app.command("init")
def workspace_init(
    path: str = typer.Option("~/.openclaw/workspace", "--path", "-p", help="Workspace directory"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing files"),
):
    """Initialize workspace with default identity templates."""
    from openclaw.agents.system_prompt import SystemPromptBuilder

    root = SystemPromptBuilder.init_workspace(workspace_path=path, overwrite=overwrite)
    typer.echo(f"✅ Workspace initialized at: {root}")
    typer.echo("   Edit SOUL.md and IDENTITY.md to customize your agent's personality.")


@workspace_app.command("show")
def workspace_show(
    path: str = typer.Option("~/.openclaw/workspace", "--path", "-p", help="Workspace directory"),
):
    """Show the fully assembled system prompt."""
    from openclaw.agents.system_prompt import SystemPromptBuilder, WorkspaceConfig

    builder = SystemPromptBuilder(config=WorkspaceConfig(workspace_path=path))
    prompt = builder.assemble()
    typer.echo(prompt)


@workspace_app.command("status")
def workspace_status(
    path: str = typer.Option("~/.openclaw/workspace", "--path", "-p", help="Workspace directory"),
):
    """Show workspace file status."""
    from openclaw.agents.system_prompt import SystemPromptBuilder, WorkspaceConfig

    builder = SystemPromptBuilder(config=WorkspaceConfig(workspace_path=path))
    typer.echo(builder.format_workspace_status())


# ─── Skills Subcommand ─────────────────────────────────────────────

skills_app = typer.Typer(help="Inspect agent skills and tools.")
app.add_typer(skills_app, name="skills")


@skills_app.command("list")
def skills_list():
    """List enabled tools on the running gateway."""
    import httpx
    from rich.console import Console
    from rich.table import Table

    console = Console()

    try:
        resp = httpx.get("http://127.0.0.1:8000/status", timeout=2.0)
        data = resp.json()
        tools = data.get("tools", [])
        
        if not tools:
            console.print("[yellow]No tools registered on the gateway.[/yellow]")
            return

        table = Table(title=f"Active Skills ({len(tools)})")
        table.add_column("Tool Name", style="cyan")
        
        for tool in tools:
            table.add_row(tool)
            
        console.print(table)
            
    except httpx.ConnectError:
        console.print("[red]❌ Gateway is not running. Start it with: `openclaw gateway`[/red]")
    except Exception as e:
        console.print(f"[red]Error querying gateway: {e}[/red]")


# ─── Gateway ───────────────────────────────────────────────────────

@app.command()
def gateway(
    host: str = typer.Option("127.0.0.1", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to bind to"),
    reload: bool = typer.Option(False, help="Enable auto-reload for development"),
):
    """Start the OpenClaw Gateway server."""
    typer.echo(f"🚀 Starting OpenClaw Gateway at http://{host}:{port}")
    uvicorn.run(
        "openclaw.gateway.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


# ─── Channel Commands ─────────────────────────────────────────────

@app.command()
def telegram():
    """Start the Telegram channel bot."""
    from openclaw.channels.plugins.telegram.adapter import TelegramChannel
    from openclaw.gateway.server import dock

    typer.echo("🤖 Starting Telegram channel...")

    async def _run():
        channel = TelegramChannel(on_message=dock.handle_incoming_message)
        dock.register_channel("telegram", channel)
        await channel.start()

    asyncio.run(_run())


@app.command()
def discord():
    """Start the Discord channel bot."""
    from openclaw.channels.plugins.discord.adapter import DiscordChannel
    from openclaw.gateway.server import dock

    typer.echo("🤖 Starting Discord channel...")

    async def _run():
        channel = DiscordChannel(on_message=dock.handle_incoming_message)
        dock.register_channel("discord", channel)
        await channel.start()

    asyncio.run(_run())


@app.command()
def slack():
    """Start the Slack channel bot (Socket Mode)."""
    from openclaw.channels.plugins.slack.adapter import SlackChannel
    from openclaw.gateway.server import dock

    typer.echo("🤖 Starting Slack channel...")

    async def _run():
        channel = SlackChannel(on_message=dock.handle_incoming_message)
        dock.register_channel("slack", channel)
        await channel.start()

    asyncio.run(_run())


# ─── Update Command ───────────────────────────────────────────────

@app.command()
def update(
    check: bool = typer.Option(False, "--check", "-c", help="Only check for updates, don't install"),
    force: bool = typer.Option(False, "--force", "-f", help="Force update even if up-to-date"),
):
    """Check for and install updates."""
    from openclaw.version import check_for_updates, perform_update, get_version

    typer.echo(f"🔍 OpenClaw v{get_version()}\n")

    if check:
        result = check_for_updates()
        typer.echo(result["message"])
        if result.get("changelog"):
            typer.echo(f"\n📋 New commits:\n{result['changelog']}")
        return

    # Perform the update
    typer.echo("📦 Checking for updates...")
    check_result = check_for_updates()

    if not check_result["available"] and not force:
        typer.echo(check_result["message"])
        return

    if check_result["available"]:
        typer.echo(check_result["message"])
        if check_result.get("changelog"):
            typer.echo(f"\n📋 Changes:\n{check_result['changelog']}\n")

    typer.echo("⬇️  Downloading and installing...")
    result = perform_update(force=force)

    typer.echo(result["message"])

    if result.get("needs_restart"):
        typer.echo("\n♻️  Restart your gateway and bot to apply changes.")


@app.command()
def version():
    """Show OpenClaw version."""
    from openclaw.version import get_version, get_project_root, _is_git_repo
    import subprocess

    root = get_project_root()
    ver = get_version()
    typer.echo(f"OpenClaw v{ver}")

    if _is_git_repo(root):
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(root), capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            typer.echo(f"Commit: {result.stdout.strip()}")

        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(root), capture_output=True, text=True, timeout=5,
        )
        if branch.returncode == 0:
            typer.echo(f"Branch: {branch.stdout.strip()}")
    else:
        typer.echo("(not a git repo)")


# ─── System Commands ──────────────────────────────────────────────

@app.command()
def status():
    """Show system status."""
    import httpx

    try:
        resp = httpx.get("http://127.0.0.1:8000/status", timeout=5)
        data = resp.json()
        typer.echo("📊 OpenClaw Status:")
        typer.echo(f"  Gateway: {data.get('gateway', 'unknown')}")
        typer.echo(f"  Model:   {data.get('model', 'unknown')}")
        typer.echo(f"  Sessions: {data.get('sessions', 0)}")
        typer.echo(f"  Channels: {', '.join(data.get('channels', {}).keys()) or 'none'}")
        typer.echo(f"  Tools: {', '.join(data.get('tools', []))}")
    except Exception:
        typer.echo("❌ Gateway is not running. Start it with: openclaw gateway")


@app.command()
def doctor():
    """Run diagnostics and health checks."""
    typer.echo("🩺 Running diagnostics...\n")

    checks = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
        "TELEGRAM_BOT_TOKEN": os.getenv("TELEGRAM_BOT_TOKEN"),
        "DISCORD_BOT_TOKEN": os.getenv("DISCORD_BOT_TOKEN"),
        "SLACK_BOT_TOKEN": os.getenv("SLACK_BOT_TOKEN"),
        "WHATSAPP_API_TOKEN": os.getenv("WHATSAPP_API_TOKEN"),
    }

    for key, value in checks.items():
        result = "✅ Set" if value else "❌ Not set"
        typer.echo(f"  {key}: {result}")

    # Quick model check
    from openclaw.agents.model_catalog import ModelSelector
    selector = ModelSelector()
    primary = selector.get_primary()
    typer.echo(f"\n  Primary model: {primary.full_id if primary else 'not set'}")

    typer.echo("\nDone.")


@app.command()
def onboard():
    """Interactive onboarding wizard."""
    from openclaw.agents.model_catalog import ModelSelector

    typer.echo("🧙 OpenClaw Setup Wizard\n")

    # Step 1: Pick a provider
    selector = ModelSelector()
    providers = selector.list_providers()

    typer.echo("Step 1: Choose your LLM provider\n")
    for i, p in enumerate(providers, 1):
        typer.echo(f"  {i}. {p.display_name} ({p.env_key})")

    choice = typer.prompt("\nProvider number", default="1")
    try:
        provider = providers[int(choice) - 1]
    except (IndexError, ValueError):
        provider = providers[0]

    api_key = typer.prompt(f"\nEnter your {provider.display_name} API key", hide_input=True)
    typer.echo(f"  ✅ {provider.display_name} configured\n")

    # Step 2: Pick a model from that provider
    typer.echo(f"Step 2: Choose a model from {provider.display_name}\n")
    for i, m in enumerate(provider.models, 1):
        alias = f" ({m.alias})" if m.alias else ""
        typer.echo(f"  {i}. {m.name}{alias}  —  ctx:{m.context_window:,}")

    model_choice = typer.prompt("\nModel number", default="1")
    try:
        model = provider.models[int(model_choice) - 1]
    except (IndexError, ValueError):
        model = provider.models[0]

    selector.set_model(model.full_id)
    typer.echo(f"  ✅ Model set to: {model.full_id}\n")

    # Step 3: Channels
    typer.echo("Step 3: Configure messaging channels (optional)")
    channels = typer.prompt(
        "Which channels? (comma-separated: telegram,discord,slack,whatsapp)",
        default="",
    )

    typer.echo(f"\n{'─' * 40}")
    typer.echo(f"  ✅ Setup complete!")
    typer.echo(f"  Provider: {provider.display_name}")
    typer.echo(f"  Model:    {model.full_id} ({model.name})")
    typer.echo(f"  Channels: {channels or 'none (WebSocket only)'}")
    typer.echo(f"\n  Start with: openclaw gateway")
    typer.echo(f"  List models: openclaw models list")


if __name__ == "__main__":
    app()
