"""
TripleTriple CLI — Full Command Interface

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
    name="tripletriple",
    help="TripleTriple AI Gateway — Local-first AI agent for all your messaging channels.",
    add_completion=False,
)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """
    TripleTriple AI Gateway.
    
    Running without a command starts the gateway (or onboarding if not configured).
    """
    if ctx.invoked_subcommand is None:
        from tripletriple.config.manager import ConfigManager
        from tripletriple.cli.onboard import run_onboarding
        
        cm = ConfigManager()
        
        # If not configured, run onboarding
        if not cm.is_configured():
            should_start = run_onboarding()
            if not should_start:
                return
        
        # Start gateway (pass default args)
        ctx.invoke(gateway, host="127.0.0.1", port=8000, reload=False)


# ─── Models Subcommand ─────────────────────────────────────────────

models_app = typer.Typer(help="Manage LLM models and providers.")
app.add_typer(models_app, name="models")


@models_app.command("list")
def models_list(
    provider: str = typer.Option(None, "--provider", "-p", help="Filter by provider name"),
    all_models: bool = typer.Option(False, "--all", help="Show full catalog"),
):
    """List available models, grouped by provider."""
    from tripletriple.agents.model_catalog import ModelSelector

    selector = ModelSelector()
    output = selector.format_model_list(provider=provider)
    typer.echo(output)


@models_app.command("set")
def models_set(
    model_ref: str = typer.Argument(..., help="Model ref (provider/model or alias)"),
):
    """Set the primary model."""
    from tripletriple.agents.model_catalog import ModelSelector

    selector = ModelSelector()
    model = selector.set_model(model_ref)
    if model:
        typer.echo(f"✅ Primary model set to: {model.full_id} ({model.name})")
    else:
        typer.echo(f'❌ Model "{model_ref}" not found. Use `tripletriple models list` to see available models.')
        raise typer.Exit(1)


@models_app.command("status")
def models_status():
    """Show current model selection, fallbacks, and aliases."""
    from tripletriple.agents.model_catalog import ModelSelector

    selector = ModelSelector()
    typer.echo(selector.format_status())


@models_app.command("fallbacks")
def models_fallbacks(
    action: str = typer.Argument("list", help="list | add | remove | clear"),
    model_ref: str = typer.Argument(None, help="Model ref for add/remove"),
):
    """Manage fallback model chain."""
    from tripletriple.agents.model_catalog import ModelSelector

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
        typer.echo("Usage: tripletriple models fallbacks [list|add|remove|clear] [model_ref]")


@models_app.command("aliases")
def models_aliases(
    action: str = typer.Argument("list", help="list | add | remove"),
    alias: str = typer.Argument(None, help="Alias name"),
    model_ref: str = typer.Argument(None, help="Model ref for add"),
):
    """Manage model aliases."""
    from tripletriple.agents.model_catalog import ModelSelector

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
        typer.echo("Usage: tripletriple models aliases [list|add|remove] [alias] [model_ref]")


# ─── Sessions Subcommand ───────────────────────────────────────────

sessions_app = typer.Typer(help="Manage sessions.")
app.add_typer(sessions_app, name="sessions")


@sessions_app.command("list")
def sessions_list(
    active: int = typer.Option(None, "--active", "-a", help="Only show sessions active in last N minutes"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """List all sessions."""
    from tripletriple.gateway.session import SessionManager, SessionConfig

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
    from tripletriple.gateway.session import SessionManager, SessionConfig

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
    from tripletriple.gateway.session import SessionManager, SessionConfig

    sm = SessionManager(config=SessionConfig())
    typer.echo(sm.format_session_detail(key))


# ─── Workspace Subcommand ──────────────────────────────────────────

workspace_app = typer.Typer(help="Manage workspace identity files (SOUL.md, IDENTITY.md, etc).")
app.add_typer(workspace_app, name="workspace")


@workspace_app.command("init")
def workspace_init(
    path: str = typer.Option("~/.tripletriple/workspace", "--path", "-p", help="Workspace directory"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing files"),
):
    """Initialize workspace with default identity templates."""
    from tripletriple.agents.system_prompt import SystemPromptBuilder

    root = SystemPromptBuilder.init_workspace(workspace_path=path, overwrite=overwrite)
    typer.echo(f"✅ Workspace initialized at: {root}")
    typer.echo("   Edit SOUL.md and IDENTITY.md to customize your agent's personality.")


@workspace_app.command("show")
def workspace_show(
    path: str = typer.Option("~/.tripletriple/workspace", "--path", "-p", help="Workspace directory"),
):
    """Show the fully assembled system prompt."""
    from tripletriple.agents.system_prompt import SystemPromptBuilder, WorkspaceConfig

    builder = SystemPromptBuilder(config=WorkspaceConfig(workspace_path=path))
    prompt = builder.assemble()
    typer.echo(prompt)


@workspace_app.command("status")
def workspace_status(
    path: str = typer.Option("~/.tripletriple/workspace", "--path", "-p", help="Workspace directory"),
):
    """Show workspace file status."""
    from tripletriple.agents.system_prompt import SystemPromptBuilder, WorkspaceConfig

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
        console.print("[red]❌ Gateway is not running. Start it with: `tripletriple gateway`[/red]")
    except Exception as e:
        console.print(f"[red]Error querying gateway: {e}[/red]")


# ─── Gateway ───────────────────────────────────────────────────────

@app.command()
def gateway(
    host: str = typer.Option("127.0.0.1", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to bind to"),
    reload: bool = typer.Option(False, help="Enable auto-reload for development"),
):
    """Start the TripleTriple Gateway server."""
    typer.echo(f"🚀 Starting TripleTriple Gateway at http://{host}:{port}")
    uvicorn.run(
        "tripletriple.gateway.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


# ─── Channel Commands ─────────────────────────────────────────────

@app.command()
def telegram():
    """Start the Telegram channel bot."""
    from tripletriple.channels.plugins.telegram.adapter import TelegramChannel
    from tripletriple.gateway.server import dock

    typer.echo("🤖 Starting Telegram channel...")

    async def _run():
        channel = TelegramChannel(on_message=dock.handle_incoming_message)
        dock.register_channel("telegram", channel)
        await channel.start()

    asyncio.run(_run())


@app.command()
def discord():
    """Start the Discord channel bot."""
    from tripletriple.channels.plugins.discord.adapter import DiscordChannel
    from tripletriple.gateway.server import dock

    typer.echo("🤖 Starting Discord channel...")

    async def _run():
        channel = DiscordChannel(on_message=dock.handle_incoming_message)
        dock.register_channel("discord", channel)
        await channel.start()

    asyncio.run(_run())


@app.command()
def slack():
    """Start the Slack channel bot (Socket Mode)."""
    from tripletriple.channels.plugins.slack.adapter import SlackChannel
    from tripletriple.gateway.server import dock

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
    from tripletriple.version import check_for_updates, perform_update, get_version

    typer.echo(f"🔍 TripleTriple v{get_version()}\n")

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
    """Show TripleTriple version."""
    from tripletriple.version import get_version, get_project_root, _is_git_repo
    import subprocess

    root = get_project_root()
    ver = get_version()
    typer.echo(f"TripleTriple v{ver}")

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
        typer.echo("📊 TripleTriple Status:")
        typer.echo(f"  Gateway: {data.get('gateway', 'unknown')}")
        typer.echo(f"  Model:   {data.get('model', 'unknown')}")
        typer.echo(f"  Sessions: {data.get('sessions', 0)}")
        typer.echo(f"  Channels: {', '.join(data.get('channels', {}).keys()) or 'none'}")
        typer.echo(f"  Tools: {', '.join(data.get('tools', []))}")
    except Exception:
        typer.echo("❌ Gateway is not running. Start it with: tripletriple gateway")


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
    from tripletriple.agents.model_catalog import ModelSelector
    selector = ModelSelector()
    primary = selector.get_primary()
    typer.echo(f"\n  Primary model: {primary.full_id if primary else 'not set'}")

    typer.echo("\nDone.")


@app.command()
def onboard():
    """Interactive onboarding wizard."""
    from tripletriple.cli.onboard import run_onboarding
    run_onboarding()


if __name__ == "__main__":
    app()
