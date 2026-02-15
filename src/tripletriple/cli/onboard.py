"""
Interactive Onboarding Wizard

Guides the user through initial configuration:
1. Select LLM Provider & API Key
2. Select Model (persisted via TRIPLETREBLE_MODEL)
3. Select Channels & Tokens
4. Install Dependencies (Playwright)
5. Launch Gateway
"""

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..config.manager import ConfigManager
from ..version import get_version
from ..agents.model_catalog import ModelSelector

console = Console()
config = ConfigManager()


def install_playwright():
    """Install Playwright browsers."""
    console.print("\n[bold cyan]📦 Installing browser dependencies (Playwright)...[/bold cyan]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task(description="Installing browsers...", total=None)
        
        try:
            # Run playwright install
            subprocess.run(
                [sys.executable, "-m", "playwright", "install"],
                check=True,
                capture_output=True,
            )
            progress.update(task, completed=True)
            console.print("[green]✅ Browsers installed successfully.[/green]")
        except subprocess.CalledProcessError as e:
            console.print(f"[red]❌ Failed to install browsers: {e}[/red]")
            console.print("[yellow]You may need to run 'playwright install' manually.[/yellow]")
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")


def run_onboarding():
    """Run the interactive onboarding wizard."""
    console.clear()
    
    # Banner
    console.print(Panel.fit(
        f"[bold blue]TripleTriple AI Gateway v{get_version()}[/bold blue]\n"
        "[dim]Local-first AI agent for all your messaging channels.[/dim]",
        border_style="blue",
        padding=(1, 2),
    ))

    console.print("\n👋 [bold]Welcome![/bold] Let's get you set up.\n")

    # ── Step 1: Intelligence (LLM) ────────────────────────────────
    console.print("[bold yellow]1. Intelligence[/bold yellow]")
    console.print("Choose your primary AI provider. You can change this later.\n")

    selector = ModelSelector()
    providers = selector.list_providers()

    for i, p in enumerate(providers, 1):
        console.print(f"  {i}. {p.display_name}")

    choice = IntPrompt.ask(
        "\nSelect Provider",
        choices=[str(i) for i in range(1, len(providers) + 1)],
        default="1",
        show_choices=False,
    )
    
    provider_info = providers[int(choice) - 1]
    provider_name = provider_info.name
    key_name = provider_info.env_key
    
    # Help URLs
    help_urls = {
        "gemini": "https://aistudio.google.com/app/apikey",
        "openai": "https://platform.openai.com/api-keys",
        "anthropic": "https://console.anthropic.com/settings/keys",
    }
    help_url = help_urls.get(provider_name, "")

    console.print(f"\n[dim]Get your key at: {help_url}[/dim]")
    console.print(f"[bold]Enter {key_name}[/bold] :")
    api_key = Prompt.ask("")
    
    if api_key:
        config.set(key_name, api_key)
        console.print(f"[green]✅ Saved {key_name}[/green]")
    else:
        console.print("[red]❌ flexible, but you need a key to run.[/red]")

    # ── Step 2: Model Selection ───────────────────────────────────
    console.print(f"\n[bold yellow]2. Model Selection ({provider_info.display_name})[/bold yellow]")
    
    models = selector.list_models(provider=provider_name)
    if not models:
        console.print("[red]No models found for this provider![/red]")
    else:
        for i, m in enumerate(models, 1):
            alias = f" ({m.alias})" if m.alias else ""
            console.print(f"  {i}. {m.name}{alias}")
            
        model_choice = IntPrompt.ask(
            "\nSelect Model",
            choices=[str(i) for i in range(1, len(models) + 1)],
            default="1",
            show_choices=False,
        )
        selected_model = models[int(model_choice) - 1]
        
        # Save explicit model choice
        config.set("TRIPLETREBLE_MODEL", selected_model.full_id)
        # Update selector in-memory so status check works later
        selector.set_model(selected_model.full_id)
        console.print(f"[green]✅ Model set to: {selected_model.name}[/green]")

    # ── Step 3: Communication (Channels) ──────────────────────────
    console.print("\n[bold yellow]3. Communication[/bold yellow]")
    console.print("Which channels do you want to connect now? (Skip if none)\n")

    # Telegram
    if Confirm.ask("Enable [bold blue]Telegram[/bold blue]?", default=False):
        console.print("[bold]Enter TELEGRAM_BOT_TOKEN[/bold] :")
        token = Prompt.ask("")
        if token:
            config.set("TELEGRAM_BOT_TOKEN", token)
            console.print("[green]✅ Saved Telegram token[/green]")

    # Discord
    if Confirm.ask("Enable [bold purple]Discord[/bold purple]?", default=False):
        console.print("[bold]Enter DISCORD_BOT_TOKEN[/bold] :")
        token = Prompt.ask("")
        if token:
            config.set("DISCORD_BOT_TOKEN", token)
            console.print("[green]✅ Saved Discord token[/green]")

    # Slack
    if Confirm.ask("Enable [bold green]Slack[/bold green]?", default=False):
        console.print("[bold]Enter SLACK_BOT_TOKEN[/bold] :")
        token = Prompt.ask("")
        console.print("[bold]Enter SLACK_APP_TOKEN[/bold] (Socket Mode):")
        app_token = Prompt.ask("")
        if token and app_token:
            config.set("SLACK_BOT_TOKEN", token)
            config.set("SLACK_APP_TOKEN", app_token)
            console.print("[green]✅ Saved Slack tokens[/green]")
    
    # ── Step 4: Identity (Basic) ──────────────────────────────────
    console.print("\n[bold yellow]4. Identity[/bold yellow]")
    agent_name = Prompt.ask("Name your agent", default="TripleTriple")
    
    # Initialize workspace identity
    workspace_dir = Path(os.path.expanduser("~/.tripletriple/workspace"))
    workspace_dir.mkdir(parents=True, exist_ok=True)
    
    identity_file = workspace_dir / "IDENTITY.md"
    if not identity_file.exists():
        identity_content = f"""# IDENTITY.md — Who Am I?

- **Name:** {agent_name}
- **Creature:** AI assistant
- **Vibe:** Helpful, concise, and proactive
- **Emoji:** 🤖
"""
        identity_file.write_text(identity_content)
        console.print(f"[green]✅ Identity initialized ({agent_name})[/green]")
    else:
        # Update name in existing file? simple replace for now if standard format
        content = identity_file.read_text()
        if "**Name:**" in content:
            import re
            content = re.sub(r"\*\*Name:\*\* .*", f"**Name:** {agent_name}", content)
            identity_file.write_text(content)
            console.print(f"[green]✅ Identity updated ({agent_name})[/green]")


    # ── Step 5: Dependencies ──────────────────────────────────────
    if Confirm.ask("\nInstall browser dependencies (required for web tools)?", default=True):
        install_playwright()

    # ── Finish ────────────────────────────────────────────────────
    console.print("\n[bold green]🎉 Setup Complete![/bold green]")
    console.print(f"Configuration saved to: [underline]{config.env_path}[/underline]")
    # Final prompt
    console.print("\nOptions:")
    console.print("  [bold green]y[/bold green] = Start now (foreground)")
    console.print("  [bold blue]d[/bold blue] = Run in background (detach)")
    console.print("  [bold red]n[/bold red] = Do nothing")

    choice = Prompt.ask("\n🚀 Start the Gateway now?", choices=["y", "n", "d"], default="y")

    if choice == "y":
        return "start"
    elif choice == "d":
        return "detach"
    
    return "no"
