"""
Interactive Onboarding Wizard

Guides the user through initial configuration:
1. Select LLM Provider & API Key
2. Select Channels & Tokens
3. Install Dependencies (Playwright)
4. Launch Gateway
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
            # Run playwright install-deps (system deps - might need sudo, skip if fails)
            # subprocess.run([sys.executable, "-m", "playwright", "install-deps"], ...)
            
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

    providers = ["Google Gemini", "OpenAI", "Anthropic Claude"]
    choice = IntPrompt.ask(
        "Select Provider",
        choices=["1", "2", "3"],
        default="1",
        show_choices=False,
    )
    
    provider_name = providers[int(choice) - 1]
    
    if provider_name == "Google Gemini":
        key_name = "GEMINI_API_KEY"
        help_url = "https://aistudio.google.com/app/apikey"
    elif provider_name == "OpenAI":
        key_name = "OPENAI_API_KEY"
        help_url = "https://platform.openai.com/api-keys"
    else:
        key_name = "ANTHROPIC_API_KEY"
        help_url = "https://console.anthropic.com/settings/keys"

    console.print(f"\n[dim]Get your key at: {help_url}[/dim]")
    api_key = Prompt.ask(f"[bold]Enter {key_name}[/bold]", password=True)
    
    if api_key:
        config.set(key_name, api_key)
        console.print(f"[green]✅ Saved {key_name}[/green]")
    else:
        console.print("[red]❌ flexible, but you need a key to run.[/red]")
        # We allow proceeding, user might add it manually later

    # ── Step 2: Communication (Channels) ──────────────────────────
    console.print("\n[bold yellow]2. Communication[/bold yellow]")
    console.print("Which channels do you want to connect now? (Skip if none)\n")

    # Telegram
    if Confirm.ask("Enable [bold blue]Telegram[/bold blue]?", default=False):
        token = Prompt.ask("Enter [bold]TELEGRAM_BOT_TOKEN[/bold]", password=True)
        if token:
            config.set("TELEGRAM_BOT_TOKEN", token)
            console.print("[green]✅ Saved Telegram token[/green]")

    # Discord
    if Confirm.ask("Enable [bold purple]Discord[/bold purple]?", default=False):
        token = Prompt.ask("Enter [bold]DISCORD_BOT_TOKEN[/bold]", password=True)
        if token:
            config.set("DISCORD_BOT_TOKEN", token)
            console.print("[green]✅ Saved Discord token[/green]")

    # Slack
    if Confirm.ask("Enable [bold green]Slack[/bold green]?", default=False):
        token = Prompt.ask("Enter [bold]SLACK_BOT_TOKEN[/bold]", password=True)
        app_token = Prompt.ask("Enter [bold]SLACK_APP_TOKEN[/bold] (Socket Mode)", password=True)
        if token and app_token:
            config.set("SLACK_BOT_TOKEN", token)
            config.set("SLACK_APP_TOKEN", app_token)
            console.print("[green]✅ Saved Slack tokens[/green]")
    
    # ── Step 3: Identity (Basic) ──────────────────────────────────
    console.print("\n[bold yellow]3. Identity[/bold yellow]")
    agent_name = Prompt.ask("Name your agent", default="TripleTriple")
    # We don't save this to .env usually, but we could init workspace here.
    # For now, let's just use it to personalize the finish message.

    # ── Step 4: Dependencies ──────────────────────────────────────
    if Confirm.ask("\nInstall browser dependencies (required for web tools)?", default=True):
        install_playwright()

    # ── Finish ────────────────────────────────────────────────────
    console.print("\n[bold green]🎉 Setup Complete![/bold green]")
    console.print(f"Configuration saved to: [underline]{config.env_path}[/underline]")
    
    if Confirm.ask("\n  Start with: tripletriple\n🚀 Start the Gateway now?", default=True):
        # We need to run the gateway command.
        # Since we are in the CLI process, we can't easily replace ourselves with uvicorn via shell without
        # losing the context or making main.py logic complex.
        # But we can just return, and main.py will handle the start.
        return True
    
    return False
