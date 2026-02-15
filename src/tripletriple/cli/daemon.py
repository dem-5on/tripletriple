"""
Daemon Management Utilities

Handles starting, stopping, and restarting the TripleTriple Gateway in background mode.
Stores PID in ~/.tripletriple/gateway.pid.
Logs to ~/.tripletriple/gateway.log.
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

from rich.console import Console

console = Console()

PID_FILE = Path(os.path.expanduser("~/.tripletriple/gateway.pid"))
LOG_FILE = Path(os.path.expanduser("~/.tripletriple/gateway.log"))


def get_pid() -> Optional[int]:
    """Read the PID from the PID file if it exists."""
    if PID_FILE.exists():
        try:
            return int(PID_FILE.read_text().strip())
        except ValueError:
            return None
    return None


def is_running(pid: int) -> bool:
    """Check if a process with the given PID is running."""
    try:
        os.kill(pid, 0)  # Signal 0 checks if process exists
        return True
    except OSError:
        return False


def start_daemon(host: str = "127.0.0.1", port: int = 8000):
    """Start the gateway in detached mode."""
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if already running
    pid = get_pid()
    if pid and is_running(pid):
        console.print(f"[yellow]⚠️  Gateway is already running (PID: {pid}).[/yellow]")
        return

    console.print(f"[bold green]🚀 Starting TripleTriple Gateway in background...[/bold green]")
    console.print(f"To stop: [blue]tripletriple stop[/blue]")
    console.print(f"Logs:    [blue]{LOG_FILE}[/blue]")

    with open(LOG_FILE, "a") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "tripletriple.cli.main", "gateway", "--host", host, "--port", str(port)],
            stdout=log,
            stderr=log,
            cwd=os.getcwd(),
            start_new_session=True,  # Detach from terminal
        )
    
    PID_FILE.write_text(str(process.pid))
    console.print(f"[green]✅ Started successfully (PID: {process.pid})[/green]")


def stop_daemon() -> bool:
    """Stop the running gateway daemon."""
    pid = get_pid()
    if not pid:
        console.print("[yellow]⚠️  No running gateway found (no PID file).[/yellow]")
        return False

    if not is_running(pid):
        console.print(f"[yellow]⚠️  Process {pid} not found (stale PID file). Cleaning up.[/yellow]")
        PID_FILE.unlink()
        return False

    console.print(f"[bold red]🛑 Stopping TripleTriple Gateway (PID: {pid})...[/bold red]")
    try:
        os.kill(pid, signal.SIGTERM)
        
        # Wait for shutdown
        for _ in range(30):  # Wait up to 3 seconds
            if not is_running(pid):
                break
            time.sleep(0.1)
        
        if is_running(pid):
            console.print("[red]⚠️  Force killing...[/red]")
            os.kill(pid, signal.SIGKILL)
            
        PID_FILE.unlink()
        console.print("[green]✅ Gateway stopped.[/green]")
        return True
    except ProcessLookupError:
        console.print("[yellow]⚠️  Process already gone.[/yellow]")
        PID_FILE.unlink()
        return True
    except PermissionError:
        console.print(f"[red]❌ Permission denied to stop PID {pid}.[/red]")
        return False


def restart_daemon(host: str = "127.0.0.1", port: int = 8000):
    """Restart the gateway."""
    stop_daemon()
    time.sleep(1)
    start_daemon(host, port)


def get_status() -> Tuple[str, Optional[int]]:
    """Return status string and PID."""
    pid = get_pid()
    if pid and is_running(pid):
        return "running", pid
    return "stopped", None
