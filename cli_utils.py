"""
cli_utils.py — Pretty CLI formatting utilities for Orthello.

Provides colored output, formatted tables, and styled progress bars.
"""

import sys
from typing import Optional

# Try to use rich for beautiful output, fall back to basic colors if not available
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, DownloadColumn, TransferSpeedColumn
    from rich.style import Style
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# Try colorama for basic color support
try:
    from colorama import Fore, Back, Style as ColoramaStyle, init as colorama_init
    colorama_init(autoreset=True)
    HAS_COLORAMA = True
except ImportError:
    HAS_COLORAMA = False


class CLIFormatter:
    """Provides pretty CLI formatting with fallbacks for different terminal capabilities."""
    
    def __init__(self):
        self.use_rich = HAS_RICH
        self.use_colorama = HAS_COLORAMA and not HAS_RICH
        if HAS_RICH:
            self.console = Console()
        
    def header(self, title: str, width: int = 60):
        """Print a styled header."""
        if self.use_rich:
            from rich.style import Style
            panel = Panel(
                Text(title.center(width - 4), style="bold white"),
                style="bold cyan",
                expand=False,
                padding=(1, 2)
            )
            self.console.print(panel)
        elif self.use_colorama:
            print(f"\n{Fore.CYAN}{'═' * width}")
            print(f"{Fore.CYAN}  {title}")
            print(f"{Fore.CYAN}{'═' * width}{ColoramaStyle.RESET_ALL}\n")
        else:
            print(f"\n{'═' * width}")
            print(f"  {title}")
            print(f"{'═' * width}\n")
    
    def subheader(self, title: str, width: int = 60):
        """Print a styled subheader."""
        if self.use_rich:
            self.console.print(f"\n[bold yellow]→ {title}[/bold yellow]")
        elif self.use_colorama:
            print(f"\n{Fore.YELLOW}→ {title}{ColoramaStyle.RESET_ALL}")
        else:
            print(f"\n→ {title}")
    
    def success(self, message: str):
        """Print a success message."""
        if self.use_rich:
            self.console.print(f"[bold green]✓ {message}[/bold green]")
        elif self.use_colorama:
            print(f"{Fore.GREEN}✓ {message}{ColoramaStyle.RESET_ALL}")
        else:
            print(f"✓ {message}")
    
    def info(self, message: str):
        """Print an info message."""
        if self.use_rich:
            self.console.print(f"[bold blue]ℹ {message}[/bold blue]")
        elif self.use_colorama:
            print(f"{Fore.BLUE}ℹ {message}{ColoramaStyle.RESET_ALL}")
        else:
            print(f"ℹ {message}")
    
    def warning(self, message: str):
        """Print a warning message."""
        if self.use_rich:
            self.console.print(f"[bold yellow]⚠ {message}[/bold yellow]")
        elif self.use_colorama:
            print(f"{Fore.YELLOW}⚠ {message}{ColoramaStyle.RESET_ALL}")
        else:
            print(f"⚠ {message}")
    
    def error(self, message: str):
        """Print an error message."""
        if self.use_rich:
            self.console.print(f"[bold red]✗ {message}[/bold red]")
        elif self.use_colorama:
            print(f"{Fore.RED}✗ {message}{ColoramaStyle.RESET_ALL}")
        else:
            print(f"✗ {message}")
    
    def highlight(self, message: str, value: str, color: str = "cyan"):
        """Print a message with a highlighted value."""
        if self.use_rich:
            self.console.print(f"{message} [bold {color}]{value}[/bold {color}]")
        elif self.use_colorama:
            color_map = {
                "cyan": Fore.CYAN,
                "green": Fore.GREEN,
                "yellow": Fore.YELLOW,
                "red": Fore.RED,
                "magenta": Fore.MAGENTA,
            }
            color_code = color_map.get(color, Fore.WHITE)
            print(f"{message} {color_code}{value}{ColoramaStyle.RESET_ALL}")
        else:
            print(f"{message} {value}")
    
    def stats_line(self, **kwargs):
        """Print a line of statistics."""
        if self.use_rich:
            parts = []
            for key, value in kwargs.items():
                if isinstance(value, float):
                    parts.append(f"[cyan]{key}[/cyan]= [bold yellow]{value:.3f}[/bold yellow]")
                else:
                    parts.append(f"[cyan]{key}[/cyan]= [bold yellow]{value}[/bold yellow]")
            self.console.print("  " + "  ".join(parts))
        elif self.use_colorama:
            parts = []
            for key, value in kwargs.items():
                if isinstance(value, float):
                    parts.append(f"{Fore.CYAN}{key}{ColoramaStyle.RESET_ALL}= {Fore.YELLOW}{value:.3f}{ColoramaStyle.RESET_ALL}")
                else:
                    parts.append(f"{Fore.CYAN}{key}{ColoramaStyle.RESET_ALL}= {Fore.YELLOW}{value}{ColoramaStyle.RESET_ALL}")
            print("  " + "  ".join(parts))
        else:
            parts = [f"{key}= {value}" for key, value in kwargs.items()]
            print("  " + "  ".join(parts))
    
    def table(self, title: str, rows: list, columns: list):
        """Print a formatted table."""
        if self.use_rich:
            table = Table(title=title, style="cyan")
            for col in columns:
                table.add_column(col, style="magenta")
            for row in rows:
                table.add_row(*[str(v) for v in row])
            self.console.print(table)
        else:
            # Fallback: print simple format
            print(f"\n{title}")
            print("  " + "  ".join(columns))
            for row in rows:
                print("  " + "  ".join(str(v) for v in row))
    
    def progress_bar(self, total: int, description: str = "Processing"):
        """Return a progress bar context manager."""
        if self.use_rich:
            return Progress(
                SpinnerColumn(),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("[cyan]{task.description}"),
                console=self.console,
            ).wrap_progress(Description=description, total=total)
        else:
            # Return a simple wrapper for compatibility
            from tqdm import tqdm
            return tqdm(total=total, desc=description, unit="it", 
                       bar_format='{desc}: {percentage:.0f}%|{bar:20}| {n_fmt}/{total_fmt}')
    
    def section(self, title: str):
        """Print a section separator."""
        if self.use_rich:
            self.console.print(f"\n[bold magenta]{'─' * 50}\n{title}\n{'─' * 50}[/bold magenta]\n")
        elif self.use_colorama:
            print(f"\n{Fore.MAGENTA}{'─' * 50}\n{title}\n{'─' * 50}{ColoramaStyle.RESET_ALL}\n")
        else:
            print(f"\n{'─' * 50}\n{title}\n{'─' * 50}\n")


# Global formatter instance
formatter = CLIFormatter()


def get_formatter() -> CLIFormatter:
    """Get the global CLI formatter instance."""
    return formatter
