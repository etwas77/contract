from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from candidate_reader.config import Settings
from candidate_reader.errors import CandidateReaderError
from company_reader.workflow import CompanyReadWorkflow

app = typer.Typer(
    name="company-reader",
    help="Convert a company employment offer into validated structured data.",
    no_args_is_help=True,
)
console = Console()
error_console = Console(stderr=True)


@app.callback()
def application() -> None:
    """Company employment-offer reader."""


def _default_offer() -> Path | None:
    matches = sorted(Path.cwd().glob("company/contract/*"))
    return matches[0] if matches else None


@app.command("read")
def read_company_offer(
    offer: Annotated[
        Path | None,
        typer.Option("--offer", help="Company employment offer PDF or TXT path."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Regenerate structured data even when the cache is valid."),
    ] = False,
    non_interactive: Annotated[
        bool,
        typer.Option("--non-interactive", help="Disable prompts; --offer is required."),
    ] = False,
) -> None:
    """Read a company employment offer and write validated JSON."""
    try:
        if offer is None:
            if non_interactive:
                raise typer.BadParameter("--offer is required in non-interactive mode")
            default = _default_offer()
            entered = typer.prompt(
                "Company offer path",
                default=str(default) if default is not None else None,
            )
            offer = Path(entered)

        if not non_interactive:
            table = Table(title="Selected input")
            table.add_column("Role")
            table.add_column("Path")
            table.add_row("Company offer", str(offer))
            console.print(table)
            if not typer.confirm("Continue?", default=True):
                raise typer.Abort()

        result = CompanyReadWorkflow(Settings()).run(
            offer,
            force=force,
            authorize=lambda _: False,
        )
        table = Table(title="Company structured data")
        table.add_column("Status")
        table.add_column("Output")
        table.add_column("Clauses", justify="right")
        table.add_column("Warnings", justify="right")
        table.add_row(
            "reused" if result.reused else "generated",
            str(result.output_path),
            str(len(result.offer.data.clauses)),
            str(len(result.offer.source.warnings)),
        )
        console.print(table)
        console.print(
            f"Conservative OpenAI cost reserved: [bold]${result.reserved_cost_usd}[/bold]"
        )
    except CandidateReaderError as exc:
        error_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


def main() -> None:
    app()
