from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from candidate_reader.config import Settings
from candidate_reader.errors import CandidateReaderError
from comparison.workflow import ComparisonRunResult, ComparisonWorkflow

app = typer.Typer(
    name="comparison",
    help="Compare structured candidate requirements with a company offer.",
    no_args_is_help=True,
)
console = Console()
error_console = Console(stderr=True)


@app.callback()
def application() -> None:
    """Structured candidate and offer comparison."""


def _authorization_callback(
    *,
    automatically_confirm: bool,
    non_interactive: bool,
) -> Callable[[tuple[str, ...], Decimal], bool]:
    def authorize(filenames: tuple[str, ...], reserved: Decimal) -> bool:
        if automatically_confirm:
            return True
        if non_interactive:
            return False
        console.print(
            "[yellow]A minimal structured subset from these files will be sent to the "
            f"configured OpenAI model:[/yellow] {', '.join(filenames)}"
        )
        console.print(f"Conservative two-call reservation: [bold]${reserved:.4f}[/bold]")
        return typer.confirm("Authorize semantic analysis and verification?", default=False)

    return authorize


def _summary(result: ComparisonRunResult) -> None:
    data = result.data
    summary = data.summary
    table = Table(title="Candidate and offer comparison")
    table.add_column("Status")
    table.add_column("Outcome")
    table.add_column("Satisfied", justify="right")
    table.add_column("Violated", justify="right")
    table.add_column("Unknown", justify="right")
    table.add_column("Report")
    table.add_row(
        "reused" if result.reused else "generated",
        summary.overall_outcome.value,
        str(summary.mandatory_satisfied),
        str(summary.mandatory_violated),
        str(summary.mandatory_unknown),
        str(result.report_path),
    )
    console.print(table)
    console.print(f"Conservative OpenAI cost reserved: [bold]${data.reserved_cost_usd}[/bold]")


@app.command("run")
def run_comparison(
    cv: Annotated[
        Path,
        typer.Option("--cv", help="Structured candidate CV JSON."),
    ] = Path("structured_data/candidate_cv.json"),
    constraints: Annotated[
        Path,
        typer.Option("--constraints", help="Structured candidate constraints JSON."),
    ] = Path("structured_data/candidate_constraints.json"),
    offer: Annotated[
        Path,
        typer.Option("--offer", help="Structured company offer JSON."),
    ] = Path("structured_data/company_offer.json"),
    output: Annotated[
        Path,
        typer.Option("--output", help="Markdown report path."),
    ] = Path("report.md"),
    force: Annotated[
        bool,
        typer.Option("--force", help="Regenerate comparison even when the cache is valid."),
    ] = False,
    non_interactive: Annotated[
        bool,
        typer.Option("--non-interactive", help="Disable authorization prompts."),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Authorize minimal structured-data transmission."),
    ] = False,
) -> None:
    """Compare structured inputs and write report.md."""
    try:
        result = ComparisonWorkflow(Settings()).run(
            cv,
            constraints,
            offer,
            report_path=output,
            force=force,
            authorize=_authorization_callback(
                automatically_confirm=yes,
                non_interactive=non_interactive,
            ),
        )
        _summary(result)
    except CandidateReaderError as exc:
        error_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


def main() -> None:
    app()
