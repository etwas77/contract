from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from candidate_reader.config import Settings
from candidate_reader.errors import CandidateReaderError
from candidate_reader.models import CandidateConstraints, CandidateCV
from candidate_reader.workflow import CandidateReadWorkflow, ReadResult

app = typer.Typer(
    name="candidate-reader",
    help="Convert candidate CV and constraints into validated structured data.",
    no_args_is_help=True,
)
console = Console()
error_console = Console(stderr=True)


@app.callback()
def application() -> None:
    """Candidate document reader."""


def _discover_default(pattern: str) -> Path | None:
    matches = sorted(Path.cwd().glob(pattern))
    return matches[0] if matches else None


def _resolve_input(
    supplied: Path | None,
    *,
    prompt: str,
    default: Path | None,
    non_interactive: bool,
) -> Path:
    if supplied is not None:
        return supplied
    if non_interactive:
        raise typer.BadParameter(f"{prompt} is required in non-interactive mode")
    entered = typer.prompt(prompt, default=str(default) if default is not None else None)
    return Path(entered)


def _authorization_callback(
    *,
    automatically_confirm: bool,
    non_interactive: bool,
) -> Callable[[tuple[str, ...]], bool]:
    def authorize(filenames: tuple[str, ...]) -> bool:
        if automatically_confirm:
            return True
        if non_interactive:
            return False
        console.print(
            "[yellow]The following extracted candidate data will be sent to the configured "
            f"OpenAI model:[/yellow] {', '.join(filenames)}"
        )
        return typer.confirm("Authorize this transmission?", default=False)

    return authorize


def _summary(result: ReadResult) -> None:
    table = Table(title="Candidate structured data")
    table.add_column("Role")
    table.add_column("Status")
    table.add_column("Output")
    table.add_column("Records", justify="right")
    table.add_column("Warnings", justify="right")
    for role, item in (("CV", result.cv), ("Constraints", result.constraints)):
        document = item.document
        if isinstance(document, CandidateCV):
            records = (
                len(document.data.work_experience)
                + len(document.data.skills)
                + len(document.data.education)
                + len(document.data.projects)
                + len(document.data.languages)
            )
        elif isinstance(document, CandidateConstraints):
            records = len(document.data.constraints)
        else:
            records = 0
        table.add_row(
            role,
            "reused" if item.reused else "generated",
            str(item.output_path),
            str(records),
            str(len(document.source.warnings)),
        )
    console.print(table)
    console.print(f"Conservative OpenAI cost reserved: [bold]${result.reserved_cost_usd}[/bold]")


@app.command("read")
def read_candidate(
    cv: Annotated[
        Path | None,
        typer.Option("--cv", help="Candidate CV PDF or TXT path."),
    ] = None,
    constraints: Annotated[
        Path | None,
        typer.Option("--constraints", help="Candidate constraints PDF or TXT path."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Regenerate structured data even when the cache is valid."),
    ] = False,
    non_interactive: Annotated[
        bool,
        typer.Option("--non-interactive", help="Disable prompts; paths and --yes are required."),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Authorize candidate text transmission."),
    ] = False,
) -> None:
    """Read candidate documents and write validated JSON."""
    try:
        cv_path = _resolve_input(
            cv,
            prompt="Candidate CV path",
            default=_discover_default("candidate/cv/*"),
            non_interactive=non_interactive,
        )
        constraints_path = _resolve_input(
            constraints,
            prompt="Candidate constraints path",
            default=_discover_default("candidate/constraints/*"),
            non_interactive=non_interactive,
        )
        if not non_interactive:
            table = Table(title="Selected inputs")
            table.add_column("Role")
            table.add_column("Path")
            table.add_row("CV", str(cv_path))
            table.add_row("Constraints", str(constraints_path))
            console.print(table)
            if not typer.confirm("Continue?", default=True):
                raise typer.Abort()

        settings = Settings()
        workflow = CandidateReadWorkflow(settings)
        result = workflow.run(
            cv_path,
            constraints_path,
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
