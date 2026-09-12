"""CLI subcommand for `perception`."""

import typer

app = typer.Typer(help="Run human and object perception.")


@app.callback(invoke_without_command=True)
def main() -> None:
    """Placeholder CLI callback."""
    typer.echo("perception stage placeholder")
