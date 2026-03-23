"""CLI subcommand for `optimize`."""

import typer

app = typer.Typer(help="Run staged optimization.")


@app.callback(invoke_without_command=True)
def main() -> None:
    """Placeholder CLI callback."""
    typer.echo("optimize stage placeholder")
