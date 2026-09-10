"""CLI subcommand for `tracking`."""

import typer

app = typer.Typer(help="Track points across frames.")


@app.callback(invoke_without_command=True)
def main() -> None:
    """Placeholder CLI callback."""
    typer.echo("tracking stage placeholder")
