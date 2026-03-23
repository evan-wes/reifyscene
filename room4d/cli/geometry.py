"""CLI subcommand for `geometry`."""

import typer

app = typer.Typer(help="Initialize scene geometry.")


@app.callback(invoke_without_command=True)
def main() -> None:
    """Placeholder CLI callback."""
    typer.echo("geometry stage placeholder")
