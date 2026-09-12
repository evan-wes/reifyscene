"""CLI subcommand for `graph`."""

import typer

app = typer.Typer(help="Build the dynamic scene graph.")


@app.callback(invoke_without_command=True)
def main() -> None:
    """Placeholder CLI callback."""
    typer.echo("graph stage placeholder")
