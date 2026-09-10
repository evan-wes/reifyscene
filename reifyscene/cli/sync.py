"""CLI subcommand for `sync`."""

import typer

app = typer.Typer(help="Estimate temporal offsets.")


@app.callback(invoke_without_command=True)
def main() -> None:
    """Placeholder CLI callback."""
    typer.echo("sync stage placeholder")
