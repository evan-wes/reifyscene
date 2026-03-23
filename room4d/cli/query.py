"""CLI subcommand for `query`."""

import typer

app = typer.Typer(help="Query the reconstructed scene.")


@app.callback(invoke_without_command=True)
def main() -> None:
    """Placeholder CLI callback."""
    typer.echo("query stage placeholder")
