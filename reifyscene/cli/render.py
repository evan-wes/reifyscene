"""CLI subcommand for `render`."""

import typer

app = typer.Typer(help="Fit the 4D renderer.")


@app.callback(invoke_without_command=True)
def main() -> None:
    """Placeholder CLI callback."""
    typer.echo("render stage placeholder")
