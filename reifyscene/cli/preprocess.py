"""CLI subcommand for `preprocess`."""

import typer

app = typer.Typer(help="Preprocess video inputs.")


@app.callback(invoke_without_command=True)
def main() -> None:
    """Placeholder CLI callback."""
    typer.echo("preprocess stage placeholder")
