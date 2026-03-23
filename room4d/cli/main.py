"""Typer app entrypoint."""

import typer

from room4d.cli import geometry, graph, optimize, perception, preprocess, query, render, sync, tracking

app = typer.Typer(help="room4d command-line interface.")
app.add_typer(preprocess.app, name="preprocess")
app.add_typer(sync.app, name="sync")
app.add_typer(geometry.app, name="geometry")
app.add_typer(tracking.app, name="tracking")
app.add_typer(perception.app, name="perception")
app.add_typer(graph.app, name="graph")
app.add_typer(optimize.app, name="optimize")
app.add_typer(render.app, name="render")
app.add_typer(query.app, name="query")


def main() -> None:
    """Run the Typer application."""
    app()


if __name__ == "__main__":
    main()
