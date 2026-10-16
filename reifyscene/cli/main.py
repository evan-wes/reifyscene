"""Typer app entrypoint."""

import typer

from reifyscene.cli import (
    edit,
    geometry,
    graph,
    optimize,
    perception,
    preprocess,
    query,
    render,
    serve,
    sync,
    tracking,
)

app = typer.Typer(help="reifyscene command-line interface.")
app.add_typer(preprocess.app, name="preprocess")
app.add_typer(sync.app, name="sync")
app.add_typer(geometry.app, name="geometry")
app.add_typer(tracking.app, name="tracking")
app.add_typer(perception.app, name="perception")
app.add_typer(graph.app, name="graph")
app.add_typer(optimize.app, name="optimize")
app.add_typer(render.app, name="render")
app.add_typer(query.app, name="query")
app.add_typer(edit.app, name="edit")
app.add_typer(serve.app, name="serve")


def main() -> None:
    """Run the Typer application."""
    app()


if __name__ == "__main__":
    main()
