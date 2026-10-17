"""CLI subcommand for `query`: structured queries over a finished run's scene graph."""

import json
from typing import Annotated

import typer

app = typer.Typer(help="Query the reconstructed scene (structured templates, no language model).")


@app.callback(invoke_without_command=True)
def main(
    run: Annotated[str, typer.Argument(help="a finished run, by its name in runs/")],
    template: Annotated[str, typer.Argument(help="select | count | when | relate")],
    args: Annotated[list[str] | None,
                    typer.Argument(help="key=value pairs, e.g. category=person frame=12")] = None,
) -> None:
    """Answer one query and print it as JSON.

    Examples: ``select category=person`` · ``count category=person frame=20`` ·
    ``when entity=8`` · ``relate entity=8 predicate=near``.
    """
    from reifyscene.common import data_dir
    from reifyscene.query.scene_query import QueryError, SceneQuery

    kv = {}
    for a in args or []:
        if "=" not in a:
            raise typer.BadParameter(f"expected key=value, got {a!r}")
        k, v = a.split("=", 1)
        kv[k] = v
    try:
        answer = SceneQuery.from_run(data_dir.path(data_dir.RUNS, run)).run(template, kv)
    except (QueryError, KeyError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(answer, indent=2))
