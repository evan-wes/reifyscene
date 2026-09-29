"""CLI subcommand for `edit`: execute an edit on a finished run's trained scene."""

import json
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(help="Edit the reconstructed scene (remove, so far).")


@app.command("remove")
def remove(
    run_dir: Annotated[Path, typer.Argument(help="a finished run")],
    entity: Annotated[str, typer.Argument(help="node id, or its trailing tracklet number")],
    slackness: Annotated[float, typer.Option(help="FlashSplat slackness: + surer, - more")] = 0.0,
    no_filter: Annotated[bool, typer.Option("--no-filter", help="skip the floater filter")] = False,
    checkpoint_sub: Annotated[str, typer.Option(help="checkpoint dir inside the run")] = "render",
    device: Annotated[str, typer.Option()] = "cuda",
) -> None:
    """Remove one entity; renders and result.json go to <run>/edits/<edit id>/."""
    from reifyscene.edit.remove import RemoveEditor
    from reifyscene.query.scene_query import SceneQuery

    node = SceneQuery.from_run(run_dir).entity(entity).node_id
    editor = RemoveEditor(run_dir, checkpoint_sub=checkpoint_sub, device=device)
    result = editor.remove(node, slackness=slackness, filtered=not no_filter)
    result.pop("frames", None)
    typer.echo(json.dumps(result, indent=2))
