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
    checkpoint: Annotated[str, typer.Option(
        help="the checkpoint to edit: the render stage's own, or a named retrain")] = "render",
    device: Annotated[str, typer.Option()] = "cuda",
) -> None:
    """Remove one entity; renders and result.json go to the run's output dir for the edit."""
    from reifyscene.api.run_outputs import RunOutputs
    from reifyscene.edit.remove import RemoveEditor
    from reifyscene.query.scene_query import SceneQuery

    run = RunOutputs.open(run_dir)
    node = SceneQuery(run.artifact("graph")).entity(entity).node_id
    editor = RemoveEditor(run.checkpoint(checkpoint), device=device)
    result = editor.remove(node, slackness=slackness, filtered=not no_filter)
    result.pop("frames", None)
    typer.echo(json.dumps(result, indent=2))
