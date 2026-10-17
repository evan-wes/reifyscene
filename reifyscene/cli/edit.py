"""CLI subcommand for `edit`: execute an edit on a finished run's trained scene."""

import json
from typing import Annotated

import typer

app = typer.Typer(help="Edit the reconstructed scene (remove, so far).")


@app.command("remove")
def remove(
    run: Annotated[str, typer.Argument(help="a finished run, by its name in runs/")],
    entity: Annotated[str, typer.Argument(help="node id, or its trailing tracklet number")],
    slackness: Annotated[float, typer.Option(help="FlashSplat slackness: + surer, - more")] = 0.0,
    no_filter: Annotated[bool, typer.Option("--no-filter", help="skip the floater filter")] = False,
    checkpoint: Annotated[str, typer.Option(
        help="the checkpoint to edit: the render stage's own, or a named retrain")] = "render",
    device: Annotated[str, typer.Option()] = "cuda",
) -> None:
    """Remove one entity; renders and result.json go to the run's output dir for the edit."""
    from reifyscene.api.run_outputs import RunOutputs
    from reifyscene.common import data_dir
    from reifyscene.edit.remove import RemoveEditor
    from reifyscene.query.scene_query import SceneQuery

    outputs = RunOutputs.open(data_dir.path(data_dir.RUNS, run))
    node = SceneQuery(outputs.artifact("graph")).entity(entity).node_id
    editor = RemoveEditor(outputs.checkpoint(checkpoint), device=device)
    result = editor.remove(node, slackness=slackness, filtered=not no_filter)
    result.pop("frames", None)
    typer.echo(json.dumps(result, indent=2))
