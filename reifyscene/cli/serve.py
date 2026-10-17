"""CLI subcommand for `serve`: the local demo web app."""

from typing import Annotated

import typer

app = typer.Typer(help="Serve the local web app over the data directory's runs.")


@app.callback(invoke_without_command=True)
def main(
    port: Annotated[int, typer.Option()] = 8765,
    checkpoint: Annotated[str, typer.Option(
        help="the checkpoint to serve: the render stage's own, or a named retrain beside it "
             "(a run without it is not served)")] = "render",
    runs: Annotated[list[str] | None,
                    typer.Option(help="only these runs, by name (default: every run in runs/)")
                    ] = None,
    device: Annotated[str, typer.Option()] = "cuda",
) -> None:
    """Start the app on 127.0.0.1; stop it with Ctrl-C."""
    import socket

    import uvicorn

    from reifyscene.api.web.app import create_app, gpu_power_ok
    from reifyscene.common import data_dir

    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", port)) == 0:
            raise typer.BadParameter(f"port {port} is already in use; pass --port")
    ok, msg = gpu_power_ok()
    if not ok:
        raise typer.BadParameter(msg)
    typer.echo(msg)
    app_ = create_app(data_dir.area(data_dir.RUNS), checkpoint=checkpoint, only=runs, device=device)
    uvicorn.run(app_, host="127.0.0.1", port=port, log_level="info")
