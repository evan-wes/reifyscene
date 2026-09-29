"""CLI subcommand for `serve`: the local demo web app."""

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(help="Serve the local web app over a directory of runs.")


@app.callback(invoke_without_command=True)
def main(
    runs_root: Annotated[Path, typer.Argument(help="directory holding runs, e.g. ../runs")],
    port: Annotated[int, typer.Option()] = 8765,
    checkpoint_sub: Annotated[list[str] | None, typer.Option(
        help="checkpoint dirs to prefer, in order; the first a run has is used "
             "(default: render_q3b_batch8_nodecay, then render)")] = None,
    runs: Annotated[list[str] | None,
                    typer.Option(help="only these run names (default: loop_*)")] = None,
    device: Annotated[str, typer.Option()] = "cuda",
) -> None:
    """Start the app on 127.0.0.1; stop it with Ctrl-C."""
    import socket

    import uvicorn

    from reifyscene.api.web.app import create_app, gpu_power_ok

    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", port)) == 0:
            raise typer.BadParameter(f"port {port} is already in use; pass --port")
    ok, msg = gpu_power_ok()
    if not ok:
        raise typer.BadParameter(msg)
    typer.echo(msg)
    subs = checkpoint_sub or ["render_q3b_batch8_nodecay", "render"]
    app_ = create_app(runs_root, checkpoint_subs=subs, only=runs, device=device)
    uvicorn.run(app_, host="127.0.0.1", port=port, log_level="info")
