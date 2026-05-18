"""CLI entry point for sft."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer

from sft import __version__

app = typer.Typer(
    name="sft",
    help="An interactive terminal browser for .safetensors files.",
    no_args_is_help=True,
    add_completion=False,
)


def version_callback(value: bool) -> None:
    if value:
        typer.echo(f"sft {__version__}")
        raise typer.Exit()


@app.command("browse")
def browse_cmd(
    file: Path = typer.Argument(
        ...,
        help="Path to a .safetensors file to browse.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    _version: Optional[bool] = typer.Option(
        None, "--version", "-v",
        callback=version_callback, is_eager=True, help="Show version and exit.",
    ),
) -> None:
    """Open an interactive browser for a .safetensors file."""
    if file.suffix.lower() != ".safetensors":
        typer.secho(
            f"Error: Expected a .safetensors file, got '{file.suffix}'",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(code=1)
    from sft.browser import SftApp
    SftApp(file).run()


@app.command("diff")
def diff_cmd(
    file_a: Path = typer.Argument(..., exists=True, dir_okay=False, resolve_path=True),
    file_b: Path = typer.Argument(..., exists=True, dir_okay=False, resolve_path=True),
    rtol: float = typer.Option(1e-5, "--rtol", help="Relative tolerance for 'close'."),
    atol: float = typer.Option(1e-8, "--atol", help="Absolute tolerance for 'close'."),
    show: str = typer.Option(
        "diff", "--show",
        help="Which entries to list: diff, all, missing, incompatible.",
    ),
    limit: int = typer.Option(50, "--limit", help="Max rows to list."),
) -> None:
    """Compare two .safetensors files tensor-by-tensor."""
    from sft.data import diff_safetensors

    result = diff_safetensors(file_a, file_b, rtol=rtol, atol=atol)

    buckets: dict[str, list] = {
        "left_only": [], "right_only": [], "incompatible": [],
        "equal": [], "close": [], "differ": [],
    }
    for e in result.entries:
        buckets[e.status].append(e)

    colors = {
        "equal": typer.colors.GREEN,
        "close": typer.colors.CYAN,
        "differ": typer.colors.RED,
        "incompatible": typer.colors.MAGENTA,
        "left_only": typer.colors.YELLOW,
        "right_only": typer.colors.YELLOW,
    }

    typer.echo(f"A: {file_a}")
    typer.echo(f"B: {file_b}")
    typer.echo(f"rtol={rtol:g}  atol={atol:g}")
    typer.echo("")
    typer.echo("Summary:")
    for status in ("equal", "close", "differ", "incompatible", "left_only", "right_only"):
        n = len(buckets[status])
        if n:
            typer.secho(f"  {status:13s} {n}", fg=colors[status])

    if show == "missing":
        listing = buckets["left_only"] + buckets["right_only"]
    elif show == "incompatible":
        listing = buckets["incompatible"]
    elif show == "all":
        listing = result.entries
    else:
        listing = (
            buckets["differ"] + buckets["incompatible"]
            + buckets["left_only"] + buckets["right_only"]
        )

    if not listing:
        typer.echo("\n(nothing to list)")
        return

    typer.echo("")
    header = (
        f"{'status':13s} {'name':60s} {'shape':20s} "
        f"{'max_abs':>11s} {'mean_abs':>11s} {'rel_L2':>10s} {'cos':>8s}"
    )
    typer.echo(header)
    typer.echo("-" * len(header))
    for e in listing[:limit]:
        if e.shape_a == e.shape_b:
            shape = str(e.shape_a)
        elif e.shape_a and e.shape_b:
            shape = f"{e.shape_a}!={e.shape_b}"
        else:
            shape = str(e.shape_a or e.shape_b)
        name = e.name if len(e.name) <= 60 else "…" + e.name[-59:]
        if e.status in ("equal", "close", "differ"):
            stats = (
                f"{e.max_abs:11.3e} {e.mean_abs:11.3e} "
                f"{e.rel_l2:10.3e} {e.cosine:8.4f}"
            )
        else:
            stats = f"{'-':>11s} {'-':>11s} {'-':>10s} {'-':>8s}"
        typer.secho(
            f"{e.status:13s} {name:60s} {shape:20s} {stats}",
            fg=colors.get(e.status),
        )
    if len(listing) > limit:
        typer.echo(f"... ({len(listing) - limit} more — use --limit to see more)")


def _entry() -> None:
    """Entry point with backward-compat: `sft <file>` opens the browser."""
    argv = sys.argv[1:]
    known = {"browse", "diff", "--help", "-h", "--install-completion", "--show-completion"}
    if argv and argv[0] not in known and not argv[0].startswith("-"):
        # Backward compat: `sft file.safetensors` → `sft browse file.safetensors`
        sys.argv.insert(1, "browse")
    app()


if __name__ == "__main__":
    _entry()
