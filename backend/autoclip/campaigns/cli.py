"""``autoclip campaign ...``: Podcast Campaign Mode from the command line.

``autoclip campaign run jack_neel`` is the one-command acceptance run: it picks
the newest qualifying episode (or the one given), checks it against the
campaign's source rules before downloading, runs the whole pipeline in
campaign mode, and prints ``selected_clips.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .. import config, paths
from . import list_presets, load_preset
from . import report as campaign_report
from . import source as campaign_source
from .preset import PresetError

campaign_app = typer.Typer(help="Podcast Campaign Mode presets.", no_args_is_help=True)
console = Console(stderr=True)


def _preset(key: str):
    try:
        return load_preset(key)
    except PresetError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc


@campaign_app.command("list")
def list_cmd() -> None:
    """List the available campaign presets."""
    table = Table(header_style="bold")
    table.add_column("Preset")
    table.add_column("Name")
    table.add_column("Platform")
    for preset in list_presets():
        table.add_row(preset["key"], preset["name"], preset["platform"])
    Console().print(table)


@campaign_app.command("episodes")
def episodes_cmd(key: str = typer.Argument("jack_neel", help="Preset name.")) -> None:
    """Show the episodes a campaign currently accepts (newest first)."""
    preset = _preset(key)
    try:
        episodes = campaign_source.latest_episodes(preset, config.load().ingest)
    except Exception as exc:
        console.print(f"[red]Couldn't read the channel:[/red] {exc}")
        raise typer.Exit(1) from exc
    table = Table(header_style="bold", title=f"{preset.name}: latest {len(episodes)}")
    table.add_column("#", width=3)
    table.add_column("Length", width=8)
    table.add_column("Title")
    table.add_column("URL")
    for index, episode in enumerate(episodes, start=1):
        minutes = int(episode.get("duration_s") or 0) // 60
        table.add_row(str(index), f"{minutes} min", episode["title"], episode["url"])
    Console().print(table)


@campaign_app.command("run")
def run_cmd(
    key: str = typer.Argument("jack_neel", help="Preset name."),
    target: str = typer.Argument(
        "", help="Episode URL or a local file. Default: pick with --episode."
    ),
    episode: int = typer.Option(
        1, "--episode", "-e", help="Which of the latest episodes to use when no target is given."
    ),
    whisper_model: str = typer.Option("", "--whisper-model", help="Override the Whisper model."),
    provider: str = typer.Option("", "--provider", "-p", help="Override the LLM provider."),
    max_clips: int = typer.Option(0, "--max-clips", "-n", help="Override the preset's top N."),
) -> None:
    """Run the whole pipeline in campaign mode and print selected_clips.json."""
    import asyncio

    from .. import db
    from ..db import store
    from ..db.models import Job, new_id
    from ..pipeline import ingest, runner
    from . import apply

    preset = _preset(key)
    db.init()
    settings = config.load()
    settings.campaign.enabled = True
    settings.campaign.preset = key
    settings.campaign.rules = None
    if provider:
        settings.active_provider = provider  # type: ignore[assignment]
    if whisper_model:
        settings.whisper.model = whisper_model
    settings = apply(settings)
    if max_clips:
        settings.viral_hook.top_n = max_clips

    # --- pick and check the source before downloading anything -----------
    if not target:
        try:
            episodes = campaign_source.latest_episodes(preset, settings.ingest)
        except Exception as exc:
            console.print(f"[red]Couldn't read the channel's latest episodes:[/red] {exc}")
            raise typer.Exit(1) from exc
        if not 1 <= episode <= len(episodes):
            console.print(f"[red]--episode must be between 1 and {len(episodes)}.[/red]")
            raise typer.Exit(2)
        target = episodes[episode - 1]["url"]
        console.print(f"Episode {episode}: {episodes[episode - 1]['title']}")

    is_url = ingest.is_youtube_url(target)
    if is_url:
        check = campaign_source.check_url(preset, target, settings.ingest)
        if not check.allowed:
            console.print(f"[red]Source rejected:[/red] {check.message}")
            raise typer.Exit(1)
        console.print(f"Source: {check.status} — {check.message}")

    try:
        with console.status("Downloading the episode..."):
            source = (
                ingest.ingest_youtube(target, settings.ingest)
                if is_url
                else ingest.ingest_file(Path(target))
            )
    except ingest.IngestError as exc:
        console.print(f"[red]Ingest failed.[/red] {exc}")
        raise typer.Exit(1) from exc
    store.create_source(source)

    check = campaign_source.check_source(preset, source, settings.ingest)
    if not check.allowed:
        console.print(f"[red]Source rejected:[/red] {check.message}")
        raise typer.Exit(1)
    settings.campaign.source_check = check.as_dict()
    if check.status != "verified":
        console.print(f"[yellow]{check.message}[/yellow]")

    job = store.create_job(
        Job(
            id=new_id(),
            source_id=source.id,
            provider=settings.active_provider,
            settings=settings.model_dump(mode="json"),
        )
    )
    console.print(f"Job {job.id} — {source.title}")

    last: dict[str, object] = {"message": None, "pct": -10.0}

    def on_progress(event: runner.ProgressEvent) -> None:
        # One line per new step, or every 10% within a long one.
        pct = event.overall * 100
        if event.message != last["message"] or pct - float(last["pct"]) >= 10:
            console.print(f"  {pct:5.1f}%  {event.message}", highlight=False)
            last.update(message=event.message, pct=pct)

    try:
        asyncio.run(
            runner.PipelineRunner(job, source, settings=settings, on_progress=on_progress).run()
        )
    except Exception as exc:
        console.print(f"[red]Pipeline failed.[/red] {exc}")
        raise typer.Exit(1) from exc

    selected = campaign_report.read(job.id)
    out_dir = paths.exports_dir() / job.id
    console.print(f"\n[green]Clips, captions and selected_clips.json in[/green] {out_dir}")
    if selected and selected.get("host_check") != "verified":
        console.print(
            f"[yellow]No speaker labels: every clip opens on a question, but whether "
            f"{preset.host.name or 'the host'} asked it could not be verified. Install the "
            "diarization extra and set a HuggingFace token to check it.[/yellow]"
        )
    # The JSON goes to stdout on its own, so it can be piped or redirected.
    print(json.dumps(selected, indent=2, ensure_ascii=False))
