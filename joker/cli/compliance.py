"""Compliance commands for OPRA data governance."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from joker.compliance.opra_scanner import (
    discover_db_paths,
    quarantine_opra_artifacts,
    scan_local_opra,
)

compliance_app = typer.Typer(help="OPRA data-governance compliance tools")
console = Console()


@compliance_app.command("scan-local-opra")
def scan_local_opra_cmd(
    root: Optional[str] = typer.Option(None, help="Project root to scan"),
    db: Optional[str] = typer.Option(None, help="Extra SQLite database path"),
) -> None:
    """Scan local captures, logs, reports, replays, and SQLite DBs for raw OPRA fields."""
    from joker.config.loader import load_app_settings

    app_settings, _ = load_app_settings()
    base = Path(root) if root else Path(".")
    db_paths = discover_db_paths(base, app_settings.db_path)
    if db:
        extra = Path(db)
        if extra.exists() and extra.resolve() not in db_paths:
            db_paths.append(extra.resolve())

    result = scan_local_opra(root=base, db_paths=db_paths)
    grouped = result.by_category()

    console.print(
        f"[bold]OPRA local scan[/bold] — {result.files_scanned} file(s), "
        f"{len(result.db_paths_scanned)} DB(s)"
    )
    if result.db_paths_scanned:
        console.print(f"DB paths: {', '.join(result.db_paths_scanned)}")

    summary = Table(title="Scan summary by category")
    summary.add_column("Category")
    summary.add_column("Count")
    for category in ("possible_raw_opra", "stock_data_not_opra", "synthetic_ignored", "safe_metadata"):
        summary.add_row(category, str(len(grouped.get(category, []))))
    console.print(summary)

    violations = result.violations
    if not violations:
        console.print("[green]No possible raw OPRA violations detected.[/green]")
        return

    table = Table(title="Possible raw OPRA violations")
    table.add_column("File")
    table.add_column("Line")
    table.add_column("Reason")
    for v in violations[:50]:
        table.add_row(v.path, str(v.line_number or "—"), v.reason)
    console.print(table)
    if len(violations) > 50:
        console.print(f"[yellow]... and {len(violations) - 50} more[/yellow]")

    console.print("\n[bold]Recommended actions:[/bold]")
    for cmd in result.recommended_quarantine_commands():
        console.print(f"  {cmd}")


@compliance_app.command("quarantine-opra-artifacts")
def quarantine_opra_artifacts_cmd(
    root: Optional[str] = typer.Option(None, help="Project root"),
    delete: bool = typer.Option(False, "--delete", help="Delete instead of move"),
    db: Optional[str] = typer.Option(None, help="Extra SQLite database path"),
) -> None:
    """Move suspicious OPRA artifacts to quarantine/opra_raw_<timestamp>/."""
    from joker.config.loader import load_app_settings

    app_settings, _ = load_app_settings()
    base = Path(root) if root else Path(".")
    db_paths = discover_db_paths(base, app_settings.db_path)
    if db:
        extra = Path(db)
        if extra.exists():
            db_paths.append(extra.resolve())
    scan = scan_local_opra(root=base, db_paths=db_paths)
    if not scan.violations:
        console.print("[green]Nothing to quarantine.[/green]")
        raise typer.Exit(code=0)

    dest = quarantine_opra_artifacts(scan, root=base, delete=delete)
    action = "Deleted" if delete else "Quarantined"
    console.print(f"[yellow]{action} artifacts under {dest}[/yellow]")
    console.print("Review manifest.json before permanent deletion.")
