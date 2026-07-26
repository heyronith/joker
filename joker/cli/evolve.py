"""Task 3 evolution CLI (paper-only; never enables live trading)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

import typer
from rich.console import Console

from joker.config.loader import load_app_settings
from joker.evolution.champion_registry import ChampionRegistry
from joker.evolution.config import EvolutionSettings
from joker.evolution.migrations import apply_task3_migrations
from joker.evolution.repositories import build_evolution_repositories
from joker.evolution.runtime import EvolutionRuntime, build_status_report

evolve_app = typer.Typer(help="Task 3 cognitive evolution (paper-only)")
console = Console()


def _db_path(config: str | None) -> Path:
    app, _env = load_app_settings(config_path=config)
    return Path(app.db_path)


def _settings(config: str | None) -> EvolutionSettings:
    app, _env = load_app_settings(config_path=config)
    return getattr(app, "evolution", EvolutionSettings())


def _emit(data: Any, as_json: bool) -> None:
    if as_json:
        typer.echo(json.dumps(data, default=str, indent=2, sort_keys=True))
    else:
        console.print(data)


@evolve_app.command("status")
def evolve_status(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Show evolution runtime / champion status."""

    async def _run() -> dict[str, Any]:
        settings = _settings(config)
        db = _db_path(config)
        apply_task3_migrations(db)
        runtime = EvolutionRuntime(db_path=db, settings=settings)
        if settings.enabled:
            await runtime.start()
            report = await build_status_report(runtime)
            await runtime.shutdown()
            return report
        registry = ChampionRegistry(db)
        await registry.initialize()
        champ = await registry.get_current_champion()
        await registry.close()
        return {
            "enabled": False,
            "champion": None if champ is None else champ.model_dump(mode="json"),
            "paper_only": True,
            "live_trading_enabled": False,
        }

    _emit(asyncio.run(_run()), as_json)


@evolve_app.command("champion")
def evolve_champion(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    async def _run() -> dict[str, Any]:
        db = _db_path(config)
        apply_task3_migrations(db)
        registry = ChampionRegistry(db)
        champ = await registry.bootstrap_champion()
        await registry.close()
        return {
            "configuration_version_id": str(champ.configuration_version_id),
            "content_hash": champ.content_hash,
            "status": champ.status,
            "paper_only": True,
        }

    _emit(asyncio.run(_run()), as_json)


@evolve_app.command("history")
def evolve_history(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    async def _run() -> list[dict[str, Any]]:
        db = _db_path(config)
        apply_task3_migrations(db)
        registry = ChampionRegistry(db)
        await registry.bootstrap_champion()
        history = await registry.compare_champion_history()
        await registry.close()
        return [h.model_dump(mode="json") for h in history]

    _emit(asyncio.run(_run()), as_json)


@evolve_app.command("drift")
def evolve_drift(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    async def _run() -> dict[str, Any]:
        db = _db_path(config)
        repos = build_evolution_repositories(db)
        await repos["drift"].initialize()
        registry = ChampionRegistry(db)
        champ = await registry.bootstrap_champion()
        rows = await repos["drift"].list_by_configuration(champ.configuration_version_id)
        await registry.close()
        return {
            "champion_version_id": str(champ.configuration_version_id),
            "observations": [r.model_dump(mode="json") for r in rows],
        }

    _emit(asyncio.run(_run()), as_json)


@evolve_app.command("episodes")
def evolve_episodes(
    action: str = typer.Argument("inspect"),
    episode_id: Optional[str] = typer.Argument(None),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """episodes build|inspect <id>."""

    async def _run() -> Any:
        db = _db_path(config)
        repos = build_evolution_repositories(db)
        await repos["episodes"].initialize()
        if action == "build":
            return {"status": "ok", "note": "use runtime workers to compile episodes"}
        if action == "inspect":
            if not episode_id:
                raise typer.BadParameter("episode-id required")
            ep = await repos["episodes"].get_by_id(episode_id)
            return None if ep is None else ep.model_dump(mode="json")
        raise typer.BadParameter(f"unknown action {action}")

    _emit(asyncio.run(_run()), as_json)


@evolve_app.command("rollback")
def evolve_rollback(
    to_version: str = typer.Option(..., "--to"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    as_json: bool = typer.Option(False, "--json"),
    yes: bool = typer.Option(False, "--yes", help="Confirm champion-changing action"),
) -> None:
    """Human-requested rollback to a prior immutable configuration version."""

    async def _run() -> dict[str, Any]:
        db = _db_path(config)
        registry = ChampionRegistry(db)
        current = await registry.bootstrap_champion()
        target = await registry._configs.get_by_id(to_version)
        if target is None:
            raise typer.Exit(code=1)
        payload = {
            "current_champion": str(current.configuration_version_id),
            "current_hash": current.content_hash,
            "proposed_target": str(target.configuration_version_id),
            "target_hash": target.content_hash,
            "action": "rollback",
            "automatic": False,
            "human_requested": True,
            "paper_only": True,
        }
        if not yes:
            payload["status"] = "confirmation_required"
            return payload
        transition = await registry.rollback(
            restore_version_id=UUID(to_version),
            expected_champion_id=current.configuration_version_id,
            reason="rollback:human",
        )
        await registry.close()
        payload["status"] = "rolled_back"
        payload["transition"] = transition.model_dump(mode="json")
        return payload

    _emit(asyncio.run(_run()), as_json)


@evolve_app.command("experiments")
def evolve_experiments(
    action: str = typer.Argument("inspect"),
    experiment_id: Optional[str] = typer.Argument(None),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """experiments create|run|resume|inspect <experiment-id>."""

    async def _run() -> Any:
        db = _db_path(config)
        repos = build_evolution_repositories(db)
        await repos["experiments"].initialize()
        if action == "inspect":
            if not experiment_id:
                raise typer.BadParameter("experiment-id required")
            definition = await repos["experiments"].get_definition(experiment_id)
            result = await repos["experiments"].get_result(experiment_id)
            return {
                "definition": None
                if definition is None
                else definition.model_dump(mode="json"),
                "result": None if result is None else result.model_dump(mode="json"),
                "paper_only": True,
            }
        if action == "resume":
            pending = await repos["experiments"].list_resumable()
            return {
                "action": "resume",
                "experiment_id": experiment_id,
                "resumable": [p.model_dump(mode="json") for p in pending],
                "paper_only": True,
            }
        return {"action": action, "status": "use_programmatic_api_for_create_run"}

    _emit(asyncio.run(_run()), as_json)


@evolve_app.command("shadow")
def evolve_shadow(
    action: str = typer.Argument(...),
    challenger_id: Optional[str] = typer.Argument(None),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """shadow start|stop <challenger-id>."""

    async def _run() -> dict[str, Any]:
        db = _db_path(config)
        repos = build_evolution_repositories(db)
        await repos["shadow"].initialize()
        if action == "start":
            if not challenger_id:
                raise typer.BadParameter("challenger-id required")
            return {
                "action": "start",
                "challenger_id": challenger_id,
                "status": "registered_via_runtime",
                "paper_only": True,
                "broker_authority": "champion_only",
            }
        if action == "stop":
            if challenger_id:
                await repos["shadow"].mark_status(challenger_id, "stopped")
            return {"action": "stop", "challenger_id": challenger_id, "paper_only": True}
        raise typer.BadParameter(f"unknown action {action}")

    _emit(asyncio.run(_run()), as_json)


@evolve_app.command("promote")
def evolve_promote(
    experiment_id: str = typer.Argument(...),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    as_json: bool = typer.Option(False, "--json"),
    yes: bool = typer.Option(False, "--yes"),
) -> None:
    """Inspect promotion eligibility for an experiment (paper-only)."""

    async def _run() -> dict[str, Any]:
        db = _db_path(config)
        repos = build_evolution_repositories(db)
        await repos["experiments"].initialize()
        await repos["promotions"].initialize()
        registry = ChampionRegistry(db)
        current = await registry.bootstrap_champion()
        result = await repos["experiments"].get_result(experiment_id)
        decision = await repos["promotions"].get_by_experiment(experiment_id)
        payload = {
            "experiment_id": experiment_id,
            "current_champion": str(current.configuration_version_id),
            "current_hash": current.content_hash,
            "eligibility_outcome": None if result is None else result.eligibility_outcome,
            "gate_codes": None if result is None else result.gate_rejection_codes,
            "existing_decision": None
            if decision is None
            else decision.model_dump(mode="json"),
            "paper_only": True,
            "automatic": False,
            "confirmation_required": not yes,
        }
        await registry.close()
        return payload

    _emit(asyncio.run(_run()), as_json)


@evolve_app.command("reject")
def evolve_reject(
    experiment_id: str = typer.Argument(...),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    async def _run() -> dict[str, Any]:
        db = _db_path(config)
        repos = build_evolution_repositories(db)
        await repos["promotions"].initialize()
        decision = await repos["promotions"].get_by_experiment(experiment_id)
        return {
            "experiment_id": experiment_id,
            "decision": None if decision is None else decision.model_dump(mode="json"),
            "paper_only": True,
        }

    _emit(asyncio.run(_run()), as_json)
