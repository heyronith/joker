#!/usr/bin/env python3
"""Reproducible local verification with a SHA-tied evidence manifest.

GitHub Actions is optional (workflow_dispatch only) and is not an acceptance gate.
This command is the acceptance path: same safety/warning gates as CI, plus
SQLite integrity and a defined soak suite.

Usage (from repo root, with the target venv active)::

    python scripts/local_verify.py
    python scripts/local_verify.py --python .venv311/bin/python
    python scripts/local_verify.py --focused-runs 5 --full-runs 3 --soak-runs 3

Evidence is written under::

    artifacts/local-verify/<commit_sha>/
        manifest.json
        summary.txt
        logs/*.log
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

PYTEST_WARNINGS = [
    "-W",
    "error::ResourceWarning",
    "-W",
    "error::pytest.PytestUnhandledThreadExceptionWarning",
]

FOCUSED_TESTS = [
    "tests/evolution/test_episode_compiler.py",
    "tests/objectives/test_historical_outcomes.py",
    "tests/objectives/test_execution_repricing.py",
    "tests/integration/test_goal_driven_production_wiring.py",
    "tests/integration/test_goal_driven_live_runner.py",
]

SOAK_TESTS = [
    "tests/integration/test_task3_recovery_matrix.py",
    "tests/integration/test_goal_driven_live_runner.py",
    "tests/integration/test_kill_switch_blocks_webull_paper_entry.py",
    "tests/integration/test_goal_driven_full_graph.py::test_kill_switch_blocks_positive_ev_before_paper_submission",
]


@dataclass
class StepResult:
    name: str
    ok: bool
    seconds: float
    command: list[str]
    exit_code: int | None = None
    detail: dict = field(default_factory=dict)
    log_file: str | None = None


@dataclass
class EvidenceManifest:
    schema_version: str
    commit_sha: str
    git_dirty: bool
    git_status_porcelain: str
    started_at: str
    finished_at: str | None
    environment: dict
    steps: list[dict]
    sqlite_integrity: dict
    overall_ok: bool
    notes: list[str]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _run(
    cmd: list[str],
    *,
    cwd: Path,
    log_path: Path,
    env: dict[str, str] | None = None,
) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {' '.join(cmd)}\n\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return proc.returncode, time.perf_counter() - started


def _git(cmd: list[str], *, cwd: Path) -> str:
    out = subprocess.check_output(["git", *cmd], cwd=cwd, text=True)
    return out.strip()


def _environment(python: str) -> dict:
    py_info = subprocess.check_output(
        [
            python,
            "-c",
            "import platform,sqlite3,sys;"
            "print(sys.version);"
            "print(platform.python_implementation());"
            "print(sqlite3.sqlite_version);"
            "print(sys.executable)",
        ],
        text=True,
    ).strip().splitlines()
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": py_info[0] if py_info else "unknown",
        "python_implementation": py_info[1] if len(py_info) > 1 else "unknown",
        "sqlite_version": py_info[2] if len(py_info) > 2 else "unknown",
        "python_executable": py_info[3] if len(py_info) > 3 else python,
        "tz": os.environ.get("TZ", ""),
        "cwd": str(REPO_ROOT),
    }


def _sqlite_integrity_check() -> dict:
    """Migrate a fresh DB and require PRAGMA integrity_check == ok."""
    from joker.persistence.migrations import apply_task1_migrations

    with tempfile.TemporaryDirectory(prefix="joker-verify-sqlite-") as tmp:
        db_path = Path(tmp) / "verify.db"
        apply_task1_migrations(db_path)
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute(
                "INSERT INTO domain_events_seen (event_id, session_id, event_type) "
                "VALUES ('verify-1', 'verify-sess', 'verify')"
            )
            conn.commit()
            integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
            foreign_keys = conn.execute("PRAGMA foreign_key_check").fetchall()
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        finally:
            conn.close()

        required = {
            "ledger_events",
            "trading_episodes",
            "session_objective_definitions",
            "capital_reservations",
        }
        missing = sorted(required - tables)
        ok = integrity == "ok" and not foreign_keys and not missing
        return {
            "ok": ok,
            "integrity_check": integrity,
            "foreign_key_violations": len(foreign_keys),
            "missing_required_tables": missing,
            "table_count": len(tables),
            "db_path_ephemeral": True,
        }


def _parse_pytest_summary(log_path: Path) -> dict:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    # Prefer the last summary line like "830 passed in 120.5s"
    passed = failed = errors = skipped = None
    for line in reversed(text.splitlines()):
        if " passed" in line or " failed" in line or " error" in line:
            # crude but stable for pytest -q
            parts = line.replace(",", " ").split()
            counts: dict[str, int] = {}
            for i, tok in enumerate(parts):
                if tok.isdigit() and i + 1 < len(parts):
                    label = parts[i + 1]
                    if label in {"passed", "failed", "error", "errors", "skipped", "xfailed"}:
                        key = "errors" if label == "error" else label
                        counts[key] = int(tok)
            if counts:
                passed = counts.get("passed", 0)
                failed = counts.get("failed", 0)
                errors = counts.get("errors", 0)
                skipped = counts.get("skipped", 0)
                break
    return {
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
        "summary_line": next(
            (
                ln
                for ln in reversed(text.splitlines())
                if "passed" in ln or "failed" in ln
            ),
            None,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used for pytest/ruff (default: current)",
    )
    parser.add_argument("--focused-runs", type=int, default=5)
    parser.add_argument("--full-runs", type=int, default=3)
    parser.add_argument("--soak-runs", type=int, default=3)
    parser.add_argument(
        "--skip-full",
        action="store_true",
        help="Skip full-suite repeats (not for acceptance)",
    )
    parser.add_argument(
        "--skip-soak",
        action="store_true",
        help="Skip soak repeats (not for acceptance)",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow dirty working tree (manifest still records porcelain status)",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=REPO_ROOT / "artifacts" / "local-verify",
        help="Directory for evidence manifests",
    )
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    env = os.environ.copy()
    env.setdefault("TZ", "America/New_York")
    env.setdefault("OPENAI_API_KEY", "test-local-verify-key-not-real")
    env["PYTHONUNBUFFERED"] = "1"

    commit_sha = _git(["rev-parse", "HEAD"], cwd=REPO_ROOT)
    porcelain = _git(["status", "--porcelain"], cwd=REPO_ROOT)
    dirty = bool(porcelain)
    if dirty and not args.allow_dirty:
        print(
            "ERROR: working tree is dirty. Commit or stash, or pass --allow-dirty.\n"
            f"{porcelain}",
            file=sys.stderr,
        )
        return 2

    out_dir = args.out_root / commit_sha
    log_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    started_at = _utc_now()
    steps: list[StepResult] = []
    notes = [
        "GitHub Actions is not an acceptance requirement; this local run is.",
        "Warning gates match CI: ResourceWarning and PytestUnhandledThreadExceptionWarning as errors.",
    ]

    print(f"local_verify commit={commit_sha} dirty={dirty}")
    print(f"evidence_dir={out_dir}")

    # --- Ruff ---
    ruff_log = log_dir / "ruff.log"
    ruff_cmd = [args.python, "-m", "ruff", "check", "."]
    code, secs = _run(ruff_cmd, cwd=REPO_ROOT, log_path=ruff_log, env=env)
    steps.append(
        StepResult(
            name="ruff",
            ok=code == 0,
            seconds=secs,
            command=ruff_cmd,
            exit_code=code,
            log_file=str(ruff_log.relative_to(out_dir)),
        )
    )
    print(f"[{'ok' if code == 0 else 'FAIL'}] ruff ({secs:.1f}s)")

    # --- CLI smoke ---
    help_log = log_dir / "cli_help.log"
    help_cmd = [args.python, "-m", "joker", "--help"]
    code, secs = _run(help_cmd, cwd=REPO_ROOT, log_path=help_log, env=env)
    steps.append(
        StepResult(
            name="cli_help",
            ok=code == 0,
            seconds=secs,
            command=help_cmd,
            exit_code=code,
            log_file=str(help_log.relative_to(out_dir)),
        )
    )
    print(f"[{'ok' if code == 0 else 'FAIL'}] cli_help ({secs:.1f}s)")

    # --- SQLite integrity ---
    sqlite_started = time.perf_counter()
    try:
        sqlite_result = _sqlite_integrity_check()
    except Exception as exc:  # noqa: BLE001 — recorded in manifest
        sqlite_result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    steps.append(
        StepResult(
            name="sqlite_integrity",
            ok=bool(sqlite_result.get("ok")),
            seconds=time.perf_counter() - sqlite_started,
            command=["python", "-c", "apply_task1_migrations + PRAGMA integrity_check"],
            detail=sqlite_result,
        )
    )
    print(
        f"[{'ok' if sqlite_result.get('ok') else 'FAIL'}] "
        f"sqlite_integrity ({steps[-1].seconds:.1f}s)"
    )

    def _pytest_repeat(name: str, paths: list[str], runs: int) -> None:
        for i in range(1, runs + 1):
            log = log_dir / f"{name}_run{i}.log"
            cmd = [
                args.python,
                "-m",
                "pytest",
                "-q",
                *paths,
                *PYTEST_WARNINGS,
                "--tb=short",
            ]
            code, secs = _run(cmd, cwd=REPO_ROOT, log_path=log, env=env)
            summary = _parse_pytest_summary(log)
            ok = code == 0
            steps.append(
                StepResult(
                    name=f"{name}_run{i}",
                    ok=ok,
                    seconds=secs,
                    command=cmd,
                    exit_code=code,
                    detail=summary,
                    log_file=str(log.relative_to(out_dir)),
                )
            )
            print(
                f"[{'ok' if ok else 'FAIL'}] {name} run {i}/{runs} "
                f"({secs:.1f}s) {summary.get('summary_line') or ''}"
            )
            if not ok:
                break

    _pytest_repeat("focused", FOCUSED_TESTS, args.focused_runs)

    if not args.skip_full:
        _pytest_repeat("full", ["tests"], args.full_runs)
    else:
        notes.append("full suite skipped via --skip-full (not acceptance-grade)")

    if not args.skip_soak:
        _pytest_repeat("soak", SOAK_TESTS, args.soak_runs)
    else:
        notes.append("soak skipped via --skip-soak (not acceptance-grade)")

    finished_at = _utc_now()
    overall_ok = all(s.ok for s in steps) and bool(sqlite_result.get("ok"))

    # Re-check git status after runs (tests must not dirty the tree)
    porcelain_after = _git(["status", "--porcelain"], cwd=REPO_ROOT)
    if porcelain_after != porcelain:
        notes.append(
            "git status changed during verification; "
            f"before={porcelain!r} after={porcelain_after!r}"
        )
        overall_ok = False

    manifest = EvidenceManifest(
        schema_version="1.0.0",
        commit_sha=commit_sha,
        git_dirty=dirty,
        git_status_porcelain=porcelain_after,
        started_at=started_at,
        finished_at=finished_at,
        environment=_environment(args.python),
        steps=[asdict(s) for s in steps],
        sqlite_integrity=sqlite_result,
        overall_ok=overall_ok,
        notes=notes,
    )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary_lines = [
        f"local_verify overall={'PASS' if overall_ok else 'FAIL'}",
        f"commit_sha={commit_sha}",
        f"git_dirty={dirty}",
        f"python={manifest.environment.get('python_executable')}",
        f"python_version={manifest.environment.get('python_version')}",
        f"platform={manifest.environment.get('platform')}",
        f"sqlite_version={manifest.environment.get('sqlite_version')}",
        f"sqlite_integrity={sqlite_result.get('integrity_check')}",
        f"started_at={started_at}",
        f"finished_at={finished_at}",
        "",
        "steps:",
    ]
    for s in steps:
        summary_lines.append(
            f"  [{'ok' if s.ok else 'FAIL'}] {s.name} "
            f"{s.seconds:.1f}s exit={s.exit_code} "
            f"{s.detail.get('summary_line') or ''}".rstrip()
        )
    summary_lines.append("")
    summary_lines.append(f"manifest={manifest_path}")
    summary_text = "\n".join(summary_lines) + "\n"
    (out_dir / "summary.txt").write_text(summary_text, encoding="utf-8")
    # Also pin a "latest" pointer for convenience (not SHA-specific acceptance).
    latest = args.out_root / "LATEST"
    latest.write_text(f"{commit_sha}\n{manifest_path}\n", encoding="utf-8")

    print()
    print(summary_text)
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
