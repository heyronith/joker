"""Scan local artifacts for persisted raw OPRA-like fields."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from joker.compliance.data_classification import DataClassification, classify_market_event

OPRA_VALUE_FIELDS = frozenset(
    {
        "bid",
        "ask",
        "mid",
        "last",
        "spread_pct",
        "volume",
        "open_interest",
        "implied_volatility",
        "delta",
        "gamma",
        "theta",
        "vega",
        "quote_timestamp",
    }
)

VIOLATION_VALUE_PATTERN = re.compile(
    r'"(' + "|".join(sorted(OPRA_VALUE_FIELDS)) + r')"\s*:\s*[-+]?\d',
    re.IGNORECASE,
)

SCAN_DIRS = ("data/captures", "logs", "reports", "data/replays")

SCAN_CATEGORIES = (
    "possible_raw_opra",
    "stock_data_not_opra",
    "synthetic_ignored",
    "safe_metadata",
)


@dataclass
class OpraScanFinding:
    path: str
    category: str
    reason: str
    line_number: int | None = None


@dataclass
class OpraScanResult:
    files_scanned: int = 0
    findings: list[OpraScanFinding] = field(default_factory=list)
    scanned_paths: list[str] = field(default_factory=list)
    db_paths_scanned: list[str] = field(default_factory=list)

    @property
    def violations(self) -> list[OpraScanFinding]:
        return [f for f in self.findings if f.category == "possible_raw_opra"]

    def by_category(self) -> dict[str, list[OpraScanFinding]]:
        grouped: dict[str, list[OpraScanFinding]] = {c: [] for c in SCAN_CATEGORIES}
        for finding in self.findings:
            grouped.setdefault(finding.category, []).append(finding)
        return grouped

    def recommended_quarantine_commands(self) -> list[str]:
        cmds = ["joker compliance quarantine-opra-artifacts"]
        if self.violations:
            cmds.append("# Review quarantine/opra_raw_* before deleting")
            cmds.append("joker compliance quarantine-opra-artifacts --delete  # optional")
        return cmds


def discover_db_paths(root: Path, config_db: Path | None = None) -> list[Path]:
    """Discover local SQLite databases for compliance scanning."""
    discovered: list[Path] = []
    if config_db and config_db.exists():
        discovered.append(config_db.resolve())
    data_dir = root / "data"
    if data_dir.exists():
        discovered.extend(data_dir.glob("*.db"))
        discovered.extend(data_dir.glob("*.sqlite"))
        discovered.extend(data_dir.rglob("*.db"))
        discovered.extend(data_dir.rglob("*.sqlite"))
    discovered.extend(root.glob("joker*.db"))
    return sorted({p.resolve() for p in discovered if p.is_file()})


def _parse_line_object(line: str) -> dict | None:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _classify_line(path: Path, line: str, line_number: int) -> OpraScanFinding | None:
    lower = line.lower()
    if line.strip().startswith("#"):
        return None

    obj = _parse_line_object(line)
    if obj is not None:
        classification = classify_market_event(obj)
        if classification == DataClassification.SYNTHETIC_DATA or obj.get("is_synthetic") is True:
            return OpraScanFinding(
                path=str(path),
                category="synthetic_ignored",
                reason="synthetic labeled data",
                line_number=line_number,
            )
        if classification == DataClassification.STOCK_MARKET_DATA:
            if VIOLATION_VALUE_PATTERN.search(line):
                return OpraScanFinding(
                    path=str(path),
                    category="stock_data_not_opra",
                    reason="stock market data (not OPRA)",
                    line_number=line_number,
                )
            return None
        if classification == DataClassification.NON_PRICE_DECISION_METADATA:
            return OpraScanFinding(
                path=str(path),
                category="safe_metadata",
                reason="non-price OPRA metadata",
                line_number=line_number,
            )
        if classification in (DataClassification.RAW_OPRA, DataClassification.DERIVED_OPRA_PRICE):
            if VIOLATION_VALUE_PATTERN.search(line):
                return OpraScanFinding(
                    path=str(path),
                    category="possible_raw_opra",
                    reason="raw OPRA option values",
                    line_number=line_number,
                )

    source = lower
    is_opra_context = (
        '"source": "webull_opra"' in lower
        or '"data_classification": "raw_opra"' in lower
        or "option.snapshot" in lower
        or '"event_type": "option_quote"' in lower
    )
    is_stock_context = '"source": "webull_stock"' in lower or '"event_type": "spy_quote"' in lower

    if not VIOLATION_VALUE_PATTERN.search(line):
        if "field_types" in lower or "top_level_keys" in lower or "presence" in lower:
            if "option_snapshot" in lower or "spread_check" in lower:
                return OpraScanFinding(
                    path=str(path),
                    category="safe_metadata",
                    reason="shape/presence metadata only",
                    line_number=line_number,
                )
        if "spread_check" in lower or "bid_ask_available" in lower:
            return OpraScanFinding(
                path=str(path),
                category="safe_metadata",
                reason="pass/fail metadata only",
                line_number=line_number,
            )
        return None

    if is_stock_context and not is_opra_context:
        return OpraScanFinding(
            path=str(path),
            category="stock_data_not_opra",
            reason="stock bid/ask (not OPRA)",
            line_number=line_number,
        )

    if '"is_synthetic": true' in lower or "synthetic_option" in source or "synthetic_stock" in source:
        return OpraScanFinding(
            path=str(path),
            category="synthetic_ignored",
            reason="synthetic replay data",
            line_number=line_number,
        )

    if is_opra_context or (
        VIOLATION_VALUE_PATTERN.search(line)
        and ("option" in lower or "webull_opra" in lower)
        and "webull_stock" not in lower
    ):
        return OpraScanFinding(
            path=str(path),
            category="possible_raw_opra",
            reason="possible raw OPRA field value",
            line_number=line_number,
        )

    if '"bid": true' in lower or '"ask": true' in lower:
        return OpraScanFinding(
            path=str(path),
            category="safe_metadata",
            reason="field availability boolean",
            line_number=line_number,
        )

    return None


def _scan_text(path: Path, text: str) -> list[OpraScanFinding]:
    findings: list[OpraScanFinding] = []
    for idx, line in enumerate(text.splitlines(), start=1):
        finding = _classify_line(path, line, idx)
        if finding:
            findings.append(finding)
    return findings


def _scan_file(path: Path) -> list[OpraScanFinding]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    return _scan_text(path, text)


def _scan_sqlite(db_path: Path) -> list[OpraScanFinding]:
    hits: list[OpraScanFinding] = []
    if not db_path.exists():
        return hits
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        tables = [row[0] for row in cur.fetchall()]
        for table in tables:
            try:
                rows = conn.execute(f"SELECT * FROM {table} LIMIT 200").fetchall()
            except sqlite3.Error:
                continue
            blob = json.dumps(rows, default=str)
            hits.extend(_scan_text(db_path, blob))
        conn.close()
    except sqlite3.Error:
        pass
    return hits


def scan_local_opra(
    *,
    root: Path | None = None,
    extra_paths: Iterable[Path] | None = None,
    db_paths: Iterable[Path] | None = None,
) -> OpraScanResult:
    base = root or Path(".")
    result = OpraScanResult()
    paths: list[Path] = []
    for rel in SCAN_DIRS:
        directory = base / rel
        if directory.exists():
            paths.extend(directory.rglob("*"))

    if extra_paths:
        paths.extend(extra_paths)

    for db_path in db_paths or []:
        if db_path.exists():
            result.findings.extend(_scan_sqlite(db_path))
            result.files_scanned += 1
            result.db_paths_scanned.append(str(db_path))
            result.scanned_paths.append(str(db_path))

    for path in sorted(set(paths)):
        if not path.is_file():
            continue
        if path.suffix.lower() not in (".jsonl", ".json", ".md", ".log", ".txt"):
            continue
        result.files_scanned += 1
        result.scanned_paths.append(str(path))
        result.findings.extend(_scan_file(path))

    return result


def quarantine_opra_artifacts(
    scan: OpraScanResult,
    *,
    root: Path | None = None,
    delete: bool = False,
) -> Path:
    base = root or Path(".")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    quarantine_dir = base / "quarantine" / f"opra_raw_{ts}"
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    moved: set[str] = set()
    for violation in scan.violations:
        src = Path(violation.path)
        if not src.exists() or str(src) in moved:
            continue
        dest = quarantine_dir / src.name
        if delete:
            src.unlink()
        else:
            src.rename(dest)
        moved.add(str(src))
    manifest = {
        "quarantined_at": datetime.now(timezone.utc).isoformat(),
        "delete_mode": delete,
        "files": sorted(moved),
        "violation_count": len(scan.violations),
    }
    (quarantine_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return quarantine_dir
