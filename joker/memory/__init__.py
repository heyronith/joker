"""Day memory for agent context — prior outcomes only, no raw OPRA."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from joker.schemas.domain import DayMemoryBundle, SessionLesson
from joker.storage.database import Database
from joker.storage.models import AgentDecisionRecord, RiskDecisionRecord


def memory_dir(data_dir: Path) -> Path:
    path = Path(data_dir) / "memory"
    path.mkdir(parents=True, exist_ok=True)
    return path


def lessons_path(data_dir: Path) -> Path:
    return memory_dir(data_dir) / "session_lessons.jsonl"


def save_session_lesson(data_dir: Path, lesson: SessionLesson) -> Path:
    path = lessons_path(data_dir)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(lesson.model_dump_json() + "\n")
    return path


def load_session_lessons(
    data_dir: Path,
    *,
    lookback_days: int = 5,
    as_of: date | None = None,
) -> list[SessionLesson]:
    path = lessons_path(data_dir)
    if not path.exists():
        return []
    as_of = as_of or date.today()
    cutoff = as_of - timedelta(days=max(lookback_days, 1))
    lessons: list[SessionLesson] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            lesson = SessionLesson.model_validate_json(line)
        except Exception:
            continue
        if lesson.trading_day < cutoff or lesson.trading_day >= as_of:
            continue
        lessons.append(lesson)
    lessons.sort(key=lambda x: x.trading_day)
    return lessons[-lookback_days:]


def build_day_memory(
    *,
    data_dir: Path,
    db: Database | None = None,
    as_of: date | None = None,
    lookback_days: int = 5,
) -> DayMemoryBundle:
    """Assemble compact prior-session memory for agent prompts."""
    as_of = as_of or date.today()
    lessons = load_session_lessons(data_dir, lookback_days=lookback_days, as_of=as_of)

    recent_pnl = sum(l.final_pnl_usd for l in lessons)
    recent_trades = sum(l.trades_entered for l in lessons)
    reject_codes: list[str] = []
    titles: list[str] = []

    for lesson in lessons:
        reject_codes.extend(lesson.risk_notes[:5])
        if lesson.summary:
            titles.append(lesson.summary[:80])

    if db is not None:
        try:
            with db.session() as session:
                from sqlmodel import select

                stmt = (
                    select(RiskDecisionRecord)
                    .where(RiskDecisionRecord.approved == False)  # noqa: E712
                    .order_by(RiskDecisionRecord.created_at.desc())
                    .limit(20)
                )
                rows = list(session.exec(stmt).all())
                for row in rows:
                    for code in row.reason_codes or []:
                        if code not in reject_codes:
                            reject_codes.append(str(code))
                reject_codes = reject_codes[:15]
        except Exception:
            pass

        try:
            with db.session() as session:
                from sqlmodel import select

                stmt = (
                    select(AgentDecisionRecord)
                    .where(AgentDecisionRecord.decision_type == "premarket")
                    .order_by(AgentDecisionRecord.created_at.desc())
                    .limit(5)
                )
                for row in session.exec(stmt).all():
                    payload = row.payload or {}
                    syn = payload.get("synthesis_summary") or payload.get("summary")
                    if syn and syn not in titles:
                        titles.append(str(syn)[:80])
        except Exception:
            pass

    return DayMemoryBundle(
        as_of=as_of,
        prior_lessons=lessons,
        recent_pnl_usd=recent_pnl,
        recent_trade_count=recent_trades,
        recent_risk_reject_codes=reject_codes[:15],
        recent_playbook_titles=titles[:5],
        memory_available=bool(lessons) or bool(reject_codes) or bool(titles),
    )


def memory_prompt_dict(bundle: DayMemoryBundle) -> dict:
    """Sanitize-friendly dict for LLM context (no OPRA)."""
    return {
        "as_of": bundle.as_of.isoformat(),
        "memory_available": bundle.memory_available,
        "recent_pnl_usd": bundle.recent_pnl_usd,
        "recent_trade_count": bundle.recent_trade_count,
        "recent_risk_reject_codes": bundle.recent_risk_reject_codes,
        "recent_playbook_titles": bundle.recent_playbook_titles,
        "prior_lessons": [
            {
                "trading_day": l.trading_day.isoformat(),
                "summary": l.summary,
                "what_worked": l.what_worked,
                "what_failed": l.what_failed,
                "risk_notes": l.risk_notes,
                "next_day_hints": l.next_day_hints,
                "final_pnl_usd": l.final_pnl_usd,
                "trades_entered": l.trades_entered,
            }
            for l in bundle.prior_lessons
        ],
    }
