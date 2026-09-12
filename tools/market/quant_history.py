"""Vault-native Forward-tracking history สำหรับ Equity Quant Signals

มิเรอร์ tools/macro/baselines.py (เก็บ snapshot เป็น Markdown ในตัว Vault เอง ไม่ใช่ DB แยก —
Single Source of Truth ตามกฎ 5.3) แต่ต่างจาก baselines.py ตรงที่เขียนไฟล์แบบ atomic (os.replace)
ตามกฎ 5.1 อย่างเคร่งครัด

หมายเหตุสำคัญ: นี่คือ Forward-tracking (สะสมคะแนนไปข้างหน้าตั้งแต่วันที่ระบบเริ่มบันทึก)
ไม่ใช่ Backtest ย้อนอดีต — yfinance ไม่มี point-in-time fundamentals ให้ backtest จริงได้
"""
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.logger import get_logger
from schemas.micro_quant_schemas import QuantSignals
from application.knowledge.note_write_ports import KnowledgeNoteWritePort
from application.knowledge.write_context import current_note_writer
from tools._atomic_io import _atomic_write_to as _shared_atomic_write_to

log = get_logger(__name__)

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _history_dir(ticker: str) -> Path:
    from tools.archivist.vault_paths import VaultPaths
    vp = VaultPaths()
    t = ticker.upper()
    v2_dir = vp.root / "30_Knowledge_Base" / "Stocks" / t / "Quant"
    if vp.layout_version >= 2 or v2_dir.exists():
        return v2_dir
    legacy_dir = vp.root / "30_Knowledge_Base" / "Equities" / "QuantHistory" / t
    return legacy_dir


def _atomic_write_text(path: Path, content: str, max_retries: int = 8, backoff: float = 0.05) -> None:
    """Compatibility alias; delegate to the shared atomic writer."""
    _shared_atomic_write_to(path, content)


def save_equity_quant_snapshot(signals: QuantSignals, *, note_writer: KnowledgeNoteWritePort | None = None) -> None:
    """บันทึก QuantSignals เป็นไฟล์ Markdown แยกต่อ ticker+วัน (re-run วันเดียวกัน = overwrite ไฟล์เดิม)

    ไม่ raise exception ออกไป — เป็น side-effect เสริมของ equity_intel pipeline เท่านั้น
    ถ้าเขียนไม่สำเร็จให้ log แล้วปล่อยให้ pipeline หลักทำงานต่อได้ตามปกติ (ไม่ block การวิเคราะห์)
    """
    try:
        note_writer = note_writer or current_note_writer(Path(os.getenv("OBSIDIAN_VAULT_PATH", "./memories")).resolve())
        date_str = signals.evaluated_at[:10]
        ticker_upper = signals.ticker.upper()
        file_path = _history_dir(ticker_upper) / f"{ticker_upper}_{date_str}.md"

        json_content = json.dumps(signals.model_dump(mode="json"), ensure_ascii=False, indent=2)

        def _fmt(v: Any) -> str:
            return "N/A" if v is None else str(v)

        body_content = f"""# Quant Signals: {ticker_upper} ({date_str})

| Metric | Value |
|---|---|
| **Composite Score** | {_fmt(signals.composite_score)} |
| Value Score | {_fmt(signals.value_score)} |
| Quality Score | {_fmt(signals.quality_score)} |
| Growth Score | {_fmt(signals.growth_score)} |
| Momentum Score | {_fmt(signals.momentum_score)} |
| Dividend Score | {_fmt(signals.dividend_score)} |
| Solvency Score (Risk Gate, ไม่รวมใน Composite) | {_fmt(signals.solvency_score)} |
| Beta | {_fmt(signals.beta)} |
| Volatility % | {_fmt(signals.volatility_pct)} |
| MDD % | {_fmt(signals.mdd_pct)} |

```json
{json_content}
```
"""
        note_writer.write_note(
            metadata={
                "title": f"{ticker_upper} Quant Signals {date_str}",
                "entity_type": "equity_quant_snapshot",
                "ticker": ticker_upper,
                "market": signals.market,
                "date": date_str,
                "tags": ["equity_quant", signals.ticker.lower(), f"market_{signals.market.lower()}"],
            },
            body=body_content,
            filename=f"{ticker_upper}_{date_str}.md",
        )
    except Exception as e:
        log.warning("save_equity_quant_snapshot failed for %s (non-fatal): %s", signals.ticker, e)


def get_equity_score_trend(ticker: str, days: int = 90) -> list[dict[str, Any]]:
    """อ่าน Forward-tracking snapshot ย้อนหลังของ ticker จาก Vault — เรียงจากเก่าไปใหม่ตามชื่อไฟล์

    Returns:
        list[dict]: แต่ละ dict คือ QuantSignals.model_dump() ของวันนั้นๆ — [] ถ้าไม่มีประวัติ
    """
    history_dir = _history_dir(ticker)
    if not history_dir.exists():
        return []

    cutoff_date = datetime.now(timezone.utc).date()
    t_up = ticker.upper()
    candidate_files = list(history_dir.glob(f"{t_up}_*.md")) + list(history_dir.glob(f"*{t_up}*.md"))
    seen_paths: set[Path] = set()
    sorted_files = []
    for p in sorted(candidate_files, key=lambda x: x.name):
        if p not in seen_paths and p.is_file():
            seen_paths.add(p)
            sorted_files.append(p)

    results: list[dict[str, Any]] = []
    for f in sorted_files:
        try:
            content = f.read_text(encoding="utf-8")
            match = _JSON_BLOCK_RE.search(content)
            if not match:
                continue
            data = json.loads(match.group(1))
            evaluated_at = data.get("evaluated_at", "")
            if evaluated_at:
                snap_date = datetime.fromisoformat(evaluated_at.replace("Z", "+00:00")).date()
                if (cutoff_date - snap_date).days > days:
                    continue
            results.append(data)
        except Exception as e:
            log.warning("get_equity_score_trend: failed to parse %s: %s", f, e)
            continue

    return results
