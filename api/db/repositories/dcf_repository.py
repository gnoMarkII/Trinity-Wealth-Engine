"""DCF Evaluations Ledger SQLite DAO repository."""
import json
import sqlite3
import time
from typing import Optional, List, Dict


def record_dcf_evaluation(
    conn: sqlite3.Connection,
    evaluation_id: str,
    ticker: str,
    market: str,
    evaluated_at: str,
    scenarios: dict,
    model_version: str = "dcf_v1.0",
    valuation_price_basis: str = "split_adjusted_only",
    current_price_at_eval: float | None = None,
    wacc_pct: float | None = None,
    valuation_verdict: str = "unknown",
    corporate_action_evidence: list | None = None,
    input_snapshot: dict | None = None,
) -> None:
    """บันทึกการประเมิน DCF ลง Immutable Ledger แบบ Canonical Record"""
    now = time.time()
    conn.execute(
        """
        INSERT OR REPLACE INTO dcf_evaluations_ledger (
            evaluation_id, ticker, market, evaluated_at, model_version,
            valuation_price_basis, current_price_at_eval, wacc_pct,
            valuation_verdict, scenarios_json, corporate_action_evidence_json,
            input_snapshot_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            evaluation_id,
            ticker.upper(),
            market.upper(),
            evaluated_at,
            model_version,
            valuation_price_basis,
            current_price_at_eval,
            wacc_pct,
            valuation_verdict,
            json.dumps(scenarios),
            json.dumps(corporate_action_evidence or []),
            json.dumps(input_snapshot or {}),
            now,
        ),
    )
    conn.commit()


def get_latest_dcf_evaluation(conn: sqlite3.Connection, ticker: str) -> sqlite3.Row | None:
    """ดึงผลการประเมิน DCF ล่าสุดสำหรับ ticker จาก Canonical Ledger"""
    cur = conn.execute(
        "SELECT * FROM dcf_evaluations_ledger WHERE ticker = ? ORDER BY evaluated_at DESC, created_at DESC LIMIT 1",
        (ticker.upper(),),
    )
    return cur.fetchone()


def get_dcf_evaluation_by_id(conn: sqlite3.Connection, evaluation_id: str) -> sqlite3.Row | None:
    """ดึงผลการประเมิน DCF ตาม evaluation_id ที่เจาะจง"""
    cur = conn.execute(
        "SELECT * FROM dcf_evaluations_ledger WHERE evaluation_id = ?",
        (evaluation_id,),
    )
    return cur.fetchone()
