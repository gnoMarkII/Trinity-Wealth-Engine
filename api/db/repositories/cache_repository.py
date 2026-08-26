"""Analyst Context and Financial Statements Caches SQLite DAO repository."""
import json
import sqlite3
from typing import Optional, List, Dict
from datetime import datetime, timezone


def get_analyst_context_cache(conn: sqlite3.Connection, ticker: str) -> dict | None:
    """ดึงข้อมูล Analyst Context จาก SQLite cache พร้อม decode JSON และ format synced_at เป็น ISO string"""
    row = conn.execute(
        "SELECT * FROM analyst_context_cache WHERE ticker = ?", (ticker.upper(),)
    ).fetchone()
    if not row:
        return None
    d = dict(row)
    try:
        raw_json = d.pop("eps_history_json", "[]") or "[]"
        decoded = json.loads(raw_json)
        if not isinstance(decoded, list):
            return None  # Invalid JSON structure -> cache miss
        d["earnings_history"] = decoded
    except Exception:
        return None  # Corrupted JSON -> cache miss
    d["synced_at"] = datetime.fromtimestamp(d["synced_at"], tz=timezone.utc).isoformat()
    return d


def upsert_analyst_context_cache(conn: sqlite3.Connection, ticker: str, data: dict) -> None:
    """บันทึกข้อมูล Analyst Context ลง SQLite cache เฉพาะสถานะ ok และ partial"""
    conn.execute(
        """
        INSERT INTO analyst_context_cache (
            ticker, provider_symbol, market, currency, exchange_tz,
            target_mean, target_high, target_low, num_analysts,
            next_earnings_date, eps_history_json, source_as_of,
            data_status, synced_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker) DO UPDATE SET
            provider_symbol=excluded.provider_symbol,
            market=excluded.market,
            currency=excluded.currency,
            exchange_tz=excluded.exchange_tz,
            target_mean=excluded.target_mean,
            target_high=excluded.target_high,
            target_low=excluded.target_low,
            num_analysts=excluded.num_analysts,
            next_earnings_date=excluded.next_earnings_date,
            eps_history_json=excluded.eps_history_json,
            source_as_of=excluded.source_as_of,
            data_status=excluded.data_status,
            synced_at=excluded.synced_at
        """,
        (
            ticker.upper(),
            data["provider_symbol"],
            data["market"],
            data["currency"],
            data["exchange_tz"],
            data.get("target_mean"),
            data.get("target_high"),
            data.get("target_low"),
            data.get("num_analysts"),
            data.get("next_earnings_date"),
            json.dumps(data.get("earnings_history", [])),
            data.get("source_as_of"),
            data["data_status"],
            data["synced_at"],
        ),
    )


def get_financial_statements_cache(
    conn: sqlite3.Connection, market: str, provider_symbol: str
) -> dict | None:
    """ดึงข้อมูลงบการเงิน V5 จาก SQLite cache พร้อม decode JSON และ backward compatibility กับ V4"""
    row = conn.execute(
        "SELECT * FROM financial_statements_cache WHERE market = ? AND provider_symbol = ?",
        (market.upper(), provider_symbol.upper()),
    ).fetchone()
    if not row:
        return None
    d = dict(row)
    try:
        raw_json = d.get("data_json", "{}") or "{}"
        decoded = json.loads(raw_json)
        if not isinstance(decoded, dict):
            # Non-dict JSON -> Corrupted row
            conn.execute(
                "DELETE FROM financial_statements_cache WHERE market = ? AND provider_symbol = ?",
                (market.upper(), provider_symbol.upper()),
            )
            return None

        # Support Schema V6 (Do not reuse V5/V4 caches with potentially faulty FCF fields)
        s_ver = decoded.get("schema_version")
        if s_ver != 6:
            return None

        decoded["synced_at"] = datetime.fromtimestamp(d["synced_at"], tz=timezone.utc).isoformat()
        decoded["_raw_synced_at"] = d["synced_at"]
        return decoded
    except Exception:
        # Corrupted JSON string -> Auto-heal by deleting
        try:
            conn.execute(
                "DELETE FROM financial_statements_cache WHERE market = ? AND provider_symbol = ?",
                (market.upper(), provider_symbol.upper()),
            )
        except Exception:
            pass
        return None


def get_raw_financial_statements_cache(
    conn: sqlite3.Connection, market: str, provider_symbol: str
) -> dict | None:
    """ดึงข้อมูล cache ดิบทั้งหมด (รวม legacy schema) โดยไม่ลบแถว เพื่อใช้สำหรับ fallback เมื่อจำเป็น"""
    row = conn.execute(
        "SELECT * FROM financial_statements_cache WHERE market = ? AND provider_symbol = ?",
        (market.upper(), provider_symbol.upper()),
    ).fetchone()
    if not row:
        return None
    d = dict(row)
    try:
        raw_json = d.get("data_json", "{}") or "{}"
        decoded = json.loads(raw_json)
        if isinstance(decoded, dict):
            decoded["synced_at"] = datetime.fromtimestamp(d["synced_at"], tz=timezone.utc).isoformat()
            decoded["_raw_synced_at"] = d["synced_at"]
            return decoded
    except Exception:
        pass
    return None


def upsert_financial_statements_cache(
    conn: sqlite3.Connection,
    market: str,
    provider_symbol: str,
    provider: str,
    data_json: str,
    synced_at: float,
) -> None:
    """บันทึกข้อมูลงบการเงินลง SQLite cache แบบ (market, provider_symbol)"""
    conn.execute(
        """
        INSERT INTO financial_statements_cache (
            market, provider_symbol, provider, data_json, synced_at
        ) VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(market, provider_symbol) DO UPDATE SET
            provider=excluded.provider,
            data_json=excluded.data_json,
            synced_at=excluded.synced_at
        """,
        (market.upper(), provider_symbol.upper(), provider, data_json, synced_at),
    )
