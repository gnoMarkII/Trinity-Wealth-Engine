"""Semantic & Quote-Anchored Guidance Validation Engine (Phase 3 & v3.1).

Binds earnings call transcripts to reported fiscal periods, extracts structured guidance,
and performs Two-Stage Quote Verification (verbatim text presence + parameter checks)
before feeding into the Deterministic Scorecard.
"""
import hashlib
import os
import re
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple

import yaml

from core.equity_analytics.canary_guard import scan_untrusted_text
from core.logger import get_logger
from schemas.micro_quant_schemas import (
    DataStatus,
    EarningsGuidanceContext,
    VerifiedGuidanceClaim,
)
from tools.archivist.core import VAULT_PATH

log = get_logger(__name__)

_MARGIN_TRAJECTORY_MAP = {
    "expand": "expanding",
    "expanding": "expanding",
    "expansion": "expanding",
    "grow": "expanding",
    "higher": "expanding",
    "increase": "expanding",
    "stable": "stable",
    "flat": "stable",
    "maintain": "stable",
    "contract": "contracting",
    "contracting": "contracting",
    "contraction": "contracting",
    "lower": "contracting",
    "decrease": "contracting",
    "compress": "contracting",
    "compression": "contracting",
}


def compute_quote_hash(quote_text: str) -> str:
    """Computes a SHA256 substring hash for source references."""
    normalized = " ".join(quote_text.strip().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def verify_quote_presence(quote_text: str, note_content: str) -> Tuple[bool, int, int]:
    """Verifies that quote exists verbatim inside the transcript/note content.

    Returns:
        tuple[bool, int, int]: (is_verified, start_char, end_char)
    """
    clean_quote = quote_text.strip()
    if not clean_quote:
        return False, -1, -1

    pos = note_content.find(clean_quote)
    if pos != -1:
        return True, pos, pos + len(clean_quote)

    # Secondary whitespace-tolerant search
    pattern = re.escape(clean_quote).replace(r"\ ", r"\s+")
    match = re.search(pattern, note_content)
    if match:
        return True, match.start(), match.end()

    return False, -1, -1


def _parse_iso_fiscal_quarter(period_str: Optional[str]) -> Optional[Tuple[str, str]]:
    """Map 'YYYY-MM-DD' to calendar fiscal ('YYYY', 'Qn') when possible."""
    if not period_str:
        return None
    m = re.match(r"^\s*(\d{4})-(\d{2})-(\d{2})", str(period_str))
    if not m:
        return None
    year, month = m.group(1), int(m.group(2))
    if not 1 <= month <= 12:
        return None
    quarter = f"Q{((month - 1) // 3) + 1}"
    return year, quarter


def _read_frontmatter_quick(path: Path) -> Tuple[dict[str, Any], str]:
    """Read file text and parse optional YAML frontmatter safely."""
    content = path.read_text(encoding="utf-8")
    metadata: dict[str, Any] = {}
    body = content
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            try:
                loaded = yaml.safe_load(parts[1])
                if isinstance(loaded, dict):
                    metadata = loaded
            except Exception:
                metadata = {}
            body = parts[2]
    return metadata, body


def _parse_range_pair(raw_val: Any, *, convert_usd_to_billions: bool = False) -> Optional[Tuple[float, float]]:
    if not isinstance(raw_val, (list, tuple)) or len(raw_val) < 2:
        return None
    try:
        vals = [float(v) for v in raw_val[:2]]
    except (ValueError, TypeError):
        return None
    if convert_usd_to_billions and max(abs(vals[0]), abs(vals[1])) >= 100_000.0:
        vals = [v / 1e9 for v in vals]
    return vals[0], vals[1]


def _parse_scalar_usd_b(raw_val: Any, *, convert_usd_to_billions: bool = False) -> Optional[float]:
    if raw_val is None:
        return None
    try:
        val = float(raw_val)
    except (ValueError, TypeError):
        return None
    if convert_usd_to_billions and abs(val) >= 100_000.0:
        val /= 1e9
    return val


def extract_verified_earnings_guidance(
    ticker: str,
    target_fiscal_period: Optional[str] = None,
    vault_base: Optional[Path] = None,
) -> Tuple[Optional[EarningsGuidanceContext], list[str]]:
    """Extracts and verifies earnings guidance for given ticker and fiscal period.

    Returns:
        tuple[Optional[EarningsGuidanceContext], list[str]]: (guidance_context, data_quality_flags)
    """
    flags: list[str] = []
    base_path = vault_base or VAULT_PATH
    san_ticker = ticker.strip().upper()

    search_dirs = [
        base_path / "30_Knowledge_Base" / "Earnings_Calls" / san_ticker,
        base_path / "30_Knowledge_Base" / "Stocks" / san_ticker / "Earnings",
    ]
    parsed_candidates: list[Tuple[Tuple[str, str, str], Path, dict[str, Any], str]] = []
    seen_paths: set[Path] = set()
    for target_dir in search_dirs:
        if not target_dir.exists():
            continue
        for md_path in target_dir.glob("*.md"):
            if md_path in seen_paths:
                continue
            seen_paths.add(md_path)
            if md_path.name.lower() == "index.md":
                continue
            try:
                meta, body_text = _read_frontmatter_quick(md_path)
            except Exception as e:
                log.warning("Failed to inspect earnings call note %s: %s", md_path, e)
                continue
            doc_role = str(meta.get("document_role") or "").strip().lower()
            entity_type = str(meta.get("entity_type") or "").strip().lower()
            if doc_role == "navigation" or entity_type == "navigation":
                continue
            sort_key = (
                str(meta.get("date") or meta.get("published_at") or ""),
                str(meta.get("period") or ""),
                md_path.name,
            )
            parsed_candidates.append((sort_key, md_path, meta, body_text))

    if not parsed_candidates:
        flags.append("missing_earnings_call_transcript")
        return None, flags

    parsed_candidates.sort(key=lambda item: item[0], reverse=True)

    selected_entry: Optional[Tuple[Tuple[str, str, str], Path, dict[str, Any], str]] = None
    is_lagging = False

    if target_fiscal_period:
        clean_p = target_fiscal_period.strip().upper()
        tokens = [t for t in re.split(r"[\s_\-]+", clean_p) if t]
        iso_fq = _parse_iso_fiscal_quarter(target_fiscal_period)
        for entry in parsed_candidates:
            _, f_path, f_meta, _ = entry
            fname_upper = f_path.name.upper()
            period_upper = str(f_meta.get("period") or "").upper()
            combined_tag = f"{fname_upper} {period_upper}"
            if tokens and (all(tok in fname_upper for tok in tokens) or all(tok in period_upper for tok in tokens)):
                selected_entry = entry
                break
            if iso_fq is not None:
                iso_year, iso_q = iso_fq
                if iso_year in combined_tag and iso_q in combined_tag:
                    selected_entry = entry
                    break

    if selected_entry is None:
        selected_entry = parsed_candidates[0]
        if target_fiscal_period:
            is_lagging = True
            flags.append(f"transcript_lags_reported_period:{target_fiscal_period}")

    _, selected_file, metadata, body = selected_entry

    # Guard untrusted external transcript text against prompt injection / delimiter escapes
    body_scan = scan_untrusted_text(body, raise_on_critical=False)
    if not body_scan.is_safe:
        flags.append("prompt_injection_detected:earnings_guidance")
        body = body_scan.sanitized_text

    period = str(metadata.get("period") or selected_file.stem.split("_")[0])
    rel_path = str(selected_file.relative_to(base_path)).replace("\\", "/")

    guidance_meta = metadata.get("guidance") if isinstance(metadata.get("guidance"), dict) else {}

    raw_tone = str(guidance_meta.get("management_tone") or "neutral").lower().strip()
    management_tone: Literal["bullish", "neutral", "cautious"] = "neutral"
    if "bull" in raw_tone or "optimist" in raw_tone:
        management_tone = "bullish"
    elif "caut" in raw_tone or "bear" in raw_tone or "concern" in raw_tone:
        management_tone = "cautious"

    # Quotes extraction & verification
    candidate_quotes = guidance_meta.get("key_executive_quotes") or []
    verified_quotes: list[str] = []
    verified_claims: list[VerifiedGuidanceClaim] = []

    for q in candidate_quotes:
        quote_str = str(q).strip()
        q_scan = scan_untrusted_text(quote_str, raise_on_critical=False)
        if not q_scan.is_safe:
            if "prompt_injection_quote_dropped:earnings_guidance" not in flags:
                flags.append("prompt_injection_quote_dropped:earnings_guidance")
            continue
        is_verified, start, end = verify_quote_presence(quote_str, body)
        if is_verified:
            q_hash = compute_quote_hash(quote_str)
            ref_str = f"{rel_path}#char_{start}_{end}:{q_hash}"
            verified_quotes.append(f'"{quote_str}" (ref: {ref_str})')
        else:
            flags.append("unverified_candidate_quote_dropped")

    # If no quotes in frontmatter, scan body for explicit quote callouts
    if not verified_quotes:
        for match in re.finditer(r'>\s*["“](.+?)["”]', body):
            q_text = match.group(1).strip()
            if len(q_text) >= 15:
                q_scan = scan_untrusted_text(q_text, raise_on_critical=False)
                if not q_scan.is_safe:
                    if "prompt_injection_quote_dropped:earnings_guidance" not in flags:
                        flags.append("prompt_injection_quote_dropped:earnings_guidance")
                    continue
                is_v, start, end = verify_quote_presence(q_text, body)
                if is_v:
                    q_hash = compute_quote_hash(q_text)
                    ref_str = f"{rel_path}#offset_{start}_{end}:{q_hash}"
                    verified_quotes.append(f'"{q_text}" (ref: {ref_str})')
                    if len(verified_quotes) >= 3:
                        break

    # Extract explicit Revenue Growth % bound to verified quote
    rev_growth: Optional[float] = None
    if guidance_meta.get("revenue_growth_guidance_pct") is not None and verified_quotes:
        try:
            rev_growth = float(guidance_meta["revenue_growth_guidance_pct"])
        except (ValueError, TypeError):
            rev_growth = None

    # Guidance Ranges & Baselines (STRICTLY from note YAML metadata — zero hardcoded ticker defaults)
    rev_range = _parse_range_pair(guidance_meta.get("revenue_guidance_range_usd_b")) or _parse_range_pair(
        guidance_meta.get("revenue_guidance_range_usd"), convert_usd_to_billions=True
    )
    billings_range = _parse_range_pair(guidance_meta.get("billings_guidance_range_usd_b")) or _parse_range_pair(
        guidance_meta.get("billings_guidance_range_usd"), convert_usd_to_billions=True
    )
    op_margin_range = _parse_range_pair(guidance_meta.get("non_gaap_operating_margin_range_pct"))

    fy_rev_b = (
        _parse_scalar_usd_b(guidance_meta.get("fy2025_audited_revenue_usd_b"))
        or _parse_scalar_usd_b(guidance_meta.get("prior_fy_revenue_usd_b"))
        or _parse_scalar_usd_b(guidance_meta.get("prior_fy_revenue_usd"), convert_usd_to_billions=True)
    )
    fy_billings_b = (
        _parse_scalar_usd_b(guidance_meta.get("fy2025_audited_billings_usd_b"))
        or _parse_scalar_usd_b(guidance_meta.get("prior_fy_billings_usd_b"))
        or _parse_scalar_usd_b(guidance_meta.get("prior_fy_billings_usd"), convert_usd_to_billions=True)
    )

    if rev_range is not None and fy_rev_b is not None and fy_rev_b > 0:
        rev_mid = (rev_range[0] + rev_range[1]) / 2.0
        computed_rev_growth = ((rev_mid - fy_rev_b) / fy_rev_b) * 100.0
        if rev_growth is None:
            rev_growth = round(computed_rev_growth, 1)

        primary_quote = verified_quotes[0] if verified_quotes else f"Transcript Guidance ({period})"
        verified_claims.append(
            VerifiedGuidanceClaim(
                metric_name="revenue_growth_guidance_pct",
                numeric_value=round(computed_rev_growth, 1),
                unit="%",
                denominator="YoY_vs_Prior_FY",
                fiscal_period=period,
                quote_text=primary_quote,
                quote_hash=compute_quote_hash(primary_quote),
                char_start=0,
                char_end=len(primary_quote),
                source_ref=f"vault:///{rel_path}",
            )
        )
    elif rev_growth is not None and verified_quotes:
        primary_quote = verified_quotes[0]
        verified_claims.append(
            VerifiedGuidanceClaim(
                metric_name="revenue_growth_guidance_pct",
                numeric_value=round(rev_growth, 1),
                unit="%",
                denominator="YoY_Management_Guidance",
                fiscal_period=period,
                quote_text=primary_quote,
                quote_hash=compute_quote_hash(primary_quote),
                char_start=0,
                char_end=len(primary_quote),
                source_ref=f"vault:///{rel_path}",
            )
        )

    if billings_range is not None and fy_billings_b is not None and fy_billings_b > 0:
        billings_mid = (billings_range[0] + billings_range[1]) / 2.0
        computed_billings_growth = ((billings_mid - fy_billings_b) / fy_billings_b) * 100.0
        primary_quote = verified_quotes[0] if verified_quotes else f"Transcript Billings Guidance ({period})"
        verified_claims.append(
            VerifiedGuidanceClaim(
                metric_name="billings_growth_guidance_pct",
                numeric_value=round(computed_billings_growth, 1),
                unit="%",
                denominator="YoY_vs_Prior_FY",
                fiscal_period=period,
                quote_text=primary_quote,
                quote_hash=compute_quote_hash(primary_quote),
                char_start=0,
                char_end=len(primary_quote),
                source_ref=f"vault:///{rel_path}",
            )
        )

    # Operating Margin Trajectory & Guidance (STRICTLY when explicitly present in YAML)
    margin_trajectory: Optional[Literal["expanding", "stable", "contracting", "unspecified"]] = None
    raw_margin_val = guidance_meta.get("operating_margin_trajectory") or guidance_meta.get("margin_guidance_direction")
    if raw_margin_val:
        raw_margin = str(raw_margin_val).lower().strip()
        if raw_margin in _MARGIN_TRAJECTORY_MAP:
            margin_trajectory = _MARGIN_TRAJECTORY_MAP[raw_margin]  # type: ignore

    if op_margin_range is not None:
        mid_margin = round((op_margin_range[0] + op_margin_range[1]) / 2.0, 2)
        margin_quote = (
            verified_quotes[1]
            if len(verified_quotes) > 1
            else (verified_quotes[0] if verified_quotes else f"Transcript Margin Guidance ({period})")
        )
        verified_claims.append(
            VerifiedGuidanceClaim(
                metric_name="non_gaap_operating_margin_midpoint_pct",
                numeric_value=mid_margin,
                unit="%",
                denominator=period,
                fiscal_period=period,
                quote_text=margin_quote,
                quote_hash=compute_quote_hash(margin_quote),
                char_start=0,
                char_end=len(margin_quote),
                source_ref=f"vault:///{rel_path}",
            )
        )
    elif guidance_meta.get("non_gaap_operating_margin_pct") is not None:
        try:
            scalar_margin = float(guidance_meta["non_gaap_operating_margin_pct"])
            margin_quote = verified_quotes[0] if verified_quotes else f"Transcript Margin Guidance ({period})"
            verified_claims.append(
                VerifiedGuidanceClaim(
                    metric_name="non_gaap_operating_margin_midpoint_pct",
                    numeric_value=round(scalar_margin, 2),
                    unit="%",
                    denominator=period,
                    fiscal_period=period,
                    quote_text=margin_quote,
                    quote_hash=compute_quote_hash(margin_quote),
                    char_start=0,
                    char_end=len(margin_quote),
                    source_ref=f"vault:///{rel_path}",
                )
            )
        except (ValueError, TypeError):
            pass

    status: DataStatus = "unavailable"
    if is_lagging and (rev_growth is not None or verified_claims or verified_quotes):
        status = "partial"
    elif rev_growth is not None or verified_claims:
        status = "available"
    elif verified_quotes:
        status = "partial"

    guidance_context = EarningsGuidanceContext(
        fiscal_quarter=period,
        revenue_growth_guidance_pct=rev_growth,
        operating_margin_trajectory=margin_trajectory,
        management_tone=management_tone,
        verified_claims=verified_claims,
        key_executive_quotes=verified_quotes,
        source_note_path=rel_path,
        status=status,
    )

    return guidance_context, flags
