from datetime import datetime
from enum import Enum
from typing import Literal, Optional, Any
from pydantic import BaseModel, Field, computed_field, field_validator, model_validator
import re
import logging

logger = logging.getLogger(__name__)
from schemas.warning_registry import (
    WarningMessage,
    DEFENSIVE_LOW_SUPPORTING_DATA,
    SINGLE_SOURCE_PENALTY,
    SOURCE_REF_PENALTY,
    ACTIVE_ALLOC_GUARDRAIL,
    PT_MANDATORY_FIELD_MISSING,
    PT_DEFENSIVE_LOW,
    PT_EXECUTION_GUARDRAIL,
    PT_RISK_BUDGET_HIGH_TO_MED,
    PT_RISK_BUDGET_HIGH_TO_SMALL,
    PT_RISK_BUDGET_MED_TO_SMALL,
    PT_GRACEFUL_DOWNGRADE,
    RS_DEFENSIVE_LOW,
    PORTFOLIO_DEFENSIVE_LOW,
    STALE_DATA_DEGRADATION,
    GRACEFUL_DROP_PAIR_TRADES,
    GRACEFUL_DROP_RISK_SCENARIOS,
    COVERAGE_BACKFILL_EXPANDED,
    COVERAGE_WARNING_INCOMPLETE,
    GOLD_RATIONALE_WARNING,
    GOLD_CONTRADICTION,
    US_EQUITY_CONTRADICTION,
    REGIME_CONTRADICTION,
    CONVICTION_CONTRADICTION,
    MISSING_ASSET_BUCKET,
    STATISTICAL_OVERCLAIM,
    ALLOCATION_DELTA_INVALID,
    FX_STANCE_MISMATCH,
    HALLUCINATED_ATTRIBUTION_CLEANED,
    SUPPORTING_DATA_MISMATCH,
    MISSING_OBSERVABLE_REFS,
)
from schemas.report_labels import DOWNGRADE_WARNING_IDS, WHY_NOT_HIGH_MESSAGES

class GeographicScope(str, Enum):
    GLOBAL = "Global"
    REGIONAL = "Regional"
    COUNTRY = "Country"

class Region(str, Enum):
    GLOBAL = "Global"
    # Regional
    US_REGIONAL = "US_Regional"
    EUROPE = "Europe"
    CHINA = "China"
    JAPAN = "Japan"
    INDIA = "India"
    LATAM = "Latin_America"
    # Country
    THAILAND = "Thailand"
    USA = "USA"

class EconomicIndicator(str, Enum):
    MONETARY_POLICY = "Monetary_Policy"
    ECONOMIC_GROWTH = "Economic_Growth"
    INFLATION = "Inflation"
    GEOPOLITICS = "Geopolitics"

class EconomicState(str, Enum):
    GOLDILOCKS = "Goldilocks"
    REFLATION = "Reflation"
    STAGFLATION = "Stagflation"
    RECESSION = "Recession"
    UNKNOWN = "Unknown"

class TrendDirection(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    FLAT = "FLAT"

class IndicatorComponent(BaseModel):
    symbol_or_id: str = Field(description="Ticker หรือ FRED Series ID หรือคีย์หลักของดัชนี")
    name: str = Field(description="ชื่อเรียกตัวแปรหรือดัชนี")
    value: float = Field(description="ค่าล่าสุดของตัวแปร")
    unit: str = Field(description="หน่วยของค่า เช่น %, % YoY, Index, USD")
    date: str = Field(description="วันที่ของตัวเลขล่าสุดที่มีรายงาน")
    change_pct: Optional[float] = Field(default=None, description="เปอร์เซ็นต์การเปลี่ยนแปลงเมื่อเทียบกับรอบก่อนหน้า (ถ้ามี)")

class CellMetrics(BaseModel):
    score: float = Field(description="คะแนนที่คำนวณและ Normalize ได้ ช่วง -1.0 ถึง 1.0 (เช่น Growth: -1.0 = หดตัวแรง, 1.0 = โตแรง)")
    trend: TrendDirection = Field(default=TrendDirection.FLAT, description="แนวโน้มระยะสั้น (UP, DOWN, FLAT)")
    status_label: str = Field(description="คำอธิบายสถานะย่อย เช่น 'Hawkish', 'Expansion', 'High Inflation'")
    components: list[IndicatorComponent] = Field(default_factory=list, description="รายการดัชนีย่อยที่ใช้ประกอบในเซลล์นี้")
    updated_at: datetime = Field(default_factory=datetime.now)

class RegionStateEvaluation(BaseModel):
    region: Region = Field(description="ภูมิภาคหรือประเทศที่ถูกประเมิน")
    scope: GeographicScope = Field(description="ระดับขอบเขตของข้อมูล (Global, Regional, Country)")
    evaluated_state: EconomicState = Field(description="ผลการระบุสภาวะเศรษฐกิจ")
    confidence_score: float = Field(description="คะแนนความเชื่อมั่นต่อการจัดกลุ่ม 0.0 - 1.0")
    recommended_assets: list[str] = Field(description="ประเภทสินทรัพย์เด่นที่แนะนำ")
    rationale: str = Field(description="สรุปคำอธิบาย/เหตุผลสนับสนุน")
    cells: dict[EconomicIndicator, CellMetrics] = Field(description="ผลวิเคราะห์แต่ละ Indicator สำหรับพื้นที่นี้")
    updated_at: datetime = Field(default_factory=datetime.now)

class MacroEconomicMatrix(BaseModel):
    evaluated_at: datetime = Field(default_factory=datetime.now)
    evaluations: dict[Region, RegionStateEvaluation] = Field(description="ผลประเมินของแต่ละภูมิภาค/ประเทศ")

# === Macro Intelligence Team Schemas ===

class RegionQuantMetrics(BaseModel):
    growth_score: float = Field(ge=-1.0, le=1.0, description="-1.0 ถึง 1.0")
    inflation_score: float = Field(ge=-1.0, le=1.0, description="-1.0 ถึง 1.0")
    monetary_score: float = Field(ge=-1.0, le=1.0, description="-1.0 ถึง 1.0")
    economic_state: EconomicState = Field(description="สภาวะเศรษฐกิจ")
    confidence: float = Field(ge=0.0, le=1.0, description="ความเชื่อมั่น 0.0 - 1.0")

class MarketObservable(BaseModel):
    observable_id: str = Field(description="Stable ID of observable")
    asset_bucket: Literal["equities", "fixed_income", "commodities", "fx", "cash", "risk"] = Field(description="Asset bucket category")
    region: str = Field(description="Region or country")
    indicator: str = Field(description="Indicator name")
    value: str = Field(description="Latest value")
    unit: str = Field(description="Unit of measurement")
    observed_at: str = Field(description="Date of observation YYYY-MM-DD")
    source_file: str = Field(description="Source filename")
    source_section: str = Field(default="", description="Source section heading")
    provider: str = Field(default="FRED", description="Data provider")
    confidence: Literal["high", "medium", "low"] = Field(default="high", description="Confidence level")
    is_valid: bool = Field(default=True, description="Validity flag")
    stale_reason: str = Field(default="", description="Reason if stale or invalid")
    observable_type: str = Field(default="macro", description="Type of observable: macro, valuation, derived_ratio, risk_correlation, leading_indicator")
    lookback_days: Optional[int] = Field(default=None, description="Number of historical lookback days for derived metrics")
    calculation_method: str = Field(default="", description="Calculation methodology for derived/valuation metrics")
    input_observable_ids: list[str] = Field(default_factory=list, description="List of input observable IDs used to derive this metric")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Structured metadata for machine-readable derived metrics and statistical values")

    @field_validator("observed_at")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        v = str(v).strip()
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
            raise ValueError(f"observed_at must be in YYYY-MM-DD format, got {v!r}")
        return v


class QuantScore(BaseModel):
    evaluated_at: str = Field(description="ISO format string (ไม่ใช่ datetime object)")
    regions: dict[str, RegionQuantMetrics] = Field(description="Key คือชื่อ Region")
    global_geopolitics_score: float = Field(ge=-1.0, le=1.0, description="-1.0 ถึง 1.0")
    recession_probability: float = Field(ge=0.0, le=1.0, description="0.0 - 1.0")
    data_freshness_note: str = Field(description="หมายเหตุความสดใหม่ของข้อมูล")
    market_observables: list[MarketObservable] = Field(default_factory=list, description="Structured market and macro observables with provenance")

class ThemeCategory(str, Enum):
    POLICY = "policy"
    GROWTH = "growth"
    INFLATION = "inflation"
    LIQUIDITY = "liquidity"
    GEOPOLITICS = "geopolitics"
    EARNINGS = "earnings"
    RISK_SENTIMENT = "risk_sentiment"

class PivotStrength(str, Enum):
    NONE = "none"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"

class MacroTheme(BaseModel):
    category: ThemeCategory
    theme_title: str
    deduplicated_summary: str
    age_hours: int = Field(ge=0)
    freshness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    freshness_reason: str = Field(default="")
    sources_count: int = Field(ge=0)

    asset_impacts: dict[str, Literal["bullish", "bearish", "neutral"]] = Field(default_factory=dict, description="ทิศทางต่อ Asset Classes เช่น {'equity': 'bullish', 'bond': 'bearish'}")
    market_impact_score: float = Field(default=0.0, ge=0.0, le=1.0, description="ขนาดแรงกระแทกต่อตลาด (0.0 - 1.0)")
    event_confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="ความแน่ใจในเหตุการณ์นี้ (0.0 - 1.0) จาก Python")

    pivot_strength: PivotStrength = PivotStrength.NONE
    changed_from: Optional[str] = None
    baseline_date: Optional[str] = Field(default=None, description="วันที่ของ Baseline (Short/Medium)")
    pivot_evidence: Optional[str] = Field(default=None, description="อธิบายหลักฐานเทียบกับ Explicit Baseline")

    validation_warnings: list[str] = Field(default_factory=list)

    @computed_field
    def investment_conviction_contribution(self) -> dict[str, float]:
        direction_map = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0}
        convictions = {}
        if not self.asset_impacts:
            return convictions
        for asset, direction in self.asset_impacts.items():
            dir_val = direction_map.get(direction, 0.0)
            convictions[asset] = dir_val * self.market_impact_score * self.event_confidence * self.freshness_score
        return convictions

    @model_validator(mode='after')
    def enforce_computed_and_pivot(self) -> 'MacroTheme':
        from core.nlp_utils import calculate_freshness, calculate_event_confidence

        if self.sources_count > 0:
            self.event_confidence = calculate_event_confidence(self.sources_count)

        if self.age_hours >= 0:
            if self.age_hours >= 9999:
                self.freshness_score = 0.0
                self.freshness_reason = "Missing or unparseable timestamp"
            else:
                score, reason = calculate_freshness(self.age_hours, self.category)
                self.freshness_score = score
                self.freshness_reason = reason

        if self.pivot_strength != PivotStrength.NONE:
            if not self.changed_from or not self.baseline_date or not self.pivot_evidence:
                self.pivot_strength = PivotStrength.NONE
                self.validation_warnings.append(
                    "Downgraded pivot_strength to NONE due to missing baseline evidence (changed_from, baseline_date, or pivot_evidence)"
                )
        return self

class NarrativeContext(BaseModel):
    evaluated_at: str = Field(description="ISO format string")
    dominant_themes: list[MacroTheme] = Field(description="ธีมหลักที่กำลังขับเคลื่อนตลาด")
    market_sentiment: Literal["bullish", "neutral", "bearish"] = Field(description="อารมณ์ตลาด")
    tail_risks: list[str] = Field(description="ความเสี่ยงที่ซ่อนอยู่")
    policy_signals: list[str] = Field(description="สัญญาณจากผู้กำหนดนโยบาย (Fed, BOT)")
    key_narratives_by_region: dict[str, str] = Field(description="สรุป narrative ของแต่ละภูมิภาค")
    sources_summary: str = Field(description="สรุปแหล่งที่มาของข้อมูล")

def _has_hard_data_numbers(data_list: list[str]) -> bool:
    """Minimal numeric evidence heuristic.

    Note: This is NOT a deep semantic validator. It serves as a fast structural screen
    to ensure the supporting text contains numbers coupled with actionable financial
    units or indicators (%, bps, USD, Index, Yield, etc.) rather than just dates or general text.
    In the long term, validation should prioritize checking for valid `observable_refs`.
    """
    if not data_list:
        return False
    num_pattern = re.compile(r'\d+([.,]\d+)?')
    fin_pattern = re.compile(
        r'(?:%|bps|pts?|points?|score|level|target|usd|thb|eur|jpy|index|yield|rate|cpi|ppi|pmi|'
        r'vix|dxy|spx|sp500|s&p|nasdaq|gold|oil|brent|wti|spread|ratio|curve|inversion|y|years?|ปี|p/e|eps|gdp|nfp|bn|mn|mb|gb|mil|bil|million|billion|'
        r'ล้าน|พันล้าน|บาท|baht|contracts?|สัญญา|shares?|หุ้น|oz|ออนซ์|barrels?|บาร์เรล|จุด|ดัชนี|ผลตอบแทน|อัตรา|\$|€|£|¥)',
        re.IGNORECASE
    )
    for item in data_list:
        if num_pattern.search(item) and fin_pattern.search(item):
            return True
    return False

def _normalize_time_horizon_str(val: Any) -> str:
    return str(val or "3-6 Months").strip()


def _add_warning_idempotent(warnings_list: list[str], warning: Any) -> None:
    warning_str = str(warning)
    if warning_str not in warnings_list:
        try:
            msg = WarningMessage.from_str(warning_str)
            if msg:
                for existing in warnings_list:
                    ex_msg = WarningMessage.from_str(existing)
                    if ex_msg and ex_msg.id == msg.id and ex_msg.params == msg.params:
                        return
        except Exception:
            pass
        warnings_list.append(warning_str)


def _valid_observables_for_refs(
    observable_refs: list[str],
    observable_registry: dict[str, "MarketObservable"],
) -> list["MarketObservable"]:
    valid = []
    for ref in observable_refs:
        obs = observable_registry.get(ref)
        if (
            obs
            and obs.is_valid
            and bool(str(obs.source_file).strip())
            and bool(str(obs.observed_at).strip())
        ):
            valid.append(obs)
    return valid


def _valid_source_files_for_refs(
    observable_refs: list[str],
    observable_registry: dict[str, "MarketObservable"],
) -> set[str]:
    return {
        obs.source_file
        for obs in _valid_observables_for_refs(observable_refs, observable_registry)
        if obs.source_file
    }


def _all_execution_fields_present(pair_trade: "PairTradeStrategy") -> bool:
    required = [
        pair_trade.instrument_proxy,
        pair_trade.hedge_ratio,
        pair_trade.stop_loss_trigger,
        pair_trade.target_gain_or_rebalance,
        pair_trade.max_drawdown_limit,
    ]
    return all(str(value).strip() for value in required)


def _has_pair_trade_execution_evidence(
    pair_trade: "PairTradeStrategy",
    observable_registry: dict[str, "MarketObservable"] | None = None,
) -> bool:
    if not _all_execution_fields_present(pair_trade):
        return False
    exec_values = [
        pair_trade.hedge_ratio,
        pair_trade.stop_loss_trigger,
        pair_trade.target_gain_or_rebalance,
        pair_trade.max_drawdown_limit,
    ] + pair_trade.supporting_data
    if not any(re.search(r'\d', str(val)) for val in exec_values):
        return False
    if bool(observable_registry):
        valid_count = len(_valid_observables_for_refs(pair_trade.observable_refs, observable_registry))
        if pair_trade.confidence in {"high", "medium"}:
            return valid_count >= 2
        return valid_count >= 1
    return bool(pair_trade.observable_refs)


def _refs_supported_by_source_files(
    observable_refs: list[str],
    observable_registry: dict[str, "MarketObservable"],
    source_files: list[str],
) -> bool:
    if not observable_refs or not observable_registry or not source_files:
        return False
    source_set = {str(src).strip() for src in source_files if str(src).strip()}
    for obs in _valid_observables_for_refs(observable_refs, observable_registry):
        if obs.source_file in source_set:
            return True
    return False


def _source_files_from_observable_refs(
    observable_refs: list[str],
    observable_registry: dict[str, "MarketObservable"],
) -> list[str]:
    files = []
    for obs in _valid_observables_for_refs(observable_refs, observable_registry):
        if obs.source_file and obs.source_file not in files:
            files.append(obs.source_file)
    return files



def _is_allowed_cross_bucket(target_bucket: str | None, obs: "MarketObservable") -> bool:
    if not target_bucket:
        return False
    if target_bucket == obs.asset_bucket:
        return True
    if target_bucket == "commodities" and obs.asset_bucket == "fixed_income":
        text = (obs.indicator + " " + obs.observable_id).lower()
        if any(k in text for k in ["credit", "spread", "high yield", "hyg", "lqd", "baa", "corporate"]):
            return False
        if any(k in text for k in ["yield", "tips", "dfii10", "rate", "treasury", "policy", "fed"]):
            return True
    if target_bucket == "commodities" and obs.asset_bucket == "fx":
        text = (obs.indicator + " " + obs.observable_id).lower()
        if any(k in text for k in ["dollar", "dtwexbgs", "dxy", "usd"]):
            return True
    return False




def _clear_resolved_hard_data_warnings(asset: "AssetAllocationView") -> None:
    if not _has_hard_data_numbers(asset.supporting_data):
        return
    asset.validation_warnings = [
        warning
        for warning in asset.validation_warnings
        if "lack of numeric hard data in supporting_data" not in warning
        and "missing numeric hard data" not in warning
        and "Auto-extracted hard data lacks valid observable_refs" not in warning
    ]




def _downgrade_pair_trade_statistical_overclaim(
    pair_trade: "PairTradeStrategy",
    observable_registry: dict[str, "MarketObservable"],
) -> None:
    text = " ".join([
        pair_trade.thesis,
        pair_trade.catalyst,
        pair_trade.risk,
        pair_trade.hedge_ratio,
        " ".join(pair_trade.supporting_data),
    ]).lower()
    statistical_terms = ["beta", "correlation", "tracking error"]
    if not any(term in text for term in statistical_terms):
        return
    valid_obs = _valid_observables_for_refs(pair_trade.observable_refs, observable_registry)
    obs_text = " ".join(f"{obs.indicator} {obs.source_section}" for obs in valid_obs).lower()
    has_single_point_relative = any(
        term in obs_text for term in ["spread", "differential", "ratio", "carry", "relative"]
    )
    has_historical = any(
        term in obs_text for term in ["historical", "time series", "rolling", "beta", "correlation", "tracking error"]
    )
    if has_single_point_relative and not has_historical:
        pair_trade.confidence = "medium"
        if pair_trade.sizing_guidance == "high_risk_budget":
            pair_trade.sizing_guidance = "medium_risk_budget"
        _add_warning_idempotent(
            pair_trade.validation_warnings,
            str(WarningMessage(PT_GRACEFUL_DOWNGRADE)),
        )


def _clean_hallucinated_attributions(text: str) -> str:
    """ทำความสะอาดการอ้างอิงที่หลุดรูปแบบหรืออ้างข่าวในรูปแบบ YouTube Channel เช่น (Channel: Yahoo Finance: 0h)"""
    if not text or not isinstance(text, str):
        return text
    def replace_fake(m):
        source = m.group(1).strip()
        ref = m.group(2).strip()
        if re.match(r'^(age_)?\d+[hdms]$', ref, re.IGNORECASE) or any(k in source.lower() for k in ["yahoo", "reuters", "bloomberg", "cnbc", "news", "fred", "investing.com", "infoquest", "wsj", "ft.com", "settrade", "bangkok post"]):
            return f"(Source: {source})"
        return m.group(0)
    return re.sub(r'[\(\[]\s*Channel\s*:\s*([^:\]\)]+)\s*:\s*([^\]\)]+)\s*[\)\]]', replace_fake, text, flags=re.IGNORECASE)


def _split_supporting_clauses(item: str) -> list[str]:
    """แตก supporting_data string เป็นท่อนย่อยตาม comma/semicolon
    เพื่อไม่ให้ตัวเลขจากคนละสถิติ (เช่น Ratio vs Z-score) ถูกเทียบข้ามกัน"""
    return [c.strip() for c in re.split(r'[,;]', str(item)) if c.strip()]


def _is_excluded_number(num: float) -> bool:
    """ตัวเลขปีหรือตัวเลขจำนวนเต็มเล็กๆ (เช่น เปอร์เซ็นต์แบบง่าย) ไม่ต้องเช็ค"""
    if 1990 <= num <= 2050 and num == int(num):
        return True
    if num == int(num) and num <= 100:
        return True
    return False


def _validate_supporting_data_against_registry(
    supporting_data: list[str],
    observable_refs: list[str],
    observable_registry: dict[str, Any],
    warnings_list: list[str]
) -> None:
    """ตรวจสอบความสอดคล้องของตัวเลขใน supporting_data เทียบกับ observable_registry
    หลักการ: แต่ละตัวเลขต้อง 'ใกล้เคียง' กับ indicator ที่ token คาบเกี่ยวอย่างน้อย 1 ตัว
    ถ้ามี indicator ที่ match ได้อย่างสมเหตุสมผล ไม่ถือว่าผิด แม้จะมี indicator อื่นที่ token
    คาบเกี่ยวกันด้วยแต่ค่าต่างกันมาก (เพราะ token ทั่วไปเช่น 'yield'/'rate' คาบเกี่ยวหลาย indicator ได้)
    """
    if not supporting_data or not observable_registry:
        return
    indicator_val_map: list[tuple[str, float]] = []
    for oid, obs in observable_registry.items():
        try:
            val_str = str(getattr(obs, "value", "")).replace(",", "").strip()
            v = float(val_str)
            ind_name = str(getattr(obs, "indicator", oid)).lower()
            indicator_val_map.append((ind_name, v))
        except (ValueError, TypeError):
            continue

    mismatch_found = False
    for item in supporting_data:
        if mismatch_found:
            break
        for clause in _split_supporting_clauses(item):
            if mismatch_found:
                break
            clause_str = clause.lower()
            nums_in_clause = [
                float(x.replace(",", "")) for x in re.findall(r'\b\d+(?:\.\d+)?\b', clause)
            ]
            if not nums_in_clause:
                continue

            matched_vals = [
                reg_v for ind_name, reg_v in indicator_val_map
                if abs(reg_v) > 1e-5
                and any(
                    token in clause_str
                    for token in re.findall(r'[a-z0-9/]+', ind_name)
                    if len(token) >= 3
                    and token not in ("ratio", "the", "and", "for", "from", "with", "market", "equities", "equity")
                )
            ]
            if not matched_vals:
                continue

            for num in nums_in_clause:
                if _is_excluded_number(num):
                    continue
                # ถือว่า "ตรง" ถ้าใกล้เคียงกับ indicator ที่ match ได้ตัวใดตัวหนึ่งก็พอ
                close_to_any = any(
                    num / abs(v) <= 10.0 and abs(v) / (num if num > 0 else 1e-9) <= 10.0
                    for v in matched_vals
                )
                if not close_to_any:
                    mismatch_found = True
                    break

    if mismatch_found:
        _add_warning_idempotent(warnings_list, str(WarningMessage(SUPPORTING_DATA_MISMATCH)))



class RegimeEvidenceComponent(BaseModel):
    dimension: str = Field(description="มิติที่พิจารณา เช่น Growth, Inflation")
    signal: str = Field(description="ทิศทางสัญญาณ")
    evidence: str = Field(description="ตัวเลข Hard Data รองรับ")
    conflict: str = Field(default="", description="ข้อขัดแย้งกับมิติอื่น")
    confidence: Literal["high", "medium", "low"] = Field(default="high", description="ระดับความมั่นใจ")
    observable_refs: list[str] = Field(default_factory=list, description="รหัสอ้างอิง MarketObservable ใน registry")
    source_refs: list[str] = Field(default_factory=list, description="ไฟล์หรือแหล่งข้อมูลอ้างอิง")

    @model_validator(mode='after')
    def normalize_conflicting_dimension(self) -> 'RegimeEvidenceComponent':
        if "conflict" in self.dimension.lower():
            if not self.conflict or self.conflict == "-":
                self.conflict = f"Reflation vs {self.signal}" if "stagflation" in self.signal.lower() else f"Baseline vs {self.signal}"
            if self.dimension.lower() in {"conflicting evidence", "conflicting", "conflict"}:
                self.dimension = "Regional Inflation & Growth Risk" if "inflation" in self.signal.lower() or "cpi" in self.evidence.lower() else "Cross-Regional Divergence Risk"
        return self

RegimeEvidence = RegimeEvidenceComponent

class AssetStance(str, Enum):
    OVERWEIGHT = "Overweight"
    NEUTRAL = "Neutral"
    UNDERWEIGHT = "Underweight"

class AssetAllocationView(BaseModel):
    asset_class: str = Field(description="ชื่อสินทรัพย์เจาะลึกระดับภูมิภาคและสไตล์ เช่น US Equities (AI/Tech Growth), EU Equities (Defensive)")
    asset_bucket: Optional[Literal["equities", "fixed_income", "commodities", "fx", "cash"]] = Field(default=None, description="หมวดหมู่สินทรัพย์หลัก")
    stance: AssetStance = Field(description="มุมมองการจัดสรร")
    rationale: str = Field(description="เหตุผลรองรับ")
    confidence: Literal["high", "medium", "low"] = Field(default="medium", description="ระดับความมั่นใจต่อมุมมองนี้")
    supporting_data: list[str] = Field(default_factory=list, description="ตัวเลข Hard Data รองรับ เช่น ['Quant Score US = +0.85', '10Y Yield = 4.40%']")
    validation_warnings: list[str] = Field(default_factory=list, description="คำเตือนคุณภาพข้อมูล")
    data_confidence: Literal["high", "medium", "low"] = Field(default="high", description="ความมั่นใจด้านคุณภาพข้อมูล")
    signal_confidence: Literal["high", "medium", "low"] = Field(default="high", description="ความมั่นใจด้านสัญญาณเทคนิค/ปริมาณ")
    implementation_confidence: Literal["high", "medium", "low"] = Field(default="high", description="ความมั่นใจด้านการนำไปปฏิบัติจริง")
    observable_refs: list[str] = Field(default_factory=list, description="รหัสอ้างอิง MarketObservable ใน registry")
    source_refs: list[str] = Field(default_factory=list, description="ไฟล์หรือแหล่งข้อมูลอ้างอิง")
    why_not_high: str = Field(default="", description="เหตุผลที่ความมั่นใจไม่ถึงระดับ HIGH")
    invalidation_conditions: list[str] = Field(default_factory=list, description="เงื่อนไขที่จะทำให้มุมมองนี้ถูกยกเลิก")
    allocation_delta: str = Field(default="", description="การปรับน้ำหนักเทียบกับเกณฑ์มาตรฐาน")
    benchmark_ref: str = Field(default="", description="เกณฑ์มาตรฐานอ้างอิง")
    time_horizon: str = Field(default="3-6 Months", description="กรอบระยะเวลาลงทุน")
    stale_data_warning: str = Field(default="", description="คำเตือนข้อมูลล่าช้า")

    @field_validator("time_horizon", mode="before")
    @classmethod
    def _val_time_horizon(cls, v: Any) -> str:
        return _normalize_time_horizon_str(v)

    @model_validator(mode='after')
    def validate_hard_data(self) -> 'AssetAllocationView':
        self.rationale = _clean_hallucinated_attributions(self.rationale)
        self.supporting_data = [_clean_hallucinated_attributions(sd) for sd in self.supporting_data]
        if self.asset_bucket is None:
            _add_warning_idempotent(self.validation_warnings, str(WarningMessage(MISSING_ASSET_BUCKET)))
        if len(self.source_refs) == 1:
            if self.confidence == "high":
                self.confidence = "medium"
            _add_warning_idempotent(self.validation_warnings, str(WarningMessage(SINGLE_SOURCE_PENALTY)))
        if not _has_hard_data_numbers(self.supporting_data):
            self.confidence = "low"
            _add_warning_idempotent(self.validation_warnings, str(WarningMessage(DEFENSIVE_LOW_SUPPORTING_DATA)))
        if self.confidence == "low" and self.stance != AssetStance.NEUTRAL:
            self.stance = AssetStance.NEUTRAL
            _add_warning_idempotent(self.validation_warnings, str(WarningMessage(ACTIVE_ALLOC_GUARDRAIL)))

        ac_lower = str(self.asset_class).lower()
        is_usd_base = any(k in ac_lower for k in ["usd/thb", "usd vs thb", "ค่าเงินบาท", "เงินบาท", "currenci", "fx"]) and "thb vs usd" not in ac_lower and "thb/usd" not in ac_lower
        if is_usd_base:
            is_baht_weakening = any(k in str(self.rationale).lower() for k in ["บาทอ่อน", "เงินบาทมีแนวโน้มอ่อน", "ดอลลาร์แข็ง", "thb depreciation", "usd appreciation"])
            if is_baht_weakening and self.stance == AssetStance.UNDERWEIGHT:
                _add_warning_idempotent(self.validation_warnings, str(WarningMessage(FX_STANCE_MISMATCH)))

        delta = str(self.allocation_delta or "").strip().lower()
        if delta in {"overweight", "underweight", "neutral"} or delta == self.stance.value.lower():
            _add_warning_idempotent(self.validation_warnings, str(WarningMessage(ALLOCATION_DELTA_INVALID)))
        return self

class PairTradeStrategy(BaseModel):
    long_leg: str = Field(description="สินทรัพย์ฝั่ง Long เช่น US Tech Equities")
    short_leg: str = Field(description="สินทรัพย์ฝั่ง Short เช่น European Industrials")
    thesis: str = Field(description="สมมติฐานการลงทุนและ Divergence")
    catalyst: str = Field(description="ปัจจัยเร่ง (Catalyst) เช่น Q2 Big Tech earnings")
    risk: str = Field(description="ความเสี่ยงของกลยุทธ์")
    time_horizon: str = Field(default="1-3 เดือน", description="กรอบระยะเวลาลงทุน เช่น 1-3 เดือน")
    confidence: Literal["high", "medium", "low"] = Field(description="ระดับความมั่นใจ")
    sizing_guidance: Literal["small_risk_budget", "medium_risk_budget", "high_risk_budget"] = Field(default="small_risk_budget", description="คำแนะนำกรอบความเสี่ยงเชิงคุณภาพ")
    implementation_idea: str = Field(default="", description="แนวทางการนำไปปรับใช้เชิงกลยุทธ์ เช่น Relative Overweight/Underweight ไม่ใช่คำสั่งส่งออเดอร์")
    supporting_data: list[str] = Field(default_factory=list, description="ตัวเลขเชิงปริมาณรองรับ")
    validation_warnings: list[str] = Field(default_factory=list, description="คำเตือนคุณภาพข้อมูล")
    observable_refs: list[str] = Field(default_factory=list, description="รหัสอ้างอิง MarketObservable ใน registry")
    source_refs: list[str] = Field(default_factory=list, description="ไฟล์หรือแหล่งข้อมูลอ้างอิง")
    instrument_proxy: str = Field(default="", description="เครื่องมือจริงที่ใช้เทรด เช่น Long QQQ ETF / Short VGK ETF (ต้องระบุเสมอ)")
    hedge_ratio: str = Field(default="", description="สัดส่วนการเทรด เช่น 1.0 : 1.0 Notional / Price Ratio หรือ 1 : 1 Dollar-equivalent (ต้องมีตัวเลขเสมอ ห้ามใช้คำว่า Beta-adjusted หากไม่มีข้อมูลสถิติ Beta ในตาราง)")
    fx_handling: str = Field(default="", description="การบริหารค่าเงิน เช่น Unhedged หรือ USD/EUR 50% Forward Hedge")
    entry_trigger: str = Field(default="", description="จุดเข้าเทรด เช่น Relative spread divergence > 2.0% หรือ Ratio > 1.5x (ต้องมีตัวเลขเสมอ ห้ามใช้คำว่า SD หากไม่มีข้อมูล Historical SD ในตาราง)")
    stop_loss_trigger: str = Field(default="", description="จุดตัดขาดทุน เช่น -3.0% relative spread divergence (ต้องระบุตัวเลขเสมอ)")
    target_gain_or_rebalance: str = Field(default="", description="เป้าหมายทำกำไรหรือปรับพอร์ต เช่น +6.0% relative spread convergence (ต้องระบุตัวเลขเสมอ)")
    max_drawdown_limit: str = Field(default="", description="ขีดจำกัดผลขาดทุนสูงสุด เช่น -4.5% of allocated risk budget (ต้องระบุตัวเลขเสมอ)")
    review_frequency: str = Field(default="", description="ความถี่ทบทวนกลยุทธ์ เช่น Weekly หรือ Bi-weekly review")

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: Any, handler: Any
    ) -> dict[str, Any]:
        json_schema = handler(core_schema)
        json_schema = handler.resolve_ref_schema(json_schema)
        if "properties" in json_schema:
            json_schema["required"] = list(json_schema["properties"].keys())
        return json_schema

    @field_validator("time_horizon", mode="before")
    @classmethod
    def _val_time_horizon(cls, v: Any) -> str:
        return _normalize_time_horizon_str(v)

    @model_validator(mode='after')
    def validate_evidence_and_sizing(self) -> 'PairTradeStrategy':
        self.thesis = _clean_hallucinated_attributions(self.thesis)
        self.supporting_data = [_clean_hallucinated_attributions(sd) for sd in self.supporting_data]
        for fname, val in [
            ("long_leg", self.long_leg),
            ("short_leg", self.short_leg),
            ("thesis", self.thesis),
            ("catalyst", self.catalyst),
            ("risk", self.risk),
        ]:
            if not str(val).strip():
                self.confidence = "low"
                self.sizing_guidance = "small_risk_budget"
                _add_warning_idempotent(self.validation_warnings, str(WarningMessage(PT_MANDATORY_FIELD_MISSING, {"field": fname})))

        if not _has_hard_data_numbers(self.supporting_data):
            self.confidence = "low"
            _add_warning_idempotent(self.validation_warnings, str(WarningMessage(PT_DEFENSIVE_LOW)))

        import re
        has_stat_evidence = any(
            re.search(r'(?i)\b(beta|z-score|standard deviation|sd|correlation)\b', str(d))
            for d in (self.supporting_data + self.observable_refs)
        )
        if not has_stat_evidence:
            if re.search(r'(?i)\bbeta[- ]?adjusted\b', str(self.hedge_ratio)) or re.search(r'(?i)\b(\d+(?:\.\d+)?)\s*(?:sd|z-score)\b', str(self.entry_trigger)):
                if self.confidence == "high":
                    self.confidence = "medium"
                _add_warning_idempotent(self.validation_warnings, str(WarningMessage(STATISTICAL_OVERCLAIM)))

        trigger_str = str(self.entry_trigger).lower() + " " + str(self.stop_loss_trigger).lower()
        needs_rel = any(k in trigger_str for k in ["spread", "divergence", "convergence", "relative", "ratio", "beta", "sd", "z-score"])
        has_rel = any(any(k in str(d).lower() for k in ["/", "ratio", "spread", "diff", "diverging", "vs", "relative", "beta", "z-score"]) and not "hyg/lqd" in str(d).lower() for d in self.supporting_data)
        if needs_rel and not has_rel:
            if self.confidence == "high":
                self.confidence = "medium"
            if self.sizing_guidance == "high_risk_budget":
                self.sizing_guidance = "medium_risk_budget"
            _add_warning_idempotent(self.validation_warnings, str(WarningMessage(PT_GRACEFUL_DOWNGRADE)))

        if not _all_execution_fields_present(self):
            if self.confidence in ["high", "medium"]:
                self.confidence = "low"
            _add_warning_idempotent(self.validation_warnings, str(WarningMessage(PT_EXECUTION_GUARDRAIL)))

        if self.sizing_guidance == "high_risk_budget":
            if self.confidence == "medium":
                self.sizing_guidance = "medium_risk_budget"
                _add_warning_idempotent(self.validation_warnings, str(WarningMessage(PT_RISK_BUDGET_HIGH_TO_MED)))
            elif self.confidence == "low":
                self.sizing_guidance = "small_risk_budget"
                _add_warning_idempotent(self.validation_warnings, str(WarningMessage(PT_RISK_BUDGET_HIGH_TO_SMALL)))
        elif self.sizing_guidance == "medium_risk_budget" and self.confidence == "low":
            self.sizing_guidance = "small_risk_budget"
            _add_warning_idempotent(self.validation_warnings, str(WarningMessage(PT_RISK_BUDGET_MED_TO_SMALL)))
        return self

class RiskMitigationScenario(BaseModel):
    tail_risk: str = Field(description="ความเสี่ยงรุนแรง (Tail Risk)")
    probability: Literal["high", "medium", "low"] = Field(description="โอกาสในการเกิด")
    impact: Literal["severe", "moderate", "manageable"] = Field(description="ผลกระทบต่อพอร์ต")
    early_warning_indicators: list[str] = Field(default_factory=list, description="ตัวเลขเตือนภัยล่วงหน้า")
    hedge_instruments: list[str] = Field(default_factory=list, description="เครื่องมือ Hedge เช่น Long TLT, Put Options on S&P 500")
    trigger_to_activate: str = Field(description="เงื่อนไขจุดตัดในการเริ่ม Hedge เช่น VIX Index > 25.0 หรือ Gold > 4,200 USD/oz (ต้องมีตัวเลขเชิงปริมาณเสมอ ห้ามใส่ค่าว่างหรือคำคุณศัพท์ลอยๆ)")
    cost_or_tradeoff: str = Field(description="ต้นทุนหรือ Negative carry ของการ Hedge เช่น -1.5% p.a. option premium")
    supporting_data: list[str] = Field(default_factory=list, description="ตัวเลขเชิงปริมาณที่ประเมินความเสี่ยง")
    confidence: Literal["high", "medium", "low"] = Field(default="medium", description="ระดับความมั่นใจในแผนป้องกันนี้")
    validation_warnings: list[str] = Field(default_factory=list, description="คำเตือนคุณภาพข้อมูล")
    mitigation_strategy: str = Field(default="", description="กลยุทธ์ป้องกันความเสี่ยง เช่น Put Option Protection")
    trigger_type: str = Field(default="", description="ประเภทจุดตัด เช่น Quantitative Market Volatility Trigger")
    volume_threshold: str = Field(default="", description="ปริมาณซื้อขายยืนยัน เช่น Daily trading volume > 50,000 contracts (ต้องระบุตัวเลขเสมอ)")
    unwind_or_cover_condition: str = Field(default="", description="เงื่อนไขยกเลิกหรือ Cover เช่น VIX Index drops below 18.0 (ต้องระบุตัวเลขเสมอ)")
    hedge_size: str = Field(default="", description="ขนาดป้องกันความเสี่ยง เช่น 10% of portfolio value (ต้องระบุตัวเลขเสมอ)")
    hedge_purpose: str = Field(default="portfolio_hedge", description="วัตถุประสงค์การป้องกันความเสี่ยง")

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: Any, handler: Any
    ) -> dict[str, Any]:
        json_schema = handler(core_schema)
        json_schema = handler.resolve_ref_schema(json_schema)
        if "properties" in json_schema:
            json_schema["required"] = list(json_schema["properties"].keys())
        return json_schema

    @model_validator(mode='after')
    def validate_quality_gate(self) -> 'RiskMitigationScenario':
        self.supporting_data = [_clean_hallucinated_attributions(sd) for sd in self.supporting_data]
        has_numeric_execution = (
            bool(re.search(r'\d', str(self.trigger_to_activate)))
            and bool(re.search(r'\d', str(self.volume_threshold)))
            and bool(str(self.trigger_type).strip())
            and bool(str(self.unwind_or_cover_condition).strip())
        )
        if (
            not _has_hard_data_numbers(self.supporting_data)
            or not self.early_warning_indicators
            or not self.hedge_instruments
            or not has_numeric_execution
        ):
            self.confidence = "low"
            _add_warning_idempotent(self.validation_warnings, str(WarningMessage(RS_DEFENSIVE_LOW)))
        return self

def _asset_has_downgrade_warning(asset: AssetAllocationView) -> bool:
    return any(
        any(wid in warning for wid in DOWNGRADE_WARNING_IDS)
        for warning in asset.validation_warnings
    )


def _why_not_high_is_weak(asset: AssetAllocationView) -> bool:
    current = str(asset.why_not_high or "").strip().lower()
    if not current:
        return True
    return any(
        marker in current
        for marker in [
            "-",
            "none",
            "n/a",
            "no reason",
            "ไม่มีเหตุผล",
            WHY_NOT_HIGH_MESSAGES["default"].lower(),
            WHY_NOT_HIGH_MESSAGES["low_confidence"].lower(),
        ]
    )


def _normalize_why_not_high(asset: AssetAllocationView) -> None:
    if asset.confidence == "high":
        asset.why_not_high = "-"
        return
    current = str(asset.why_not_high or "").strip().lower()
    weak_values = {
        "",
        "-",
        "none",
        "n/a",
        "ไม่มี",
        "ไม่มีเหตุผล",
        "ไม่มีเหตุผลที่ความมั่นใจไม่ถึงระดับสูง",
        WHY_NOT_HIGH_MESSAGES["default"].lower(),
        WHY_NOT_HIGH_MESSAGES["low_confidence"].lower(),
    }
    if current not in weak_values and "ไม่มีเหตุผล" not in current:
        return
    warnings = [str(w) for w in asset.validation_warnings]
    if any(f"[{SINGLE_SOURCE_PENALTY}]" in w for w in warnings):
        asset.why_not_high = WHY_NOT_HIGH_MESSAGES["single_source"]
    elif any(f"[{GOLD_CONTRADICTION}]" in w or f"[{GOLD_RATIONALE_WARNING}]" in w for w in warnings):
        asset.why_not_high = WHY_NOT_HIGH_MESSAGES["gold_schema"]
    elif any("CONTRADICTION" in w for w in warnings):
        asset.why_not_high = WHY_NOT_HIGH_MESSAGES["contradiction"]
    elif any(f"[{SOURCE_REF_PENALTY}]" in w for w in warnings):
        asset.why_not_high = WHY_NOT_HIGH_MESSAGES["source_ref_inferred"]
    elif asset.confidence == "low":
        asset.why_not_high = WHY_NOT_HIGH_MESSAGES["low_confidence"]
    else:
        asset.why_not_high = WHY_NOT_HIGH_MESSAGES["default"]



def _is_leading_indicator(obs: Any) -> bool:
    id_lower = getattr(obs, "observable_id", "").lower()
    ind_lower = getattr(obs, "indicator", "").lower()
    leading_ids = {"nfci", "icsa", "ccsa", "t5yie", "t10yie", "t10y2y", "umcsent", "houst", "dgs2", "dfii10", "pmi"}
    if any(lid in id_lower for lid in leading_ids):
        return True
    leading_keywords = [
        "pmi", "leading", "claims", "breakeven", "sentiment", "housing starts",
        "nfci", "icsa", "ccsa", "financial conditions", "ดัชนีสภาวะทางการเงิน", "สวัสดิการว่างงาน"
    ]
    if any(kw in ind_lower for kw in leading_keywords):
        return True
    return False


class MacroStrategyDirection(BaseModel):
    evaluated_at: str = Field(description="ISO format string")
    overall_regime: EconomicState = Field(description="สภาวะเศรษฐกิจองค์รวม")
    time_horizon: str = Field(default="3-6 เดือน", description="กรอบเวลาของกลยุทธ์")
    key_assumptions: list[str] = Field(default_factory=list, description="สมมติฐานหลักของสภาวะเศรษฐกิจ")
    asset_allocation: list[AssetAllocationView] = Field(description="มุมมองจัดสรรสินทรัพย์")
    focus_themes: list[str] = Field(description="ธีมการลงทุนที่ควรเน้น")
    conviction_level: Literal["high", "medium", "low"] = Field(description="ระดับความมั่นใจ")
    conviction_rationale: str = Field(description="เหตุผลรองรับแบบเจาะลึก")
    quant_narrative_alignment: Literal["aligned", "divergent", "partially_aligned"] = Field(description="ความสอดคล้องระหว่าง Quant และ Narrative")
    divergence_note: str = Field(default="", description="ข้อสังเกตกรณีขัดแย้ง")
    pair_trades: list[PairTradeStrategy] = Field(default_factory=list, description="กลยุทธ์จับคู่เทรด Relative Value")
    risk_scenarios: list[RiskMitigationScenario] = Field(default_factory=list, description="การบริหารความเสี่ยงแบบจำลองและ Hedge")
    validation_warnings: list[str] = Field(default_factory=list, description="คำเตือนกรณีถูกลดระดับความน่าเชื่อถือ")
    observable_registry: dict[str, MarketObservable] = Field(default_factory=dict, exclude=True, description="Registry ของ MarketObservable ทั้งหมดในระบบ")
    generated_by: str = Field(default="strategic_allocator", description="ระบบหรือ agent ที่สร้างรายงานนี้")
    source_files: list[str] = Field(default_factory=list, description="ไฟล์ต้นทางทั้งหมดที่ใช้ในการประเมิน")
    data_timestamp_notes: list[str] = Field(default_factory=list, description="บันทึกเวลาของข้อมูลที่ใช้")
    stale_data_warnings: list[str] = Field(default_factory=list, description="คำเตือนกรณีพบข้อมูลล่าช้าหรือหมดอายุ")
    regime_probabilities: dict[str, Any] = Field(default_factory=dict, description="การกระจายความน่าจะเป็นของสภาวะเศรษฐกิจ")
    regime_evidence: list[RegimeEvidenceComponent] = Field(default_factory=list, description="หลักฐานรองรับสภาวะเศรษฐกิจใน 5 มิติ")

    @field_validator("time_horizon", mode="before")
    @classmethod
    def _val_time_horizon(cls, v: Any) -> str:
        return _normalize_time_horizon_str(v)

    def revalidate_with_registry(
        self, registry: dict[str, MarketObservable]
    ) -> "MacroStrategyDirection":
        """Inject registry Explicitly และรันการตรวจสอบ Guardrail ใหม่ทั้งหมด"""
        data = self.model_dump()
        data["observable_registry"] = registry
        return type(self).model_validate(data)

    @model_validator(mode='after')
    def validate_portfolio_conviction(self) -> 'MacroStrategyDirection':
        old_cr = self.conviction_rationale
        self.conviction_rationale = _clean_hallucinated_attributions(self.conviction_rationale)
        cleaned_att = (old_cr != self.conviction_rationale)

        for a in self.asset_allocation:
            old_r = a.rationale
            a.rationale = _clean_hallucinated_attributions(a.rationale)
            a.supporting_data = [_clean_hallucinated_attributions(sd) for sd in a.supporting_data]
            if old_r != a.rationale:
                cleaned_att = True
            _validate_supporting_data_against_registry(a.supporting_data, a.observable_refs, self.observable_registry, a.validation_warnings)
            if any("SUPPORTING_DATA_MISMATCH" in w for w in a.validation_warnings):
                if a.confidence == "high":
                    a.confidence = "medium"

        for pt in self.pair_trades:
            old_t = pt.thesis
            pt.thesis = _clean_hallucinated_attributions(pt.thesis)
            pt.supporting_data = [_clean_hallucinated_attributions(sd) for sd in pt.supporting_data]
            if old_t != pt.thesis:
                cleaned_att = True
            _validate_supporting_data_against_registry(pt.supporting_data, pt.observable_refs, self.observable_registry, pt.validation_warnings)

        for rs in self.risk_scenarios:
            rs.supporting_data = [_clean_hallucinated_attributions(sd) for sd in rs.supporting_data]
            _validate_supporting_data_against_registry(rs.supporting_data, [], self.observable_registry, rs.validation_warnings)

        old_ft = list(self.focus_themes or [])
        self.focus_themes = [_clean_hallucinated_attributions(ft) for ft in old_ft]
        if old_ft != self.focus_themes:
            cleaned_att = True

        old_dn = str(self.divergence_note or "")
        self.divergence_note = _clean_hallucinated_attributions(old_dn)
        if old_dn != self.divergence_note:
            cleaned_att = True

        if cleaned_att:
            _add_warning_idempotent(self.validation_warnings, str(WarningMessage(HALLUCINATED_ATTRIBUTION_CLEANED)))

        if self.observable_registry and not self.source_files:
            self.source_files = []
            for obs in self.observable_registry.values():
                if obs.is_valid and obs.source_file and obs.source_file not in self.source_files:
                    self.source_files.append(obs.source_file)

        was_stale = any(
            "STALE_DATA_DEGRADATION" in str(w) or "ข้อมูลล่าช้า" in str(w) or "Stale Data Degradation:" in str(w)
            for w in self.validation_warnings
        )
        if self.stale_data_warnings or was_stale:
            is_exempt_from_stale = False
            if self.observable_registry:
                cited_ids = set()
                for ev in getattr(self, "regime_evidence", []):
                    cited_ids.update(getattr(ev, "observable_refs", []) or [])
                for a in getattr(self, "asset_allocation", []):
                    cited_ids.update(getattr(a, "observable_refs", []) or [])
                valid_leading_count = sum(
                    1 for oid in cited_ids
                    if oid in self.observable_registry
                    and getattr(self.observable_registry[oid], "is_valid", False)
                    and _is_leading_indicator(self.observable_registry[oid])
                )
                if valid_leading_count >= 2:
                    is_exempt_from_stale = True

            self.validation_warnings = [
                w for w in self.validation_warnings
                if not ("STALE_DATA_DEGRADATION" in str(w) or "ข้อมูลล่าช้า" in str(w) or "Stale Data Degradation:" in str(w))
            ]
            if not is_exempt_from_stale:
                if self.conviction_level == "high" or (self.conviction_level == "low" and was_stale):
                    self.conviction_level = "medium"
                _add_warning_idempotent(self.validation_warnings, str(WarningMessage(STALE_DATA_DEGRADATION)))
            elif is_exempt_from_stale and was_stale and self.conviction_level == "low":
                self.conviction_level = "medium"

        for a in self.asset_allocation:
            logger.debug(f"[DEBUG] asset={a.asset_class} observable_refs={a.observable_refs}")
            bucket = a.asset_bucket
            if self.observable_registry and a.observable_refs and bucket:
                valid_refs = []
                for ref in a.observable_refs:
                    obs = self.observable_registry.get(ref)
                    if obs and obs.is_valid and _is_allowed_cross_bucket(bucket, obs):
                        valid_refs.append(ref)
                a.observable_refs = valid_refs
            if self.observable_registry and a.observable_refs:
                valid_ref_files = _source_files_from_observable_refs(a.observable_refs, self.observable_registry)
                if valid_ref_files:
                    a.source_refs = sorted(list(set(a.source_refs or []) | set(valid_ref_files)))
            valid_source_files = _valid_source_files_for_refs(a.observable_refs, self.observable_registry) if self.observable_registry else set()
            if len(valid_source_files) >= 2:
                was_single = any("Single-Source Penalty" in str(w) or "SINGLE_SOURCE_PENALTY" in str(w) for w in a.validation_warnings)
                a.validation_warnings = [
                    w for w in a.validation_warnings if "Single-Source Penalty" not in w and "SINGLE_SOURCE_PENALTY" not in w
                ]
                if was_single and a.confidence == "medium" and not _asset_has_downgrade_warning(a):
                    a.confidence = "high"

        valid_pair_trades = []
        dropped_pair_trades = 0
        for pt in self.pair_trades:
            if _has_pair_trade_execution_evidence(pt, self.observable_registry):
                _downgrade_pair_trade_statistical_overclaim(pt, self.observable_registry)
                valid_pair_trades.append(pt)
            else:
                dropped_pair_trades += 1
        self.pair_trades = valid_pair_trades
        if dropped_pair_trades > 0:
            _add_warning_idempotent(self.validation_warnings, str(WarningMessage(GRACEFUL_DROP_PAIR_TRADES, {"count": str(dropped_pair_trades)})))

        valid_risk_scenarios = []
        dropped_risk_scenarios = 0
        for rs in self.risk_scenarios:
            exec_params = [
                getattr(rs, "volume_threshold", ""),
                getattr(rs, "trigger_to_activate", ""),
                getattr(rs, "hedge_size", ""),
                getattr(rs, "unwind_or_cover_condition", ""),
            ] + rs.supporting_data
            if any(re.search(r'\d', str(val)) for val in exec_params) and bool(getattr(rs, "hedge_instruments", [])):
                valid_risk_scenarios.append(rs)
            else:
                dropped_risk_scenarios += 1
        self.risk_scenarios = valid_risk_scenarios
        if dropped_risk_scenarios > 0:
            _add_warning_idempotent(self.validation_warnings, str(WarningMessage(GRACEFUL_DROP_RISK_SCENARIOS, {"count": str(dropped_risk_scenarios)})))

        for a in self.asset_allocation:
            if a.confidence != "low" and not a.observable_refs:
                _add_warning_idempotent(
                    a.validation_warnings,
                    str(WarningMessage(MISSING_OBSERVABLE_REFS, {"asset_class": a.asset_class}))
                )

            inferred_source_refs = not a.source_refs or (self.source_files and set(a.source_refs) == set(self.source_files))

            if self.source_files and inferred_source_refs:
                filtered_sources = []
                bucket = (a.asset_bucket or "").lower()
                for sf in self.source_files:
                    sf_lower = sf.lower()
                    if "global" in sf_lower:
                        filtered_sources.append(sf)
                    elif bucket == "equities" and any(k in sf_lower for k in ["country", "stocks", "val", "youtube", "eq", "spy"]):
                        filtered_sources.append(sf)
                    elif bucket == "fixed_income" and any(k in sf_lower for k in ["country", "regional", "fred", "macro", "bond", "yield", "rate"]):
                        filtered_sources.append(sf)
                    elif bucket == "fx" and any(k in sf_lower for k in ["regional", "country", "th_", "us_", "fx", "curr"]):
                        filtered_sources.append(sf)
                    elif bucket in ("commodities", "cash") and any(k in sf_lower for k in ["global", "country", "fred", "comm", "cash", "gold"]):
                        filtered_sources.append(sf)
                a.source_refs = sorted(list(set(filtered_sources))) if filtered_sources else sorted(list(self.source_files))

            if inferred_source_refs and self.source_files and self.observable_registry:
                has_valid_obs = _refs_supported_by_source_files(
                    a.observable_refs,
                    self.observable_registry,
                    list(self.source_files),
                )
                if not has_valid_obs:
                    if a.confidence == "high":
                        a.confidence = "medium"
                    msg = str(WarningMessage(SOURCE_REF_PENALTY))
                    _add_warning_idempotent(a.validation_warnings, msg)
                    _add_warning_idempotent(self.validation_warnings, msg)

        from validators.contradiction_rules import validate_all_contradictions
        validate_all_contradictions(self)

        if self.asset_allocation is not None:
            core_buckets = {"equities", "fixed_income", "commodities", "fx", "cash"}
            present_buckets = {
                str(getattr(a, "asset_bucket", "")).lower().strip()
                for a in self.asset_allocation
                if getattr(a, "asset_bucket", None) is not None
            }
            missing_buckets = core_buckets - present_buckets
            if len(self.asset_allocation) < 5 or missing_buckets:
                _add_warning_idempotent(self.validation_warnings, str(WarningMessage(COVERAGE_WARNING_INCOMPLETE, {"count": str(len(self.asset_allocation)), "missing": sorted(list(missing_buckets))})))
            else:
                self.validation_warnings = [
                    w for w in self.validation_warnings
                    if "COVERAGE_WARNING_INCOMPLETE" not in str(w) and "Coverage Warning" not in str(w)
                ]

        for a in self.asset_allocation:
            if a.confidence == "high":
                a.why_not_high = "-"
            elif _asset_has_downgrade_warning(a) or _why_not_high_is_weak(a):
                _normalize_why_not_high(a)

        if not self.asset_allocation:
            return self

        low_conf_count = sum(1 for a in self.asset_allocation if a.confidence == "low" or not _has_hard_data_numbers(a.supporting_data))
        if (low_conf_count / len(self.asset_allocation)) >= 0.5:
            self.conviction_level = "low"
            _add_warning_idempotent(self.validation_warnings, str(WarningMessage(PORTFOLIO_DEFENSIVE_LOW, {"ratio": f"{low_conf_count}/{len(self.asset_allocation)}"})))
        else:
            defensive_prefix = "Defensive Degradation: Portfolio conviction downgraded to LOW because >=50%"
            defensive_id = f"[{PORTFOLIO_DEFENSIVE_LOW}]"
            had_defensive = any(w.startswith(defensive_prefix) or w.startswith(defensive_id) for w in self.validation_warnings)
            if had_defensive:
                self.validation_warnings = [
                    w for w in self.validation_warnings if not w.startswith(defensive_prefix) and not w.startswith(defensive_id)
                ]
                if self.conviction_level == "low":
                    self.conviction_level = "medium"

        clevel = str(self.conviction_level or "medium").upper()
        if clevel in {"MEDIUM", "LOW"}:
            for high_term in ["ความเชื่อมั่นระดับสูง", "ความมั่นใจระดับสูง", "high conviction"]:
                if high_term in self.conviction_rationale.lower() or high_term in self.conviction_rationale:
                    _add_warning_idempotent(self.validation_warnings, str(WarningMessage(CONVICTION_CONTRADICTION)))

        return self
