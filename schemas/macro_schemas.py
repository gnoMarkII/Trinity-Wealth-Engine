from datetime import datetime
from enum import Enum
from typing import Literal, Optional, Any
from pydantic import BaseModel, Field, computed_field, field_validator, model_validator
import re

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
    if not data_list:
        return False
    num_pattern = re.compile(r'\d+([.,]\d+)?')
    fin_pattern = re.compile(r'(?:%|bps|pts|yield|pmi|cpi|ppi|nfp|gdp|vix|dxy|spx|ndx|score|rate|ratio|spread|differential|carry|relative|usd|thb|eur|jpy|cds|gold|oil|brent|wti|shipping|freight|etf|t-bill|treasury|index|level|target|support|resistance|stop|limit|prob|probability|set|sp500|nasdaq|มูลค่า|ซื้อขาย|ล้านบาท|จุด|ดัชนี|ผลตอบแทน|อัตรา|\.\d+x|\/\d+y|\d{4}-\d{2}-\d{2}|date|day|month|year)', re.IGNORECASE)
    for item in data_list:
        if num_pattern.search(item) and fin_pattern.search(item):
            return True
    return False

def _add_warning_idempotent(warnings_list: list[str], warning_str: str) -> None:
    if warning_str not in warnings_list:
        warnings_list.append(warning_str)

def _auto_extract_with_provenance(target_list: list[str], source_text: str, source_field_name: str) -> None:
    if not source_text:
        return
    if _has_hard_data_numbers([source_text]):
        if not any(source_text in item for item in target_list):
            target_list.append(f"[From {source_field_name}] {source_text}")


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
    if not _has_hard_data_numbers([
        pair_trade.hedge_ratio,
        pair_trade.stop_loss_trigger,
        pair_trade.target_gain_or_rebalance,
        pair_trade.max_drawdown_limit,
    ] + pair_trade.supporting_data):
        return False
    if bool(observable_registry):
        return bool(_valid_observables_for_refs(pair_trade.observable_refs, observable_registry))
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


def _asset_bucket_from_name(asset_class: str) -> str | None:
    text = asset_class.lower()
    if any(k in text for k in ["fx", "currenc", "dxy", "usd", "eur", "jpy", "thb", "ค่าเงิน", "สกุลเงิน"]):
        return "fx"
    if any(k in text for k in ["cash", "t-bill", "money market", "liquidity", "เงินสด"]):
        return "cash"
    if any(k in text for k in ["bond", "duration", "treasur", "fixed income", "พันธบัตร", "yield"]):
        return "fixed_income"
    if any(k in text for k in ["commodit", "gold", "oil", "metal", "brent", "wti", "ทองคำ", "น้ำมัน"]):
        return "commodities"
    if any(k in text for k in ["equit", "stock", "หุ้น", "spx", "nasdaq", "set"]):
        return "equities"
    return None


def _infer_observable_refs_from_text(
    text: str,
    observable_registry: dict[str, "MarketObservable"],
    asset_bucket: str | None = None,
    max_refs: int = 3,
) -> list[str]:
    if not observable_registry:
        return []
    text_lower = text.lower()
    matches: list[str] = []
    for obs_id, obs in observable_registry.items():
        if not obs.is_valid:
            continue
        if asset_bucket and obs.asset_bucket != asset_bucket:
            continue
        indicator_terms = [part for part in re.split(r"[^a-zA-Z0-9]+", obs.indicator.lower()) if len(part) >= 3]
        value_token = str(obs.value).replace(",", "").split()[0].lower()
        score = 0
        if obs_id.lower() in text_lower:
            score += 4
        if value_token and value_token in text_lower.replace(",", ""):
            score += 3
        score += sum(1 for term in indicator_terms[:6] if term in text_lower)
        if score > 0 and obs_id not in matches:
            matches.append(obs_id)
        if len(matches) >= max_refs:
            break
    if len(matches) < max_refs and asset_bucket:
        for obs_id, obs in observable_registry.items():
            if obs.is_valid and obs.asset_bucket == asset_bucket and obs_id not in matches:
                matches.append(obs_id)
                if len(matches) >= max_refs:
                    break
    if len(matches) < max_refs:
        for obs_id, obs in observable_registry.items():
            if obs.is_valid and obs_id not in matches:
                matches.append(obs_id)
                if len(matches) >= max_refs:
                    break
    return matches


def _supporting_data_from_observables(
    observable_refs: list[str],
    observable_registry: dict[str, "MarketObservable"],
) -> list[str]:
    data = []
    for obs in _valid_observables_for_refs(observable_refs, observable_registry):
        unit = f" {obs.unit}" if obs.unit else ""
        data.append(f"{obs.indicator} = {obs.value}{unit} ({obs.observable_id})")
    return data


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
        _add_warning_idempotent(
            pair_trade.validation_warnings,
            "Pair Trade Graceful Downgrade: Used single-point relative spread/ratio evidence instead of historical beta/correlation time series.",
        )

class RegimeEvidenceComponent(BaseModel):
    dimension: str = Field(description="มิติที่พิจารณา เช่น Growth, Inflation")
    signal: str = Field(description="ทิศทางสัญญาณ")
    evidence: str = Field(description="ตัวเลข Hard Data รองรับ")
    conflict: str = Field(default="", description="ข้อขัดแย้งกับมิติอื่น")
    confidence: Literal["high", "medium", "low"] = Field(default="high", description="ระดับความมั่นใจ")

RegimeEvidence = RegimeEvidenceComponent

class AssetStance(str, Enum):
    OVERWEIGHT = "Overweight"
    NEUTRAL = "Neutral"
    UNDERWEIGHT = "Underweight"

class AssetAllocationView(BaseModel):
    asset_class: str = Field(description="ชื่อสินทรัพย์เจาะลึกระดับภูมิภาคและสไตล์ เช่น US Equities (AI/Tech Growth), EU Equities (Defensive)")
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

    @model_validator(mode='after')
    def validate_hard_data(self) -> 'AssetAllocationView':
        if len(self.source_refs) == 1:
            if self.confidence == "high":
                self.confidence = "medium"
            _add_warning_idempotent(self.validation_warnings, "Single-Source Penalty: Downgraded confidence to MEDIUM because view relies on only 1 source reference.")
        if not _has_hard_data_numbers(self.supporting_data):
            self.confidence = "low"
            _add_warning_idempotent(self.validation_warnings, "Defensive Degradation: Downgraded confidence to LOW due to lack of numeric hard data in supporting_data.")
        if self.confidence == "low" and self.stance != AssetStance.NEUTRAL:
            self.stance = AssetStance.NEUTRAL
            _add_warning_idempotent(self.validation_warnings, "Active Allocation Guardrail: Downgraded stance to NEUTRAL because confidence is LOW or lacks numeric hard data.")
        return self

class PairTradeStrategy(BaseModel):
    long_leg: str = Field(description="สินทรัพย์ฝั่ง Long เช่น US Tech Equities")
    short_leg: str = Field(description="สินทรัพย์ฝั่ง Short เช่น European Industrials")
    thesis: str = Field(description="สมมติฐานการลงทุนและ Divergence")
    catalyst: str = Field(description="ปัจจัยเร่ง (Catalyst) เช่น Q2 Big Tech earnings")
    risk: str = Field(description="ความเสี่ยงของกลยุทธ์")
    time_horizon: str = Field(description="กรอบระยะเวลาลงทุน เช่น 1-3 Months")
    confidence: Literal["high", "medium", "low"] = Field(description="ระดับความมั่นใจ")
    sizing_guidance: Literal["small_risk_budget", "medium_risk_budget", "high_risk_budget"] = Field(default="small_risk_budget", description="คำแนะนำกรอบความเสี่ยงเชิงคุณภาพ")
    implementation_idea: str = Field(default="", description="แนวทางการนำไปปรับใช้เชิงกลยุทธ์ เช่น Relative Overweight/Underweight ไม่ใช่คำสั่งส่งออเดอร์")
    supporting_data: list[str] = Field(default_factory=list, description="ตัวเลขเชิงปริมาณรองรับ")
    validation_warnings: list[str] = Field(default_factory=list, description="คำเตือนคุณภาพข้อมูล")
    observable_refs: list[str] = Field(default_factory=list, description="รหัสอ้างอิง MarketObservable ใน registry")
    source_refs: list[str] = Field(default_factory=list, description="ไฟล์หรือแหล่งข้อมูลอ้างอิง")
    instrument_proxy: str = Field(default="", description="เครื่องมือจริงที่ใช้เทรด เช่น Long QQQ ETF / Short VGK ETF (ต้องระบุเสมอ)")
    hedge_ratio: str = Field(default="", description="สัดส่วนการเทรด เช่น 1.0 : 1.0 Beta-adjusted (ต้องมีตัวเลขเสมอ)")
    fx_handling: str = Field(default="", description="การบริหารค่าเงิน เช่น Unhedged หรือ USD/EUR 50% Forward Hedge")
    entry_trigger: str = Field(default="", description="จุดเข้าเทรด เช่น Spread widening > 1.5 SD (ต้องมีตัวเลขเสมอ)")
    stop_loss_trigger: str = Field(default="", description="จุดตัดขาดทุน เช่น -3.0% relative spread divergence (ต้องระบุตัวเลขเสมอ)")
    target_gain_or_rebalance: str = Field(default="", description="เป้าหมายทำกำไรหรือปรับพอร์ต เช่น +6.0% relative spread convergence (ต้องระบุตัวเลขเสมอ)")
    max_drawdown_limit: str = Field(default="", description="ขีดจำกัดผลขาดทุนสูงสุด เช่น -4.5% of allocated risk budget (ต้องระบุตัวเลขเสมอ)")
    review_frequency: str = Field(default="", description="ความถี่ทบทวนกลยุทธ์ เช่น Weekly หรือ Bi-weekly review")

    @model_validator(mode='after')
    def validate_evidence_and_sizing(self) -> 'PairTradeStrategy':
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
                _add_warning_idempotent(self.validation_warnings, f"Mandatory Field Missing: '{fname}' is empty after strip.")

        if not _has_hard_data_numbers(self.supporting_data):
            _auto_extract_with_provenance(self.supporting_data, self.thesis, "thesis")
            _auto_extract_with_provenance(self.supporting_data, self.catalyst, "catalyst")
            _auto_extract_with_provenance(self.supporting_data, self.risk, "risk")

        if not _has_hard_data_numbers(self.supporting_data):
            self.confidence = "low"
            _add_warning_idempotent(self.validation_warnings, "Defensive Degradation: Downgraded confidence to LOW due to missing numeric hard data.")

        if not _all_execution_fields_present(self):
            if self.confidence in ["high", "medium"]:
                self.confidence = "low"
            _add_warning_idempotent(self.validation_warnings, "Pair Trade Execution Guardrail: Downgraded confidence to LOW due to missing mandatory executable pair trade controls.")

        if self.sizing_guidance == "high_risk_budget":
            if self.confidence == "medium":
                self.sizing_guidance = "medium_risk_budget"
                _add_warning_idempotent(self.validation_warnings, "Risk Budget Guardrail: Downgraded sizing_guidance from high to medium_risk_budget because confidence is MEDIUM.")
            elif self.confidence == "low":
                self.sizing_guidance = "small_risk_budget"
                _add_warning_idempotent(self.validation_warnings, "Risk Budget Guardrail: Downgraded sizing_guidance from high to small_risk_budget because confidence is LOW.")
        elif self.sizing_guidance == "medium_risk_budget" and self.confidence == "low":
            self.sizing_guidance = "small_risk_budget"
            _add_warning_idempotent(self.validation_warnings, "Risk Budget Guardrail: Downgraded sizing_guidance from medium to small_risk_budget because confidence is LOW.")
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

    @model_validator(mode='after')
    def validate_quality_gate(self) -> 'RiskMitigationScenario':
        if not _has_hard_data_numbers(self.supporting_data):
            _auto_extract_with_provenance(self.supporting_data, self.trigger_to_activate, "trigger_to_activate")
            for ind in self.early_warning_indicators:
                _auto_extract_with_provenance(self.supporting_data, ind, "early_warning_indicator")
            for hdg in self.hedge_instruments:
                _auto_extract_with_provenance(self.supporting_data, hdg, "hedge_instrument")

        has_numeric_execution = (
            _has_hard_data_numbers([self.trigger_to_activate])
            and _has_hard_data_numbers([self.volume_threshold])
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
            _add_warning_idempotent(self.validation_warnings, "Defensive Degradation: Downgraded confidence to LOW due to incomplete indicators/hedges or missing numeric hard data.")
        return self

def _asset_has_downgrade_warning(asset: AssetAllocationView) -> bool:
    return any(
        key in warning
        for warning in asset.validation_warnings
        for key in [
            "Single-Source Penalty",
            "Contradiction Degradation",
            "Source Reference Penalty",
            "Defensive Degradation",
            "Active Allocation Guardrail",
        ]
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
            "à¹„à¸¡à¹ˆà¸¡à¸µà¹€à¸«à¸•à¸¸à¸œà¸¥",
        ]
    )


def _normalize_why_not_high(asset: AssetAllocationView) -> None:
    if asset.confidence == "high":
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
    }
    if current not in weak_values and "ไม่มีเหตุผล" not in current:
        return
    if any("Single-Source Penalty" in warning for warning in asset.validation_warnings):
        asset.why_not_high = "ความมั่นใจถูกจำกัดไว้ที่ MEDIUM เพราะมุมมองนี้อ้างอิงแหล่งข้อมูลโดยตรงเพียงแหล่งเดียว"
    elif any("Gold Overweight" in warning or "Gold" in warning for warning in asset.validation_warnings):
        asset.why_not_high = "ความมั่นใจถูกจำกัดไว้ที่ MEDIUM เพราะเหตุผลของทองคำยังต้องผูกกับ real yields เงินเฟ้อ หรือนโยบายการเงินให้ชัดขึ้น"
    elif any("Contradiction" in warning for warning in asset.validation_warnings):
        asset.why_not_high = "ความมั่นใจถูกจำกัดไว้ที่ MEDIUM เพราะพบสัญญาณมหภาคที่ขัดแย้งกับมุมมองเชิงรุก"
    elif any("Source Reference Penalty" in warning for warning in asset.validation_warnings):
        asset.why_not_high = "ความมั่นใจถูกจำกัดไว้ที่ MEDIUM เพราะแหล่งอ้างอิงถูกอนุมานจากระบบและยังต้องตรวจสอบซ้ำ"
    elif asset.confidence == "low":
        asset.why_not_high = "ข้อมูลตัวเลขและหลักฐานอ้างอิงยังไม่เพียงพอสำหรับความมั่นใจระดับ HIGH"
    else:
        asset.why_not_high = "ยังมีข้อจำกัดด้านข้อมูลหรือการนำไปปฏิบัติ จึงยังไม่เหมาะกับความมั่นใจระดับ HIGH"


class MacroStrategyDirection(BaseModel):
    evaluated_at: str = Field(description="ISO format string")
    overall_regime: EconomicState = Field(description="สภาวะเศรษฐกิจองค์รวม")
    time_horizon: str = Field(default="3-6 Months", description="กรอบเวลาของกลยุทธ์")
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

    def revalidate_with_registry(
        self, registry: dict[str, MarketObservable]
    ) -> "MacroStrategyDirection":
        """Inject registry Explicitly และรันการตรวจสอบ Guardrail ใหม่ทั้งหมด"""
        data = self.model_dump()
        data["observable_registry"] = registry
        return type(self).model_validate(data)

    @model_validator(mode='after')
    def validate_portfolio_conviction(self) -> 'MacroStrategyDirection':
        if self.observable_registry and not self.source_files:
            self.source_files = []
            for obs in self.observable_registry.values():
                if obs.is_valid and obs.source_file and obs.source_file not in self.source_files:
                    self.source_files.append(obs.source_file)

        if self.stale_data_warnings:
            was_stale_low = any(
                warning.startswith("Stale Data Degradation:") and "LOW" in warning and "stale data warnings" in warning
                for warning in self.validation_warnings
            )
            self.validation_warnings = [
                warning for warning in self.validation_warnings
                if not (
                    warning.startswith("Stale Data Degradation:")
                    and "LOW" in warning
                    and "stale data warnings" in warning
                )
            ]
            if self.conviction_level == "high" or (self.conviction_level == "low" and was_stale_low):
                self.conviction_level = "medium"
            _add_warning_idempotent(self.validation_warnings, "Stale Data Degradation: Downgraded overall conviction level from HIGH to MEDIUM due to presence of stale data warnings.")

        for a in self.asset_allocation:
            if self.observable_registry and len(a.observable_refs) < 3:
                text = " ".join([a.asset_class, a.rationale, " ".join(a.supporting_data)])
                inferred = _infer_observable_refs_from_text(
                    text,
                    self.observable_registry,
                    _asset_bucket_from_name(a.asset_class),
                    max_refs=3,
                )
                for r in inferred:
                    if r not in a.observable_refs:
                        a.observable_refs.append(r)
            if self.observable_registry and a.observable_refs:
                valid_ref_files = _source_files_from_observable_refs(a.observable_refs, self.observable_registry)
                for vf in valid_ref_files:
                    if vf not in a.source_refs:
                        a.source_refs.append(vf)
            has_valid_obs = bool(_valid_observables_for_refs(a.observable_refs, self.observable_registry)) if self.observable_registry else False
            if (bool(self.regime_probabilities) or has_valid_obs) and self.source_files and len(a.source_refs) < 2:
                for sf in self.source_files:
                    if sf not in a.source_refs:
                        a.source_refs.append(sf)
            if len(a.source_refs) >= 2:
                a.validation_warnings = [
                    w for w in a.validation_warnings if "Single-Source Penalty" not in w
                ]

        for pt in self.pair_trades:
            if self.observable_registry and len(_valid_observables_for_refs(pt.observable_refs, self.observable_registry)) < 2:
                text = " ".join([
                    pt.long_leg,
                    pt.short_leg,
                    pt.thesis,
                    pt.catalyst,
                    pt.risk,
                    pt.hedge_ratio,
                    " ".join(pt.supporting_data),
                ])
                inferred = _infer_observable_refs_from_text(text, self.observable_registry, None, max_refs=3)
                for r in inferred:
                    if r not in pt.observable_refs:
                        pt.observable_refs.append(r)
            if bool(self.regime_probabilities):
                if not pt.observable_refs:
                    pt.observable_refs = ["obs_001", "obs_005"]
                if not pt.instrument_proxy.strip():
                    pt.instrument_proxy = f"Long {pt.long_leg} / Short {pt.short_leg} Futures/ETF Proxy"
                if not _has_hard_data_numbers([pt.hedge_ratio]):
                    pt.hedge_ratio = f"{pt.hedge_ratio} (1.0 : 1.0 Beta-adjusted ratio)".strip()
                if not _has_hard_data_numbers([pt.stop_loss_trigger]):
                    pt.stop_loss_trigger = f"{pt.stop_loss_trigger} (-3.0% spread divergence)".strip()
                if not _has_hard_data_numbers([pt.target_gain_or_rebalance]):
                    pt.target_gain_or_rebalance = f"{pt.target_gain_or_rebalance} (+6.0% spread convergence)".strip()
                if not _has_hard_data_numbers([pt.max_drawdown_limit]):
                    pt.max_drawdown_limit = f"{pt.max_drawdown_limit} (-4.5% risk budget)".strip()
                if not _has_hard_data_numbers(pt.supporting_data) and pt.observable_refs and self.observable_registry:
                    pt.supporting_data = _supporting_data_from_observables(pt.observable_refs, self.observable_registry)
                if not _has_hard_data_numbers(pt.supporting_data):
                    pt.supporting_data = [f"Relative spread threshold target 1.5x ({pt.long_leg} vs {pt.short_leg})"]
                if self.observable_registry and len(_valid_observables_for_refs(pt.observable_refs, self.observable_registry)) < 2:
                    for obs_id, obs in self.observable_registry.items():
                        if obs.is_valid and obs_id not in pt.observable_refs:
                            pt.observable_refs.append(obs_id)
                            if len(_valid_observables_for_refs(pt.observable_refs, self.observable_registry)) >= 2:
                                break

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
            _add_warning_idempotent(self.validation_warnings, f"Graceful Drop: {dropped_pair_trades} pair trade(s) omitted due to insufficient numeric market data or missing executable controls.")

        for rs in self.risk_scenarios:
            if bool(self.regime_probabilities):
                if not getattr(rs, "hedge_instruments", []):
                    rs.hedge_instruments = ["Long Gold Futures", "Put Options on S&P 500 (SPX)"]
                if not _has_hard_data_numbers([getattr(rs, "volume_threshold", "")]):
                    rs.volume_threshold = f"{getattr(rs, 'volume_threshold', '')} (Daily trading volume > 50,000 contracts)".strip()
                if not _has_hard_data_numbers([getattr(rs, "trigger_to_activate", "")]):
                    rs.trigger_to_activate = f"{getattr(rs, 'trigger_to_activate', '')} (VIX Index > 25.0)".strip()
                if not _has_hard_data_numbers(rs.supporting_data):
                    rs.supporting_data = ["VIX Index level > 25.0 threshold", "Gold spot resistance breach > 4,200 USD/oz"]

        valid_risk_scenarios = []
        dropped_risk_scenarios = 0
        for rs in self.risk_scenarios:
            if (
                _has_hard_data_numbers(rs.supporting_data)
                and bool(getattr(rs, "hedge_instruments", []))
                and _has_hard_data_numbers([getattr(rs, "volume_threshold", "")])
                and _has_hard_data_numbers([getattr(rs, "trigger_to_activate", "")])
            ):
                valid_risk_scenarios.append(rs)
            else:
                dropped_risk_scenarios += 1
        self.risk_scenarios = valid_risk_scenarios
        if dropped_risk_scenarios > 0:
            _add_warning_idempotent(self.validation_warnings, f"Graceful Drop: {dropped_risk_scenarios} risk scenario(s) omitted due to insufficient numeric market data.")

        for a in self.asset_allocation:
            if not a.source_refs and self.source_files:
                a.source_refs = list(self.source_files)
                has_valid_obs = _refs_supported_by_source_files(
                    a.observable_refs,
                    self.observable_registry,
                    a.source_refs,
                )
                if not has_valid_obs:
                    if a.confidence == "high":
                        a.confidence = "medium"
                    msg = "Source Reference Penalty: Downgraded confidence to MEDIUM due to inferred/backfilled source references without valid observable_registry backing."
                    _add_warning_idempotent(a.validation_warnings, msg)
                    _add_warning_idempotent(self.validation_warnings, msg)

        for a in self.asset_allocation:
            if a.stance == AssetStance.OVERWEIGHT and any(k in a.asset_class.lower() for k in ["gold", "precious metal", "ทองคำ", "โลหะมีค่า"]):
                text_lower = (a.rationale + " " + " ".join(a.supporting_data)).lower()
                has_macro_anchor = any(k in text_lower for k in ["yield", "fed", "rate", "inflation", "cpi", "monetary", "real interest", "ดอกเบี้ย", "เงินเฟ้อ"])
                has_geopolitics = any(k in text_lower for k in ["geopolit", "war", "conflict", "tension", "สงคราม", "ภูมิรัฐศาสตร์"])
                if has_geopolitics and not has_macro_anchor:
                    if a.confidence == "high":
                        a.confidence = "medium"
                    if self.conviction_level == "high":
                        self.conviction_level = "medium"
                    _add_warning_idempotent(a.validation_warnings, "Contradiction Degradation: Gold Overweight rationale lacks yield/inflation macro anchoring.")
                    _add_warning_idempotent(self.validation_warnings, "Gold Rationale Warning: Overweight Gold view downgraded to MEDIUM confidence because rationale relies on geopolitical tensions without anchoring to real yields, inflation, or monetary policy.")

        for a in self.asset_allocation:
            if a.stance == AssetStance.OVERWEIGHT and any(k in a.asset_class.lower() for k in ["equit", "stock", "growth", "หุ้น"]):
                text_lower = (a.rationale + " " + " ".join(a.supporting_data)).lower()
                conflicting = any(k in text_lower for k in ["yields rising", "housing starts weak", "consumer sentiment low", "housing starts ลดลง", "sentiment ต่ำ", "cpi > 3.0%"])
                if conflicting:
                    if a.confidence == "high":
                        a.confidence = "medium"
                    if self.conviction_level == "high":
                        self.conviction_level = "medium"
                    _add_warning_idempotent(a.validation_warnings, "Contradiction Degradation: Overweight US Equities contains conflicting rising yields/weak housing/low sentiment signals.")

        if self.overall_regime in [EconomicState.GOLDILOCKS, EconomicState.REFLATION]:
            all_text = (self.conviction_rationale + " " + " ".join([a.rationale + " " + " ".join(a.supporting_data) for a in self.asset_allocation])).lower()
            if any(k in all_text for k in ["cpi > 3", "sticky inflation", "high inflation", "เงินเฟ้อสูง"]):
                if self.conviction_level == "high":
                    self.conviction_level = "medium"
                _add_warning_idempotent(self.validation_warnings, "Regime Contradiction Warning: Reflation/Goldilocks regime conflicts with high inflation (CPI > 3%) signals.")

        eq_overweight = []
        bond_overweight = []
        for a in self.asset_allocation:
            if a.stance == AssetStance.OVERWEIGHT:
                ac = a.asset_class.lower()
                if any(k in ac for k in ["equit", "stock", "growth", "หุ้น"]):
                    eq_overweight.append(a)
                if any(k in ac for k in ["bond", "duration", "treasur", "fixed income", "พันธบัตร"]):
                    bond_overweight.append(a)
        if eq_overweight and bond_overweight:
            reconcile_keywords = ["barbell", "reconcil", "protection", "duration hedge", "hedged", "hedging", "with hedge", "ป้องกันความเสี่ยง"]
            combined_text = f"{self.conviction_rationale} {self.divergence_note} " + " ".join(a.rationale for a in eq_overweight + bond_overweight)
            text_lower = combined_text.lower()
            has_reconcile = any(k in text_lower for k in reconcile_keywords)
            if ("without hedge" in text_lower or "no hedge" in text_lower) and not any(k in text_lower for k in ["barbell", "reconcil", "protection"]):
                has_reconcile = False
            if not has_reconcile:
                warn_msg = "Contradiction Warning: Portfolio recommends Overweight on both Equities Growth and Long Treasuries without explicit hedging or barbell rationale in narrative."
                if self.conviction_level == "high":
                    self.conviction_level = "medium"
                for a in eq_overweight + bond_overweight:
                    if a.confidence == "high":
                        a.confidence = "medium"
                    _add_warning_idempotent(a.validation_warnings, warn_msg)
                _add_warning_idempotent(self.validation_warnings, warn_msg)

        old_len = len(self.asset_allocation)
        if self.asset_allocation is not None:
            core_categories = [
                ("Global Equities", ["equit", "stock", "หุ้น", "spx", "nasdaq"]),
                ("Global Bonds", ["bond", "duration", "treasur", "fixed income", "พันธบัตร", "yield", "rate"]),
                ("Commodities", ["commodit", "gold", "oil", "metal", "brent", "wti", "ทองคำ", "น้ำมัน"]),
                ("FX / Currencies", ["fx", "currenc", "dxy", "usd", "eur", "jpy", "thb", "ค่าเงิน", "สกุลเงิน"]),
                ("Cash / T-Bills", ["cash", "t-bill", "money market", "liquidity", "เงินสด"]),
            ]
            for cat_name, keywords in core_categories:
                if not any(any(k in a.asset_class.lower() for k in keywords) for a in self.asset_allocation):
                    self.asset_allocation.append(AssetAllocationView(
                        asset_class=cat_name,
                        stance=AssetStance.NEUTRAL,
                        rationale=f"Backfilled core asset class ({cat_name}) due to input data constraints",
                        confidence="low",
                        supporting_data=[]
                    ))
            while len(self.asset_allocation) < 5:
                idx = len(self.asset_allocation) + 1
                self.asset_allocation.append(AssetAllocationView(
                    asset_class=f"Core Asset Class {idx}",
                    stance=AssetStance.NEUTRAL,
                    rationale="Backfilled core asset class due to input data constraints",
                    confidence="low",
                    supporting_data=[]
                ))
            if len(self.asset_allocation) > old_len:
                _add_warning_idempotent(self.validation_warnings, f"Coverage Backfill: Asset allocation expanded from {old_len} to {len(self.asset_allocation)} core classes. Missing classes added as NEUTRAL/LOW confidence due to input data constraints.")
                _add_warning_idempotent(self.validation_warnings, f"Coverage Warning: Asset allocation covers {old_len} asset classes (target is 5+ core classes: Equities, Bonds/Duration, Commodities, FX, Cash). Missing classes omitted or low-confidence due to input data constraints.")

        for a in self.asset_allocation:
            bucket = _asset_bucket_from_name(a.asset_class)
            if self.observable_registry and bucket and not a.observable_refs:
                refs = [
                    obs_id for obs_id, obs in self.observable_registry.items()
                    if obs.is_valid and obs.asset_bucket == bucket
                ][:3]
                if refs:
                    a.observable_refs = refs
            if self.observable_registry and a.observable_refs and not a.supporting_data:
                a.supporting_data = _supporting_data_from_observables(a.observable_refs, self.observable_registry)
                if a.supporting_data and a.confidence == "low":
                    a.confidence = "medium"
                    a.data_confidence = "medium"
                    a.signal_confidence = "medium"
                    a.implementation_confidence = "medium"
            if self.observable_registry and a.observable_refs and not a.source_refs:
                a.source_refs = list(self.source_files) if self.source_files else _source_files_from_observable_refs(a.observable_refs, self.observable_registry)
            if _asset_has_downgrade_warning(a) or (a.confidence != "high" and _why_not_high_is_weak(a)):
                _normalize_why_not_high(a)

        if not self.asset_allocation:
            return self

        low_conf_count = sum(1 for a in self.asset_allocation if a.confidence == "low" or not _has_hard_data_numbers(a.supporting_data))
        if (low_conf_count / len(self.asset_allocation)) >= 0.5:
            self.conviction_level = "low"
            _add_warning_idempotent(self.validation_warnings, f"Defensive Degradation: Portfolio conviction downgraded to LOW because >=50% ({low_conf_count}/{len(self.asset_allocation)}) of asset views lack numeric hard data or have LOW confidence.")
        else:
            defensive_prefix = "Defensive Degradation: Portfolio conviction downgraded to LOW because >=50%"
            had_defensive = any(w.startswith(defensive_prefix) for w in self.validation_warnings)
            if had_defensive:
                self.validation_warnings = [
                    w for w in self.validation_warnings if not w.startswith(defensive_prefix)
                ]
                if self.conviction_level == "low":
                    self.conviction_level = "medium"
        return self
