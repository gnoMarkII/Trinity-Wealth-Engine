"""Tests for MOF Thailand Public Debt and SEC Thailand Industry Adapters."""
import pytest
from tools.market.terminal_v2.adapters.mof_th_adapter import (
    parse_public_debt_csv,
    MofThailandAdapter,
)
from tools.market.terminal_v2.adapters.sec_th_adapter import (
    parse_industry_csv,
    SecThailandAdapter,
    SECTOR_GROUP,
    GROUP_CODES,
)
from tools.market.terminal_v2.domain.errors import ProviderError

MOF_SAMPLE_CSV = """\ufeff"ลำดับที่","กุมภาพันธ์ 2569","มกราคม 2569"
"1. หนี้รัฐบาล","10,500,000.50","10,400,000.00"
"2. หนี้รัฐวิสาหกิจ","1,050,000.00","1,040,000.00"
"3. หนี้รัฐวิสาหกิจที่ทำธุรกิจในภาคการเงินฯ (รัฐบาลค้ำประกัน)","200,000.00","200,000.00"
"4. หนี้กองทุนเพื่อการฟื้นฟูและพัฒนาระบบสถาบันการเงิน","600,000.00","610,000.00"
"5. หนื้หน่วยงานอื่นของรัฐ","50,000.00","49,000.00"
"สัดส่วนหนี้สาธารณะต่อ GDP (Debt : GDP) (%)","63.85","63.50"
"อัตราแลกเปลี่ยน (บาท/ดอลลาร์สหรัฐ)","35.50","35.40"
"หมายเหตุเพิ่มเติม: 1. ข้อมูลเบื้องต้น","",""
"""

SEC_SAMPLE_CSV = """\ufeff"วันที่มีผลบังคับใช้","ตลาด","กลุ่มอุตสาหกรรม/หมวดธุรกิจ","ปี","ไตรมาส","มูลค่าหลักทรัพย์ตามราคาตลาด (ล้านบาท)"
"30/06/2569","SET","AGRI   (ธุรกิจการเกษตร)","2569","ไตรมาส 2","150,000.00"
"30/06/2569","SET","FOOD   (อาหารและเครื่องดื่ม)","2569","ไตรมาส 2","450,000.00"
"30/06/2569","SET","BANK   (ธนาคาร)","2569","ไตรมาส 2","1,200,000.00"
"30/06/2569","SET","FIN   (เงินทุนและหลักทรัพย์)","2569","ไตรมาส 2","300,000.00"
"30/06/2569","SET","INSUR  (ประกันภัยและประกันชีวิต)","2569","ไตรมาส 2","100,000.00"
"30/06/2569","SET","COMM   (พาณิชย์)","2569","ไตรมาส 2","800,000.00"
"30/06/2569","SET","HELTH  (การแพทย์)","2569","ไตรมาส 2","600,000.00"
"30/06/2569","SET","MEDIA  (สื่อและสิ่งพิมพ์)","2569","ไตรมาส 2","50,000.00"
"30/06/2569","SET","PROF   (บริการเฉพาะกิจ)","2569","ไตรมาส 2","20,000.00"
"30/06/2569","SET","TOURISM (การท่องเที่ยว)","2569","ไตรมาส 2","180,000.00"
"30/06/2569","SET","TRANS  (ขนส่งและโลจิสติกส์)","2569","ไตรมาส 2","750,000.00"
"30/06/2569","SET","ETRON  (ชิ้นส่วนอิเล็กทรอนิกส์)","2569","ไตรมาส 2","900,000.00"
"30/06/2569","SET","ICT    (เทคโนโลยีสารสนเทศ)","2569","ไตรมาส 2","850,000.00"
"30/06/2569","SET","ENERG  (พลังงานและสาธารณูปโภค)","2569","ไตรมาส 2","2,100,000.00"
"30/06/2569","SET","MINE   (เหมืองแร่)","2569","ไตรมาส 2","10,000.00"
"30/06/2569","SET","CONMAT (วัสดุก่อสร้าง)","2569","ไตรมาส 2","350,000.00"
"30/06/2569","SET","CONS   (บริการรับเหมาก่อสร้าง)","2569","ไตรมาส 2","70,000.00"
"30/06/2569","SET","PROP   (พัฒนาอสังหาริมทรัพย์)","2569","ไตรมาส 2","650,000.00"
"30/06/2569","SET","PF&REIT (กองทุนรวมอสังหาฯ)","2569","ไตรมาส 2","250,000.00"
"30/06/2569","SET","AUTO   (ยานยนต์)","2569","ไตรมาส 2","60,000.00"
"30/06/2569","SET","IMM    (วัสดุอุตสาหกรรม)","2569","ไตรมาส 2","30,000.00"
"30/06/2569","SET","PAPER  (กระดาษ)","2569","ไตรมาส 2","10,000.00"
"30/06/2569","SET","PETRO  (ปิโตรเคมี)","2569","ไตรมาส 2","220,000.00"
"30/06/2569","SET","PKG    (บรรจุภัณฑ์)","2569","ไตรมาส 2","180,000.00"
"30/06/2569","SET","STEEL  (เหล็ก)","2569","ไตรมาส 2","40,000.00"
"30/06/2569","SET","FASHION (แฟชั่น)","2569","ไตรมาส 2","35,000.00"
"30/06/2569","SET","HOME   (ของใช้ในครัวเรือน)","2569","ไตรมาส 2","25,000.00"
"30/06/2569","SET","PERSON (ของใช้ส่วนตัว)","2569","ไตรมาส 2","40,000.00"
"""

WAF_REJECTION_HTML = """<html>
<head><title>Request Rejected</title></head>
<body>The requested URL was rejected. Please consult with your administrator.<br><br>Your support ID is: 1234567890</body>
</html>"""


def test_mof_public_debt_parsing_and_last_day_iso():
    snapshot = parse_public_debt_csv(MOF_SAMPLE_CSV)
    # Check that date is the last day of February 2026
    assert snapshot.reporting_month == "2026-02-28"
    assert snapshot.debt_to_gdp_pct == 63.85
    assert snapshot.fx_rate_usd_thb == 35.50

    # 5 components
    assert len(snapshot.components) == 5
    comp_map = {c.component_number: c.amount_thb for c in snapshot.components}
    assert comp_map[1] == 10500000.50 * 1e6
    assert comp_map[2] == 1050000.00 * 1e6
    assert comp_map[3] == 200000.00 * 1e6
    assert comp_map[4] == 600000.00 * 1e6
    # Row 5 with typo "หนื้" was correctly captured
    assert comp_map[5] == 50000.00 * 1e6

    total = sum(comp_map.values())
    assert snapshot.total_debt_thb == total


def test_mof_macro_observables():
    snapshot = parse_public_debt_csv(MOF_SAMPLE_CSV)
    adapter = MofThailandAdapter()
    observables = adapter.as_macro_observables(snapshot)

    assert len(observables) == 2
    obs_debt = next(o for o in observables if o.observable_id == "obs_th_public_debt_mof")
    obs_gdp = next(o for o in observables if o.observable_id == "obs_th_debt_to_gdp_mof")

    assert obs_debt.is_valid is True
    assert obs_debt.status == "verified"
    assert obs_debt.observed_at == "2026-02-28"
    assert obs_debt.provider == "MOF Thailand"

    assert obs_gdp.is_valid is True
    assert obs_gdp.status == "verified"
    assert obs_gdp.observed_at == "2026-02-28"
    assert obs_gdp.value == "63.85"
    assert obs_gdp.metadata["is_within_limit"] is True


def test_sec_industry_stat_parsing_and_sector_group():
    snapshot = parse_industry_csv(SEC_SAMPLE_CSV, market="SET")
    assert snapshot.market == "SET"
    assert snapshot.as_of == "2026-06-30"
    assert snapshot.reporting_period == "2026 Q2"

    # All 28 sectors are represented
    assert len(snapshot.sectors) == 28
    sector_map = {s.sector_code: s for s in snapshot.sectors}
    assert "BANK" in sector_map
    assert sector_map["BANK"].group_code == "FINCIAL"
    assert sector_map["BANK"].market_cap_thb == 1200000.0 * 1e6

    assert "ENERG" in sector_map
    assert sector_map["ENERG"].group_code == "RESOURC"

    # All 8 groups are represented
    assert len(snapshot.groups) == 8
    group_map = {g.group_code: g for g in snapshot.groups}
    assert set(group_map.keys()) == GROUP_CODES

    # Agro group = AGRI (150,000) + FOOD (450,000) = 600,000 million THB
    assert group_map["AGRO"].market_cap_thb == 600000.0 * 1e6

    # Financials group = BANK (1,200,000) + FIN (300,000) + INSUR (100,000) = 1,600,000 million THB
    assert group_map["FINCIAL"].market_cap_thb == 1600000.0 * 1e6


def test_sec_waf_rejection_raises_provider_error():
    with pytest.raises(ProviderError) as exc_info:
        parse_industry_csv(WAF_REJECTION_HTML, market="SET")
    assert "WAF" in str(exc_info.value) or "rejected" in str(exc_info.value).lower()


def test_thai_yield_curve_and_policy_rate_observables():
    from tools.macro.adapters.thai_hard_data_adapter import ThaiHardDataAdapter, ThaiHardDataRecord

    # 1. Matching dates: 10Y 3.0% and 2Y 2.5% on 2026-09-30 -> spread 50.0 bps (TH-AC14)
    records_matching = {
        "TH_GOV_YIELD_2Y": ThaiHardDataRecord(
            series_id="TH_GOV_YIELD_2Y",
            indicator_name="Thailand 2Y Gov Bond Yield",
            source_authority="ThaiBMA",
            frequency="Daily",
            unit="%",
            value=2.50,
            observed_at="2026-09-30",
            is_verified=True,
            status="verified",
        ),
        "TH_GOV_YIELD_10Y": ThaiHardDataRecord(
            series_id="TH_GOV_YIELD_10Y",
            indicator_name="Thailand 10Y Gov Bond Yield",
            source_authority="ThaiBMA",
            frequency="Daily",
            unit="%",
            value=3.00,
            observed_at="2026-09-30",
            is_verified=True,
            status="verified",
        ),
        "TH_POLICY_RATE": ThaiHardDataRecord(
            series_id="TH_POLICY_RATE",
            indicator_name="Bank of Thailand Policy Rate",
            source_authority="Bank of Thailand (BOT / ธปท.)",
            frequency="Event / Daily",
            unit="% per annum",
            value=2.50,
            observed_at="2026-09-30",
            is_verified=True,
            status="verified",
        ),
    }

    adapter = ThaiHardDataAdapter(override_records=records_matching)
    observables = adapter.as_observables("2026-09-30")

    spread_obs = next(o for o in observables if o.observable_id == "obs_th_gov_10y_2y_spread")
    assert spread_obs.is_valid is True
    assert spread_obs.status == "verified"
    assert spread_obs.value == "50.0"
    assert spread_obs.metadata["diff_bps"] == 50.0

    rate_obs = next(o for o in observables if o.observable_id == "obs_thai_policy_rate_bot")
    assert rate_obs.is_valid is True
    assert rate_obs.value == "2.50"

    # 2. Mismatched dates: 10Y on 2026-09-30, 2Y on 2026-09-28 -> spread invalid (TH-AC14)
    records_mismatched = {
        "TH_GOV_YIELD_2Y": ThaiHardDataRecord(
            series_id="TH_GOV_YIELD_2Y",
            indicator_name="Thailand 2Y Gov Bond Yield",
            source_authority="ThaiBMA",
            frequency="Daily",
            unit="%",
            value=2.50,
            observed_at="2026-09-28",  # Different date
            is_verified=True,
            status="verified",
        ),
        "TH_GOV_YIELD_10Y": ThaiHardDataRecord(
            series_id="TH_GOV_YIELD_10Y",
            indicator_name="Thailand 10Y Gov Bond Yield",
            source_authority="ThaiBMA",
            frequency="Daily",
            unit="%",
            value=3.00,
            observed_at="2026-09-30",
            is_verified=True,
            status="verified",
        ),
    }
    adapter_mismatch = ThaiHardDataAdapter(override_records=records_mismatched)
    obs_mismatch = adapter_mismatch.as_observables("2026-09-30")
    spread_mismatched_obs = next(o for o in obs_mismatch if o.observable_id == "obs_th_gov_10y_2y_spread")
    assert spread_mismatched_obs.is_valid is False
    assert spread_mismatched_obs.status == "mismatched_date"

    # 3. Explicit empty records -> blocked with ThaiBMA access gate reason (TH-07)
    adapter_empty = ThaiHardDataAdapter(override_records={})
    obs_empty = adapter_empty.as_observables("2026-09-30")
    spread_blocked_obs = next(o for o in obs_empty if o.observable_id == "obs_th_gov_10y_2y_spread")
    assert spread_blocked_obs.is_valid is False
    assert spread_blocked_obs.status == "blocked"
    assert "ThaiBMA" in spread_blocked_obs.stale_reason

    # 4. Default loaded records from ThaiBMA sync -> verified
    adapter_default = ThaiHardDataAdapter()
    obs_default = adapter_default.as_observables()
    spread_default_obs = next(o for o in obs_default if o.observable_id == "obs_th_gov_10y_2y_spread")
    assert spread_default_obs.is_valid is True
    assert spread_default_obs.status == "verified"


def test_official_csv_file_import_with_bom_and_buddhist_era():
    from tools.macro.adapters.thai_hard_data_adapter import ThaiHardDataAdapter

    sample_official_csv = """\ufeff"period","value","prev","ma","published_at"
"2569-Q2","2.5","2.3","2.2","2026-08-18"
"""
    adapter = ThaiHardDataAdapter()
    rec = adapter.import_official_csv(sample_official_csv, "TH_REAL_GDP", default_authority="NESDC")
    assert rec.is_verified is True
    assert rec.status == "verified"
    assert rec.value == 2.5
    assert rec.prev == 2.3
    assert rec.ma == 2.2
    assert rec.period == "2026-Q2"  # 2569 converted to 2026

    # Test missing value '-' does not convert to 0 (TH-AC40)
    sample_blank_csv = """\ufeff"period","value","prev","ma"
"2569-08","-","0.6","0.7"
"""
    rec_blank = adapter.import_official_csv(sample_blank_csv, "TH_CPI_YOY", default_authority="TPSO MOC")
    assert rec_blank.value is None  # Never converted to 0
    assert rec_blank.is_verified is False
    assert rec_blank.status == "missing"

    # Test WAF error rejection handling (TH-AC40)
    rec_waf = adapter.import_official_csv(WAF_REJECTION_HTML, "TH_REAL_GDP")
    assert rec_waf.is_verified is False
    assert rec_waf.status == "waf_blocked"


def test_thai_yield_curve_in_scoring():
    from tools.macro.scoring import _calculate_matrix_scores_from_observables
    from schemas.macro_schemas import MarketObservable

    obs_spread = MarketObservable(
        observable_id="obs_th_gov_10y_2y_spread",
        asset_bucket="fixed_income",
        region="Thailand",
        indicator="Thailand Gov Bond 10Y-2Y Spread",
        value="50.0",
        unit="bps",
        observed_at="2026-09-30",
        source_file="ThaiBMA",
        is_valid=True,
        status="verified",
        metadata={"val": 50.0},
    )

    scores = _calculate_matrix_scores_from_observables([obs_spread])
    th = scores.get("Thailand", {})
    assert "thai_yield_curve" in th
    assert th["thai_yield_curve"]["is_available"] is True
    assert th["thai_yield_curve"]["spread_10y_2y_bps"] == 50.0


def test_thaibma_adapter_snapshot_and_observables(monkeypatch):
    import requests
    from tools.market.terminal_v2.adapters.thaibma_adapter import ThaiBmaPublicAdapter
    from unittest.mock import MagicMock

    mock_avail_resp = MagicMock()
    mock_avail_resp.status_code = 200
    mock_avail_resp.json.return_value = ["2000-01-01T00:00:00", "2026-10-02T00:00:00"]

    mock_gov_resp = MagicMock()
    mock_gov_resp.status_code = 200
    mock_gov_resp.json.return_value = {
        "Curve": [
            {"X": 1.0, "Y": 1.25},
            {"X": 2.0, "Y": 1.3877},
            {"X": 5.0, "Y": 1.75},
            {"X": 10.0, "Y": 2.3914},
        ]
    }

    def mock_get(url, *args, **kwargs):
        if "avail" in url:
            return mock_avail_resp
        return mock_gov_resp

    monkeypatch.setattr(requests, "get", mock_get)

    adapter = ThaiBmaPublicAdapter()
    snap = adapter.get_government_yield_curve()
    assert snap.observation_date == "2026-10-02"
    assert snap.spread_10y_2y_bps == 100.4
    assert snap.source == "ThaiBMA"
    assert len(snap.yields) == 4

    observables = adapter.as_macro_observables(snap)
    assert len(observables) == 5  # 2Y, 10Y, Spread, 1Y, 5Y
    obs_map = {o.observable_id: o for o in observables}
    assert obs_map["obs_th_gov_yield_2y"].value == "1.39"
    assert obs_map["obs_th_gov_yield_10y"].value == "2.39"
    assert obs_map["obs_th_gov_10y_2y_spread"].value == "100.4"
    assert obs_map["obs_th_gov_10y_2y_spread"].status == "verified"


