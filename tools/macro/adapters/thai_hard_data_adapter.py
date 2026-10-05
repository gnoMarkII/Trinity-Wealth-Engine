"""Thai Hard Data Feasibility & Status Adapter (Dual-Track Macro Intelligence - Phase 4 & Revision 3).

Tracks feasibility, status, and verified release sources for Thai official macroeconomic hard data:
- Real GDP: Office of the National Economic and Social Development Council (NESDC / สภาพัฒน์)
- CPI / Headline Inflation: Trade Policy and Strategy Office, Ministry of Commerce (TPSO / MOC / กระทรวงพาณิชย์)
- Core CPI: Trade Policy and Strategy Office, Ministry of Commerce (TPSO / MOC / กระทรวงพาณิชย์)
- Policy Interest Rate: Monetary Policy Committee, Bank of Thailand (BOT / ธปท.)
- Manufacturing Production Index (MPI): Office of Industrial Economics (OIE / สศอ.)
- Public Debt & Debt to GDP: Public Debt Management Office, Ministry of Finance (PDMO / MOF / สบน. กระทรวงการคลัง)

Fail-Closed Policy:
When automated official feeds are unavailable, the adapter emits structured data gaps
rather than generating synthetic or mock numbers.
"""
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from core.logger import get_logger
from schemas.macro_schemas import MarketObservable

log = get_logger(__name__)


@dataclass(frozen=True)
class ThaiHardDataRecord:
    series_id: str
    indicator_name: str
    source_authority: str
    frequency: str
    unit: str
    value: Optional[float] = None
    prev: Optional[float] = None
    ma: Optional[float] = None
    period: Optional[str] = None
    observed_at: Optional[str] = None
    published_at: Optional[str] = None
    is_verified: bool = False
    status: str = "missing"
    gap_reason: str = ""


DEFAULT_HARD_DATA_PATH = Path("data/macro/thailand/official_hard_data.json")


def _to_iso_date(d: Optional[str], default_date: str) -> str:
    if not d:
        return default_date
    d_clean = str(d).strip()
    if len(d_clean) == 10 and d_clean[4] == "-" and d_clean[7] == "-":
        return d_clean
    if len(d_clean) == 7 and d_clean[4] == "-":
        try:
            import calendar
            y, m = int(d_clean[:4]), int(d_clean[5:7])
            last_day = calendar.monthrange(y, m)[1]
            return f"{y:04d}-{m:02d}-{last_day:02d}"
        except Exception:
            return f"{d_clean}-01"
    return default_date


def load_default_thai_records(path: Optional[Path] = None) -> dict[str, ThaiHardDataRecord]:
    target_path = path or DEFAULT_HARD_DATA_PATH
    if not target_path.exists():
        return {}
    try:
        data = json.loads(target_path.read_text(encoding="utf-8"))
        records_raw = data.get("records", data)
        records = {}
        for k, v in records_raw.items():
            if isinstance(v, dict):
                records[k] = ThaiHardDataRecord(**v)
        return records
    except Exception as e:
        log.warning(f"Failed to load default Thai hard data from {target_path}: {e}")
        return {}


def sync_thai_macro_data(store_path: Optional[Path] = None) -> dict[str, ThaiHardDataRecord]:
    """Refresh and synchronize Thai macroeconomic indicators from official and live feeds.

    1. Loads current official ground truth records from disk.
    2. Pulls live MOF Public Debt and Debt-to-GDP from dataservices.mof.go.th (Keyless Open CSV).
    3. Pulls live BoT Policy Rate from BIS SDMX REST API.
    4. Persists the synchronized snapshot to disk and returns the updated record map.
    """
    target_path = store_path or DEFAULT_HARD_DATA_PATH
    records = load_default_thai_records(target_path)

    # 1. Sync live MOF Public Debt
    try:
        from tools.market.terminal_v2.adapters.mof_th_adapter import MofThailandAdapter
        mof_adapter = MofThailandAdapter()
        snap = mof_adapter.get_public_debt()
        if snap and snap.total_debt_thb > 0:
            records["TH_PUBLIC_DEBT_TOTAL"] = ThaiHardDataRecord(
                series_id="TH_PUBLIC_DEBT_TOTAL",
                indicator_name="Thailand Public Debt Total",
                source_authority="Public Debt Management Office, Ministry of Finance (PDMO / MOF / สบน. กระทรวงการคลัง)",
                frequency="Monthly",
                unit="Million THB",
                value=round(snap.total_debt_thb / 1e6, 2),
                period=snap.reporting_month,
                observed_at=snap.reporting_month,
                is_verified=True,
                status="verified",
                gap_reason="",
            )
            if snap.debt_to_gdp_pct is not None:
                records["TH_DEBT_TO_GDP_PCT"] = ThaiHardDataRecord(
                    series_id="TH_DEBT_TO_GDP_PCT",
                    indicator_name="Thailand Debt to GDP Ratio",
                    source_authority="Public Debt Management Office, Ministry of Finance (PDMO / MOF / สบน. กระทรวงการคลัง)",
                    frequency="Monthly",
                    unit="%",
                    value=snap.debt_to_gdp_pct,
                    period=snap.reporting_month,
                    observed_at=snap.reporting_month,
                    is_verified=True,
                    status="verified",
                    gap_reason="",
                )
    except Exception as exc:
        log.warning(f"Could not refresh live MOF public debt in sync_thai_macro_data: {exc}")

    # 2. Sync live BoT Policy Rate from BIS SDMX
    try:
        from tools.market.terminal_v2.adapters.bis_adapter import BisPolicyRatesHttpAdapter
        bis_adapter = BisPolicyRatesHttpAdapter()
        rates_snap = bis_adapter.fetch_global_policy_rates()
        if rates_snap and rates_snap.rates:
            th_rate = next((r for r in rates_snap.rates if r.country == "TH"), None)
            if th_rate and th_rate.rate_value is not None and not th_rate.is_stale:
                prev_val = th_rate.previous_rate if th_rate.previous_rate is not None else th_rate.rate_value
                raw_date = th_rate.effective_date or rates_snap.as_of_date
                obs_date = _to_iso_date(raw_date, datetime.now().strftime("%Y-%m-%d"))
                records["TH_POLICY_RATE"] = ThaiHardDataRecord(
                    series_id="TH_POLICY_RATE",
                    indicator_name="Bank of Thailand Policy Rate",
                    source_authority="Bank of Thailand (BOT / ธปท.)",
                    frequency="Event / Daily",
                    unit="% per annum",
                    value=th_rate.rate_value,
                    prev=prev_val,
                    ma=th_rate.rate_value,
                    period=th_rate.effective_date[:7] if th_rate.effective_date else rates_snap.as_of_date[:7],
                    observed_at=obs_date,
                    published_at=obs_date,
                    is_verified=True,
                    status="verified",
                    gap_reason="",
                )
    except Exception as exc:
        log.warning(f"Could not refresh live BoT policy rate from BIS in sync_thai_macro_data: {exc}")

    # 3. Sync live Thai Government Bond Yield Curve from ThaiBMA
    try:
        from tools.market.terminal_v2.adapters.thaibma_adapter import ThaiBmaPublicAdapter
        thaibma = ThaiBmaPublicAdapter()
        curve_snap = thaibma.get_government_yield_curve()
        if curve_snap and curve_snap.yields:
            yields_by_tenor = {p.tenor: p.yield_percent for p in curve_snap.yields if p.yield_percent is not None}
            obs_date = curve_snap.observation_date

            y2 = yields_by_tenor.get("2Y")
            if y2 is not None:
                records["TH_GOV_YIELD_2Y"] = ThaiHardDataRecord(
                    series_id="TH_GOV_YIELD_2Y",
                    indicator_name="Thailand 2Y Gov Bond Yield",
                    source_authority="Thai Bond Market Association (ThaiBMA)",
                    frequency="Daily",
                    unit="%",
                    value=y2,
                    period=obs_date,
                    observed_at=obs_date,
                    published_at=obs_date,
                    is_verified=True,
                    status="verified",
                    gap_reason="",
                )

            y10 = yields_by_tenor.get("10Y")
            if y10 is not None:
                records["TH_GOV_YIELD_10Y"] = ThaiHardDataRecord(
                    series_id="TH_GOV_YIELD_10Y",
                    indicator_name="Thailand 10Y Gov Bond Yield",
                    source_authority="Thai Bond Market Association (ThaiBMA)",
                    frequency="Daily",
                    unit="%",
                    value=y10,
                    period=obs_date,
                    observed_at=obs_date,
                    published_at=obs_date,
                    is_verified=True,
                    status="verified",
                    gap_reason="",
                )

            if curve_snap.spread_10y_2y_bps is not None:
                records["TH_GOV_10Y_2Y_SPREAD"] = ThaiHardDataRecord(
                    series_id="TH_GOV_10Y_2Y_SPREAD",
                    indicator_name="Thailand Gov Bond 10Y-2Y Spread",
                    source_authority="Thai Bond Market Association (ThaiBMA)",
                    frequency="Daily",
                    unit="bps",
                    value=curve_snap.spread_10y_2y_bps,
                    period=obs_date,
                    observed_at=obs_date,
                    published_at=obs_date,
                    is_verified=True,
                    status="verified",
                    gap_reason="",
                )
    except Exception as exc:
        log.warning(f"Could not refresh live ThaiBMA yield curve in sync_thai_macro_data: {exc}")

    # 4. Persist updated records
    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        dump_data = {
            "records": {k: v.__dict__ for k, v in records.items()},
            "last_refreshed_at": datetime.now().isoformat(),
        }
        target_path.write_text(json.dumps(dump_data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        log.warning(f"Could not save synced Thai hard data to {target_path}: {exc}")

    return records


class ThaiHardDataAdapter:
    """Provides status, metadata, and structured data gaps for Thai macroeconomic releases."""

    def __init__(
        self,
        override_records: Optional[dict[str, ThaiHardDataRecord]] = None,
        mof_adapter: Optional[Any] = None,
        thaibma_adapter: Optional[Any] = None,
        load_defaults: bool = True,
    ):
        if override_records is not None:
            self._records = override_records
        elif load_defaults:
            self._records = load_default_thai_records()
        else:
            self._records = {}
        self._mof_adapter = mof_adapter
        self._thaibma_adapter = thaibma_adapter

    def get_thai_gdp_status(self) -> ThaiHardDataRecord:
        if "TH_REAL_GDP" in self._records:
            return self._records["TH_REAL_GDP"]
        return ThaiHardDataRecord(
            series_id="TH_REAL_GDP",
            indicator_name="Thailand Real GDP YoY",
            source_authority="Office of the National Economic and Social Development Council (NESDC / สภาพัฒน์)",
            frequency="Quarterly",
            unit="% YoY",
            value=None,
            is_verified=False,
            status="missing",
            gap_reason="Direct NESDC automated API integration not yet active; synthetic mocks disabled by production guardrail.",
        )

    def get_thai_cpi_status(self) -> ThaiHardDataRecord:
        if "TH_CPI_YOY" in self._records:
            return self._records["TH_CPI_YOY"]
        return ThaiHardDataRecord(
            series_id="TH_CPI_YOY",
            indicator_name="Thailand CPI Inflation YoY",
            source_authority="Trade Policy and Strategy Office, Ministry of Commerce (TPSO / MOC / กระทรวงพาณิชย์)",
            frequency="Monthly",
            unit="% YoY",
            value=None,
            is_verified=False,
            status="missing",
            gap_reason="Direct MOC/TPSO automated API integration not yet active; synthetic mocks disabled by production guardrail.",
        )

    def get_thai_core_cpi_status(self) -> ThaiHardDataRecord:
        if "TH_CORE_CPI_YOY" in self._records:
            return self._records["TH_CORE_CPI_YOY"]
        return ThaiHardDataRecord(
            series_id="TH_CORE_CPI_YOY",
            indicator_name="Thailand Core CPI YoY",
            source_authority="Trade Policy and Strategy Office, Ministry of Commerce (TPSO / MOC / กระทรวงพาณิชย์)",
            frequency="Monthly",
            unit="% YoY",
            value=None,
            is_verified=False,
            status="missing",
            gap_reason="Direct MOC/TPSO Core CPI automated API integration not yet active; diagnostic indicator.",
        )

    def get_thai_mpi_status(self) -> ThaiHardDataRecord:
        if "TH_MPI_YOY" in self._records:
            return self._records["TH_MPI_YOY"]
        return ThaiHardDataRecord(
            series_id="TH_MPI_YOY",
            indicator_name="Thailand Manufacturing Production Index YoY (MPI)",
            source_authority="Office of Industrial Economics (OIE / สศอ. กระทรวงอุตสาหกรรม)",
            frequency="Monthly",
            unit="% YoY",
            value=None,
            is_verified=False,
            status="missing",
            gap_reason="Direct OIE MPI feed integration not yet active; growth corroboration component.",
        )

    def get_thai_public_debt_status(self) -> ThaiHardDataRecord:
        if "TH_PUBLIC_DEBT_TOTAL" in self._records:
            return self._records["TH_PUBLIC_DEBT_TOTAL"]
        if self._mof_adapter is not None:
            try:
                snap = self._mof_adapter.get_public_debt()
                return ThaiHardDataRecord(
                    series_id="TH_PUBLIC_DEBT_TOTAL",
                    indicator_name="Thailand Public Debt Total",
                    source_authority="Public Debt Management Office, Ministry of Finance (PDMO / MOF / สบน. กระทรวงการคลัง)",
                    frequency="Monthly",
                    unit="Million THB",
                    value=snap.total_debt_thb / 1e6,
                    period=snap.reporting_month,
                    observed_at=snap.reporting_month,
                    is_verified=True,
                    status="verified",
                )
            except Exception as e:
                pass
        return ThaiHardDataRecord(
            series_id="TH_PUBLIC_DEBT_TOTAL",
            indicator_name="Thailand Public Debt Total",
            source_authority="Public Debt Management Office, Ministry of Finance (PDMO / MOF / สบน. กระทรวงการคลัง)",
            frequency="Monthly",
            unit="Million THB",
            value=None,
            is_verified=False,
            status="missing",
            gap_reason="MOF Open Data public debt feed unavailable or not connected.",
        )

    def get_thai_debt_to_gdp_status(self) -> ThaiHardDataRecord:
        if "TH_DEBT_TO_GDP_PCT" in self._records:
            return self._records["TH_DEBT_TO_GDP_PCT"]
        if self._mof_adapter is not None:
            try:
                snap = self._mof_adapter.get_public_debt()
                if snap.debt_to_gdp_pct is not None:
                    return ThaiHardDataRecord(
                        series_id="TH_DEBT_TO_GDP_PCT",
                        indicator_name="Thailand Debt to GDP Ratio",
                        source_authority="Public Debt Management Office, Ministry of Finance (PDMO / MOF / สบน. กระทรวงการคลัง)",
                        frequency="Monthly",
                        unit="%",
                        value=snap.debt_to_gdp_pct,
                        period=snap.reporting_month,
                        observed_at=snap.reporting_month,
                        is_verified=True,
                        status="verified",
                    )
            except Exception as e:
                pass
        return ThaiHardDataRecord(
            series_id="TH_DEBT_TO_GDP_PCT",
            indicator_name="Thailand Debt to GDP Ratio",
            source_authority="Public Debt Management Office, Ministry of Finance (PDMO / MOF / สบน. กระทรวงการคลัง)",
            frequency="Monthly",
            unit="%",
            value=None,
            is_verified=False,
            status="missing",
            gap_reason="MOF Open Data Debt to GDP ratio unavailable or not connected.",
        )

    def get_thai_policy_rate_status(self) -> ThaiHardDataRecord:
        if "TH_POLICY_RATE" in self._records:
            return self._records["TH_POLICY_RATE"]
        return ThaiHardDataRecord(
            series_id="TH_POLICY_RATE",
            indicator_name="Bank of Thailand Policy Rate",
            source_authority="Bank of Thailand (BOT / ธปท.)",
            frequency="Event / Daily",
            unit="% per annum",
            value=None,
            is_verified=False,
            status="missing",
            gap_reason="Direct BOT Policy Rate automated API integration not yet active; BIS reconciliation or official record input required.",
        )

    def get_thai_yield_curve_status(self) -> dict[str, ThaiHardDataRecord]:
        y2 = self._records.get("TH_GOV_YIELD_2Y")
        y10 = self._records.get("TH_GOV_YIELD_10Y")
        spread_override = self._records.get("TH_GOV_10Y_2Y_SPREAD")

        if (y2 is None or y10 is None) and self._thaibma_adapter is not None:
            try:
                snap = self._thaibma_adapter.get_government_yield_curve()
                if snap and snap.yields:
                    ymap = {p.tenor: p.yield_percent for p in snap.yields if p.yield_percent is not None}
                    obs = snap.observation_date
                    if "2Y" in ymap and y2 is None:
                        y2 = ThaiHardDataRecord(
                            series_id="TH_GOV_YIELD_2Y",
                            indicator_name="Thailand 2Y Gov Bond Yield",
                            source_authority="Thai Bond Market Association (ThaiBMA)",
                            frequency="Daily",
                            unit="%",
                            value=ymap["2Y"],
                            period=obs,
                            observed_at=obs,
                            published_at=obs,
                            is_verified=True,
                            status="verified",
                        )
                    if "10Y" in ymap and y10 is None:
                        y10 = ThaiHardDataRecord(
                            series_id="TH_GOV_YIELD_10Y",
                            indicator_name="Thailand 10Y Gov Bond Yield",
                            source_authority="Thai Bond Market Association (ThaiBMA)",
                            frequency="Daily",
                            unit="%",
                            value=ymap["10Y"],
                            period=obs,
                            observed_at=obs,
                            published_at=obs,
                            is_verified=True,
                            status="verified",
                        )
                    if snap.spread_10y_2y_bps is not None and spread_override is None:
                        spread_override = ThaiHardDataRecord(
                            series_id="TH_GOV_10Y_2Y_SPREAD",
                            indicator_name="Thailand Gov Bond 10Y-2Y Spread",
                            source_authority="Thai Bond Market Association (ThaiBMA)",
                            frequency="Daily",
                            unit="bps",
                            value=snap.spread_10y_2y_bps,
                            period=obs,
                            observed_at=obs,
                            published_at=obs,
                            is_verified=True,
                            status="verified",
                        )
            except Exception as exc:
                log.warning(f"Could not load yield curve from ThaiBMA adapter: {exc}")

        if spread_override is not None:
            spread_record = spread_override
        elif y2 is not None and y10 is not None and y2.value is not None and y10.value is not None:
            # Check same-date requirement per TH-07 and TH-AC14
            if y2.observed_at == y10.observed_at and y2.is_verified and y10.is_verified:
                diff_bps = round((y10.value - y2.value) * 100.0, 1)
                spread_record = ThaiHardDataRecord(
                    series_id="TH_GOV_10Y_2Y_SPREAD",
                    indicator_name="Thailand Gov Bond 10Y-2Y Spread",
                    source_authority="Thai Bond Market Association (ThaiBMA)",
                    frequency="Daily",
                    unit="bps",
                    value=diff_bps,
                    observed_at=y10.observed_at,
                    published_at=y10.published_at,
                    is_verified=True,
                    status="verified",
                )
            else:
                spread_record = ThaiHardDataRecord(
                    series_id="TH_GOV_10Y_2Y_SPREAD",
                    indicator_name="Thailand Gov Bond 10Y-2Y Spread",
                    source_authority="Thai Bond Market Association (ThaiBMA)",
                    frequency="Daily",
                    unit="bps",
                    value=None,
                    is_verified=False,
                    status="mismatched_date",
                    gap_reason="Thai government bond 10Y and 2Y yields must share the same observation date.",
                )
        else:
            spread_record = ThaiHardDataRecord(
                series_id="TH_GOV_10Y_2Y_SPREAD",
                indicator_name="Thailand Gov Bond 10Y-2Y Spread",
                source_authority="Thai Bond Market Association (ThaiBMA)",
                frequency="Daily",
                unit="bps",
                value=None,
                is_verified=False,
                status="blocked",
                gap_reason="ThaiBMA authorized access required (separate access gate per Dual-Track policy).",
            )

        return {
            "2Y": y2 or ThaiHardDataRecord(
                series_id="TH_GOV_YIELD_2Y",
                indicator_name="Thailand 2Y Gov Bond Yield",
                source_authority="Thai Bond Market Association (ThaiBMA)",
                frequency="Daily",
                unit="%",
                value=None,
                is_verified=False,
                status="blocked",
                gap_reason="ThaiBMA authorized access required (separate access gate per Dual-Track policy).",
            ),
            "10Y": y10 or ThaiHardDataRecord(
                series_id="TH_GOV_YIELD_10Y",
                indicator_name="Thailand 10Y Gov Bond Yield",
                source_authority="Thai Bond Market Association (ThaiBMA)",
                frequency="Daily",
                unit="%",
                value=None,
                is_verified=False,
                status="blocked",
                gap_reason="ThaiBMA authorized access required (separate access gate per Dual-Track policy).",
            ),
            "SPREAD": spread_record,
        }

    def import_official_csv(
        self,
        csv_text: str,
        series_id: str,
        default_authority: str = "Official Source",
    ) -> ThaiHardDataRecord:
        """Parses official CSV with BOM handling, Buddhist Era translation, WAF check, and missing value guards."""
        # 1. WAF check
        if "<title>Request Rejected</title>" in csv_text or "The requested URL was rejected" in csv_text:
            rec = ThaiHardDataRecord(
                series_id=series_id,
                indicator_name=series_id,
                source_authority=default_authority,
                frequency="Monthly",
                unit="%",
                value=None,
                is_verified=False,
                status="waf_blocked",
                gap_reason="Request rejected by upstream WAF (F5 BIG-IP).",
            )
            self._records[series_id] = rec
            return rec

        clean_text = csv_text.lstrip("\ufeff").strip()
        lines = [line.strip() for line in clean_text.splitlines() if line.strip()]
        if not lines:
            rec = ThaiHardDataRecord(
                series_id=series_id,
                indicator_name=series_id,
                source_authority=default_authority,
                frequency="Monthly",
                unit="%",
                value=None,
                is_verified=False,
                status="missing",
                gap_reason="Empty official CSV content.",
            )
            self._records[series_id] = rec
            return rec

        import csv
        reader = csv.DictReader(lines)
        rows = list(reader)
        if not rows:
            rec = ThaiHardDataRecord(
                series_id=series_id,
                indicator_name=series_id,
                source_authority=default_authority,
                frequency="Monthly",
                unit="%",
                value=None,
                is_verified=False,
                status="missing",
                gap_reason="No data rows found in CSV.",
            )
            self._records[series_id] = rec
            return rec

        latest_row = rows[-1]

        def _parse_val(k: str) -> Optional[float]:
            for col in [k, k.lower(), k.upper(), "val", "value", "latest", "obs_value", "ค่าล่าสุด"]:
                if col in latest_row:
                    v = latest_row[col].strip()
                    if v and v != "-":
                        try:
                            return float(v.replace(",", ""))
                        except ValueError:
                            pass
            return None

        val = _parse_val("value")
        prev = _parse_val("prev")
        ma = _parse_val("ma")

        period = (
            latest_row.get("period")
            or latest_row.get("TIME_PERIOD")
            or latest_row.get("งวด")
            or latest_row.get("date")
            or latest_row.get("วันที่")
        )
        if period:
            period = period.strip()
            import re
            m = re.search(r"25[5-7]\d", period)
            if m:
                be_year = int(m.group(0))
                period = period.replace(str(be_year), str(be_year - 543))

        published_at = latest_row.get("published_at") or latest_row.get("release_date")
        if published_at:
            published_at = published_at.strip()

        rec = ThaiHardDataRecord(
            series_id=series_id,
            indicator_name=series_id,
            source_authority=default_authority,
            frequency=latest_row.get("frequency", "Monthly"),
            unit=latest_row.get("unit", "% YoY"),
            value=val,
            prev=prev,
            ma=ma,
            period=period,
            observed_at=period,
            published_at=published_at,
            is_verified=val is not None,
            status="verified" if val is not None else "missing",
            gap_reason="" if val is not None else "Value absent in official CSV row.",
        )
        self._records[series_id] = rec
        return rec

    def get_hard_data_gaps(self) -> list[str]:
        gaps = []
        gdp = self.get_thai_gdp_status()
        if not gdp.is_verified or gdp.value is None:
            gaps.append(f"{gdp.indicator_name} ({gdp.source_authority})")
        cpi = self.get_thai_cpi_status()
        if not cpi.is_verified or cpi.value is None:
            gaps.append(f"{cpi.indicator_name} ({cpi.source_authority})")
        return gaps

    def as_observables(self, as_of_date: Optional[str] = None) -> list[MarketObservable]:
        today_str = as_of_date or datetime.now().strftime("%Y-%m-%d")
        observables: list[MarketObservable] = []

        # 1. Real GDP
        gdp = self.get_thai_gdp_status()
        gdp_meta: dict[str, Any] = {}
        if gdp.value is not None:
            gdp_meta["val"] = gdp.value
        if gdp.prev is not None:
            gdp_meta["prev"] = gdp.prev
        if gdp.ma is not None:
            gdp_meta["ma"] = gdp.ma

        observables.append(MarketObservable(
            observable_id="obs_th_gdp_nesdc",
            asset_bucket="equities",
            region="Thailand",
            indicator=gdp.indicator_name,
            value=f"{gdp.value:.2f}" if gdp.value is not None else "N/A",
            unit=gdp.unit,
            observed_at=gdp.observed_at if (gdp.is_verified and gdp.observed_at) else today_str,
            published_at=gdp.published_at,
            source_file="NESDC_Official_Releases",
            provider="NESDC",
            confidence="high" if gdp.is_verified else "low",
            is_valid=gdp.is_verified and gdp.value is not None,
            status="verified" if (gdp.is_verified and gdp.value is not None) else "missing",
            stale_reason="" if gdp.is_verified else gdp.gap_reason,
            period=gdp.period,
            metadata=gdp_meta,
        ))

        # 2. Headline CPI
        cpi = self.get_thai_cpi_status()
        cpi_meta: dict[str, Any] = {}
        if cpi.value is not None:
            cpi_meta["val"] = cpi.value
        if cpi.prev is not None:
            cpi_meta["prev"] = cpi.prev
        if cpi.ma is not None:
            cpi_meta["ma"] = cpi.ma

        observables.append(MarketObservable(
            observable_id="obs_th_cpi_moc",
            asset_bucket="cash",
            region="Thailand",
            indicator=cpi.indicator_name,
            value=f"{cpi.value:.2f}" if cpi.value is not None else "N/A",
            unit=cpi.unit,
            observed_at=cpi.observed_at if (cpi.is_verified and cpi.observed_at) else today_str,
            published_at=cpi.published_at,
            source_file="MOC_Official_Releases",
            provider="MOC TPSO",
            confidence="high" if cpi.is_verified else "low",
            is_valid=cpi.is_verified and cpi.value is not None,
            status="verified" if (cpi.is_verified and cpi.value is not None) else "missing",
            stale_reason="" if cpi.is_verified else cpi.gap_reason,
            period=cpi.period,
            metadata=cpi_meta,
        ))

        # 3. Core CPI (Diagnostic)
        core_cpi = self.get_thai_core_cpi_status()
        if core_cpi.is_verified and core_cpi.value is not None:
            core_meta: dict[str, Any] = {"val": core_cpi.value}
            if core_cpi.prev is not None:
                core_meta["prev"] = core_cpi.prev
            if core_cpi.ma is not None:
                core_meta["ma"] = core_cpi.ma
            observables.append(MarketObservable(
                observable_id="obs_th_core_cpi_moc",
                asset_bucket="cash",
                region="Thailand",
                indicator=core_cpi.indicator_name,
                value=f"{core_cpi.value:.2f}",
                unit=core_cpi.unit,
                observed_at=core_cpi.observed_at or today_str,
                published_at=core_cpi.published_at,
                source_file="MOC_Official_Releases",
                provider="MOC TPSO",
                confidence="high",
                is_valid=True,
                status="verified",
                period=core_cpi.period,
                metadata=core_meta,
            ))

        # 4. Manufacturing Production Index (OIE MPI - Corroboration)
        mpi = self.get_thai_mpi_status()
        if mpi.is_verified and mpi.value is not None:
            mpi_meta: dict[str, Any] = {"val": mpi.value}
            if mpi.prev is not None:
                mpi_meta["prev"] = mpi.prev
            if mpi.ma is not None:
                mpi_meta["ma"] = mpi.ma
            observables.append(MarketObservable(
                observable_id="obs_th_mpi_oie",
                asset_bucket="equities",
                region="Thailand",
                indicator=mpi.indicator_name,
                value=f"{mpi.value:.2f}",
                unit=mpi.unit,
                observed_at=mpi.observed_at or today_str,
                published_at=mpi.published_at,
                source_file="OIE_Official_Releases",
                provider="OIE",
                confidence="high",
                is_valid=True,
                status="verified",
                period=mpi.period,
                metadata=mpi_meta,
            ))

        # 5. Public Debt & Debt to GDP (MOF)
        debt = self.get_thai_public_debt_status()
        if debt.is_verified and debt.value is not None:
            observables.append(MarketObservable(
                observable_id="obs_th_public_debt_mof",
                asset_bucket="cash",
                region="Thailand",
                indicator=debt.indicator_name,
                value=f"{debt.value:,.2f}",
                unit=debt.unit,
                observed_at=debt.observed_at or today_str,
                source_file="MOF_Public_Debt_CSV",
                provider="MOF Thailand",
                confidence="high",
                is_valid=True,
                status="verified",
                period=debt.period,
                metadata={"val": debt.value, "debt_million_thb": debt.value},
            ))

        debt_gdp = self.get_thai_debt_to_gdp_status()
        if debt_gdp.is_verified and debt_gdp.value is not None:
            observables.append(MarketObservable(
                observable_id="obs_th_debt_to_gdp_mof",
                asset_bucket="cash",
                region="Thailand",
                indicator=debt_gdp.indicator_name,
                value=f"{debt_gdp.value:.2f}",
                unit=debt_gdp.unit,
                observed_at=debt_gdp.observed_at or today_str,
                source_file="MOF_Public_Debt_CSV",
                provider="MOF Thailand",
                confidence="high",
                is_valid=True,
                status="verified",
                period=debt_gdp.period,
                metadata={
                    "val": debt_gdp.value,
                    "statutory_limit_pct": 70.0,
                    "is_within_limit": debt_gdp.value <= 70.0,
                },
            ))

        # 6. Policy Rate (BOT)
        policy_rate = self.get_thai_policy_rate_status()
        if policy_rate.is_verified and policy_rate.value is not None:
            observables.append(MarketObservable(
                observable_id="obs_thai_policy_rate_bot",
                asset_bucket="cash",
                region="Thailand",
                indicator=policy_rate.indicator_name,
                value=f"{policy_rate.value:.2f}",
                unit=policy_rate.unit,
                observed_at=_to_iso_date(policy_rate.observed_at, today_str),
                published_at=policy_rate.published_at,
                source_file="BOT_Official_Releases",
                provider="Bank of Thailand",
                confidence="high",
                is_valid=True,
                status="verified",
                period=policy_rate.period,
                metadata={"val": policy_rate.value, "prev": policy_rate.prev, "ma": policy_rate.ma},
            ))

        # 7. Thai Yield Curve & Spread (ThaiBMA / Authorized Export)
        curve_data = self.get_thai_yield_curve_status()
        y2 = curve_data.get("2Y")
        if y2 and y2.is_verified and y2.value is not None:
            observables.append(MarketObservable(
                observable_id="obs_th_gov_yield_2y",
                asset_bucket="fixed_income",
                region="Thailand",
                indicator=y2.indicator_name,
                value=f"{y2.value:.2f}",
                unit=y2.unit,
                observed_at=y2.observed_at or today_str,
                source_file="ThaiBMA_Yield_Curve",
                provider="ThaiBMA",
                confidence="high",
                is_valid=True,
                status="verified",
                metadata={"val": y2.value, "tenor": "2Y"},
            ))

        y10 = curve_data.get("10Y")
        if y10 and y10.is_verified and y10.value is not None:
            observables.append(MarketObservable(
                observable_id="obs_th_gov_yield_10y",
                asset_bucket="fixed_income",
                region="Thailand",
                indicator=y10.indicator_name,
                value=f"{y10.value:.2f}",
                unit=y10.unit,
                observed_at=y10.observed_at or today_str,
                source_file="ThaiBMA_Yield_Curve",
                provider="ThaiBMA",
                confidence="high",
                is_valid=True,
                status="verified",
                metadata={"val": y10.value, "tenor": "10Y"},
            ))

        spread = curve_data.get("SPREAD")
        if spread and spread.is_verified and spread.value is not None:
            observables.append(MarketObservable(
                observable_id="obs_th_gov_10y_2y_spread",
                asset_bucket="fixed_income",
                region="Thailand",
                indicator=spread.indicator_name,
                value=f"{spread.value:.1f}",
                unit=spread.unit,
                observed_at=spread.observed_at or today_str,
                source_file="ThaiBMA_Yield_Curve",
                provider="ThaiBMA",
                confidence="high",
                is_valid=True,
                status="verified",
                metadata={"val": spread.value, "diff_bps": spread.value},
            ))
        elif spread:
            observables.append(MarketObservable(
                observable_id="obs_th_gov_10y_2y_spread",
                asset_bucket="fixed_income",
                region="Thailand",
                indicator=spread.indicator_name,
                value="N/A",
                unit="bps",
                observed_at=today_str,
                source_file="ThaiBMA_Yield_Curve",
                provider="ThaiBMA",
                confidence="low",
                is_valid=False,
                status=spread.status,
                stale_reason=spread.gap_reason,
            ))

        # A locally verified record may become stale between refreshes. Preserve
        # its real observation date and exclude stale/missing dates from scoring.
        records_by_id = {
            "obs_th_gdp_nesdc": gdp, "obs_th_cpi_moc": cpi,
            "obs_th_core_cpi_moc": core_cpi, "obs_th_mpi_oie": mpi,
            "obs_th_public_debt_mof": debt, "obs_th_debt_to_gdp_mof": debt_gdp,
            "obs_thai_policy_rate_bot": policy_rate,
            "obs_th_gov_yield_2y": y2, "obs_th_gov_yield_10y": y10,
            "obs_th_gov_10y_2y_spread": spread,
        }
        for observable in observables:
            record = records_by_id.get(observable.observable_id)
            if record is None or not observable.is_valid:
                continue
            observed_at = _to_iso_date(record.observed_at, "1970-01-01")
            observable.observed_at = observed_at
            observable.metadata["frequency"] = record.frequency
            try:
                age = (datetime.strptime(today_str, "%Y-%m-%d") - datetime.strptime(observed_at, "%Y-%m-%d")).days
                frequency = record.frequency.lower()
                cutoff = 180 if "quarter" in frequency or "event" in frequency else 90 if "month" in frequency else 7
                reason = "Missing real observation date" if observed_at == "1970-01-01" else "Observation date is in the future" if age < 0 else f"Observation exceeds {cutoff}-day freshness window ({age} days)" if age > cutoff else ""
            except ValueError:
                reason = "Invalid observation date"
            if reason:
                observable.is_valid = False
                observable.confidence = "low"
                observable.status = "stale" if "exceeds" in reason else "unverified"
                observable.stale_reason = reason
        return observables
