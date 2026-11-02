"""Tests for Farside Investors ETF Flows Adapter and SoSoValue Failover."""
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from tools.market.terminal_v2.adapters.etf_flows_adapter import SoSoValueEtfFlowsAdapter
from tools.market.terminal_v2.adapters.farside_adapter import (
    FarsideEtfFlowsAdapter,
    _parse_farside_number,
)
from tools.market.terminal_v2.domain.errors import ProviderError
from tools.market.terminal_v2.domain.models import SpotEtfFlowSnapshot

SAMPLE_BTC_HTML = """
<html><body>
<table class="etf">
<tr><th>Date</th><th>IBIT</th><th>FBTC</th><th>BITB</th><th>Total</th></tr>
<tr><td>27 Oct 2026</td><td>50.0</td><td>10.0</td><td>5.0</td><td>65.0</td></tr>
<tr><td>28 Oct 2026</td><td>136.0</td><td>0.0</td><td>1.3</td><td>137.3</td></tr>
<tr><td>Total</td><td>50,000</td><td>10,000</td><td>2,000</td><td>62,000</td></tr>
</table>
</body></html>
"""

SAMPLE_ETH_HTML = """
<html><body>
<table class="etf">
<tr><th></th><th>ETHA</th><th>FETH</th><th>ETHE</th><th>Total</th></tr>
<tr><td>28 Oct 2026</td><td>(44.8)</td><td>18.8</td><td>0.0</td><td>(26.0)</td></tr>
<tr><td>Total</td><td>10,000</td><td>2,000</td><td>(5,000)</td><td>7,000</td></tr>
</table>
</body></html>
"""


def test_parse_farside_number():
    assert _parse_farside_number("136.0") == 136.0
    assert _parse_farside_number("(44.8)") == -44.8
    assert _parse_farside_number("12,017") == 12017.0
    assert _parse_farside_number("(1,018.8)") == -1018.8
    assert _parse_farside_number("9,199*") == 9199.0
    assert _parse_farside_number("-") is None
    assert _parse_farside_number("") is None
    assert _parse_farside_number("N/A") is None


def test_farside_adapter_btc_parsing(tmp_path: Path):
    fixture = tmp_path / "farside_btc.html"
    fixture.write_text(SAMPLE_BTC_HTML, encoding="utf-8")

    adapter = FarsideEtfFlowsAdapter(btc_fixture_path=fixture)
    snap = adapter.get_spot_etf_flows("BTC")

    assert snap.asset == "BTC"
    assert snap.report_date == "2026-10-28"
    assert snap.daily_total_usd == 137_300_000.0
    assert snap.cumulative_total_usd == 62_000_000_000.0
    assert snap.source == "Farside"
    assert snap.is_stale is False

    issuers_map = {i.ticker: i.daily_net_inflow_usd for i in snap.issuers}
    assert issuers_map.get("IBIT") == 136_000_000.0
    assert issuers_map.get("BITB") == 1_300_000.0


def test_farside_adapter_eth_parsing(tmp_path: Path):
    fixture = tmp_path / "farside_eth.html"
    fixture.write_text(SAMPLE_ETH_HTML, encoding="utf-8")

    adapter = FarsideEtfFlowsAdapter(eth_fixture_path=fixture)
    snap = adapter.get_spot_etf_flows("ETH")

    assert snap.asset == "ETH"
    assert snap.report_date == "2026-10-28"
    assert snap.daily_total_usd == -26_000_000.0
    assert snap.cumulative_total_usd == 7_000_000_000.0
    assert snap.source == "Farside"


def test_sosovalue_failover_to_farside(tmp_path: Path):
    btc_fixture = tmp_path / "farside_btc.html"
    btc_fixture.write_text(SAMPLE_BTC_HTML, encoding="utf-8")

    farside_mock = FarsideEtfFlowsAdapter(btc_fixture_path=btc_fixture)
    soso_adapter = SoSoValueEtfFlowsAdapter(secondary_adapter=farside_mock)

    # Simulate SoSoValue 403 Forbidden
    with patch.object(soso_adapter, "_fetch_snapshot", side_effect=ProviderError("Cloudflare 403 Forbidden", source="SoSoValue")):
        snap = soso_adapter.get_spot_etf_flows("BTC")
        assert snap.source == "Farside"
        assert snap.daily_total_usd == 137_300_000.0
        assert snap.is_stale is False

