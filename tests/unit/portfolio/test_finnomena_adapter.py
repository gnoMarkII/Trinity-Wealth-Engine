import json
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from tools.portfolio.adapters.thai_fund.finnomena_adapter import FinnomenaFundAdapter
from tools.portfolio.ports.thai_fund_port import FundNavData


@pytest.fixture
def sample_catalog_json():
    return [
        {"id": "F00000ZMXD", "short_code": "PRINCIPAL VNEQ-A", "name_th": "พรินซิเพิล เวียดนาม อิควิตี้"},
        {"id": "F000016LB4", "short_code": "KT-ASIAG-A", "name_th": "เคแทม เอเชีย โกรท อิควิตี้"},
        {"id": "F00001LRQS", "short_code": "DAOL-KOREAEQ", "name_th": "ดาโอ โคเรีย อิควิตี้"},
    ]


@pytest.fixture
def sample_nav_json():
    return {
        "status": True,
        "data": {
            "fund_id": "F00000ZMXD",
            "short_code": "PRINCIPAL VNEQ-A",
            "navs": [
                {
                    "date": "2026-09-03T00:00:00Z",
                    "value": 12.0125,
                    "percent_change": 0.5,
                },
                {
                    "date": "2026-09-04T00:00:00Z",
                    "value": 12.1371,
                    "percent_change": 1.04,
                },
            ],
        },
    }


def test_has_fund_with_mock_catalog(tmp_path, sample_catalog_json):
    adapter = FinnomenaFundAdapter(cache_dir=str(tmp_path))
    
    mock_resp = MagicMock()
    mock_resp.json.return_value = sample_catalog_json
    mock_resp.raise_for_status.return_value = None

    with patch.object(adapter._session, "get", return_value=mock_resp):
        assert adapter.has_fund("PRINCIPAL VNEQ-A") is True
        assert adapter.has_fund("KT-ASIAG-A") is True
        assert adapter.has_fund("DAOL-KOREAEQ") is True
        assert adapter.has_fund("UNKNOWN-FUND") is False


def test_fetch_nav_success(tmp_path, sample_catalog_json, sample_nav_json):
    adapter = FinnomenaFundAdapter(cache_dir=str(tmp_path))
    
    mock_catalog_resp = MagicMock()
    mock_catalog_resp.json.return_value = sample_catalog_json
    mock_catalog_resp.raise_for_status.return_value = None

    mock_nav_resp = MagicMock()
    mock_nav_resp.json.return_value = sample_nav_json
    mock_nav_resp.raise_for_status.return_value = None

    def mock_get(url, *args, **kwargs):
        if "public/list" in url:
            return mock_catalog_resp
        return mock_nav_resp

    with patch.object(adapter._session, "get", side_effect=mock_get):
        nav_data = adapter.fetch_nav("PRINCIPAL VNEQ-A")
        assert nav_data is not None
        assert isinstance(nav_data, FundNavData)
        assert nav_data.symbol == "PRINCIPAL VNEQ-A"
        assert nav_data.nav == 12.1371
        assert nav_data.nav_date == "2026-09-04"
        assert nav_data.percent_change == 1.04
        assert nav_data.currency == "THB"


def test_fetch_nav_unknown_fund_returns_none(tmp_path, sample_catalog_json):
    adapter = FinnomenaFundAdapter(cache_dir=str(tmp_path))
    
    mock_catalog_resp = MagicMock()
    mock_catalog_resp.json.return_value = sample_catalog_json
    mock_catalog_resp.raise_for_status.return_value = None

    with patch.object(adapter._session, "get", return_value=mock_catalog_resp):
        nav_data = adapter.fetch_nav("NOT-A-FUND")
        assert nav_data is None


def test_fetch_nav_network_error_returns_none(tmp_path, sample_catalog_json):
    adapter = FinnomenaFundAdapter(cache_dir=str(tmp_path))
    adapter._catalog = {"PRINCIPAL VNEQ-A": "F00000ZMXD"}
    adapter._catalog_loaded_at = 9999999999.0

    with patch.object(adapter._session, "get", side_effect=Exception("Connection reset")):
        nav_data = adapter.fetch_nav("PRINCIPAL VNEQ-A")
        assert nav_data is None
