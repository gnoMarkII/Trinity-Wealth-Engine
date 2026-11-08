from decimal import Decimal

from tools.thesis.market_observation_adapter import VaultMarketObservationAdapter


class _Sidecars:
    def __init__(self, values):
        self.values = values

    def get_all_equity_sidecars(self):
        return self.values


def test_actual_quant_signals_zero_and_raw_price_are_mapped():
    adapter = VaultMarketObservationAdapter(_Sidecars({
        "ACME": {
            "quant_signals": {
                "revenue_growth_yoy_pct": 0,
                "fcf_margin_pct": 0,
                "raw_analysis_price_str": "0.00",
            }
        }
    }))
    assert adapter.get_financial_metric("ACME", "revenue_growth_yoy") == Decimal("0")
    assert adapter.get_financial_metric("ACME", "fcf_margin") == Decimal("0")
    assert adapter.get_financial_metric("ACME", "current_price") == Decimal("0")


def test_unsupported_margin_is_unavailable_instead_of_fabricated():
    adapter = VaultMarketObservationAdapter(_Sidecars({
        "ACME": {"quant_signals": {"revenue_growth_yoy_pct": 0}}
    }))
    assert adapter.get_financial_metric("ACME", "operating_margin") is None
    assert adapter.get_financial_metric("ACME", "gross_margin") is None
