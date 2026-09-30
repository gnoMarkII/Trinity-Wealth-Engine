"""Unit tests verifying Domain Layer Purity and Invariants for Terminal V2."""
import ast
import dataclasses
from pathlib import Path
import pytest

from tools.market.terminal_v2.domain.models import (
    AuctionDemandSnapshot,
    CommodityVolPoint,
    CommodityVolSnapshot,
    FinancialStressSnapshot,
    FinraShortVolumeSnapshot,
    GlobalPolicyRateSnapshot,
    GoldPriceDetail,
    InsiderTransaction,
    InvestorTypeRow,
    LivePerpsQuote,
    MacroSeries,
    MacroSeriesPoint,
    MetalsCotPositioningSnapshot,
    NasdaqEarningsConsensusSnapshot,
    NewsCandidate,
    NewsDiscoverySnapshot,
    OptionContract,
    OptionsChainSnapshot,
    OptionsMaxPainResult,
    OptionsPutCallRatios,
    PredictionMarketItem,
    ReferenceRatePoint,
    ReferenceRateSnapshot,
    SecCompanyFactsSnapshot,
    SecFact,
    SecInsiderTradeSnapshot,
    SpotEtfFlowSnapshot,
    ThaiBondMarketStats,
    ThaiCorporateBondIssuance,
    ThaiFundAssetAllocationSnapshot,
    ThaiFundFlowSnapshot,
    ThaiPublicDebtSnapshot,
    ThaiRetailGoldQuote,
    TreasuryAuctionResult,
    TreasuryYieldCurveSnapshot,
    UsNationalDebtSnapshot,
)


def test_domain_has_zero_external_dependencies():
    """Verify that domain/models.py and domain/calculations.py contain NO imports from third-party packages."""
    domain_dir = (
        Path(__file__).resolve().parent.parent.parent.parent
        / "tools"
        / "market"
        / "terminal_v2"
        / "domain"
    )
    files_to_check = [domain_dir / "models.py", domain_dir / "calculations.py"]
    allowed_standard_modules = {
        "dataclasses", "enum", "typing", "datetime", "decimal", "tools"
    }
    forbidden = ["pydantic", "fastapi", "requests", "yfinance", "pandas", "numpy", "bs4", "langchain"]

    for fpath in files_to_check:
        assert fpath.exists(), f"File {fpath} must exist"
        tree = ast.parse(fpath.read_text(encoding="utf-8"), filename=str(fpath))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    base_mod = alias.name.split(".")[0]
                    assert base_mod in allowed_standard_modules, f"Forbidden import '{alias.name}' in {fpath.name}"
                    assert base_mod not in forbidden, f"Directly forbidden import '{alias.name}' in {fpath.name}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                base_mod = node.module.split(".")[0]
                assert base_mod in allowed_standard_modules, f"Forbidden import '{node.module}' in {fpath.name}"
                assert base_mod not in forbidden, f"Directly forbidden import '{node.module}' in {fpath.name}"


def test_domain_models_are_frozen_dataclasses():
    """Domain entities must be immutable dataclasses (frozen=True)."""
    point = CommodityVolPoint(date="2026-09-25", close=22.44)
    with pytest.raises(dataclasses.FrozenInstanceError):
        point.close = 25.0

    fact = SecFact(
        concept_tag="Revenues",
        label="Revenue",
        val=100.0,
        unit="USD",
        form="10-K",
        fy=2025,
        fp="FY",
        start=None,
        end=None,
        filed=None,
        accn=None,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        fact.val = 200.0

    contract = OptionContract(
        occ_symbol="AAPL261016C00100000",
        underlying="AAPL",
        expiry="2026-10-16",
        strike=100.0,
        side="call",
        open_interest=50,
        volume=10,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        contract.open_interest = 100
