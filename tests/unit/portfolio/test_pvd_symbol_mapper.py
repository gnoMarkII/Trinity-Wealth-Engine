import pytest
from tools.portfolio.adapters.thai_fund.pvd_symbol_mapper import (
    is_pvd_symbol,
    resolve_pvd_symbol,
)


def test_is_pvd_symbol():
    assert is_pvd_symbol("PVD_SINSA_GLOB") is True
    assert is_pvd_symbol("pvd_sinsa_care") is True
    assert is_pvd_symbol("PVD_UNKNOWN") is True
    assert is_pvd_symbol("SCBS&P500E") is False
    assert is_pvd_symbol("MSFT") is False


def test_resolve_pvd_symbol():
    resolved, is_mapped = resolve_pvd_symbol("PVD_SINSA_GLOB")
    assert is_mapped is True
    assert resolved == "B-GLOBAL"

    resolved, is_mapped = resolve_pvd_symbol("PVD_SINSA_CARE")
    assert is_mapped is True
    assert resolved == "BCARE"

    resolved, is_mapped = resolve_pvd_symbol("PVD_SINSA_BTP")
    assert is_mapped is True
    assert resolved == "BTP"

    resolved, is_mapped = resolve_pvd_symbol("SCBS&P500E")
    assert is_mapped is False
    assert resolved == "SCBS&P500E"
