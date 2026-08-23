import pytest
from tools.portfolio.domain.constants import CASH_THB_SYMBOL
from tools.portfolio.domain.models import Holding, PortfolioState
from tools.portfolio.domain.errors import InvalidTradeError, InsufficientCashError, HoldingNotFoundError
from tools.portfolio.domain.validator import (
    validate_portfolio_id,
    validate_trade_request,
    validate_cash_availability,
    validate_holding_for_sell,
    validate_cash_flow_request,
)


def test_validate_portfolio_id():
    assert validate_portfolio_id("default") == "default"
    assert validate_portfolio_id("  DEFault  ") == "default"
    assert validate_portfolio_id("crypto_fund") == "crypto_fund"
    assert validate_portfolio_id("my-portfolio-1") == "my-portfolio-1"

    with pytest.raises(ValueError):
        validate_portfolio_id("invalid/portfolio")

    with pytest.raises(ValueError):
        validate_portfolio_id("invalid space")


def test_validate_trade_request():
    # Valid
    validate_trade_request("AAPL", "Stock", "buy", 10.0, 150.0, "USD")
    validate_trade_request("PTT", "Stock", "sell", 100.0, 35.0, "THB")

    # Invalid: Cash symbol
    with pytest.raises(InvalidTradeError):
        validate_trade_request("CASH_THB", "Cash", "buy", 100.0, 1.0, "THB")

    # Invalid units
    with pytest.raises(InvalidTradeError):
        validate_trade_request("AAPL", "Stock", "buy", -5.0, 150.0, "USD")

    # Invalid price
    with pytest.raises(InvalidTradeError):
        validate_trade_request("AAPL", "Stock", "buy", 5.0, 0.0, "USD")

    # Invalid action
    with pytest.raises(InvalidTradeError):
        validate_trade_request("AAPL", "Stock", "hold", 5.0, 100.0, "USD")


def test_validate_cash_availability():
    state = PortfolioState(
        last_updated="2026-08-23T12:00:00",
        holdings=[Holding(symbol=CASH_THB_SYMBOL, asset_type="Cash", units=5000.0)],
    )
    # Available 5000, need 3000 -> OK
    validate_cash_availability(state, 3000.0, "THB")

    # Available 5000, need 6000 -> Insufficient
    with pytest.raises(InsufficientCashError):
        validate_cash_availability(state, 6000.0, "THB")


def test_validate_holding_for_sell():
    h = Holding(symbol="AAPL", asset_type="Stock", units=10.0)
    # Hold 10, sell 5 -> OK
    validate_holding_for_sell(h, 5.0)

    # Hold 10, sell 15 -> Error
    with pytest.raises(InvalidTradeError):
        validate_holding_for_sell(h, 15.0)

    # None or 0 units
    with pytest.raises(HoldingNotFoundError):
        validate_holding_for_sell(None, 5.0)

    with pytest.raises(HoldingNotFoundError):
        validate_holding_for_sell(Holding(symbol="AAPL", asset_type="Stock", units=0.0), 5.0)
