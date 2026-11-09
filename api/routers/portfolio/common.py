"""Inbound error mapping shared by the canonical portfolio routers.

Legacy route helpers live in :mod:`api.compatibility.portfolio`.
"""
from api.error_mapping import handle_domain_exceptions as handle_portfolio_exceptions

__all__ = ["handle_portfolio_exceptions"]
