"""Inbound error mapping shared by the canonical portfolio routers.

Legacy route helpers intentionally live in :mod:`api.compatibility.portfolio`
and are imported only by the historical ``api.routes_portfolio`` facade.
"""
from api.error_mapping import handle_domain_exceptions as handle_portfolio_exceptions

__all__ = ["handle_portfolio_exceptions"]
