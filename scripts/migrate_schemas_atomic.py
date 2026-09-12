"""Atomic Migration Script for api/schemas.py -> api/schemas/ package."""
import ast
import os
import shutil

def main():
    pkg_dir = "api/schemas_pkg"
    os.makedirs(pkg_dir, exist_ok=True)

    with open("api/schemas.py", "r", encoding="utf-8") as f:
        full_text = f.read()

    # 1. Macro
    macro_part = full_text[full_text.find("class SourceOverrideAck"):full_text.find("class JobStatusDTO")]
    macro_code = (
        '"""Macro API Schemas."""\n'
        'from typing import Any, Literal, Optional\n'
        'from pydantic import BaseModel\n'
        'from schemas.warning_registry import WarningMessage, translate_warning\n\n'
        + macro_part
    )
    with open(os.path.join(pkg_dir, "macro.py"), "w", encoding="utf-8") as f:
        f.write(macro_code)

    # 2. Kanban
    kanban_part1 = full_text[full_text.find("class JobStatusDTO"):full_text.find("class DividendRoundDTO")]
    kanban_part2 = full_text[full_text.find("class NewsFunnelPendingItemDTO"):full_text.find("class NotebookLMAvailableSourceDTO")]
    kanban_part3 = full_text[full_text.find("class NewsFunnelFilteredItemDTO"):full_text.find("class EquitySummaryDTO")]
    kanban_code = (
        '"""Kanban and Agent Job API Schemas."""\n'
        'from typing import Any, Optional\n'
        'from pydantic import BaseModel\n\n'
        + kanban_part1 + "\n\n" + kanban_part2 + "\n\n" + kanban_part3
    )
    with open(os.path.join(pkg_dir, "kanban.py"), "w", encoding="utf-8") as f:
        f.write(kanban_code)

    # 3. NotebookLM
    nlm_part = full_text[full_text.find("class NotebookLMAvailableSourceDTO"):full_text.find("class NewsFunnelFilteredItemDTO")]
    nlm_code = (
        '"""NotebookLM Schemas."""\n'
        'from typing import Optional\n'
        'from pydantic import BaseModel\n\n'
        + nlm_part
    )
    with open(os.path.join(pkg_dir, "notebooklm.py"), "w", encoding="utf-8") as f:
        f.write(nlm_code)

    # 4. Portfolio
    portfolio_part1 = full_text[full_text.find("class DividendRoundDTO"):full_text.find("class NewsFunnelPendingItemDTO")]
    portfolio_part2 = full_text[full_text.find("class CalendarEventDTO"):full_text.find("class OHLCVCandleDTO")]
    portfolio_code = (
        '"""Portfolio, Positions, Transactions, and Goals API Schemas."""\n'
        'from typing import Any, Literal, Optional\n'
        'from pydantic import BaseModel, Field\n\n'
        + portfolio_part1 + "\n\n" + portfolio_part2
    )
    with open(os.path.join(pkg_dir, "portfolio.py"), "w", encoding="utf-8") as f:
        f.write(portfolio_code)

    # 5. Equity
    equity_part1 = full_text[full_text.find("class EquitySummaryDTO"):full_text.find("class CalendarEventDTO")]
    equity_part2 = full_text[full_text.find("class OHLCVCandleDTO"):full_text.find("class LineItemMetaDTO")]
    equity_code = (
        '"""Equity Intel, Technicals, Targets, and Filings API Schemas."""\n'
        'from typing import Any, Literal, Optional\n'
        'from pydantic import BaseModel, Field\n\n'
        + equity_part1 + "\n\n" + equity_part2
    )
    with open(os.path.join(pkg_dir, "equity.py"), "w", encoding="utf-8") as f:
        f.write(equity_code)

    # 6. Financials (Single Source of Truth -> domain.models)
    fin_code = (
        '"""Financials API Schemas (Re-exported from tools.market.financials.domain.models)."""\n'
        'from tools.market.financials.domain.models import (\n'
        '    LineItemMetaDTO,\n'
        '    FinancialCellDTO,\n'
        '    FinancialPeriodDTO,\n'
        '    FinancialStatementCategoryDTO,\n'
        '    FinancialSummaryChartPointDTO,\n'
        '    FinancialRatioPointDTO,\n'
        '    FinancialStatementsDTO,\n'
        ')\n\n'
        '__all__ = [\n'
        '    "LineItemMetaDTO",\n'
        '    "FinancialCellDTO",\n'
        '    "FinancialPeriodDTO",\n'
        '    "FinancialStatementCategoryDTO",\n'
        '    "FinancialSummaryChartPointDTO",\n'
        '    "FinancialRatioPointDTO",\n'
        '    "FinancialStatementsDTO",\n'
        ']\n'
    )
    with open(os.path.join(pkg_dir, "financials.py"), "w", encoding="utf-8") as f:
        f.write(fin_code)

    # 7. __init__.py
    init_code = (
        '"""API Schemas Package Re-exporting all DTOs for 100% Backward Compatibility."""\n'
        'from api.schemas.macro import *\n'
        'from api.schemas.kanban import *\n'
        'from api.schemas.notebooklm import *\n'
        'from api.schemas.portfolio import *\n'
        'from api.schemas.equity import *\n'
        'from api.schemas.financials import *\n'
    )
    with open(os.path.join(pkg_dir, "__init__.py"), "w", encoding="utf-8") as f:
        f.write(init_code)

    # 8. Backup schemas.py and swap
    shutil.copyfile("api/schemas.py", "api/schemas_legacy.py")
    os.remove("api/schemas.py")
    if os.path.exists("api/schemas"):
        shutil.rmtree("api/schemas")
    os.rename(pkg_dir, "api/schemas")

    print("Atomic swap complete: api/schemas/ package is active.")

if __name__ == "__main__":
    main()
