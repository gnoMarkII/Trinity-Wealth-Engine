"""Test navigation index builder (Task F10)."""
from pathlib import Path
from tools.archivist.navigation_builder import build_navigation_indices


def test_build_navigation_indices(tmp_path: Path) -> None:
    # Setup sample vault layout
    (tmp_path / "30_Knowledge_Base" / "Stocks" / "AAPL").mkdir(parents=True)
    (tmp_path / "30_Knowledge_Base" / "Stocks" / "MSFT").mkdir(parents=True)
    (tmp_path / "30_Knowledge_Base" / "Macroeconomics").mkdir(parents=True)
    (tmp_path / "30_Knowledge_Base" / "Macroeconomics" / "Global_Macro_2026-09-08.md").write_text("Macro content")
    (tmp_path / "30_Knowledge_Base" / "News").mkdir(parents=True)
    (tmp_path / "30_Knowledge_Base" / "News" / "Fed_Interest_Rate_Decision.md").write_text("News content")

    generated = build_navigation_indices(vault_root=tmp_path)

    assert "Home" in generated
    assert "Stocks_Hub" in generated
    assert "Macro_Hub" in generated
    assert "News_Hub" in generated
    assert "Audio_Hub" in generated

    home_text = generated["Home"].read_text(encoding="utf-8")
    assert len(home_text) <= 4000
    assert "AAPL" in home_text
    assert "MSFT" in home_text

    stocks_text = generated["Stocks_Hub"].read_text(encoding="utf-8")
    assert len(stocks_text) <= 6000
    assert "AAPL" in stocks_text
