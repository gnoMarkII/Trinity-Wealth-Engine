from pathlib import Path


def test_pvd_domain_modules_do_not_depend_on_api_or_adapters():
    root = Path(__file__).resolve().parents[2] / "tools" / "portfolio" / "domain"
    sources = [root / name for name in ("pvd_models.py", "pvd_calculations.py", "pvd_validation.py", "pvd_replay.py")]
    banned = ("fastapi", "tools.portfolio.adapters", "api.")
    for path in sources:
        source = path.read_text(encoding="utf-8").lower()
        assert not any(import_name in source for import_name in banned), path.name
