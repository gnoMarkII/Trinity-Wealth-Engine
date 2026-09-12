import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_machine_readable_storage_contract_is_versioned() -> None:
    contract = json.loads((ROOT / "memories" / ".system" / "storage_contract.json").read_text(encoding="utf-8"))

    assert contract["schema_version"] == 2
    assert contract["status"] == "active"
    assert contract["links"]["internal"] == "relative-markdown"
    assert contract["links"]["wikilinks_in_canonical"] is False
    assert contract["paths"]["markdown_max_length"] == 180
    assert contract["identity"]["path_independent"] is True


def test_storage_contract_document_mentions_derived_separation() -> None:
    document = (ROOT / "docs" / "contracts" / "multi-app-vault-storage-contract.md").read_text(encoding="utf-8")

    assert "canonical source" in document
    assert "derived read models" in document
    assert "fail closed" in document
    assert "universal-newline normalization" in document
