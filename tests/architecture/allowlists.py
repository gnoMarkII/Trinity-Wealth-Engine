"""Architecture Layer Validation & Dependency Rules Allowlist."""
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ArchitectureAllowlistEntry:
    source_file: str
    imported_module: str
    violation_type: str
    owner: str
    reason: str
    removal_phase: str
    removal_condition: str


# Compatibility bridges are explicit, owned, and temporary.  They are kept
# outside application services and routers so the strict dependency tests can
# distinguish an intentional migration seam from accidental leakage.
ARCHITECTURE_ALLOWLIST: list[ArchitectureAllowlistEntry] = [
    ArchitectureAllowlistEntry(
        source_file="api/db/legacy_adapter.py",
        imported_module="api",
        violation_type="compatibility_bridge",
        owner="SQLite migration owner",
        reason="Workers still need legacy state signatures during the repository/UoW migration.",
        removal_phase="Phase 6",
        removal_condition="Worker lifecycle ports are backed directly by typed repositories and no caller patches state_db.",
    ),
    ArchitectureAllowlistEntry(
        source_file="api/compatibility/equity.py",
        imported_module="tools.archivist",
        violation_type="compatibility_bridge",
        owner="Equity context owner",
        reason="Legacy direct helper imports remain available for external callers only.",
        removal_phase="Phase 6",
        removal_condition="All downstream imports move to EquityResearchQueryPort and vault adapters.",
    ),
    ArchitectureAllowlistEntry(
        source_file="api/compatibility/portfolio.py",
        imported_module="tools.archivist",
        violation_type="compatibility_bridge",
        owner="Portfolio context owner",
        reason="Legacy strategy helper import retained for compatibility facade callers.",
        removal_phase="Phase 6",
        removal_condition="No callers import the historical portfolio compatibility helpers.",
    ),
    ArchitectureAllowlistEntry(
        source_file="tools/portfolio/services/dime_sync_service.py",
        imported_module="tools.portfolio.adapters.markdown.paths",
        violation_type="adapter_in_application_service",
        owner="Portfolio sync owner",
        reason="Direct vault path helper import during migration to dedicated repository port.",
        removal_phase="Phase 6",
        removal_condition="Portfolio path resolution moved to TradeStagingPort adapter.",
    ),
    ArchitectureAllowlistEntry(
        source_file="tools/portfolio/services/scbam_sync_service.py",
        imported_module="tools.portfolio.adapters.markdown.paths",
        violation_type="adapter_in_application_service",
        owner="Portfolio sync owner",
        reason="Direct vault path helper import during migration to dedicated repository port.",
        removal_phase="Phase 6",
        removal_condition="Portfolio path resolution moved to TradeStagingPort adapter.",
    ),
    ArchitectureAllowlistEntry(
        source_file="tools/portfolio/services/scbam_sync_service.py",
        imported_module="tools.portfolio.adapters.scb.scbam_parser_adapter",
        violation_type="adapter_in_application_service",
        owner="Portfolio sync owner",
        reason="Concrete SCBAM parser adapter injected before factory extraction.",
        removal_phase="Phase 6",
        removal_condition="Parser injected via TradeDocumentParserPort factory.",
    ),
    ArchitectureAllowlistEntry(
        source_file="tools/portfolio/services/wealthx_sync_service.py",
        imported_module="tools.portfolio.adapters.markdown.paths",
        violation_type="adapter_in_application_service",
        owner="Portfolio sync owner",
        reason="Direct vault path helper import during migration to dedicated repository port.",
        removal_phase="Phase 6",
        removal_condition="Portfolio path resolution moved to TradeStagingPort adapter.",
    ),
    ArchitectureAllowlistEntry(
        source_file="api/routers/health_router.py",
        imported_module="api.db",
        violation_type="router_infrastructure_import",
        owner="API health owner",
        reason="Direct DB ping in health endpoint before repository health check extraction.",
        removal_phase="Phase 6",
        removal_condition="Database health verified via DatabaseHealthPort adapter.",
    ),
]


def is_allowed(source_file: str, imported_module: str, violation_type: str) -> bool:
    source_norm = source_file.replace("\\", "/").strip("/")
    for entry in ARCHITECTURE_ALLOWLIST:
        entry_source = entry.source_file.replace("\\", "/").strip("/")
        if source_norm.endswith(entry_source) or entry_source in source_norm:
            if (
                entry.imported_module in imported_module
                or imported_module.startswith(entry.imported_module)
                or imported_module in entry.imported_module
            ) and entry.violation_type == violation_type:
                return True
    return False
