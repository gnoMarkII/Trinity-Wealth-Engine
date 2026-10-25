"""Curated historical event catalogs for Event Studies.

Provides institutional event date presets (Apple Keynotes, Nvidia GTC, Tesla AI Days, FOMC decisions)
and helper functions to query catalog metadata.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class EventPresetItem:
    date: str           # YYYY-MM-DD
    title: str          # Descriptive event summary


@dataclass(frozen=True)
class EventPresetCatalog:
    preset_id: str
    name: str
    category: str       # "product_launch", "tech_keynote", "macro_rate", "earnings", "custom"
    description: str
    recommended_ticker: Optional[str]
    items: List[EventPresetItem]


# 1. Apple iPhone Keynote Announcements (2015 - 2024)
APPLE_IPHONE_PRESET = EventPresetCatalog(
    preset_id="apple_iphone",
    name="Apple iPhone Launch Events",
    category="product_launch",
    description="Historical announcement dates for Apple flagship iPhone keynotes.",
    recommended_ticker="AAPL",
    items=[
        EventPresetItem("2024-09-09", "iPhone 16 Series / Apple Intelligence"),
        EventPresetItem("2023-09-12", "iPhone 15 Series / USB-C & Titanium"),
        EventPresetItem("2022-09-07", "iPhone 14 Series / Dynamic Island"),
        EventPresetItem("2021-09-14", "iPhone 13 Series / ProMotion 120Hz"),
        EventPresetItem("2020-10-13", "iPhone 12 Series / 5G Supercycle"),
        EventPresetItem("2019-09-10", "iPhone 11 Series / Triple Camera System"),
        EventPresetItem("2018-09-12", "iPhone XS & XR Series"),
        EventPresetItem("2017-09-12", "iPhone X & 8 Series / OLED All-Screen"),
        EventPresetItem("2016-09-07", "iPhone 7 Series / Dual Camera & Water Resistance"),
        EventPresetItem("2015-09-09", "iPhone 6s Series / 3D Touch"),
    ],
)

# 2. Nvidia GTC Keynotes (2018 - 2024)
NVIDIA_GTC_PRESET = EventPresetCatalog(
    preset_id="nvidia_gtc",
    name="Nvidia GTC Architecture Keynotes",
    category="tech_keynote",
    description="Jensen Huang opening keynotes introducing new GPU & AI architectures.",
    recommended_ticker="NVDA",
    items=[
        EventPresetItem("2024-03-18", "GTC 2024 / Blackwell B200 Architecture"),
        EventPresetItem("2023-03-21", "GTC 2023 / Generative AI & DGX Cloud"),
        EventPresetItem("2022-03-22", "GTC 2022 / Hopper H100 GPU Architecture"),
        EventPresetItem("2021-04-12", "GTC 2021 / Grace CPU & Omniverse"),
        EventPresetItem("2020-05-14", "GTC 2020 / Ampere A100 GPU Architecture"),
        EventPresetItem("2019-03-18", "GTC 2019 / RTX Server & Mellanox Acquisition"),
        EventPresetItem("2018-03-27", "GTC 2018 / Tensor Core GPU & Drive Pegasus"),
    ],
)

# 3. Tesla Keynotes & Technology Days (2019 - 2024)
TESLA_EVENTS_PRESET = EventPresetCatalog(
    preset_id="tesla_events",
    name="Tesla Keynotes & AI/Battery Days",
    category="product_launch",
    description="Tesla flagship technology days, master plans, and product reveals.",
    recommended_ticker="TSLA",
    items=[
        EventPresetItem("2024-10-10", "We, Robot / Cybercab & Robovan Reveal"),
        EventPresetItem("2023-03-01", "Investor Day 2023 / Master Plan Part 3"),
        EventPresetItem("2022-09-30", "AI Day 2022 / Optimus Humanoid Robot Prototype"),
        EventPresetItem("2021-08-19", "AI Day 2021 / Dojo Chip & Tesla Bot Concept"),
        EventPresetItem("2020-09-22", "Battery Day 2020 / 4680 Tabless Cell Innovation"),
        EventPresetItem("2019-11-21", "Cybertruck & Cyberquad Reveal"),
    ],
)

# 4. Federal Reserve Pivotal FOMC Rate Decisions (2022 - 2024)
FED_FOMC_PRESET = EventPresetCatalog(
    preset_id="fed_fomc",
    name="FOMC Pivotal Rate Policy Decisions",
    category="macro_rate",
    description="Key Federal Reserve interest rate announcements across the hiking and cutting cycle.",
    recommended_ticker=None,  # Applicable across all assets (SPY, QQQ, AAPL, etc.)
    items=[
        EventPresetItem("2024-12-18", "FOMC 25bps Rate Cut (Fed Funds 4.25%-4.50%)"),
        EventPresetItem("2024-11-07", "FOMC 25bps Rate Cut (Fed Funds 4.50%-4.75%)"),
        EventPresetItem("2024-09-18", "FOMC 50bps Rate Cut (First cut of easing cycle)"),
        EventPresetItem("2023-07-26", "FOMC 25bps Rate Hike (Terminal hike at 5.25%-5.50%)"),
        EventPresetItem("2023-05-03", "FOMC 25bps Rate Hike"),
        EventPresetItem("2023-03-22", "FOMC 25bps Rate Hike (Post-SVB banking crisis)"),
        EventPresetItem("2023-02-01", "FOMC 25bps Rate Hike"),
        EventPresetItem("2022-12-14", "FOMC 50bps Rate Hike"),
        EventPresetItem("2022-11-02", "FOMC 75bps Rate Hike"),
        EventPresetItem("2022-09-21", "FOMC 75bps Rate Hike"),
        EventPresetItem("2022-07-27", "FOMC 75bps Rate Hike"),
        EventPresetItem("2022-06-15", "FOMC 75bps Rate Hike (First 75bps hike since 1994)"),
        EventPresetItem("2022-05-04", "FOMC 50bps Rate Hike"),
        EventPresetItem("2022-03-16", "FOMC 25bps Rate Lift-off (Cycle beginning)"),
    ],
)

PRESET_REGISTRY: Dict[str, EventPresetCatalog] = {
    APPLE_IPHONE_PRESET.preset_id: APPLE_IPHONE_PRESET,
    NVIDIA_GTC_PRESET.preset_id: NVIDIA_GTC_PRESET,
    TESLA_EVENTS_PRESET.preset_id: TESLA_EVENTS_PRESET,
    FED_FOMC_PRESET.preset_id: FED_FOMC_PRESET,
}


def list_event_presets() -> List[Dict[str, str]]:
    """Returns metadata for all available event presets."""
    results = [
        {
            "preset_id": "earnings",
            "name": "Earnings Announcements (PEAD)",
            "category": "earnings",
            "description": "Historical quarterly earnings releases extracted from market calendar.",
            "recommended_ticker": None,
        }
    ]
    for p in PRESET_REGISTRY.values():
        results.append(
            {
                "preset_id": p.preset_id,
                "name": p.name,
                "category": p.category,
                "description": p.description,
                "recommended_ticker": p.recommended_ticker or "",
            }
        )
    results.append(
        {
            "preset_id": "custom",
            "name": "Custom Event Dates",
            "category": "custom",
            "description": "User-supplied custom list of historical event dates.",
            "recommended_ticker": "",
        }
    )
    return results


def get_preset_catalog(preset_id: str) -> Optional[EventPresetCatalog]:
    """Retrieves a catalog by preset_id."""
    return PRESET_REGISTRY.get(preset_id.lower().strip())
