"""Test market_tools: news ingestion, parsing, Vault V2 persistence, and query adapter."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

import tools.market.news as news
from tools.archivist.parser import parse_company_news_items
from tools.market.adapters.equity_vault_query_adapter import EquityVaultQueryAdapter
from tools.market.terminal_v2.adapters.rss_news_adapter import RssNewsDiscoveryAdapter


class TestIngestStockNewsThaiStock:
    def test_th_market_ticker_normalization(self, mock_yf_ticker, monkeypatch):
        monkeypatch.setattr(
            "tools.market.terminal_v2.adapters.rss_news_adapter.RssNewsDiscoveryAdapter.get_news_candidates",
            lambda *args, **kwargs: None,
        )
        result = news.ingest_stock_news.invoke({"ticker": "PTT", "market": "TH"})
        assert mock_yf_ticker["ticker"] == "PTT.BK"
        assert "ข่าวล่าสุด: PTT (TH)" in result


class TestParseCompanyNewsItems:
    def test_parse_frontmatter_tickers_and_summary(self):
        sample_md = """---
schema_version: 2
title: NFLX Latest News 2026-10-17
entity_type: company_news
tickers: [NFLX]
market: US
date: 2026-10-17
---

### สรุปข่าวล่าสุด: NFLX (US)
อัปเดตล่าสุด: 2026-10-17 12:00:00

1. **Netflix Q3 Earnings Beats Expectations** (Bloomberg)
   - สรุปย่อ: Netflix added 5 million subscribers in Q3 driven by ad tier.
   - ความสดใหม่: 2 ชั่วโมงที่แล้ว
   - ลิงก์: https://bloomberg.com/news/1
"""
        parsed = parse_company_news_items(sample_md)
        assert parsed["ticker"] == "NFLX"
        assert parsed["market"] == "US"
        assert parsed["news_date"] == "2026-10-17"
        assert len(parsed["items"]) == 1

        item = parsed["items"][0]
        assert item["title"] == "Netflix Q3 Earnings Beats Expectations"
        assert item["source"] == "Bloomberg"
        assert item["summary"] == "Netflix added 5 million subscribers in Q3 driven by ad tier."
        assert item["link"] == "https://bloomberg.com/news/1"

    def test_parse_frontmatter_single_ticker_fallback(self):
        sample_md = """---
entity_type: company_news
ticker: AAPL
market: US
date: 2026-10-17
---
1. **Apple Unveils New Hardware** (Reuters)
   - ลิงก์: https://reuters.com/news/apple
"""
        parsed = parse_company_news_items(sample_md)
        assert parsed["ticker"] == "AAPL"
        assert len(parsed["items"]) == 1
        assert parsed["items"][0]["title"] == "Apple Unveils New Hardware"
        assert parsed["items"][0]["summary"] is None


class TestEquityVaultQueryAdapterNews:
    def test_get_news_vault_v2_and_regex_boundary(self, tmp_path: Path):
        vault_dir = tmp_path / "vault"
        news_v2_dir = vault_dir / "30_Knowledge_Base" / "News" / "2026" / "10"
        news_v2_dir.mkdir(parents=True)

        # Create PTT news
        ptt_data = {
            "ticker": "PTT",
            "market": "TH",
            "last_updated": "2026-10-17 10:00:00",
            "news_date": "2026-10-17",
            "items": [
                {
                    "title": "PTT Group Clean Energy Expansion",
                    "source": "Bangkok Post",
                    "link": "https://bangkokpost.com/ptt",
                    "summary": "PTT pushes into green energy initiatives.",
                    "age_hours": 1,
                    "freshness_reason": "Fresh",
                    "is_stale": False,
                }
            ],
        }
        ptt_file = news_v2_dir / "2026-10-17 PTT Latest News 2026-10-17.json"
        ptt_file.write_text(json.dumps(ptt_data), encoding="utf-8")

        # Create PTTEP news
        pttep_data = {
            "ticker": "PTTEP",
            "market": "TH",
            "last_updated": "2026-10-17 11:00:00",
            "news_date": "2026-10-17",
            "items": [
                {
                    "title": "PTTEP Deepwater Exploration Results",
                    "source": "Krungthep Turakij",
                    "link": "https://kt.com/pttep",
                    "summary": "PTTEP announces gas discovery in Gulf of Thailand.",
                    "age_hours": 2,
                    "freshness_reason": "Fresh",
                    "is_stale": False,
                }
            ],
        }
        pttep_file = news_v2_dir / "2026-10-17 PTTEP Latest News 2026-10-17.json"
        pttep_file.write_text(json.dumps(pttep_data), encoding="utf-8")

        adapter = EquityVaultQueryAdapter(vault_path=vault_dir)

        # Query PTT: must NOT match PTTEP!
        res_ptt = adapter.get_news("PTT")
        assert res_ptt is not None
        assert res_ptt["ticker"] == "PTT"
        assert len(res_ptt["items"]) == 1
        assert res_ptt["items"][0]["title"] == "PTT Group Clean Energy Expansion"
        assert res_ptt["items"][0]["summary"] == "PTT pushes into green energy initiatives."

        # Query PTTEP: returns PTTEP news
        res_pttep = adapter.get_news("PTTEP")
        assert res_pttep is not None
        assert res_pttep["ticker"] == "PTTEP"
        assert res_pttep["items"][0]["title"] == "PTTEP Deepwater Exploration Results"

    def test_get_news_vault_v1_fallback(self, tmp_path: Path):
        vault_dir = tmp_path / "vault"
        stocks_v1_dir = vault_dir / "30_Knowledge_Base" / "Stocks" / "TSLA"
        stocks_v1_dir.mkdir(parents=True)

        tsla_data = {
            "ticker": "TSLA",
            "market": "US",
            "last_updated": "2026-10-16 15:00:00",
            "news_date": "2026-10-16",
            "items": [
                {
                    "title": "Tesla Robotaxi Event",
                    "source": "CNBC",
                    "link": "https://cnbc.com/tsla",
                    "summary": "Autonomous ride-hailing vehicle revealed.",
                    "age_hours": 3,
                    "freshness_reason": "Recent",
                    "is_stale": False,
                }
            ],
        }
        tsla_file = stocks_v1_dir / "TSLA News 2026-10-16.json"
        tsla_file.write_text(json.dumps(tsla_data), encoding="utf-8")

        adapter = EquityVaultQueryAdapter(vault_path=vault_dir)
        res_tsla = adapter.get_news("TSLA")
        assert res_tsla is not None
        assert res_tsla["ticker"] == "TSLA"
        assert res_tsla["items"][0]["title"] == "Tesla Robotaxi Event"

    def test_get_news_vault_v2_ignores_article_note_and_picks_news_with_items(self, tmp_path: Path):
        vault_dir = tmp_path / "vault"
        news_v2_dir = vault_dir / "30_Knowledge_Base" / "News" / "2026" / "10"
        news_v2_dir.mkdir(parents=True)

        # 1. General macro article mentioning META with items=[]
        macro_article = {
            "title": "อนาคต Dow Jones... ผลประกอบการ Meta และ SpaceX จะกู้ชีพกลุ่ม AI ได้หรือไม่",
            "items": [],
        }
        (news_v2_dir / "2026-10-18_อนาคต Dow Jones Meta SpaceX.json").write_text(
            json.dumps(macro_article), encoding="utf-8"
        )

        # 2. Actual company news file with items
        meta_news = {
            "ticker": "META",
            "market": "US",
            "last_updated": "2026-10-18 09:00:00",
            "news_date": "2026-10-18",
            "items": [
                {
                    "title": "Meta Platforms Unveils Llama 4 AI Model",
                    "source": "Reuters",
                    "link": "https://reuters.com/meta-ai",
                    "summary": "Meta releases next-gen open weights.",
                    "age_hours": 1,
                    "freshness_reason": "Fresh",
                    "is_stale": False,
                }
            ],
        }
        (news_v2_dir / "2026-10-18 META Latest News 2026-10-18.json").write_text(
            json.dumps(meta_news), encoding="utf-8"
        )

        adapter = EquityVaultQueryAdapter(vault_path=vault_dir)
        res = adapter.get_news("META")
        assert res is not None
        assert res["ticker"] == "META"
        assert len(res["items"]) == 1
        assert res["items"][0]["title"] == "Meta Platforms Unveils Llama 4 AI Model"


class TestRssNewsDiscoveryAdapterRouting:
    def test_build_feed_urls_thai_stock(self):
        adapter = RssNewsDiscoveryAdapter()
        feeds = adapter._build_feed_urls("PTT.BK")
        feed_types = [f[0] for f in feeds]
        assert "google_news_th" in feed_types
        assert "google_news_en" in feed_types

    def test_build_feed_urls_us_stock(self):
        adapter = RssNewsDiscoveryAdapter()
        feeds = adapter._build_feed_urls("AAPL")
        feed_types = [f[0] for f in feeds]
        assert "google_news_us" in feed_types
        assert "yahoo_rss" in feed_types


class TestSaveCompanyNewsToVault:
    def test_save_company_news_creates_v2_note_and_sidecar(self, tmp_path: Path, monkeypatch):
        vault_dir = tmp_path / "vault"
        sys_dir = vault_dir / ".system"
        sys_dir.mkdir(parents=True)
        (sys_dir / "vault_config.json").write_text(json.dumps({"layout_version": 2}), encoding="utf-8")
        monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault_dir))

        sample_news_md = """---
schema_version: 2
title: NVDA Latest News 2026-10-17
entity_type: company_news
ticker: NVDA
tickers: [NVDA]
market: US
date: 2026-10-17
published_date: 2026-10-17
last_updated: 2026-10-17 14:00:00
tags: [news, nvda, market_us, stock_analysis]
---

### สรุปข่าวล่าสุด: NVDA (US)
อัปเดตล่าสุด: 2026-10-17 14:00:00

1. **Nvidia Announces Next-Gen Architecture** (Wall Street Journal)
   - สรุปย่อ: Massive bandwidth and power efficiency improvements for AI datacenters.
   - ความสดใหม่: 1 ชั่วโมงที่แล้ว
   - ลิงก์: https://wsj.com/nvda-next
"""
        news.save_company_news_to_vault(sample_news_md, "NVDA")

        # Check that file was created in 30_Knowledge_Base/News/YYYY/MM
        news_folder = vault_dir / "30_Knowledge_Base" / "News" / "2026" / "10"
        assert news_folder.exists()

        md_files = list(news_folder.glob("*.md"))
        json_files = list(news_folder.glob("*.json"))

        assert len(md_files) >= 1
        assert len(json_files) >= 1

        # Check JSON companion content
        sidecar = json_files[0]
        sidecar_data = json.loads(sidecar.read_text(encoding="utf-8"))
        assert sidecar_data["ticker"] == "NVDA"
        assert sidecar_data["items"][0]["title"] == "Nvidia Announces Next-Gen Architecture"
        assert sidecar_data["items"][0]["summary"] == "Massive bandwidth and power efficiency improvements for AI datacenters."

        # Verify query adapter can read it
        adapter = EquityVaultQueryAdapter(vault_path=vault_dir)
        vault_news = adapter.get_news("NVDA")
        assert vault_news is not None
        assert vault_news["ticker"] == "NVDA"
        assert len(vault_news["items"]) == 1
