import json


_SAMPLE_DIRECTION = {
    "evaluated_at": "2026-07-06T18:22:00",
    "overall_regime": "Recession",
    "time_horizon": "3-6 เดือน",
    "conviction_level": "medium",
    "conviction_rationale": "ตัวเลขสนับสนุนภาวะถดถอย",
    "quant_narrative_alignment": "aligned",
    "divergence_note": "",
    "focus_themes": ["Defensive"],
    "key_assumptions": ["เศรษฐกิจชะลอตัว"],
    "regime_probabilities": {"Goldilocks": 0.15, "Recession": 0.4},
    "regime_evidence": [
        {"dimension": "Growth", "signal": "ชะลอตัว", "evidence": "Real GDP = 2.68%", "conflict": "", "confidence": "medium"}
    ],
    "asset_allocation": [
        {
            "asset_class": "US Equities",
            "asset_bucket": "equities",
            "stance": "Overweight",
            "confidence": "medium",
            "rationale": "หมุนเวียนไปกลุ่ม Defensive",
            "supporting_data": ["VIX = 16.31"],
            "why_not_high": "Valuation ยังสูง",
            "allocation_delta": "+2% vs benchmark",
            "invalidation_conditions": [],
            "validation_warnings": ["[SUPPORTING_DATA_MISMATCH]"],
        }
    ],
    "pair_trades": [],
    "risk_scenarios": [],
    "validation_warnings": ["[SOME_FUTURE_WARNING_ID_NOT_YET_KNOWN]"],
    "stale_data_warnings": [],
}


def _write_sidecar(tmp_path, monkeypatch, module, payload=_SAMPLE_DIRECTION):
    vault_dir = tmp_path / "vault"
    strategy_dir = vault_dir / "30_Knowledge_Base" / "Strategies"
    strategy_dir.mkdir(parents=True)
    (strategy_dir / "Macro_Strategy_Direction_2026-07-06.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    monkeypatch.setattr(module, "VAULT_PATH", vault_dir)


def test_portfolio_latest_maps_dto_fields(authed_client, tmp_path, monkeypatch):
    import api.routes_portfolio as routes_portfolio_module

    _write_sidecar(tmp_path, monkeypatch, routes_portfolio_module)

    r = authed_client.get("/api/portfolio/latest")
    assert r.status_code == 200
    body = r.json()
    assert body["overall_regime"] == "Recession"
    assert body["asset_allocation"][0]["asset_class"] == "US Equities"
    assert body["asset_allocation"][0]["allocation_delta"] == "+2% vs benchmark"


def test_portfolio_warnings_render_generically_for_unknown_id(authed_client, tmp_path, monkeypatch):
    """Warning ID ที่ registry ยังไม่มี Thai template ต้องยัง fallback แสดงได้ ไม่ crash/หาย (Rev.5 ข้อ 7)"""
    import api.routes_portfolio as routes_portfolio_module

    _write_sidecar(tmp_path, monkeypatch, routes_portfolio_module)

    r = authed_client.get("/api/portfolio/latest")
    body = r.json()

    top_level_codes = [w["code"] for w in body["warnings"]]
    assert "SOME_FUTURE_WARNING_ID_NOT_YET_KNOWN" in top_level_codes
    for w in body["warnings"]:
        assert w["message"]  # ต้องมีข้อความเสมอ แม้ไม่มี Thai template

    asset_codes = [w["code"] for w in body["asset_allocation"][0]["warnings"]]
    assert "SUPPORTING_DATA_MISMATCH" in asset_codes


def test_macro_dashboard_maps_regime_evidence(authed_client, tmp_path, monkeypatch):
    import api.routes_portfolio as routes_portfolio_module

    _write_sidecar(tmp_path, monkeypatch, routes_portfolio_module)

    r = authed_client.get("/api/macro/dashboard")
    assert r.status_code == 200
    body = r.json()
    assert body["regime_probabilities"]["Recession"] == 0.4
    assert body["regime_evidence"][0]["dimension"] == "Growth"


def test_macro_dashboard_returns_chart_and_content_metadata(authed_client, tmp_path, monkeypatch):
    import api.routes_portfolio as routes_portfolio_module

    payload = json.loads(json.dumps(_SAMPLE_DIRECTION))
    payload["dashboard_indicators"] = [
        {
            "indicator_id": "obs_us10y_20260710",
            "series_key": "obs_us10y",
            "label": "US 10Y Yield",
            "value": 4.25,
            "display_value": "4.25%",
            "unit": "%",
            "observed_at": "2026-07-10",
            "provider": "FRED",
            "source_file": "Global_Macro_Snapshot.md",
            "is_valid": True,
            "stale_reason": "",
            "chart_available": True,
        }
    ]
    payload["report_references"] = [
        {
            "reference_id": "youtube_abc",
            "kind": "youtube",
            "title": "Macro outlook",
            "url": "https://www.youtube.com/watch?v=abcdefghijk",
            "publisher": "Macro Channel",
            "published_at": "2026-07-10",
            "thumbnail_url": "https://i.ytimg.com/vi/abcdefghijk/hqdefault.jpg",
        },
        {"reference_id": "ignored", "kind": "news", "title": "Bad", "url": "javascript:alert(1)"},
    ]
    _write_sidecar(tmp_path, monkeypatch, routes_portfolio_module, payload)

    r = authed_client.get("/api/macro/dashboard")

    assert r.status_code == 200
    body = r.json()
    assert body["dashboard_indicators"][0]["series_key"] == "obs_us10y"
    assert [item["reference_id"] for item in body["report_references"]] == ["youtube_abc"]


def test_macro_indicator_series_uses_registered_latest_indicator(authed_client, tmp_path, monkeypatch):
    import api.routes_portfolio as routes_portfolio_module

    payload = json.loads(json.dumps(_SAMPLE_DIRECTION))
    payload["dashboard_indicators"] = [
        {
            "indicator_id": "obs_us10y_20260710",
            "series_key": "obs_us10y",
            "label": "US 10Y Yield",
            "unit": "%",
        }
    ]
    _write_sidecar(tmp_path, monkeypatch, routes_portfolio_module, payload)
    series_dir = tmp_path / "vault" / "30_Knowledge_Base" / "Strategies" / "Macro_Indicator_Series"
    series_dir.mkdir(parents=True)
    (series_dir / "obs_us10y.json").write_text(
        json.dumps({"points": [{"observed_at": "2026-07-09", "value": 4.2}, {"observed_at": "2026-07-10", "value": 4.25}]}),
        encoding="utf-8",
    )

    r = authed_client.get("/api/macro/indicators/obs_us10y_20260710/series?range=1m")

    assert r.status_code == 200
    assert r.json()["points"][-1]["value"] == 4.25
    assert authed_client.get("/api/macro/indicators/obs_us10y_20260710/series?range=10y").status_code == 422


def test_portfolio_latest_404_when_no_report_exists(authed_client, tmp_path, monkeypatch):
    import api.routes_portfolio as routes_portfolio_module

    vault_dir = tmp_path / "empty_vault"
    (vault_dir / "30_Knowledge_Base" / "Strategies").mkdir(parents=True)
    monkeypatch.setattr(routes_portfolio_module, "VAULT_PATH", vault_dir)

    r = authed_client.get("/api/portfolio/latest")
    assert r.status_code == 404
