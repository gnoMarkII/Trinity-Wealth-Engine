from langchain_core.messages import AIMessage
from types import SimpleNamespace

from agents.macro_economist_agent import create_macro_economist
from schemas.macro_schemas import NarrativeContext


def test_economist_receives_all_observables_and_their_validity(monkeypatch):
    import agents.macro_economist_agent as economist
    import tools.knowledge.youtube_monitor as youtube

    monkeypatch.setattr(economist, 'get_macro_baselines', SimpleNamespace(invoke=lambda _: 'Baseline'))
    monkeypatch.setattr(economist, 'generate_news_radar_daily', SimpleNamespace(invoke=lambda _: 'News'))
    monkeypatch.setattr(economist, 'news_references_from_radar', lambda _: [])
    monkeypatch.setattr(economist, 'recent_youtube_references', lambda **_: [])
    monkeypatch.setattr(youtube, 'load_recent_youtube_insights', lambda **_: '')
    received = []

    class Model:
        def with_structured_output(self, schema):
            assert schema is NarrativeContext
            return self

        def invoke(self, messages):
            received.extend(messages)
            return NarrativeContext(
                evaluated_at='2026-10-04', dominant_themes=[], market_sentiment='neutral',
                tail_risks=[], policy_signals=[], key_narratives_by_region={}, sources_summary='Test',
            )

    observables = [{
        'observable_id': f'obs_audit_{i}', 'value': f'{i}.25',
        'observed_at': '2026-10-02', 'is_valid': i != 106,
        'metadata': {'prior_mean_bid_to_cover': 2.8},
    } for i in range(107)]
    result = create_macro_economist(Model()).invoke({'quant_score': {'market_observables': observables}})
    prompt = received[1]['content']
    for observable in observables:
        assert observable['observable_id'] in prompt
    assert '"is_valid": false' in prompt
    assert '"prior_mean_bid_to_cover": 2.8' in prompt
    assert isinstance(result['messages'][0], AIMessage)


def test_audit_detects_missing_changed_and_invalid_cited_evidence():
    from scripts.audit_macro_page_data import summarize_ai_integrity

    snapshot = [{'observable_id': 'kept', 'value': '10', 'unit': '%', 'observed_at': '2026-10-02', 'is_valid': True},
                {'observable_id': 'dropped', 'value': '20'}]
    dashboard = {'observable_registry': {'kept': {**snapshot[0], 'value': '99', 'is_valid': False}},
                 'asset_allocation': [{'observable_refs': ['kept', 'unknown']}]}
    result = summarize_ai_integrity(dashboard, snapshot)
    assert result['missing_from_report'] == ['dropped']
    assert result['changed_evidence'] == ['kept']
    assert result['invalid_or_unknown_citations'] == ['kept', 'unknown']
