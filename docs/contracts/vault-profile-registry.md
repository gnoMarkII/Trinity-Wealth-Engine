# Vault profile and policy registry

Registry digest: `6129738659145909400f2b060a79d9978feeb688f910afb70e66074595eddc1b`  
Policy digest: `e84469c85e31024f7ec67ed9e9fdc2811d0b1cef1053ba31915c3984d8e0bec1`

| Profile | Entity types | Source of truth | Manual edit | Search default | Retention |
|---|---|---|---|---|---|
| `capture` | capture | `vault_capture` | `allowed_until_normalized` | `excluded` | `operational` |
| `derived_artifact` | (none) | `external_runtime` | `forbidden` | `excluded` | `ephemeral` |
| `navigation` | navigation | `generated_view` | `forbidden` | `excluded` | `ephemeral` |
| `portfolio_projection` | portfolio_state, holding, watchlist_item, goal | `transactional_runtime_projection` | `projection_fields_forbidden_annotation_fields_allowed` | `excluded` | `operational` |
| `published` | concept, company_news, equity_analysis, stock_hub, quant_snapshot, earnings_call, macro_strategy, macro_snapshot, briefing_book, book_note, youtube_summary | `vault_markdown` | `human_revision_with_ownership_checks` | `included` | `permanent` |

Aliases are read-compatible only; new canonical writes use the entity type listed in `entity_profile`.
