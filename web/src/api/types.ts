// Type ตรงตาม api/schemas.py — DTO layer เดียวที่ frontend ผูกด้วย ไม่ใช่ schemas/macro_schemas.py ภายใน

export interface WarningDTO {
  code: string | null
  message: string
}

export type SectorRotationQuadrant = 'Leading' | 'Weakening' | 'Lagging' | 'Improving'

export interface SectorRotationPointDTO {
  as_of: string
  relative_trend: number | null
  relative_momentum: number | null
  quadrant: SectorRotationQuadrant | null
  status: 'available' | 'unavailable'
  reason: string | null
}

export interface SectorReturnMetricDTO {
  absolute_return_pct: number | null
  excess_return_pp: number | null
  relative_return_pct: number | null
  start_date: string | null
  end_date: string | null
  expected_sessions: number
  valid_sessions: number
  status: 'available' | 'partial' | 'unavailable'
  freshness: 'fresh' | 'stale' | 'unknown'
  reason: string | null
}

export interface SectorRotationRowDTO {
  ticker: string
  name: string
  status: 'available' | 'partial' | 'unavailable'
  reason: string | null
  returns_pct: Record<string, number | null>
  return_metrics: Record<string, SectorReturnMetricDTO>
  price_as_of: string | null
  rotation_as_of: string | null
  relative_strength: number | null
  relative_price_base_date: string | null
  relative_trend: number | null
  relative_momentum: number | null
  quadrant: SectorRotationQuadrant | null
  quadrant_changed_at: string | null
  momentum_direction: 'rising' | 'falling' | 'flat' | 'unavailable'
  history: SectorRotationPointDTO[]
  relative_price_history: Array<{
    as_of: string
    sector_spy_rebased_100: number | null
    status: 'available' | 'unavailable'
    reason: string | null
  }>
  quadrant_transitions: Array<{
    event_id: string
    timeframe: 'daily' | 'weekly'
    previous_valid_at: string
    changed_at: string | null
    confirmed_at: string | null
    event_type: 'transition' | 'confirmed_transition'
    from_quadrant: SectorRotationQuadrant
    to_quadrant: SectorRotationQuadrant
  }>
}

export interface SectorRotationSnapshotDTO {
  schema_version: string
  formula_version: string
  calendar_version: string
  transition_rule_version: string
  formula_config: Record<string, unknown>
  universe_version: string
  benchmark: string
  price_basis: string
  input_digest: string
  snapshot_id: string
  as_of_date: string | null
  expected_session: string | null
  expected_weekly_session: string | null
  input_start_date: string | null
  coverage: Record<string, number>
  expected_sectors: number
  available_sectors: number
  benchmark_status: 'available' | 'unavailable'
  benchmark_reason: string | null
  benchmark_returns_pct: Record<string, number | null>
  rows: SectorRotationRowDTO[]
}

export interface SectorRotationResponseDTO {
  capability_status: 'enabled' | 'disabled'
  refresh_state: 'idle' | 'running' | 'failed'
  retry_after_seconds: number | null
  error_code: string | null
  last_attempt_at: string | null
  expected_session: string | null
  freshness: 'fresh' | 'stale' | 'unknown'
  missing_sessions: number
  served_at: string
  timeframe: 'daily' | 'weekly'
  tail: number
  summary?: SectorRotationSummaryDTO | null
  snapshot: SectorRotationSnapshotDTO | null
}

export interface SectorExcessRankDTO {
  ticker: string
  name: string
  excess_return_pp: number
  as_of: string | null
  status: 'available' | 'partial'
  valid_sessions: number
  expected_sessions: number
}

export interface SectorRotationSummaryDTO {
  summary_version: string
  timeframe: 'daily' | 'weekly'
  rotation_as_of: string | null
  ranked_by_excess_3m: SectorExcessRankDTO[]
  sector_breadth_3m: {
    outperforming: number
    valid_sectors: number
    expected_sectors: number
    status: 'complete' | 'partial'
    as_of: string | null
  }
  quadrant_members: Record<string, string[]>
  periods_in_quadrant: Record<string, number | null>
  elapsed_days_in_quadrant: Record<string, number | null>
  momentum_delta: Record<string, number | null>
  heading_deg: Record<string, number | null>
}

export interface SectorRotationHistoryRowDTO {
  ticker: string
  name: string
  status: 'available' | 'partial' | 'unavailable'
  reason: string | null
  relative_price_base_date: string | null
  history: SectorRotationPointDTO[]
  relative_price_history: SectorRotationRowDTO['relative_price_history']
  quadrant_transitions: SectorRotationRowDTO['quadrant_transitions']
}

export interface SectorRotationHistoryDTO {
  snapshot_id: string
  input_digest: string
  formula_version: string
  timeframe: 'daily' | 'weekly'
  range: '3m' | '6m' | '1y' | '2y'
  from_date: string
  to_date: string | null
  rows: SectorRotationHistoryRowDTO[]
}

export interface AssetAllocationDTO {
  asset_class: string
  asset_bucket: string | null
  region?: 'Global' | 'US' | 'Thailand' | string
  stance: string
  confidence: string
  rationale: string
  supporting_data: string[]
  why_not_high: string
  allocation_delta: string
  invalidation_conditions: string[]
  source_refs?: string[]
  observable_refs?: string[]
  warnings: WarningDTO[]
}

export interface PairTradeDTO {
  long_leg: string
  short_leg: string
  thesis: string
  catalyst: string
  risk: string
  time_horizon: string
  confidence: string
  sizing_guidance: string
  instrument_proxy: string
  hedge_ratio: string
  implementation_idea?: string
  entry_trigger?: string
  stop_loss_trigger?: string
  target_gain_or_rebalance?: string
  supporting_data: string[]
  source_refs?: string[]
  observable_refs?: string[]
  warnings: WarningDTO[]
}

export interface RiskScenarioDTO {
  tail_risk: string
  probability: string
  impact: string
  trigger_to_activate: string
  hedge_instruments: string[]
  unwind_or_cover_condition: string
  early_warning_indicators?: string[]
  mitigation_strategy?: string
  cost_or_tradeoff?: string
  hedge_size?: string
  hedge_purpose?: string
  supporting_data: string[]
  warnings: WarningDTO[]
}

export interface RegimeEvidenceDTO {
  dimension: string
  signal: string
  evidence: string
  conflict: string
  confidence: string
  source_refs?: string[]
  observable_refs?: string[]
}

export interface MacroIndicatorDTO {
  indicator_id: string
  series_key: string
  label: string
  value: number | null
  display_value: string
  unit: string
  observed_at: string
  provider: string
  source_file: string
  is_valid: boolean
  stale_reason: string
  chart_available: boolean
  region?: string
  source_type?: 'provider' | 'deterministic'
  status?: string
}

export interface MacroReferenceDTO {
  reference_id: string
  kind: 'news' | 'youtube'
  title: string
  url: string
  publisher: string
  published_at: string
  age_hours: number | null
  summary: string
  thumbnail_url: string
  is_stale: boolean
  related_observable_ids: string[]
}

export interface MacroSeriesPointDTO {
  observed_at: string
  value: number
}

export interface MacroIndicatorSeriesDTO {
  indicator_id: string
  series_key: string
  label: string
  unit: string
  range: '1m' | '3m' | '1y'
  points: MacroSeriesPointDTO[]
}

export interface MacroDashboardDTO {
  evaluated_at: string
  overall_regime: string
  time_horizon: string
  conviction_level?: string
  conviction_rationale?: string
  quant_narrative_alignment?: string
  divergence_note?: string
  focus_themes?: string[]
  key_assumptions: string[]
  regime_probabilities: Record<string, number | string>
  regime_evidence: RegimeEvidenceDTO[]
  asset_allocation?: AssetAllocationDTO[]
  pair_trades?: PairTradeDTO[]
  risk_scenarios?: RiskScenarioDTO[]
  source_files?: string[]
  generated_by?: string
  dashboard_indicators?: MacroIndicatorDTO[]
  report_references?: MacroReferenceDTO[]
  thailand_market_stance?: {
    investor_flow?: {
      foreign_net_mb?: number
      institution_net_mb?: number
      prop_net_mb?: number
      retail_net_mb?: number
    }
    market_breadth?: {
      advance_decline_ratio?: number
      sentiment?: 'bullish' | 'bearish' | 'neutral'
    }
    valuation?: {
      pe_ratio?: number
      pbv_ratio?: number
      dividend_yield?: number
    }
    physical_gold?: {
      bar_sell_thb?: number
      unit?: string
    }
    policy_spread_bps?: number | null
    rationale?: string | null
    observable_refs?: string[]
  } | null
  warnings: WarningDTO[]
  run_id?: string
  job_id?: string
  snapshot_id?: string
  strategy_report_id?: string
  sector_snapshot_id?: string
  sector_analysis?: SectorAnalysisDTO | null
  run_started_at?: string
  regional_assessments?: Record<string, {
    growth_score?: number | null
    inflation_score?: number | null
    monetary_score?: number | null
    economic_state?: string
    state?: string
    growth?: number | null
    inflation?: number | null
    monetary?: number | null
    fiscal_health?: {
      debt_to_gdp_pct?: number | null
      public_debt_million_thb?: number | null
      statutory_limit_pct?: number
      status?: string
    }
    confidence?: number
    coverage?: number
    data_gaps?: string[]
    market_stance?: any
  }>
  observable_registry?: Record<string, any>
  evaluated_sources?: string[]
}

export interface SectorAnalysisDTO {
  analysis_status?: 'available' | 'limited' | 'unavailable'
  unavailable_reason?: string | null
  snapshot_id: string | null
  as_of_date: string | null
  summary_th: string
  fact_claims: Array<{
    ticker: string
    claim_kind: string
    metric_ref: string
    event_ref: string | null
    macro_observable_refs: string[]
    interpretation_th: string
  }>
  resolved_metrics: Array<{
    ticker: string
    metric_ref: string
    numeric_value: number | null
    categorical_value: string | null
    unit: string
    horizon: string
    metric_as_of: string
    snapshot_id: string
    input_refs: string[]
  }>
  watch_conditions: Array<{
    metric_ref: string
    operator: string
    future_threshold: number
    unit: string
    horizon: string
    reason: string
  }>
  validation_warnings: string[]
}

export interface NewsCandidate {
  title: string
  link: string
  source: string
  age_hours: number
  freshness_reason: string
  is_stale: boolean
  is_fetched: boolean
}

export interface YoutubeCandidate {
  channel: string
  title: string
  link: string
  video_id: string | null
  published: string
  is_fetched: boolean
}

export interface NewsYoutubeApprovalPayload {
  type: 'news_youtube_approval'
  news_candidates: NewsCandidate[]
  youtube_candidates: YoutubeCandidate[]
}

export interface NewsFunnelCandidate {
  event_id: string
  canonical_title: string
  comprehensive_summary: string
  macro_impact_score: number
  asset_impact_score: number
  extracted_tickers: string[]
  extracted_themes: string[]
  primary_tags: string[]
  sources: string[]
  links?: string[]
  /** "llm" | "mock" | "heuristic_fallback" — เมื่อเป็น heuristic_fallback คะแนนไม่ได้มาจาก LLM จริง */
  triage_source?: string
  triage_fallback_reason?: string
}

export type NewsFunnelPendingItem = NewsFunnelCandidate

export interface NewsFunnelFilteredItem extends NewsFunnelCandidate {
  status: string
  triage_reasoning?: string
  error_msg?: string
  ingested_at?: string
}

export interface NewsFunnelApprovalPayload {
  type: 'news_funnel_approval'
  candidates: NewsFunnelCandidate[]
}

export interface YoutubePitchItemDTO {
  pitch_id: string
  working_titles: string[]
  target_audience: string
  core_thesis?: string
  core_hook?: string
  primary_anchor_event_id?: string
  primary_anchor_title?: string
  parking_lot_ideas?: string[]
  key_questions_to_answer: string[]
  research_hypotheses: string[]
  source_event_ids: string[]
  source_links: string[]
  source_titles: string[]
  recommended_format: string
  estimated_impact: string
  presentation_style?: string
  investigation_mode?: 'stock' | 'macro' | 'mixed'
  counter_intuitive_lead?: string
  analogy_generator?: string
  thumbnail_concept?: string
  audience_takeaway?: string
  source_readiness?: 'ready' | 'needs_refresh' | 'blocked' | 'unknown'
  source_readiness_issues?: string[]
  unverified_draft_issue_codes?: string[]
  unverified_draft_eligible?: boolean
  unverified_draft_eligibility_token?: string
}


export interface SourceOverrideAck {
  acknowledged: true
  policy_version: 'unverified-draft-v1'
  eligibility_token: string
  reason?: string
}

export interface UnverifiedDraftSelection {
  pitch_id: string
  ack: SourceOverrideAck
}


export interface YoutubePitchApprovalPayload {
  type: 'youtube_pitch_approval'
  pitches: YoutubePitchItemDTO[]
  instruction?: string
  approval_revision?: number
  source_refresh_attempts?: number
}

export type ApprovalPayload = NewsYoutubeApprovalPayload | NewsFunnelApprovalPayload | YoutubePitchApprovalPayload

export interface JobStatusDTO {
  job_id: string
  status: 'queued' | 'running' | 'done' | 'done_with_warnings' | 'done_with_errors' | 'error' | 'awaiting_approval'
  card_id: string | null
  error_message: string | null
  current_node: string | null
  interrupt_payload: ApprovalPayload | null
  log_count: number
  created_at: number
  updated_at: number
}

export interface SpecialistOutputDTO {
  node_name: string
  label: string
  content: string
  seq: number
  created_at: number
}

export interface JobOutputsDTO {
  job_id: string
  status: JobStatusDTO['status']
  executive_summary: string | null
  executive_summary_created_at: number | null
  specialists: SpecialistOutputDTO[]
  last_seq: number
  error_message: string | null
}

export interface ActiveAgentStatusDTO {
  running: boolean
  flow: string | null
  node: string | null
  job_id: string | null
}

export interface KanbanCardDTO {
  card_id: string
  title: string
  prompt: string | null
  column_name: string
  job_id: string | null
  flow: string
  scope: string
  display_seq: number | null
  discord_notify: boolean
  is_verified: boolean
  created_at: number
  updated_at: number
}

// ---------------------------------------------------------
// Actual Portfolio Hub DTOs (Phase 1 & Phase 2)
// ---------------------------------------------------------

export interface ActualHoldingDTO {
  symbol: string
  asset_type: string
  units: number
  status?: string
  archived_at?: string | null
  bucket_id: string | null
  avg_cost_usd: number | null
  avg_cost_thb: number | null
  current_price_usd: number | null
  current_price_thb: number | null
  fx_rate?: number | null
  market_value_thb: number
  unrealized_pnl_percent: number | null
  unrealized_pnl_value: number | null
  market_cap_tier: string | null
  yield_on_cost: number | null
  company_name: string | null
  pe_ratio: number | null
  eps: number | null
  payout_ratio: number | null
  market_cap_value: number | null
  dividend_per_share: number | null
  dividend_yield: number | null
  accumulated_dividend_thb: number | null
  accumulated_dividend_native?: number | null
  upcoming_dividend_thb?: number | null
  upcoming_dividend_native?: number | null
  dividend_rounds?: DividendRoundDTO[]
  dividend_source?: 'synced' | 'manual' | null
  fundamentals_updated_at: number | null
  pvd_valuation?: PvdHoldingValuationDTO | null
}

export interface PvdHoldingValuationDTO {
  payload_version: number
  fund_code: string
  sub_fund_code: string
  units: string
  statement_nav: string
  statement_market_value: string
  reconciliation_adjustment: string
  cost_basis: string
  as_of_date: string
  statement_fingerprint: string
  archive_id: string
}

export interface PvdStatementMetadataDTO {
  payload_version: number
  fund_code: string
  masked_member_id: string
  as_of_date: string
  period_start: string | null
  period_end: string | null
  total_contribution: string
  total_nav: string
  member_principal: string
  member_benefit: string
  employer_principal: string
  employer_benefit: string
  employer_vesting_percent: string
  member_vested_amount: string
  employer_vested_amount: string
  source_pdf_sha256: string
  semantic_fingerprint: string
  archive_id: string
  reconciliation_adjustment: string
}

export interface ActualSummaryDTO {
  total_value_thb: number
  total_cost_basis_thb: number
  total_unrealized_profit: number
  total_realized_profit_ytd?: number
  passive_income_ytd: number
  total_accumulated_dividend?: number
}

export interface AllocationTargetDTO {
  bucket_id: string
  name: string
  target_percent: number
  color: string | null
}

export const DEFAULT_ALLOCATION_TARGETS: AllocationTargetDTO[] = [
  { bucket_id: 'core_equities', name: 'Core Equities', target_percent: 60, color: '#3B82F6' },
  { bucket_id: 'defensive', name: 'Defensive Assets', target_percent: 20, color: '#A855F7' },
  { bucket_id: 'cash', name: '💰 Cash & Equivalents', target_percent: 20, color: '#06B6D4' },
]

export interface ActualPortfolioStateDTO {
  name?: string | null
  last_updated: string | null
  fx_rates: Record<string, number>
  summary: ActualSummaryDTO
  allocation_targets: AllocationTargetDTO[]
  holdings: ActualHoldingDTO[]
  pvd_statements?: PvdStatementMetadataDTO[]
  pvd_import_epoch?: string | null
  price_refresh_info: Record<string, string> | null
  allocation_policy?: {
    draft_id?: string
    allocation_basis?: string
    purpose_buckets?: Array<{
      bucket_id: string
      name: string
      role?: string
      target_percent: string | number
      target_amount?: string | number | null
      minimum_amount?: string | number | null
      need_by_date?: string | null
      cash_inflow_routing_pct?: string | null
      inflow_route_mode?: string
      inflow_priority?: number
      when_full_redirect_bucket_id?: string | null
      action_rules?: string[]
      eligible_assets?: string[]
      color?: string
    }>
    mapping_weights?: Array<{
      axis_allocation_id: string
      bucket_id: string
      portfolio_weight_percent: string | number
    }>
    applied_at_iso?: string
  } | null
  allocation_plan_origin?: {
    applied_draft_id?: string
    command_id?: string
    essence_ref?: string
    axis_ref?: string
    context_ref?: string
    allocation_basis?: string
    applied_at_iso?: string
    targets_hash?: string
    is_modified_since_apply?: boolean
  } | null
}

export interface BucketAllocationSummaryDTO {
  bucket_id: string
  name: string
  target_percent: number
  actual_value_thb: number
  actual_percent: number
  variance: number
  color: string | null
}

export interface BucketAllocationResponseDTO {
  warning: string | null
  summaries: BucketAllocationSummaryDTO[]
}

export interface ActualWatchlistItemDTO {
  symbol: string
  asset_type: string
  target_price: number | null
  added_date: string
  notes: string | null
  suggested_bucket_id?: string | null
  applied_bucket_name?: string | null
  target_currency?: string | null
  evaluation_ref?: string | null
  source_refs?: string[] | null
  target_price_basis?: string | null
}

export interface ActualWatchlistStateDTO {
  last_updated: string | null
  items: ActualWatchlistItemDTO[]
}

export interface PortfolioMetaDTO {
  id: string
  name: string
  is_default?: boolean
  created_at?: string | null
}


export interface GoalAdjustmentLeversDTO {
  save_more_monthly_thb: number
  total_monthly_savings_needed_thb: number
  extend_timeline_months: number
  total_months_needed: number
  downsized_target_thb: number
  required_risk_cagr_pct: number | null
}


export interface ActualGoalItemDTO {
  name: string
  target_amount_thb: number
  goal_type: 'nav_target' | 'cash_target' | 'passive_income_ytd' | 'bucket_target'
  current_amount_thb: number
  progress_pct: number
  deadline: string | null
  deadline_days_left: number | null
  notes: string | null
  portfolio_id?: string | null
  bucket_id?: string | null
  priority_tier?: 'needs' | 'wants' | 'wishes' | null
  flexibility?: 'none' | 'timeline_flexible' | 'amount_flexible' | 'fully_flexible' | null
  monthly_savings_thb?: number | null
  required_return_pct?: number | null
  tension_level?: 'BALANCED' | 'MODERATE_TENSION' | 'CRITICAL_TENSION' | 'INSUFFICIENT_DATA' | null
  adjustment_levers?: GoalAdjustmentLeversDTO | null
}


export interface ActualGoalsResponseDTO {
  n_goals: number
  goals: ActualGoalItemDTO[]
  generated_at: string | null
}

export interface PerformanceSnapshotDTO {
  Date: string
  Total_NAV: number
  Total_Cost: number
  Unrealized_PnL: number
  Cash_Balance: number
  realized_pnl_ytd?: number | null
  passive_income_ytd?: number | null
  Asset_Class_Values_THB?: Record<string, number> | null
  coverage_warning?: string | null
}

export interface JournalEntryDTO {
  timestamp: string
  content: string
}

export interface UpsertAllocationTargetsPayload {
  targets: AllocationTargetDTO[]
}

export interface AssignBucketPayload {
  bucket_id?: string | null
}

export interface BatchAssignBucketPayload {
  symbols: string[]
  bucket_id?: string | null
}

export interface HoldingBucketAssignmentItemDTO {
  symbol: string
  bucket_id?: string | null
}

export interface BatchReassignHoldingsRequestDTO {
  assignments: HoldingBucketAssignmentItemDTO[]
}

export interface HoldingBucketRecommendationItemDTO {
  symbol: string
  asset_type: string
  current_bucket_id?: string | null
  recommended_bucket_id: string
  is_changed: boolean
  reason: string
}

export interface HoldingsBucketRecommendationResponseDTO {
  recommendations: HoldingBucketRecommendationItemDTO[]
  summary: string
  total_unassigned: number
  total_mismatched: number
}

export interface BatchRemoveHoldingsPayload {
  symbols: string[]
}

export interface TradePayload {
  symbol: string
  asset_type: string
  action: 'buy' | 'sell'
  units: number
  price: number
  currency?: 'THB' | 'USD'
  exchange_rate?: number | null
  date?: string | null
  notes?: string
  bucket_id?: string | null
}

export interface CashFlowPayload {
  amount: number
  action: 'deposit' | 'withdraw'
  currency?: 'THB' | 'USD'
  exchange_rate?: number | null
  date?: string | null
  notes?: string
}

export interface IncomePayload {
  income_type: 'Dividend' | 'Interest' | 'Rental' | 'Other'
  amount_thb: number
  source_symbol?: string | null
  date?: string | null
  notes?: string
}

export interface EditHoldingPayload {
  units?: number | null
  avg_cost?: number | null
  accumulated_dividend_thb?: number | null
  asset_type?: string | null
  reason?: string
  bucket_id?: string | null
}

export interface UpsertWatchlistItemPayload {
  asset_type: string
  target_price?: number | null
  notes?: string
  suggested_bucket_id?: string | null
  applied_bucket_name?: string | null
  target_currency?: string | null
  evaluation_ref?: string | null
  source_refs?: string[] | null
  target_price_basis?: string | null
}

export interface UpsertGoalPayload {
  goal_type: 'nav_target' | 'cash_target' | 'passive_income_ytd' | 'bucket_target'
  target_amount_thb: number
  deadline?: string | null
  years_from_now?: number | null
  notes?: string | null
  portfolio_id?: string | null
  bucket_id?: string | null
  new_name?: string | null
  priority_tier?: 'needs' | 'wants' | 'wishes' | null
  flexibility?: 'none' | 'timeline_flexible' | 'amount_flexible' | 'fully_flexible' | null
  monthly_savings_thb?: number | null
}


export interface AppendJournalPayload {
  entry: string
}

export interface NotebookLMAvailableSourceDTO {
  file_path: string
  title: string
  date_part: string | null
  is_verified: boolean
}

export interface NotebookLMGenerateResponse {
  job_id: string
  status: string
}

export interface NotebookLMStatusDTO {
  job_id: string
  status: string
  audio_path: string | null
  notebook_id: string | null
  error: string | null
  recovery_status?: string | null
}

export interface EquitySummaryDTO {
  ticker: string
  market: 'TH' | 'US'
  company_name: string | null
  analysis_date: string
  evaluated_at: string
  market_sentiment: 'bullish' | 'neutral' | 'bearish'
  composite_score: number | null
  data_quality_flags: string[]
  source_file: string
  sidecar_file: string
}

export interface EquitySentimentContextDTO {
  evaluated_at: string
  market_sentiment: 'bullish' | 'neutral' | 'bearish'
  key_themes: string[]
  tail_risks: string[]
  sources_summary: string
  report_references: any[]
}

export interface DCFScenarioDTO {
  target_price?: number | null
  upside_pct?: number | null
  margin_of_safety_pct?: number | null
  implied_exit_ebitda_multiple?: number | null
  terminal_ebitda?: number | null
}

export interface ValuationSanityAuditDTO {
  terminal_value_pct_of_ev?: number | null
  implied_exit_fcf_multiple?: number | null
  implied_exit_ebitda_multiple?: number | null
  benchmark_sector_multiple?: number | null
  model_reliability_tier?: 'high' | 'moderate' | 'terminal_dependent' | 'unavailable'
  reliability_score?: number | null
  is_terminal_dependent?: boolean
  is_growth_capped?: boolean
  sanity_warnings?: string[]
}

export interface SensitivityCellDTO {
  wacc_pct: number
  terminal_growth_pct: number
  target_price: number
  upside_pct: number
  verdict: 'undervalued' | 'fairly_valued' | 'overvalued' | 'unavailable'
  is_base?: boolean
}

export interface SensitivityMatrixDTO {
  wacc_values: number[]
  terminal_growth_values: number[]
  grid: SensitivityCellDTO[][]
  current_price: number
}

export interface SensitivitySummaryDTO {
  min_target_price: number
  max_target_price: number
  min_upside_pct: number
  max_upside_pct: number
  wacc_spread_pct: number[]
  growth_spread_pct: number[]
  growth_delta_per_1pct?: number | null
  margin_delta_per_1pct?: number | null
  primary_operating_driver?: string | null
  macro_yield_shock_50bps_delta?: number | null
  macro_yield_shock_100bps_delta?: number | null
}

export interface OperatingSensitivityCellDTO {
  revenue_growth_pct: number
  ebit_margin_pct: number
  target_price: number
  upside_pct: number
  verdict: 'undervalued' | 'fairly_valued' | 'overvalued' | 'unavailable'
  is_base?: boolean
}

export interface OperatingSensitivityMatrixDTO {
  growth_values: number[]
  margin_values: number[]
  grid: OperatingSensitivityCellDTO[][]
  current_price: number
}

export interface MacroSensitivityCellDTO {
  beta: number
  risk_free_rate_pct: number
  effective_wacc_pct: number
  target_price: number
  upside_pct: number
  verdict: 'undervalued' | 'fairly_valued' | 'overvalued' | 'unavailable'
  is_base?: boolean
}

export interface MacroSensitivityMatrixDTO {
  beta_values: number[]
  yield_values: number[]
  grid: MacroSensitivityCellDTO[][]
  current_price: number
}

export interface SensitivityMatrixSetDTO {
  valuation?: SensitivityMatrixDTO | null
  operating?: OperatingSensitivityMatrixDTO | null
  macro?: MacroSensitivityMatrixDTO | null
}

export interface DCFResultDTO {
  wacc_pct?: number | null
  cost_of_equity_pct?: number | null
  cost_of_debt_pct?: number | null
  risk_free_rate_pct?: number | null
  erp_pct?: number | null
  observable_refs?: string[]
  scenarios?: Record<'bull' | 'base' | 'bear', DCFScenarioDTO>
  valuation_verdict?: 'undervalued' | 'fairly_valued' | 'overvalued' | 'unavailable'
  is_actionable?: boolean
  actionability_reason?: string | null
  invalidation_reasons?: string[]
  valuation_sanity_audit?: ValuationSanityAuditDTO | null
  sensitivity_matrix?: SensitivityMatrixDTO | any[] | null
  sensitivity_summary?: SensitivitySummaryDTO | null
  discounting_convention?: 'mid_year' | 'end_year'
  synthetic_credit_rating?: string | null
  interest_coverage_ratio?: number | null
  credit_spread_bps?: number | null
  benchmark_cost_of_debt_pct?: number | null
  terminal_ebitda?: number | null
  implied_exit_ebitda_multiple?: number | null
  implied_exit_fcf_multiple?: number | null
  implied_perpetual_growth_from_exit_pct?: number | null
}

export type PriceSource =
  | 'ohlcv_close'
  | 'verified_live_quote'
  | 'intraday_snapshot'
  | 'stale_eod'
  | 'unavailable'

export interface MarginMetricItemDTO {
  value_pct?: number | null
  period_end?: string | null
  period_type?: 'TTM' | 'quarterly' | 'annual' | 'guidance_forward' | null
  definition?: string
  source_provenance?: string | null
}

export interface AtomicMarketSnapshotDTO {
  analysis_price?: number | null
  analysis_price_as_of?: string | null
  price_source: PriceSource
  latest_ohlcv_close?: number | null
  latest_ohlcv_date?: string | null
  shares_outstanding?: number | null
  market_cap?: number | null
  raw_analysis_price?: string | null
  raw_analysis_price_str?: string | null
  market_cap_str?: string | null
  market_cap_cents?: number | null
  is_provisional?: boolean
  volume_confirmation?: 'confirmed' | 'provisional' | 'unavailable'
  price_sync_status?: 'synced' | 'quote_ohlcv_mismatch' | 'stale' | 'unavailable'
  freshness_status?: 'fresh' | 'stale' | 'session_synced' | 'out_of_session' | 'unavailable'
  market_session_status?: 'pre_market' | 'open' | 'after_hours' | 'closed' | 'unavailable'
  data_freshness_status?: 'fresh' | 'stale' | 'stale_one_session' | 'stale_multiple_sessions' | 'unavailable' | 'unknown'
  expected_latest_session_date?: string | null
  actual_latest_session_date?: string | null
  missing_trading_sessions?: number
  retrieved_at: string
}

export interface HistoricalFinancialItemDTO {
  fiscal_year: string
  revenue: number
  revenue_growth_pct?: number | null
  ebit: number
  ebit_margin_pct: number
  depreciation?: number | null
  capex?: number | null
  fcf?: number | null
  gross_profit?: number | null
  gross_margin_pct?: number | null
  research_and_development?: number | null
  rd_pct_of_rev?: number | null
  selling_general_admin?: number | null
  sga_pct_of_rev?: number | null
  sales_and_marketing?: number | null
  general_and_administrative?: number | null
  capex_to_depreciation_ratio?: number | null
  reinvestment_lifecycle_phase?: 'aggressive_expansion' | 'sustainable_maintenance' | 'underinvestment_warning' | 'unavailable' | null
}

export interface ExplicitFCFProjectionDTO {
  year_index: number
  projected_revenue: number
  projected_revenue_growth_pct: number
  projected_gross_profit?: number | null
  projected_gross_margin_pct?: number | null
  projected_rd?: number | null
  projected_rd_pct_of_rev?: number | null
  projected_sga?: number | null
  projected_sga_pct_of_rev?: number | null
  projected_ebit: number
  projected_ebit_margin_pct: number
  projected_nopat: number
  projected_depreciation: number
  projected_capex: number
  projected_net_capex: number
  projected_delta_nwc: number
  projected_reinvestment_currency: number
  projected_fcf: number
  discount_factor: number
  pv_fcf: number
  capex_to_depreciation_ratio?: number | null
  reinvestment_lifecycle_phase?: 'aggressive_expansion' | 'sustainable_maintenance' | 'underinvestment_warning' | 'unavailable' | null
}

export interface ReverseDCFResultDTO {
  shares_outstanding?: number | null
  current_stock_price?: number | null
  terminal_growth_pct?: number | null
  beta?: number | null
  historical_financials_3y?: HistoricalFinancialItemDTO[]
  explicit_forecast_5y?: ExplicitFCFProjectionDTO[]
  explicit_forecast_5y_scenarios?: Record<'bear' | 'base' | 'bull', ExplicitFCFProjectionDTO[]> | null
  sum_pv_5y_fcf?: number | null
  terminal_value_undiscounted?: number | null
  terminal_value_pv?: number | null
  enterprise_value?: number | null
  net_cash_debt?: number | null
  equity_value?: number | null
  intrinsic_value_today?: number | null
  target_price_12m?: number | null
  upside_12m_pct?: number | null
  market_implied_growth_pct?: number | null
  market_implied_margin_pct?: number | null
  solver_status?: 'converged' | 'bounded_extreme' | 'no_solution' | 'not_applicable'
  fixed_parameters?: Record<string, any>
  valuation_horizon_months?: number
  status?: string
  is_eligible?: boolean
  exclusion_reason?: string | null
  valuation_verdict?: 'overvalued' | 'fairly_valued' | 'undervalued' | 'unavailable'
  is_actionable?: boolean
  actionability_reason?: string | null
  raw_wacc_pct?: number | null
  effective_wacc_pct?: number | null
  wacc_adjustment_reason?: string | null
  reported_ebit_margin_pct?: number | null
  ebit_margin_fiscal_period?: string | null
  ebit_margin_period_type?: 'annual' | 'quarterly' | 'ttm' | 'unknown' | null
  ebit_margin_source_tier?: 'filing_authoritative' | 'primary_best_effort' | 'fallback' | 'unknown' | null
  target_price_exit_multiple_12m?: number | null
  exit_multiple_used?: number | null
  intrinsic_value_exit_multiple?: number | null
  consensus_target_price?: number | null
  consensus_target_high?: number | null
  consensus_target_low?: number | null
  analyst_count?: number | null
  base_revenue?: number | null
  base_revenue_period_type?: 'annual' | 'quarterly' | 'ttm' | 'unknown' | null
  alternative_valuation_model?: 'justified_pbv' | 'ddm_yield_spread' | string | null
  alternative_intrinsic_value?: number | null
  alternative_target_price_12m?: number | null
  alternative_upside_12m_pct?: number | null
  market_implied_roe_pct?: number | null
  valuation_sanity_audit?: ValuationSanityAuditDTO | null
  sensitivity_matrix?: SensitivityMatrixDTO | any[] | null
  sensitivity_matrices?: SensitivityMatrixSetDTO | null
  sensitivity_summary?: SensitivitySummaryDTO | null
  discounting_convention?: 'mid_year' | 'end_year'
  synthetic_credit_rating?: string | null
  interest_coverage_ratio?: number | null
  credit_spread_bps?: number | null
  benchmark_cost_of_debt_pct?: number | null
  terminal_ebitda?: number | null
  implied_exit_ebitda_multiple?: number | null
  implied_exit_fcf_multiple?: number | null
  implied_perpetual_growth_from_exit_pct?: number | null
  exit_multiple_tv_undiscounted?: number | null
  company_archetype?: 'high_growth_tech' | 'mature_cash_cow' | 'cyclical' | 'turnaround' | 'general' | null
  normalized_mid_cycle_ebit_margin_pct?: number | null
  is_mid_cycle_normalized?: boolean
  archetype_rationale?: string | null
  margin_of_safety_pct?: number | null
  mos_entry_price_today?: number | null
  mos_target_price_12m?: number | null
  current_margin_of_safety_pct?: number | null
  is_in_buy_zone?: boolean
  required_pullback_pct?: number | null
}

export interface TacticalSetupDTO {
  price_stage: string
  current_price?: number | null
  sma_50?: number | null
  sma_200?: number | null
  atr_14?: number | null
  key_support_level?: number | null
  key_resistance_level?: number | null
  buy_zone_min?: number | null
  buy_zone_max?: number | null
  invalidation_stop_loss?: number | null
  tactical_target_price?: number | null
  tactical_risk_reward_ratio?: number | null
  current_rr_ratio?: number | null
  buy_zone_rr_min?: number | null
  buy_zone_rr_max?: number | null
  is_in_buy_zone?: boolean | null
  pullback_entry_status?: 'below_stop' | 'in_buy_zone' | 'between_zone_and_target' | 'at_or_above_target' | 'unavailable'
  breakout_trigger_price?: number | null
  breakout_target_price?: number | null
  breakout_stop_loss?: number | null
  breakout_planned_rr?: number | null
  breakout_current_rr?: number | null
  max_breakout_chase_price?: number | null
  breakout_entry_status?: 'pre_trigger' | 'eligible' | 'chased' | 'expired'
  breakout_entry_eligible?: boolean
  breakout_volume_ratio?: number | null
  breakout_volume_baseline?: number | null
  breakout_volume_confirmed?: boolean | null
  horizon_timeframe?: string
  status?: string
}

export interface DeterministicScorecardDTO {
  core_conviction_score: number
  business_conviction_score?: number | null
  investment_conviction_score?: number | null
  execution_readiness_score: number
  action_stance: 'ACCUMULATE_NOW' | 'ACCUMULATE_ON_DIP' | 'BREAKOUT_BUY' | 'HOLD_WAIT' | 'REDUCE' | 'INSUFFICIENT_DATA'
  action_stance_reason?: string | null
  stance_mode?: 'active' | 'conditional' | 'wait' | 'reduce'
  setup_readiness_score?: number | null
  execution_score_breakdown?: {
    stage?: number
    setup?: number | null
    insider?: number
    setup_reason?: string | null
  } | null
  fundamental_quality_score?: number | null
  guidance_expectation_score?: number | null
  analyst_expectations_score?: number | null
  management_guidance_score?: number | null
  valuation_margin_score?: number | null
  coverage_pct: number
  usable_coverage_pct?: number
  verified_coverage_pct?: number
  applicable_pillars_count?: number
  methodology_version?: string
  reweighting_metadata?: {
    excluded_pillars?: string[]
    weights_used?: Record<string, number>
    reason?: string
  } | null
  data_quality_flags?: string[]
}

export interface SmartMoneyFlagsDTO {
  insider_signal: 'buying' | 'selling' | 'neutral'
  insider_buy_count_90d: number
  insider_sell_count_90d: number
  institutional_ownership_pct: number | null
  insider_ownership_pct: number | null
  short_interest_pct: number | null
  short_squeeze_risk: boolean
  overall_smart_money_flag: 'bullish_signal' | 'bearish_signal' | 'neutral'
}

export interface QuantSignalsDTO {
  ticker: string
  market: 'TH' | 'US'
  company_name: string | null
  value_score: number | null
  quality_score: number | null
  momentum_score: number | null
  beta: number | null
  volatility_pct: number | null
  mdd_pct: number | null
  upside_pct: number | null
  downside_pct: number | null
  raw_analysis_price?: number | null
  raw_analysis_price_str?: string | null
  revenue_growth_yoy_pct: number | null
  net_income_growth_yoy_pct: number | null
  growth_score: number | null
  dividend_yield_pct: number | null
  payout_ratio_pct: number | null
  dividend_score: number | null
  de_ratio_pct: number | null
  current_ratio: number | null
  solvency_score: number | null
  fcf_yield_pct: number | null
  fcf_margin_pct: number | null
  fcf_cagr_3y: number | null
  interest_coverage: number | null
  net_debt_ebitda: number | null
  roic_pct: number | null
  ocf_to_net_income: number | null
  fcf_quality_score: number | null
  debt_quality_score: number | null
  adtv_local_currency: number | null
  composite_score: number | null
  peer_sector: string | null
  peer_count: number | null
  pe_vs_peer_avg_pct: number | null
  peer_relative_score: number | null
  price_percentile_5y: number | null
  price_zscore_5y: number | null
  eps_revision_net_30d: number | null
  eps_estimate_change_30d_pct: number | null
  earnings_momentum_score: number | null
  dcf_result?: DCFResultDTO | null
  smart_money_flags?: SmartMoneyFlagsDTO | null
  evaluated_at: string
  data_quality_flags: string[]
  piotroski_breakdown?: any
  reverse_dcf_result?: ReverseDCFResultDTO | null
  tactical_setup?: TacticalSetupDTO | null
  insider_conviction?: any
  deterministic_scorecard?: DeterministicScorecardDTO | null
  thesis_falsifiers?: any
  dcf_discrepancy_warning?: string | null
  atomic_market_snapshot?: AtomicMarketSnapshotDTO | null
  metric_basis?: Record<string, string>
  gaap_operating_margin?: MarginMetricItemDTO | null
  non_gaap_operating_margin?: MarginMetricItemDTO | null
  historical_gaap_operating_margin?: MarginMetricItemDTO | null
  provider_ebit_margin?: MarginMetricItemDTO | null
  valuation_margin_source_used?: string
  sector_kpis?: any | null
  net_buyback_yield_pct?: number | null
  net_shareholder_yield_pct?: number | null
}

export interface EquityDetailDTO extends EquitySummaryDTO {
  quant_signals: QuantSignalsDTO
  sentiment_context: EquitySentimentContextDTO
  narrative_analysis: string
  base_case_summary: string
  narrative_status?: 'available' | 'unavailable' | 'pending'
  error_code?: string | null
  generated_by: string
  portfolio_context?: {
    is_held: boolean
    portfolio_id?: string
    units?: number
    avg_cost?: number | null
    unrealized_pnl_pct?: number | null
    current_weight_pct?: number | null
    bucket_id?: string | null
    position_action_hint?: string | null
  } | null
  thesis_delta?: {
    previous_analysis_date: string
    price_change_pct?: number | null
    conviction_score_delta?: number | null
    previous_action_stance?: string | null
    current_action_stance?: string | null
    target_price_change_pct?: number | null
    f_score_delta?: number | null
    breached_falsifiers?: string[]
  } | null
}

export interface EquityYouTubeMentionDTO {
  video_id: string
  title?: string
  thai_title?: string
  channel: string
  key_takeaways: string[]
  alignment_with_news?: 'confirmed' | 'contrarian' | 'independent' | string | null
  alignment_note?: string | null
  link: string
  published_at?: string | null
}

export interface EquityNewsItemDTO {
  title: string
  source: string
  link: string
  published_at?: string | null
  age_hours: number
  freshness_reason: string
  is_stale: boolean
  sources_count?: number
  summary?: string
  what_happened?: string
  why_it_matters?: string
  thesis_signal?: 'thesis_breaker' | 'buy_on_dip' | 'thesis_reinforcer' | 'noise'
  impact_type?: string
  macro_impact_score?: number
  asset_impact_score?: number
}

export interface EquityNewsDTO {
  ticker: string
  market: 'TH' | 'US'
  last_updated?: string | null
  news_date?: string | null
  items: EquityNewsItemDTO[]
  youtube_mentions?: EquityYouTubeMentionDTO[]
}

export interface EquityNoteItemDTO {
  title: string
  folder: string
  relative_path: string
  obsidian_uri: string
  snippet: string
  modified_at: string
  matched_by: string
}

export interface EquityNotesDTO {
  ticker: string
  total_count: number
  items: EquityNoteItemDTO[]
}

export interface EquityNoteContentDTO {
  title: string
  relative_path: string
  content: string
  modified_at?: string | null
}


export interface CalendarEventDTO {
  ticker: string
  company_name?: string | null
  event_type: 'earnings' | 'ex_dividend'
  event_date: string
  days_until: number
  bucket: 'holding' | 'watchlist'
  eps_estimate?: number | null
  eps_low?: number | null
  eps_high?: number | null
}

export interface PortfolioCalendarDTO {
  generated_at: string
  events: CalendarEventDTO[]
  tickers_fetched: number
  tickers_failed: string[]
}

export interface TransactionItemDTO {
  transaction_id: string
  timestamp: string
  symbol: string
  action: 'BUY' | 'SELL' | string
  units: number
  price: number
  currency: string
  fx_rate?: number | null
  cost_thb: number
  realized_pnl_thb?: number | null
  notes: string
  gross_amount?: string | null
  commission?: string | null
  vat?: string | null
  other_fees?: string | null
  net_amount?: string | null
  fee_currency?: string | null
  confirmation_no?: string | null
  settlement_date?: string | null
  source?: string | null
  fingerprint?: string | null
  cash_adjusted?: string | null
  related_transaction_id?: string | null
  asset_type?: string | null
  bucket_id?: string | null
  pvd_metadata_json?: string | null
  reconciliation_adjustment?: string | null
}

export interface PvdAnalyzeResponseDTO {
  scan_id: string | null
  can_commit: boolean
  status: string
  message: string
  statement: Record<string, unknown> | null
  holdings: Array<{
    fund_code: string
    name: string
    units: string
    statement_nav: string
    statement_market_value: string
    reconciliation_adjustment: string
    adjusted_valuation: string
    allocated_cost: string
    unrealized_pnl: string
  }>
  current_nav: string | null
  new_nav: string | null
  delta_nav: string | null
  vesting_percent: string
  vested_amount: string | null
  archive_id: string | null
}

export interface PvdCommitResponseDTO {
  status: string
  message: string
  archive_id: string | null
  state: ActualPortfolioStateDTO
}

export interface DimeEmailMetadataDTO {
  message_id: string
  attachment_id: string
  subject: string
  sender: string
  received_at: string
  filename: string
  size_bytes: number
}

export interface DimeEmailListResponseDTO {
  emails: DimeEmailMetadataDTO[]
}

export interface DimeStagedItemFeeDTO {
  commission: string
  vat: string
  other_fees: string
  fee_currency: string
}

export interface DimeStagedItemDTO {
  item_id: string
  trade_date: string
  settlement_date?: string | null
  symbol: string
  action: string
  units: string
  price: string
  gross_amount: string
  fees: DimeStagedItemFeeDTO
  net_amount: string
  currency: string
  exchange_rate?: string | null
  confirmation_no: string
  order_id?: string | null
  source: string
  fingerprint: string
  line_index: number
  cash_adjusted: boolean
  asset_type?: string
}

export interface DimeScanResponseDTO {
  scan_id: string
  item_count: number
  items: DimeStagedItemDTO[]
}

export interface DimeCommitResponseDTO {
  ok: boolean
  imported_count: number
  state: ActualPortfolioStateDTO
}

export interface DimeBatchScanProgressEvent {
  current: number
  total: number
  percent: number
  items_found: number
  subject?: string
  message_id?: string
}

export interface DimeBatchScanWarningEvent {
  subject?: string
  message_id?: string
  attachment_id?: string
  filename?: string
  received_at?: string
  reason: string
  can_preview?: boolean
}

export interface DimePdfTextPageDTO {
  page_number: number
  text: string
}

export interface DimePdfTextResponseDTO {
  message_id: string
  attachment_id: string
  filename: string
  page_count: number
  pages: DimePdfTextPageDTO[]
}

export interface DimeBatchScanCompleteEvent {
  scan_id: string
  item_count: number
  items: DimeStagedItemDTO[]
  warnings: DimeBatchScanWarningEvent[]
  skipped_synced_count: number
  message?: string
}


export interface WealthXEmailMetadataDTO {
  message_id: string
  attachment_id: string
  subject: string
  sender: string
  received_at: string
  filename: string
  size_bytes: number
}

export interface WealthXEmailListResponseDTO {
  emails: WealthXEmailMetadataDTO[]
}

export interface WealthXStagedItemFeeDTO {
  commission: string
  vat: string
  other_fees: string
  fee_currency: string
}

export interface WealthXStagedItemDTO {
  item_id: string
  trade_date: string
  settlement_date?: string | null
  symbol: string
  action: string
  units: string
  price: string
  gross_amount: string
  fees: WealthXStagedItemFeeDTO
  net_amount: string
  currency: string
  exchange_rate?: string | null
  confirmation_no: string
  order_id?: string | null
  source: string
  fingerprint: string
  line_index: number
  cash_adjusted: boolean
  asset_type?: string
}

export interface WealthXScanResponseDTO {
  scan_id: string
  item_count: number
  items: WealthXStagedItemDTO[]
}

export interface WealthXCommitResponseDTO {
  ok: boolean
  imported_count: number
  state: ActualPortfolioStateDTO
}

export interface WealthXBatchScanProgressEvent {
  current: number
  total: number
  percent: number
  items_found: number
  subject?: string
  message_id?: string
}

export interface WealthXBatchScanWarningEvent {
  subject?: string
  message_id?: string
  attachment_id?: string
  filename?: string
  received_at?: string
  reason: string
  can_preview?: boolean
}

export interface WealthXPdfTextPageDTO {
  page_number: number
  text: string
}

export interface WealthXPdfTextResponseDTO {
  message_id: string
  attachment_id: string
  filename: string
  page_count: number
  pages: WealthXPdfTextPageDTO[]
}

export interface WealthXBatchScanCompleteEvent {
  scan_id: string
  item_count: number
  items: WealthXStagedItemDTO[]
  warnings: WealthXBatchScanWarningEvent[]
  skipped_synced_count: number
  message?: string
}

export interface SCBAMEmailMetadataDTO {
  message_id: string
  attachment_id: string
  subject: string
  sender: string
  received_at: string
  filename: string
  size_bytes: number
}

export interface SCBAMEmailListResponseDTO {
  emails: SCBAMEmailMetadataDTO[]
}

export interface SCBAMStagedItemFeeDTO {
  commission: string
  vat: string
  other_fees: string
  fee_currency: string
}

export interface SCBAMStagedItemDTO {
  item_id: string
  trade_date: string
  settlement_date?: string | null
  symbol: string
  action: string
  units: string
  price: string
  gross_amount: string
  fees: SCBAMStagedItemFeeDTO
  net_amount: string
  currency: string
  exchange_rate?: string | null
  confirmation_no: string
  order_id?: string | null
  source: string
  fingerprint: string
  line_index: number
  cash_adjusted: boolean
  asset_type?: string
}

export interface SCBAMScanResponseDTO {
  scan_id: string
  item_count: number
  items: SCBAMStagedItemDTO[]
}

export interface SCBAMCommitResponseDTO {
  ok: boolean
  imported_count: number
  state: ActualPortfolioStateDTO
}

export interface SCBAMBatchScanProgressEvent {
  current: number
  total: number
  percent: number
  items_found: number
  subject?: string
  message_id?: string
}

export interface SCBAMBatchScanWarningEvent {
  subject?: string
  message_id?: string
  attachment_id?: string
  filename?: string
  received_at?: string
  reason: string
  can_preview?: boolean
}

export interface SCBAMBatchScanCompleteEvent {
  scan_id: string
  item_count: number
  items: SCBAMStagedItemDTO[]
  warnings: SCBAMBatchScanWarningEvent[]
  skipped_synced_count: number
  message?: string
}

export interface ArrGoldEmailMetadataDTO {
  message_id: string
  attachment_id?: string | null
  subject: string
  sender: string
  received_at: string
  filename?: string | null
  size_bytes: number
}

export interface ArrGoldEmailListResponseDTO {
  emails: ArrGoldEmailMetadataDTO[]
}

export interface ArrGoldStagedItemFeeDTO {
  commission: string
  vat: string
  other_fees: string
  fee_currency: string
}

export interface ArrGoldStagedItemDTO {
  item_id: string
  trade_date: string
  settlement_date?: string | null
  symbol: string
  action: string
  units: string
  price: string
  gross_amount: string
  fees: ArrGoldStagedItemFeeDTO
  net_amount: string
  currency: string
  exchange_rate?: string | null
  confirmation_no: string
  order_id?: string | null
  source: string
  fingerprint: string
  line_index: number
  cash_adjusted: boolean
  asset_type?: string
}

export interface ArrGoldScanResponseDTO {
  scan_id: string
  item_count: number
  items: ArrGoldStagedItemDTO[]
}

export interface ArrGoldCommitResponseDTO {
  ok: boolean
  imported_count: number
  state: ActualPortfolioStateDTO
}



export interface TransactionSummaryDTO {
  total_buy_count: number
  total_sell_count: number
  total_buy_thb: number
  total_sell_thb: number
  total_realized_pnl_thb: number
}

export interface TransactionListResponseDTO {
  portfolio_id: string
  transactions: TransactionItemDTO[]
  summary: TransactionSummaryDTO
}

export interface UpdateTransactionNoteRequestDTO {
  notes: string
}

export interface EditTransactionPayload {
  timestamp?: string | null
  units?: number | null
  price?: number | null
  fx_rate?: number | null
  notes?: string | null
  adjust_cash?: boolean
}

export interface DeleteTransactionPayload {
  adjust_cash?: boolean
}

export interface FXRateResponseDTO {
  date: string
  currency_pair: string
  rate: number
  source: 'historical' | 'live' | 'fallback'
}

export interface DividendRoundDTO {
  symbol: string
  ex_date: string
  pay_date?: string | null
  dps: number
  currency: string
  units_held: number
  status?: 'received' | 'upcoming'
  gross_native?: number
  net_native?: number
  gross_thb: number
  tax_rate: number
  net_thb: number
  fx_rate: number
}

export interface SyncDividendsResponseDTO {
  synced_symbols: number
  total_rounds: number
  total_received_rounds?: number
  total_upcoming_rounds?: number
  total_dividend_thb: number
  total_upcoming_thb?: number
  skipped_manual: string[]
  details: Record<string, DividendRoundDTO[]>
}

export interface OHLCVCandleDTO {
  timestamp: number // Unix epoch in milliseconds
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export interface PivotLevelsDTO {
  pivot: number
  r1: number
  r2: number
  r3: number
  s1: number
  s2: number
  s3: number
  s4: number
}

export interface CorporateActionEventDTO {
  event_type: 'earnings' | 'ex_dividend' | 'split'
  timestamp: number
  date_str: string
  label: string
  color: 'green' | 'red' | 'blue' | 'purple'
  tooltip: string
  mapping_method: 'reported_date' | 'next_session' | 'period_enclosing' | 'unknown'
  eps_actual?: number | null
  eps_estimate?: number | null
  dividend_amount?: number | null
  split_numerator?: number | null
  split_denominator?: number | null
  split_formatted?: string | null
}

export interface CorporateActionsMetadataDTO {
  status: 'available' | 'partial' | 'unavailable'
  as_of?: string | null
  earnings_status: 'ok' | 'failed' | 'empty'
  earnings_as_of?: string | null
  dividends_status: 'ok' | 'failed' | 'empty'
  dividends_as_of?: string | null
  splits_status: 'ok' | 'failed' | 'empty'
  splits_as_of?: string | null
  missing_sources: string[]
  data_provenance: string
}

export interface IndicatorBurnInPolicyDTO {
  algorithm_version: string
  seed_method: string
  convergence_tolerance_pct: number
  required_burn_in_bars: number
  burn_in_bars_remaining: number
  first_reliable_timestamp?: number | null
  first_reliable_index?: number | null
}

export interface IndicatorWarmupDetailDTO {
  status: 'full' | 'partial' | 'unavailable'
  required_bars: number
  actual_warmup_bars: number
  burn_in_bars_remaining: number
  first_reliable_timestamp?: number | null
  first_reliable_index?: number | null
  burn_in_policy?: IndicatorBurnInPolicyDTO | null
}

export type ChartInterval = '15m' | '1h' | '1d' | '1wk' | '1mo'

export interface OHLCVResponseDTO {
  ticker: string
  market: 'TH' | 'US'
  currency: 'USD' | 'THB'
  price_basis?: string
  provider_name?: string
  provider_tier?: 'best_effort' | 'institutional_licensed'
  feed_latency_model?: 'realtime' | 'delayed_15m' | 'eod'
  current_price: number | null
  price_change: number | null
  price_change_pct: number | null
  price_as_of: string | null
  candles: OHLCVCandleDTO[]
  pivot_levels: PivotLevelsDTO | null
  pivot_period: string | null
  pivot_as_of: string | null
  requested_range?: string
  interval?: string
  allowed_ranges?: string[]
  effective_capabilities?: Record<string, string[]>
  capability_reasons?: Record<string, string>
  display_start_timestamp?: number | null
  available_warmup_bars?: number
  required_warmup_bars?: number
  warmup_status?: 'full' | 'partial' | 'unavailable' | 'sufficient' | 'insufficient' | 'not_applicable' | 'unknown'
  indicator_warmup?: Record<string, IndicatorWarmupDetailDTO>
  events?: CorporateActionEventDTO[]
  events_metadata?: CorporateActionsMetadataDTO | null
  week52_high?: number | null
  week52_low?: number | null
  week52_coverage_calendar_days?: number
}

export interface CorporateActionFactorDTO {
  event_type: 'split' | 'special_dividend' | 'spinoff'
  effective_date: string
  ratio?: number | null
  amount?: number | null
}

export interface DCFScenarioLevelDTO {
  scenario_name: string
  label: string
  target_price: number
  upside_pct?: number | null
  margin_of_safety_pct?: number | null
  color: 'emerald' | 'green' | 'rose' | 'zinc'
}

export interface ValuationTargetsDTO {
  evaluation_id: string
  ticker: string
  market: 'TH' | 'US'
  currency: 'USD' | 'THB'
  chart_price_basis: string
  valuation_price_basis: string
  comparability_status: 'comparable' | 'not_comparable' | 'unknown'
  comparability_reasons: string[]
  corporate_action_factors: CorporateActionFactorDTO[]
  current_price_at_eval?: number | null
  evaluated_at: string
  as_of_label: string
  model_version: string
  valuation_verdict: 'undervalued' | 'fairly_valued' | 'overvalued' | 'unknown'
  wacc_pct?: number | null
  macro_observable_refs: string[]
  data_quality_flags: string[]
  status: 'available' | 'unavailable' | 'stale'
  scenario_order_valid: boolean
  scenarios: DCFScenarioLevelDTO[]
}

export interface InsiderTransactionDTO {
  transaction_id: string
  transaction_date: string
  transaction_code: string
  shares: number
  price_per_share: number
  acquired_or_disposed: 'A' | 'D'
  shares_owned_following?: number | null
  ownership_nature?: string | null
  is_derivative: boolean
  normalized_weight: number
}

export interface InsiderFilingDTO {
  accession_number: string
  issuer_cik: string
  ticker: string
  filing_url: string
  filed_at: string
  timestamp: number
  reporting_owner_cik?: string | null
  reporting_owner_name?: string | null
  is_director: boolean
  is_officer: boolean
  is_ten_percent_owner: boolean
  officer_title?: string | null
  is_amendment: boolean
  amends_accession_number?: string | null
  is_cluster_buy: boolean
  transactions: InsiderTransactionDTO[]
}

export interface InsiderFilingsResponseDTO {
  ticker: string
  market: 'TH' | 'US'
  requested_range?: string
  interval?: string
  net_shares_30d: number
  net_shares_90d: number
  net_shares_180d: number
  cluster_buy_count: number
  total_filings_count: number
  filings: InsiderFilingDTO[]
}

export interface EarningsHistoryEntryDTO {
  date_str: string
  eps_actual: number | null
  eps_estimate: number | null
}

export interface AnalystContextDTO {
  ticker: string
  provider_symbol: string
  market: 'US' | 'TH'
  currency: 'USD' | 'THB'
  exchange_tz: string
  target_mean: number | null
  target_high: number | null
  target_low: number | null
  num_analysts: number | null
  next_earnings_date: string | null
  days_to_earnings: number | null
  earnings_history: EarningsHistoryEntryDTO[]
  source_as_of: string | null
  data_status: 'ok' | 'partial' | 'stale' | 'unavailable'
  provider_tier: 'best_effort'
  synced_at: string
}

export interface InsiderMarkerHoverDTO {
  action_type: 'buy' | 'sell' | 'cluster_buy'
  label: string
  accession_number: string
  insider_name: string
  officer_title: string | null
  shares: number
  price: number
  filing_url: string
  all_filers: Array<{
    name: string
    officer_title: string | null
    shares: number
  }> | null
}

export interface LineItemMetaDTO {
  canonical_key: string
  display_label: string
  unit_type: 'currency' | 'per_share' | 'shares' | 'ratio' | 'percentage'
  is_primary_highlight: boolean
}

export interface FinancialCellDTO {
  value: number | null
  yoy_growth_pct: number | null
  source_type?: 'reported' | 'derived' | 'not_applicable' | 'unavailable'
  source_concept?: string | null
  source_filing_url?: string | null
  derivation?: string | null
  formula?: string | null
  input_items?: string[] | null
  source_period?: string | null
  unavailable_reason?: string | null
  is_derived: boolean
}

export interface FinancialPeriodDTO {
  period_key: string
  fiscal_year: number
  fiscal_quarter?: number | null
  period_end_date: string
  period_kind: 'instant' | 'duration'
  duration_days?: number | null
  form_type: string
  filing_url?: string | null
  is_derived: boolean
  items: Record<string, FinancialCellDTO>
}

export interface FinancialStatementCategoryDTO {
  statement_type: 'income' | 'balance_sheet' | 'cash_flow'
  period_kind: 'duration' | 'instant'
  periods: FinancialPeriodDTO[]
  line_items: LineItemMetaDTO[]
}

export interface FinancialSummaryChartPointDTO {
  period_key: string
  date: string
  revenue: number | null
  gross_profit: number | null
  operating_income: number | null
  net_income: number | null
  free_cash_flow: number | null
  calculated_free_cash_flow?: number | null
  reported_free_cash_flow?: number | null
  operating_margin_pct: number | null
  net_margin_pct: number | null
}

export interface FinancialRatioPointDTO {
  period_key: string
  period_end_date: string
  gross_margin_pct: number | null
  operating_margin_pct: number | null
  net_margin_pct: number | null
  fcf_margin_pct: number | null
  debt_to_equity: number | null
  current_ratio: number | null
}

export interface FinancialStatementsDTO {
  schema_version: number
  ticker: string
  market: 'US' | 'TH'
  currency: string
  provider: 'edgartools' | 'yfinance' | null
  provider_symbol: string
  data_status: 'ok' | 'partial' | 'empty' | 'stale'
  coverage_status: 'complete' | 'partial'
  core_coverage_status: 'complete' | 'partial'
  expanded_coverage_status: 'complete' | 'partial' | 'not_available'
  expanded_data_status: 'complete' | 'partial' | 'not_available'
  core_coverage_pct?: number | null
  expanded_coverage_pct?: number | null
  missing_required_items: string[]
  missing_expanded_items?: string[]
  validation_warnings: string[]
  expanded_validation_warnings?: string[]
  expanded_error_count?: number
  error_code?: string | null
  warnings: string[]
  quarterly: FinancialStatementCategoryDTO[]
  annual: FinancialStatementCategoryDTO[]
  summary_chart_quarterly: FinancialSummaryChartPointDTO[]
  summary_chart_annual: FinancialSummaryChartPointDTO[]
  ratios_quarterly: FinancialRatioPointDTO[]
  ratios_annual: FinancialRatioPointDTO[]
  synced_at?: string | null
}

export interface EarningsCallSummarizeRequest {
  period: string
  transcript: string
}

export interface EarningsCallRunResponse {
  run_id: string
  ticker: string
  period: string
  status: 'new' | 'summarized' | 'note_written' | 'kanban_pending' | 'completed' | 'failed' | string
  kanban_status: 'none' | 'pending' | 'created' | 'existing' | 'failed' | string
  highlights?: string | null
  vault_path?: string | null
  kanban_card_id?: string | null
  reused_existing_run: boolean
  is_idempotent_replay: boolean
  last_error_code?: string | null
  created_at: number
  updated_at: number
}

export interface EarningsCallSummarizeResponse {
  run_id: string
  ticker: string
  period: string
  status: 'new' | 'summarized' | 'note_written' | 'kanban_pending' | 'completed' | 'failed' | string
  kanban_status: 'none' | 'pending' | 'created' | 'existing' | 'failed' | string
  highlights?: string | null
  vault_path?: string | null
  kanban_card_id?: string | null
  reused_existing_run: boolean
  is_idempotent_replay: boolean
}

export interface EarningsCallNoteItem {
  title: string
  ticker: string
  period: string
  vault_path: string
  highlights: string
  date: string
  last_updated: string
  has_full_transcript: boolean
}

export interface EarningsCallListResponse {
  ticker: string
  total_count: number
  items: EarningsCallNoteItem[]
}

// ---------------------------------------------------------
// Promise Check & Earnings Commitment Types
// ---------------------------------------------------------

export interface TranscriptCallDTO {
  quarter: string
  date: string
  status: 'ok' | 'missing'
  source?: string | null
  source_url?: string | null
  words?: number | null
  cached_file?: string | null
  fetch_method?: string | null
  retired?: boolean
}

export interface PromiseItemDTO {
  id: string
  made_in: string
  speaker: string
  section: 'prepared' | 'qa'
  type: 'A_numeric' | 'B_milestone'
  category: string
  scope: string
  quote: string
  metric: string
  metric_th?: string | null
  target_kind: string
  target_low?: number | null
  target_high?: number | null
  target_point?: number | null
  direction?: 'up' | 'down' | 'flat' | null
  unit?: string | null
  basis?: string | null
  due: string
  due_precision: 'explicit' | 'inferred'
  notes?: string
  notes_th?: string
  status: 'Delivered' | 'Partial' | 'Missed' | 'Restated' | 'Quietly dropped' | 'Pending' | 'Unverifiable'
  restated_direction?: 'raised' | 'lowered' | null
  supersedes?: string | null
  resolved_in?: string | null
  evidence_quote?: string | null
  actual_numeric?: number | null
  outcome_value?: number | null
  delta_pct?: number | null
  slip_quarters?: number | null
  extraction_method?: string
  retired?: boolean
}

export interface GuidanceBiasDTO {
  label: 'Sandbagger' | 'Straight shooter' | 'Overpromiser' | 'Mixed' | string
  code: string
  message: string
  median_delta_pct?: number | null
  beat_rate?: number | null
  n: number
  min_n: number
}

export interface CohortTrendItemDTO {
  quarter: string
  label: string
  date?: string | null
  status: string
  made: number
  resolved: number
  delivered: number
  pending: number
  hit_rate?: number | null
}

export interface PromiseScoresDTO {
  framework_version: string
  ticker: string
  company: string
  run_date: string
  window: {
    from: string
    to: string
    calls: number
    missing: string[]
  }
  credibility?: number | null
  grade?: string | null
  confidence: 'Low' | 'Medium' | 'High'
  delivery?: number | null
  transparency: number
  n_promises: number
  n_resolved: number
  counts: Record<string, number>
  bias: GuidanceBiasDTO
  milestone_slip_quarters?: number | null
  milestone_slip_n: number
  trend: {
    verdict: string
    older_half_hit_rate?: number | null
    newer_half_hit_rate?: number | null
    cohorts?: CohortTrendItemDTO[]
  }
  top_kept?: string[]
  top_broken?: string[]
}

export interface PromiseLedgerDTO {
  ticker: string
  company: string
  framework_version: string
  run_date: string
  calls: TranscriptCallDTO[]
  promises: PromiseItemDTO[]
}

export interface PromiseSyncResponseDTO {
  ledger: PromiseLedgerDTO
  scores: PromiseScoresDTO
  from_cache: boolean
}

// ---------------------------------------------------------
// Terminal V2 Phase 4 Types (Commodity Vol, Treasury Demand, SEC, News, Options)
// ---------------------------------------------------------

export interface CommodityVolSnapshotDTO {
  index_symbol: string
  underlying_instrument: string
  close_date: string
  implied_volatility: number
  change_1d_points?: number | null
  percentile_52w?: number | null
  sample_count: number
  regime_label?: string | null
  source: string
  as_of_date: string
  fetched_at: number
  is_stale: boolean
  stale_reason?: string | null
  limitations: string[]
}

export interface AuctionDemandSnapshotDTO {
  security_type: string
  security_term: string
  latest_auction_date: string
  latest_bid_to_cover_ratio?: number | null
  latest_high_yield?: number | null
  latest_high_investment_rate?: number | null
  latest_high_discount_rate?: number | null
  latest_offering_amount_usd?: number | null
  latest_total_accepted_usd?: number | null
  prior_mean_bid_to_cover?: number | null
  demand_delta?: number | null
  sample_count: number
  source: string
  as_of_date: string
  fetched_at: number
  is_stale: boolean
  stale_reason?: string | null
  limitations: string[]
}

export interface SecFactDTO {
  concept_tag: string
  label: string
  val?: number | null
  unit: string
  form: string
  fy?: number | null
  fp?: string | null
  start?: string | null
  end?: string | null
  filed?: string | null
  accn?: string | null
}

export interface SecCompanyFactsSnapshotDTO {
  symbol: string
  cik: string
  entity_name: string
  facts: SecFactDTO[]
  revenue_usd?: number | null
  operating_cash_flow_usd?: number | null
  capex_usd?: number | null
  free_cash_flow_usd?: number | null
  free_cash_flow_margin?: number | null
  long_term_debt_usd?: number | null
  debt_to_ocf_ratio?: number | null
  shares_outstanding?: number | null
  source: string
  as_of_date: string
  fetched_at: number
  is_stale: boolean
  stale_reason?: string | null
  limitations: string[]
}

export interface SecInsiderTransactionDTO {
  transaction_date: string
  reporting_owner: string
  officer_title?: string | null
  is_officer: boolean
  is_director: boolean
  is_ten_percent_owner: boolean
  transaction_code: string
  shares?: number | null
  price_per_share?: number | null
  notional_usd?: number | null
  direct_or_indirect: string
  accession_number: string
  is_amendment: boolean
}

export interface SecInsiderTradeSnapshotDTO {
  symbol: string
  cik: string
  transactions: SecInsiderTransactionDTO[]
  net_buy_ratio_90d?: number | null
  p_notional_sum_90d: number
  s_notional_sum_90d: number
  eligible_transaction_count: number
  source: string
  as_of_date: string
  fetched_at: number
  is_stale: boolean
  stale_reason?: string | null
  limitations: string[]
}

export interface NewsCandidateDTO {
  headline: string
  publisher: string
  source_type: string
  article_url: string
  published_at: string
  discovered_at: number
  symbol?: string | null
  is_stale: boolean
}

export interface NewsDiscoverySnapshotDTO {
  query_symbol: string
  items: NewsCandidateDTO[]
  status: 'ok' | 'rate_limited' | 'feed_unavailable'
  source: string
  as_of_date: string
  fetched_at: number
  limitations: string[]
}

export interface OptionContractDTO {
  symbol: string
  expiry: string
  strike: number
  option_type: 'call' | 'put'
  bid?: number | null
  ask?: number | null
  last_price?: number | null
  volume?: number | null
  open_interest?: number | null
  implied_volatility?: number | null
  delta?: number | null
  gamma?: number | null
  theta?: number | null
  vega?: number | null
}

export interface OptionsChainResponseDTO {
  underlying: string
  underlying_price?: number | null
  iv30_decimal?: number | null
  delay_minutes: number
  contracts: OptionContractDTO[]
  as_of_date: string
  source: string
  is_stale: boolean
}

export interface OptionsMaxPainResponseDTO {
  symbol: string
  expiry: string
  max_pain_strike: number
  current_price?: number | null
  put_call_oi_ratio?: number | null
  source: string
  as_of_date: string
}

export interface UsNationalDebtDTO {
  record_date: string
  total_public_debt_usd: number
  debt_held_by_public_usd?: number | null
  intragovernmental_holdings_usd?: number | null
  is_daily_close: boolean
  fetched_at: number
  source: string
  unit: string
  limitations: string
  is_stale: boolean
  stale_reason?: string | null
}

export interface InvestorTypeRowDTO {
  investor_type: string
  buy_value: number
  sell_value: number
  net_value: number
}

export interface ThaiFundFlowDTO {
  market: string
  as_of: string
  total_value: number
  investors: InvestorTypeRowDTO[]
  source: string
  is_stale: boolean
  stale_reason?: string | null
}

export interface GoldPriceDetailDTO {
  buy: number
  sell: number
}

export interface ThaiRetailGoldDTO {
  source: string
  unit: string
  bar: GoldPriceDetailDTO
  ornament: GoldPriceDetailDTO
  announced_at: string
  revision?: number | null
  is_stale: boolean
  stale_reason?: string | null
}

export interface MarketValuationDTO {
  market: string
  as_of: string
  market_cap?: number | null
  pe_ratio?: number | null
  pbv_ratio?: number | null
  dividend_yield?: number | null
  turnover_ratio?: number | null
  source: string
  is_stale: boolean
  stale_reason?: string | null
}

export interface MarketBreadthDTO {
  market: string
  as_of: string
  gainers: number
  losers: number
  unchanged: number
  source: string
  is_stale: boolean
  stale_reason?: string | null
}

export interface TreasuryYieldPointDTO {
  maturity: string
  yield_percent?: number | null
}

export interface TreasuryYieldCurveDTO {
  observation_date: string
  yields: TreasuryYieldPointDTO[]
  spread_10y_2y_bps?: number | null
  spread_10y_3m_bps?: number | null
  fetched_at: number
  source: string
  unit: string
  is_stale: boolean
  stale_reason?: string | null
}





export interface ThaiYieldPointDTO {
  tenor: string
  ttm_years: number
  yield_percent?: number | null
}

export interface ThaiYieldCurveDTO {
  observation_date: string
  yields: ThaiYieldPointDTO[]
  spread_10y_2y_bps?: number | null
  spread_10y_1y_bps?: number | null
  fetched_at: number
  source: string
  unit: string
  is_stale: boolean
  stale_reason?: string | null
}

export interface ThaiPublicDebtComponentDTO {
  component_number: number
  label_en: string
  label_th: string
  amount_thb: number
}

export interface ThaiPublicDebtDTO {
  reporting_month: string
  total_debt_thb: number
  debt_to_gdp_pct?: number | null
  fx_rate_usd_thb?: number | null
  components?: ThaiPublicDebtComponentDTO[]
  fetched_at?: number
  source?: string
  unit?: string
  is_stale?: boolean
  stale_reason?: string | null
}

export interface StablecoinItemDTO {
  symbol: string
  name: string
  circulating_usd?: number | null
  market_share_pct?: number | null
  price_usd?: number | null
}

export interface StablecoinSupplyDTO {
  total_circulating_usd?: number | null
  change_7d_pct?: number | null
  change_30d_pct?: number | null
  top_stablecoins: StablecoinItemDTO[]
  as_of_date: string
  is_partial: boolean
  completeness_notes: string
  fetched_at: number
  source: string
  unit: string
  is_stale: boolean
  stale_reason?: string | null
  limitations?: string | null
}

export interface CryptoMacroLiquidityDTO {
  btc_price_usd?: number | null
  btc_change_24h_pct?: number | null
  btc_change_7d_pct?: number | null
  btc_gold_ratio?: number | null
  stablecoin_total_usd?: number | null
  stablecoin_change_7d_pct?: number | null
  stablecoin_change_30d_pct?: number | null
  top_stablecoins: StablecoinItemDTO[]
  etf_daily_net_inflow_usd?: number | null
  etf_cumulative_total_usd?: number | null
  liquidity_regime: string
  as_of_date: string
  fetched_at: number
  source: string
  is_stale: boolean
  stale_reason?: string | null
  limitations?: string | null
}

// ---------------------------------------------------------
// ---------------------------------------------------------
// Macro NotebookLM Research Companion Export Types
// ---------------------------------------------------------

export type MacroNotebookLMExportState = 'queued' | 'running' | 'completed' | 'failed'

export interface MacroNotebookLMNotebookDTO {
  notebook_id: string
  title: string
  url: string
  status: string
  source_count: number
}

export interface MacroNotebookLMSourceResultDTO {
  file_name: string
  title: string
  status: string
  source_id?: string | null
  error?: string | null
}

export interface MacroNotebookLMCoverageDTO {
  strategy_report_present: boolean
  historical_reports_count: number
  catalog_notes_count: number
  indicator_series_count: number
  market_observables_cached: number
  market_observables_total: number
  thailand_hard_data_present: boolean
  sector_rotation_present: boolean
  news_events_count: number
}

export interface MacroNotebookLMExportRequestDTO {
  mode?: 'all_retained'
}

export interface MacroNotebookLMExportResponseDTO {
  export_id: string
  job_id?: string | null
  state: string
  stage: string
  message: string
}

export interface MacroNotebookLMExportStatusDTO {
  export_id: string
  job_id?: string | null
  mode: string
  state: MacroNotebookLMExportState | string
  stage: string
  snapshot_at: string
  bundle_hash: string
  strategy_report_id?: string | null
  notebooks: MacroNotebookLMNotebookDTO[]
  counts: Record<string, any>
  source_results: MacroNotebookLMSourceResultDTO[]
  coverage: MacroNotebookLMCoverageDTO
  warnings: string[]
  error_code?: string | null
  error?: string | null
  can_retry: boolean
}

// ---------------------------------------------------------
// Equity NotebookLM Research Companion Export Types
// ---------------------------------------------------------

export type EquityNotebookLMExportState =
  | 'queued'
  | 'running'
  | 'completed'
  | 'failed'
  | 'blocked'
  | 'partial'
  | 'ready'
  | 'ready_with_warnings'

export interface EquityNotebookLMNotebookDTO {
  notebook_id: string
  title: string
  url: string
  status: string
  source_count: number
}

export interface EquityNotebookLMSourceResultDTO {
  file_name: string
  title: string
  status: string
  source_id?: string | null
  error?: string | null
}

export interface EquityNotebookLMCoverageDTO {
  has_latest_analysis: boolean
  historical_analyses_count: number
  has_financials: boolean
  has_sec_filings: boolean
  has_insider_trades: boolean
  earnings_calls_count: number
  news_count: number
  has_options_derivatives: boolean
  has_portfolio_holding: boolean
  user_notes_count: number
}

export interface EquityNotebookLMExportRequestDTO {
  market?: string
  mode?: 'all_retained'
}

export interface EquityNotebookLMExportResponseDTO {
  export_id: string
  ticker: string
  market: string
  job_id?: string | null
  state: string
  stage: string
  message: string
}

export interface EquityNotebookLMExportStatusDTO {
  export_id: string
  ticker: string
  market: string
  job_id?: string | null
  mode: string
  state: EquityNotebookLMExportState | string
  stage: string
  snapshot_at: string
  bundle_hash: string
  notebooks: EquityNotebookLMNotebookDTO[]
  counts: Record<string, any>
  source_results: EquityNotebookLMSourceResultDTO[]
  coverage: EquityNotebookLMCoverageDTO
  warnings: string[]
  error_code?: string | null
  error?: string | null
  can_retry: boolean
}

// ============================================================================
// Investor Essence, Investment Axis, and Purpose Buckets (Milestones 1-3)
// ============================================================================

export interface ContentOptionDTO {
  option_id: string
  option_key: string
  text: string
}

export interface GeneratedQuestionDTO {
  question_id: string
  sequence_no: number
  text: string
  options: ContentOptionDTO[]
  evidence_type: string
  coverage_topics: string[]
  is_clarification: boolean
}

export interface AnswerDetailDTO {
  answer_id: string
  answer_kind: 'choice' | 'free_text' | 'unsure' | 'skipped' | string
  option_id?: string | null
  free_text?: string | null
  evidence_type: string
}

export interface QAItemDTO {
  question_id: string
  sequence_no: number
  text: string
  options: ContentOptionDTO[]
  evidence_type: string
  coverage_topics: string[]
  is_clarification: boolean
  answer?: AnswerDetailDTO | null
}

export interface SessionResponseDTO {
  session_id: string
  status: 'in_progress' | 'completed' | 'abandoned' | string
  revision: number
  active_branch_id: string
  questions_count: number
  answers_count: number
  is_complete: boolean
  current_question?: GeneratedQuestionDTO | null
  qa_history: QAItemDTO[]
}

export interface StartSessionRequestDTO {
  scope?: string
  prompt_version?: string
  portfolio_id?: string | null
  clear_downstream?: boolean
}

export interface RecordAnswerRequestDTO {
  answer_kind: 'choice' | 'free_text' | 'unsure' | 'skipped' | string
  option_id?: string | null
  free_text?: string | null
  expected_revision?: number | null
}

export interface AdvanceQuestionRequestDTO {
  expected_revision?: number | null
  prompt_version?: string
}

export interface ClarificationRequestDTO {
  unresolved_topic: string
  expected_revision?: number | null
  prompt_version?: string
}

export interface InterviewConfigDTO {
  interview_version: string
  prompt_version: string
  questions_count: number
  options_per_question: number
  topics: string[]
}

export interface EvidenceRefDTO {
  answer_id: string
  question_id: string
  revision: number
  quote: string
  evidence_type: string
}

export interface ClaimDTO {
  claim_id: string
  text: string
  effective_text: string
  source_kind: 'user_stated' | 'ai_interpreted' | 'user_edited' | string
  fit_rating?: 'exact' | 'partial' | 'rejected' | null
  edited_text?: string | null
  text_revision: number
  is_accepted: boolean
  evidence_refs: EvidenceRefDTO[]
}

export interface CoverageItemDTO {
  topic: string
  status: 'covered' | 'needs_review' | 'not_ready' | string
  supporting_answer_ids: string[]
}

export interface SummaryResponseDTO {
  summary_id: string
  session_id: string
  statement: string
  claims: ClaimDTO[]
  unresolved_topics: string[]
  coverage_report: CoverageItemDTO[]
  revision: number
}

export interface ReviewClaimRequestDTO {
  fit_rating?: 'exact' | 'partial' | 'rejected' | null
  edited_text?: string | null
  is_excluded?: boolean | null
  expected_revision?: number | null
}

export interface ConfirmEssenceRequestDTO {
  accepted_claim_ids?: string[] | null
  expected_summary_revision?: number | null
  idempotency_key?: string | null
}

export interface ConfirmationResponseDTO {
  confirmation_id: string
  status: string
  accepted_claims_count: number
  artifact_ref?: Record<string, string> | null
  message: string
}

export interface ConfirmedEssenceDTO {
  artifact_id: string
  scope: string
  statement: string
  accepted_claims: Array<{
    claim_id: string
    effective_text: string
    source_kind: string
    evidence_refs: EvidenceRefDTO[]
  }>
  unresolved_topics: string[]
  confirmed_at_iso: string
}

export interface SourceRefDTO {
  kind: 'question' | 'session' | 'user_input' | 'holding' | 'goal' | 'external' | string
  ref_id: string
  revision?: number | null
  quote?: string | null
  as_of?: string | null
}

export interface ValidationIssueDTO {
  code: string
  severity: 'blocking' | 'warning' | 'info' | string
  field_path: string
  message: string
  details?: Record<string, any>
  // Backward compatibility aliases
  field?: string
  issue?: string
  level?: 'blocking' | 'warning' | string
}

export type ReadinessIssueDTO = ValidationIssueDTO

export interface FinancialContextDTO {
  snapshot_id: string
  portfolio_id: string
  horizon_years?: string | null
  target_use_amount?: string | null
  target_use_range?: string | null
  target_use_timeline?: string | null
  emergency_reserves_amount?: string | null
  emergency_reserves_months?: string | null
  obligations_monthly?: string | null
  obligations_description?: string | null
  withdrawal_frequency?: string | null
  withdrawal_amount?: string | null
  experience_description?: string | null
  monthly_investable_contribution?: string | null
  essential_monthly_expense?: string | null
  reserves_outside_portfolio?: string | null
  facts?: Record<string, any>
  unknown_fields: string[]
  as_of: string
  source: string
  readiness_issues: ReadinessIssueDTO[]
  is_ready_for_numeric_policy: boolean
}

export interface UpdateFinancialContextRequestDTO {
  horizon_years?: string | null
  target_use_amount?: string | null
  target_use_range?: string | null
  target_use_timeline?: string | null
  emergency_reserves_amount?: string | null
  emergency_reserves_months?: string | null
  obligations_monthly?: string | null
  obligations_description?: string | null
  withdrawal_frequency?: string | null
  withdrawal_amount?: string | null
  experience_description?: string | null
  monthly_investable_contribution?: string | null
  essential_monthly_expense?: string | null
  reserves_outside_portfolio?: string | null
  unknown_fields?: string[] | null
}

export interface AllocationRowDTO {
  category: string
  target_percent: string
  min_percent?: string | null
  max_percent?: string | null
  notes?: string | null
  allocation_id?: string
  category_name?: string
  role_description?: string
}

export interface AxisDraftDTO {
  draft_id: string
  portfolio_id: string
  essence_ref: Record<string, any>
  context_ref: string
  basic_policy: string
  risk_limits: {
    mdd_percent?: string
    max_loss_per_trade_percent?: string
    [key: string]: any
  }
  invest_targets: string[]
  exclude_targets: string[]
  primary_methods: string[]
  secondary_methods: string[]
  investment_horizon: string
  allocation_basis: string
  allocation_rows: AllocationRowDTO[]
  rebalance_frequency: string
  role_models: string[]
  non_actions: string[]
  assumptions: string[]
  clarifications: string[]
  revision: number
  completeness_issues: string[]
  is_complete: boolean
}

export interface UpdateAxisDraftRequestDTO {
  basic_policy?: string
  risk_limits?: Record<string, any>
  invest_targets?: string[]
  exclude_targets?: string[]
  primary_methods?: string[]
  secondary_methods?: string[]
  investment_horizon?: string
  allocation_rows?: AllocationRowDTO[]
  rebalance_frequency?: string
  role_models?: string[]
  non_actions?: string[]
  expected_revision?: number
}

export interface ConfirmAxisRequestDTO {
  idempotency_key?: string
  expected_draft_revision?: number | null
}

export interface ConfirmedAxisDTO {
  artifact_id: string
  portfolio_id: string
  essence_ref: Record<string, any>
  basic_policy: string
  risk_limits: Record<string, any>
  invest_targets: string[]
  exclude_targets: string[]
  primary_methods: string[]
  secondary_methods: string[]
  investment_horizon: string
  allocation_basis: string
  allocation_rows: AllocationRowDTO[]
  rebalance_frequency: string
  role_models: string[]
  non_actions: string[]
  confirmed_at_iso: string
}

export interface PurposeBucketDTO {
  bucket_id: string
  name: string
  role: string
  color: string
  target_percent: string
  source_value_ids: string[]
  source_axis_allocation_ids: string[]
  linked_goal_name?: string | null
  eligible_assets?: string[]
  cash_inflow_routing_pct?: string | null
  action_rules?: string[]
  target_amount?: string | null
  minimum_amount?: string | null
  need_by_date?: string | null
  inflow_route_mode?: 'fixed_ratio' | 'priority_fill'
  inflow_priority?: number
  when_full_redirect_bucket_id?: string | null
  goal_id?: string | null
  schema_version?: string
}

export interface MappingWeightDTO {
  axis_allocation_id: string
  bucket_id: string
  portfolio_weight_percent: string
}

export interface BucketRemappingDTO {
  old_bucket_id: string
  target_bucket_id?: string | null
  affected_holding_count: number
}

export interface BucketPlanDraftDTO {
  draft_id: string
  portfolio_id: string
  essence_ref: Record<string, any>
  axis_ref: Record<string, any>
  context_ref: string
  portfolio_checkpoint: Record<string, any>
  purpose_buckets: PurposeBucketDTO[]
  allocation_basis: string
  mapping_weights: MappingWeightDTO[]
  constraints: string[]
  remapping: BucketRemappingDTO[]
  status: string
  revision: number
  validation_issues: string[]
  is_valid: boolean
}

export interface UpdateBucketPlanRequestDTO {
  purpose_buckets?: Array<Partial<PurposeBucketDTO>>
  mapping_weights?: Array<Partial<MappingWeightDTO>>
  remapping?: Array<Partial<BucketRemappingDTO>>
  constraints?: string[]
  expected_revision?: number
}

export interface AllocationPreviewDTO {
  checkpoint_sequence: number
  checkpoint_state_hash: string
  validated_targets: Array<{
    bucket_id: string
    name: string
    target_percent: number
    color?: string
  }>
  affected_holdings: Array<{
    symbol: string
    old_bucket_id: string
    new_bucket_id: string
  }>
  before_allocation: Record<string, any>
  after_allocation: Record<string, any>
  issues: string[]
  payload_hash: string
  preview_id?: string
  draft_id?: string
  draft_revision?: number
  source_refs?: Record<string, any>
}

export interface ApplyBucketPlanRequestDTO {
  idempotency_key?: string
  expected_draft_revision?: number
  expected_checkpoint_sequence?: number
  expected_checkpoint_state_hash?: string
  preview_id?: string
  payload_hash?: string
}

export interface AllocationApplyReceiptDTO {
  command_id: string
  portfolio_id: string
  request_hash: string
  applied_sequence: number
  applied_state_hash: string
  applied_at_iso: string
  canonical_status: string
  projection_status: string
  warnings: string[]
}

// =====================================================================
// Portfolio Advisory Desk (Tier 1 Quant + Tier 2 AI Playbook)
// =====================================================================

export type AdvisoryActionCode =
  | 'URGENT_REDUCE'
  | 'TRIM_OVERWEIGHT'
  | 'LET_WINNER_RUN'
  | 'ACCUMULATE_NOW'
  | 'BREAKOUT_BUY'
  | 'ACCUMULATE_ON_DIP'
  | 'HOLD_WITHIN_BAND'
  | 'HOLD_LOCKED_VEHICLE'
  | 'HOLD_PASSIVE_CORE'
  | 'QUARANTINED_DATA_ANOMALY'
  | 'WAIT_PRICE_REFRESH'

export type AdvisoryDataMode =
  | 'FULL_CROSS_SCALE'
  | 'PORTFOLIO_PLUS_MICRO'
  | 'PORTFOLIO_BUCKET_ONLY'

export interface LotSizingDTO {
  market: 'TH' | 'US' | 'GLOBAL'
  board_lot_shares?: number | null
  odd_lot_shares?: number | null
  whole_shares?: number | null
  fractional_shares?: number | null
  amount_thb: number
  amount_native: number
  estimated_execution_price_native: number
  native_currency: 'THB' | 'USD'
}

export interface DualCurrencyCashPlanDTO {
  cash_thb_available: number
  cash_usd_available: number
  usd_thb_fx_rate: number
  immediate_deploy_thb: number
  immediate_deploy_usd: number
  conditional_limit_order_thb: number
  conditional_limit_order_usd: number
  thb_surplus_after_immediate: number
  usd_surplus_after_immediate: number
  fx_conversion_needed_thb: number
}

export interface DataAnomalyWarningDTO {
  guardrail_code: string
  ticker: string
  message: string
  severity: 'critical' | 'warning' | 'info'
  remedy_hint?: string
}

export interface QuarantinedHoldingDTO {
  symbol: string
  reason: string
  guardrail_code: string
  current_value_thb: number
  details?: string
}

export interface HoldingAdvisoryDTO {
  symbol: string
  asset_name: string
  asset_type: string
  market: 'TH' | 'US'
  native_currency: 'THB' | 'USD'
  evaluation_track: string
  bucket_id?: string | null
  bucket_name?: string | null
  current_units: number
  current_price_native: number
  current_value_thb: number
  current_weight_pct: number
  target_weight_pct: number
  weight_drift_pct: number
  action_code: AdvisoryActionCode | string
  action_rationale: string
  lot_sizing?: LotSizingDTO | null
  is_immediate_execution: boolean
  conditional_buy_zone_min?: number | null
  conditional_buy_zone_max?: number | null
  core_conviction_score?: number | null
  business_conviction_score?: number | null
  price_stage?: string | null
  live_upside_12m_pct?: number | null
  macro_alignment: 'tailwind' | 'neutral' | 'headwind' | 'unavailable'
  sector_quadrant?: string | null
  breached_falsifiers: string[]
  data_warnings: string[]
}

export interface PortfolioRiskRadarDTO {
  weighted_beta?: number | null
  sector_crowding: Record<string, number>
  fx_exposure_pct: Record<string, number>
  purpose_bucket_drift: Record<string, number>
  unassigned_holdings_value_thb: number
  unassigned_holdings_nav_pct: number
  unassigned_drift_guard_active: boolean
  hhi?: number | null
  effective_holdings_count?: number | null
  top_1_weight_pct?: number
  top_3_weight_pct?: number
  top_5_weight_pct?: number
  concentration_risk_level?: 'LOW' | 'MODERATE' | 'HIGH'
}

export interface PortfolioAdvisoryNarrativeDTO {
  synthesized_at: string
  portfolio_state_hash: string
  macro_executive_summary: string
  immediate_action_playbook: string[]
  conditional_dip_plan: string[]
  risk_mitigation_and_hedging: string[]
  whats_working?: string[]
  what_concerns_me?: string[]
  strategic_discussion_points?: string[]
  consistency_verified: boolean
  generated_by: string
}

export interface FundHoldingItemDTO {
  name: string
  percent: number
  ticker?: string | null
  asset_type?: string
}

export interface FundSectorItemDTO {
  sector_name_th: string
  sector_name_en: string
  percent: number
}

export interface FundConstituentsDTO {
  fund_id: string
  symbol: string
  as_of_date: string
  master_fund_name?: string | null
  master_fund_ticker?: string | null
  top_holdings: FundHoldingItemDTO[]
  sector_breakdown: FundSectorItemDTO[]
  asset_allocation: Record<string, number>
}

export interface LookThroughHoldingExposureDTO {
  ticker: string
  company_name: string
  market: string
  direct_value_thb: number
  direct_weight_pct: number
  indirect_value_thb: number
  indirect_weight_pct: number
  total_value_thb: number
  total_weight_pct: number
  source_funds: Array<[string, number]>
  is_cap_breached: boolean
  is_hidden_concentration: boolean
}

export interface TrueSectorExposureDTO {
  sector_name: string
  sector_name_th: string
  total_value_thb: number
  weight_pct: number
  direct_stocks_pct: number
  funds_pct: number
  is_crowded: boolean
}

export interface PortfolioLookThroughReportDTO {
  portfolio_id: string
  evaluated_at: string
  total_nav_thb: number
  holdings_exposure: LookThroughHoldingExposureDTO[]
  top_overlapped_stocks: LookThroughHoldingExposureDTO[]
  sector_exposure: TrueSectorExposureDTO[]
  hidden_concentration_warnings: string[]
  cap_breached_count: number
}

export interface PortfolioAdvisoryResponseDTO {
  evaluated_at: string
  data_mode: AdvisoryDataMode | string
  portfolio_state_hash: string
  total_nav_thb: number
  cash_plan: DualCurrencyCashPlanDTO
  risk_radar: PortfolioRiskRadarDTO
  holdings_advisory: HoldingAdvisoryDTO[]
  quarantined_holdings: QuarantinedHoldingDTO[]
  anomaly_warnings: DataAnomalyWarningDTO[]
  macro_regime_summary?: Record<string, any> | null
  cached_narrative?: PortfolioAdvisoryNarrativeDTO | null
  look_through_summary?: PortfolioLookThroughReportDTO | null
}

export interface SynthesizeAdvisoryPlaybookRequest {
  force_refresh?: boolean
}

export interface OpportunityCandidateDTO {
  security_id: string
  issuer_id: string
  symbol: string
  market: string
  currency: string
  asset_type: string
  company_name: string
  sector_name: string
  sector_name_th: string
  source_pools: string[]
  source_refs: string[]
  already_directly_held: boolean
  already_watchlisted: boolean
  current_direct_value_thb: number
  current_indirect_value_thb: number
  current_total_exposure_thb: number
  current_total_weight_pct: number
}

export interface CandidateEvaluationResultDTO {
  candidate: OpportunityCandidateDTO
  policy_fit: 'pass' | 'fail' | 'needs_review'
  policy_reasons: string[]
  suggested_bucket_id?: string | null
  applied_bucket_name?: string | null

  evidence_readiness: 'sufficient' | 'partial' | 'stale' | 'unavailable'
  composite_score?: number | null
  valuation_upside_pct?: number | null
  target_price?: number | null
  target_price_basis?: string | null
  moat_rating?: string | null
  evidence_reasons: string[]

  macro_alignment: 'tailwind' | 'neutral' | 'headwind' | 'unavailable'
  macro_stage?: string | null
  macro_reasons: string[]

  capacity_status: 'available' | 'constrained' | 'unknown' | 'not_applicable'
  remaining_capacity_thb?: number | null
  issuer_room_thb?: number | null
  sector_room_thb?: number | null
  capacity_reasons: string[]

  research_priority: number
  priority_reasons: string[]
  is_actionable: boolean
}

export interface PortfolioOpportunitiesResponseDTO {
  portfolio_id: string
  evaluation_id: string
  evaluated_at: string
  readiness: 'ready' | 'degraded' | 'not_ready'
  total_candidates_scanned: number
  passed_candidates_count: number
  candidates: CandidateEvaluationResultDTO[]
  warnings: string[]
  missing_requirements: string[]
  snapshot_meta: Record<string, any>
}

// ---------------------------------------------------------------------------
// Event Study & Historical Scenario Analysis DTOs
// ---------------------------------------------------------------------------

export interface EventPresetItemDTO {
  preset_id: string
  name: string
  category: string
  description: string
  recommended_ticker?: string | null
}

export interface EventStudyRequestDTO {
  preset_id?: string
  custom_dates?: string[]
  custom_event_name?: string
  horizons?: number[]
}

export interface EventPathPointDTO {
  day_offset: number
  date: string
  close: number
  cumulative_return_pct: number
}

export interface EventInstanceDTO {
  event_id: string
  event_date: string
  entry_date: string
  entry_price: number
  description?: string | null
  forward_returns: Record<string, number | null>
  daily_trajectory: EventPathPointDTO[]
  is_win_by_horizon: Record<string, boolean | null>
}

export interface HorizonStatsDTO {
  horizon_days: number
  sample_size: number
  hit_rate_pct: number
  avg_return_pct: number
  median_return_pct: number
  best_return_pct?: number | null
  worst_return_pct?: number | null
  avg_win_pct?: number | null
  avg_loss_pct?: number | null
  risk_reward_ratio?: number | null
  win_count: number
  loss_count: number
}

export interface AverageTrajectoryPointDTO {
  day_offset: number
  avg_return_pct: number
  sample_count: number
}

export interface EventStudyGuardrailWarningDTO {
  code: string
  message: string
}

export interface EventStudyResponseDTO {
  ticker: string
  event_name: string
  horizons: number[]
  events: EventInstanceDTO[]
  horizon_stats: Record<string, HorizonStatsDTO>
  average_trajectory: AverageTrajectoryPointDTO[]
  warnings: EventStudyGuardrailWarningDTO[]
}

export interface PriceMoveDecompositionDTO {
  horizon_label: string
  start_date: string
  end_date: string
  start_price: number
  end_price: number
  total_return_pct: number
  start_eps: number
  end_eps: number
  eps_growth_pct: number
  start_pe: number
  end_pe: number
  pe_expansion_pct: number
  primary_driver: string
  diagnosis_tier: string
  diagnosis_summary: string
}

export interface ThenVsNowSnapshotDTO {
  is_held: boolean
  period_label: string
  then_date: string
  now_date: string
  then_price: number
  now_price: number
  then_pe?: number | null
  now_pe?: number | null
  then_growth_pct?: number | null
  now_growth_pct?: number | null
  what_was_true_then: string
  what_is_true_now: string
  thesis_reflection_question: string
}

export interface PortfolioAwareTriggerDTO {
  trigger_id: string
  title: string
  condition_text: string
  category: string
  case_impact: string
  impact_description: string
  is_currently_triggered: boolean
  metric_current_value?: number | null
  metric_threshold_value?: number | null
}

export interface HoldingSummaryDTO {
  is_held: boolean
  units: number
  avg_cost?: number | null
  unrealized_pnl_pct?: number | null
  current_weight_pct: number
  bucket_id?: string | null
  entry_date?: string | null
  added_direction?: string | null
}

export interface PositionReviewResponseDTO {
  ticker: string
  company_name?: string | null
  current_price: number
  as_of_date: string
  action_stance: string
  action_stance_reason?: string | null
  holding_summary: HoldingSummaryDTO
  decompositions: PriceMoveDecompositionDTO[]
  then_vs_now?: ThenVsNowSnapshotDTO | null
  triggers: PortfolioAwareTriggerDTO[]
  honest_gaps: string[]
}



// ---------------------------------------------------------------------------
// Market Intelligence Daily & Weekly Digest DTOs
// ---------------------------------------------------------------------------

export interface DigestPortfolioBadgeDTO {
  symbol: string
  company_name?: string | null
  impact_type: 'direct_holding' | 'fund_lookthrough' | 'watchlist'
  weight_pct?: number | null
  unrealized_pnl_pct?: number | null
  source_fund_codes?: string[]
  target_entry_price?: number | null
  sector?: string | null
}

export interface DigestNewsCandidateDTO {
  candidate_id: string
  title: string
  summary: string
  source: string
  links: string[]
  sources_count: number
  published_at?: string | null
  age_hours: number
  freshness_tier: string
  freshness_label: string
  is_stale: boolean
  matched_tickers: string[]
  portfolio_badges: DigestPortfolioBadgeDTO[]
}

export interface DigestYouTubeCandidateDTO {
  video_id: string
  title: string
  channel: string
  link: string
  published_at?: string | null
  age_hours: number
  freshness_tier: string
  freshness_label: string
  thumbnail: string
  is_fetched: boolean
  cached_summary_snippet?: string | null
  cached_tickers: string[]
}

export interface DigestCandidatesResponseDTO {
  last_fetched_at?: string | null
  news_candidates: DigestNewsCandidateDTO[]
  youtube_candidates: DigestYouTubeCandidateDTO[]
  stale_dropped_count: number
}

export interface DailyDigestSynthesizeRequestDTO {
  selected_video_ids: string[]
  excluded_news_ids: string[]
  custom_youtube_urls: string[]
  custom_news_items: string[]
}

export interface DailyDigestItemDTO {
  source_candidate_id: string
  thai_headline: string
  what_happened: string
  why_it_matters: string
  thesis_signal: 'thesis_breaker' | 'buy_on_dip' | 'thesis_reinforcer' | 'noise'
  impact_type: 'direct_holding' | 'fund_lookthrough' | 'watchlist' | 'second_order' | 'macro_general'
  affected_tickers: string[]
  affected_sectors: string[]
  macro_impact_score: number
  asset_impact_score: number
  time_horizon: 'short_term' | 'structural'
  youtube_corroboration?: string | null
  source: string
  link: string
  links: string[]
  published_at?: string | null
  age_hours: number
  freshness_tier: string
  freshness_label: string
  portfolio_badges: DigestPortfolioBadgeDTO[]
  is_read: boolean
  is_new: boolean
}

export interface DailyYouTubeHighlightDTO {
  video_id: string
  thai_title: string
  channel: string
  key_takeaways: string[]
  mentioned_tickers: string[]
  alignment_with_news: 'confirmed' | 'contrarian' | 'independent'
  alignment_note: string
}

export interface DailyDigestPayloadDTO {
  digest_id: string
  date: string
  period: string
  generated_at: string
  market_mood: 'risk_on' | 'risk_off' | 'mixed'
  market_mood_summary: string
  transmission_chain: string
  top_portfolio_focus: string
  selected_video_ids: string[]
  items: DailyDigestItemDTO[]
  youtube_highlights: DailyYouTubeHighlightDTO[]
}

export interface DailyDigestResponseDTO {
  has_digest: boolean
  digest: DailyDigestPayloadDTO | null
  snapshot_age_hours: number
  is_snapshot_stale: boolean
  fresh_pending_count: number
  available_dates?: string[]
}

export interface WeeklyPortfolioMatrixRowDTO {
  ticker: string
  thesis_status: 'strong_hold' | 'watch_closely' | 'accumulate_zone' | 'review_thesis'
  thesis_status_label: string
  weekly_summary: string
}

export interface WeeklyDigestPayloadDTO {
  digest_id: string
  from_date: string
  to_date: string
  generated_at: string
  executive_themes: string[]
  portfolio_health_summary: string
  portfolio_matrix: WeeklyPortfolioMatrixRowDTO[]
  consensus_narratives: string[]
  contrarian_and_pitches: string[]
  week_ahead_checklist: string[]
  items: DailyDigestItemDTO[]
  youtube_highlights: DailyYouTubeHighlightDTO[]
}

export interface WeeklyDigestResponseDTO {
  has_digest: boolean
  digest: WeeklyDigestPayloadDTO | null
  available_dates?: string[]
}

export interface WeeklyDigestSynthesizeRequestDTO {
  days: number
  selected_video_ids: string[]
  excluded_news_ids: string[]
  custom_youtube_urls: string[]
  custom_news_items: string[]
}

export interface DigestReadStateRequestDTO {
  item_ids: string[]
  mark_as_read: boolean
}

export interface DigestReadStateResponseDTO {
  ok: boolean
  read_item_ids: string[]
}


// ==========================================
// Trinity AI Advisor (Manager Chat) DTOs
// ==========================================
export interface AdvisorSessionDTO {
  id: string
  title: string
  created_at: string
  updated_at: string
  message_count: number
}

export interface AdvisorChatMessageDTO {
  id: string
  session_id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  citations: string[]
  created_at: string
}

export interface SkillManifestDTO {
  name: string
  description: string
  is_quarantined: boolean
  path: string
  created_at: string
}

export interface SandboxStateSnapshotDTO {
  memory_files: string[]
  active_skills: SkillManifestDTO[]
  quarantined_skills: SkillManifestDTO[]
  scratch_files_count: number
  last_audit_entry?: string | null
}

export interface HermesSyncStatusDTO {
  is_installed: boolean
  current_commit: string
  upstream_commit: string
  has_updates: boolean
  last_synced_at?: string | null
}
