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
  last_updated: string | null
  fx_rates: Record<string, number>
  summary: ActualSummaryDTO
  allocation_targets: AllocationTargetDTO[]
  holdings: ActualHoldingDTO[]
  price_refresh_info: Record<string, string> | null
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
}

export interface UpsertGoalPayload {
  goal_type: 'nav_target' | 'cash_target' | 'passive_income_ytd' | 'bucket_target'
  target_amount_thb: number
  deadline?: string | null
  years_from_now?: number | null
  notes?: string | null
  portfolio_id?: string | null
  bucket_id?: string | null
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

export interface ReverseDCFResultDTO {
  explicit_forecast_5y?: any[]
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
}

export interface EquityDetailDTO extends EquitySummaryDTO {
  quant_signals: QuantSignalsDTO
  sentiment_context: EquitySentimentContextDTO
  narrative_analysis: string
  base_case_summary: string
  narrative_status?: 'available' | 'unavailable' | 'pending'
  error_code?: string | null
  generated_by: string
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
}

export interface EquityNewsDTO {
  ticker: string
  market: 'TH' | 'US'
  last_updated?: string | null
  news_date?: string | null
  items: EquityNewsItemDTO[]
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
