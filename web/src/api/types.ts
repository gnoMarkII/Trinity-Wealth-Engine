// Type ตรงตาม api/schemas.py — DTO layer เดียวที่ frontend ผูกด้วย ไม่ใช่ schemas/macro_schemas.py ภายใน

export interface WarningDTO {
  code: string | null
  message: string
}

export interface AssetAllocationDTO {
  asset_class: string
  asset_bucket: string | null
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
  warnings: WarningDTO[]
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
  target_price: number
  upside_pct: number
  margin_of_safety_pct: number
}

export interface DCFResultDTO {
  wacc_pct: number
  cost_of_equity_pct: number
  cost_of_debt_pct: number
  risk_free_rate_pct: number
  erp_pct: number
  observable_refs: string[]
  scenarios: Record<'bull' | 'base' | 'bear', DCFScenarioDTO>
  valuation_verdict: 'undervalued' | 'fairly_valued' | 'overvalued'
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
  dcf_result: DCFResultDTO | null
  smart_money_flags: SmartMoneyFlagsDTO | null
  evaluated_at: string
  data_quality_flags: string[]
}

export interface EquityDetailDTO extends EquitySummaryDTO {
  quant_signals: QuantSignalsDTO
  sentiment_context: EquitySentimentContextDTO
  narrative_analysis: string
  base_case_summary: string
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






