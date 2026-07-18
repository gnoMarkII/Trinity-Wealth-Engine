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
  regime_probabilities: Record<string, number>
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
  core_hook: string
  key_questions_to_answer: string[]
  research_hypotheses: string[]
  source_event_ids: string[]
  source_links: string[]
  source_titles: string[]
  recommended_format: string
  estimated_impact: string
}

export interface YoutubePitchApprovalPayload {
  type: 'youtube_pitch_approval'
  pitches: YoutubePitchItemDTO[]
  instruction?: string
}

export type ApprovalPayload = NewsYoutubeApprovalPayload | NewsFunnelApprovalPayload | YoutubePitchApprovalPayload

export interface JobStatusDTO {
  job_id: string
  status: 'queued' | 'running' | 'done' | 'error' | 'awaiting_approval'
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
  bucket_id: string | null
  avg_cost_usd: number | null
  avg_cost_thb: number | null
  current_price_usd: number | null
  current_price_thb: number | null
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
  fundamentals_updated_at: number | null
}

export interface ActualSummaryDTO {
  total_value_thb: number
  total_cost_basis_thb: number
  total_unrealized_profit: number
  passive_income_ytd: number
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

export interface ActualGoalItemDTO {
  name: string
  target_amount_thb: number
  goal_type: string
  current_amount_thb: number
  progress_pct: number
  deadline: string | null
  deadline_days_left: number | null
  notes: string | null
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
  goal_type: 'nav_target' | 'cash_target' | 'passive_income_ytd'
  target_amount_thb: number
  deadline?: string | null
  years_from_now?: number | null
  notes?: string | null
}

export interface AppendJournalPayload {
  entry: string
}

