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
  /** "llm" | "mock" | "heuristic_fallback" — เมื่อเป็น heuristic_fallback คะแนนไม่ได้มาจาก LLM จริง */
  triage_source?: string
}

export interface NewsFunnelApprovalPayload {
  type: 'news_funnel_approval'
  candidates: NewsFunnelCandidate[]
}

export type ApprovalPayload = NewsYoutubeApprovalPayload | NewsFunnelApprovalPayload

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
