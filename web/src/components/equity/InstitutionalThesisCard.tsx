import React from 'react'

export interface ThesisFalsifierDTO {
  falsifier_id: string
  metric_name: string
  condition: string
  threshold_value?: number | null
  source_basis?: string
  source_ref?: string | null
  source_quote?: string | null
  narrative_explanation: string
}

export interface PiotroskiBreakdownDTO {
  f_score?: number | null
  profitability_points?: number
  leverage_liquidity_points?: number
  operating_efficiency_points?: number
  is_eligible: boolean
  exclusion_reason?: string | null
  status: string
}

export interface ReverseDCFDTO {
  target_price_12m?: number | null
  upside_12m_pct?: number | null
  intrinsic_value_today?: number | null
  market_implied_growth_pct?: number | null
  market_implied_margin_pct?: number | null
  solver_status?: string
  status: string
  is_eligible: boolean
  exclusion_reason?: string | null
}

export interface TacticalSetupDTO {
  price_stage?: string
  current_price?: number
  atr_14?: number | null
  key_support_level?: number | null
  key_resistance_level?: number | null
  buy_zone_min?: number | null
  buy_zone_max?: number | null
  invalidation_stop_loss?: number | null
  tactical_target_price?: number | null
  tactical_risk_reward_ratio?: number | null
  horizon_timeframe?: string
  status: string
}

export interface InsiderConvictionDTO {
  status: string
  data_status: string
  open_market_p_count_90d?: number
  open_market_p_value_usd?: number
  c_suite_p_count?: number
  open_market_s_count_90d?: number
  open_market_s_value_usd?: number
  insider_buy_range_min?: number | null
  insider_buy_range_max?: number | null
}

export interface DeterministicScorecardDTO {
  core_conviction_score: number
  execution_readiness_score: number
  action_stance: 'ACCUMULATE_NOW' | 'ACCUMULATE_ON_DIP' | 'BREAKOUT_BUY' | 'HOLD_WAIT' | 'REDUCE' | 'INSUFFICIENT_DATA'
  fundamental_quality_score: number
  guidance_expectation_score: number
  valuation_margin_score: number
  coverage_pct: number
  usable_coverage_pct?: number
  verified_coverage_pct?: number
  applicable_pillars_count: number
}

import { EvidenceProvenanceDrawer, type AnalysisEvidenceSnapshotDTO } from './EvidenceProvenanceDrawer'

interface InstitutionalThesisCardProps {
  scorecard?: DeterministicScorecardDTO | null
  piotroski?: PiotroskiBreakdownDTO | null
  reverseDcf?: ReverseDCFDTO | null
  tactical?: TacticalSetupDTO | null
  insider?: InsiderConvictionDTO | null
  falsifiers?: ThesisFalsifierDTO[] | null
  evidenceSnapshot?: AnalysisEvidenceSnapshotDTO | null
}

const STANCE_CONFIG: Record<
  string,
  { label: string; color: string; bg: string; border: string; desc: string }
> = {
  ACCUMULATE_NOW: {
    label: 'ACCUMULATE NOW',
    color: 'text-emerald-400',
    bg: 'bg-emerald-950/40',
    border: 'border-emerald-500/40',
    desc: 'High fundamental conviction & optimal execution timing (Within Buy Zone / Stage 2).',
  },
  ACCUMULATE_ON_DIP: {
    label: 'ACCUMULATE ON DIP',
    color: 'text-sky-400',
    bg: 'bg-sky-950/40',
    border: 'border-sky-500/40',
    desc: 'Strong fundamental quality, but price is extended. Wait for pullback into Buy Zone.',
  },
  BREAKOUT_BUY: {
    label: 'BREAKOUT BUY',
    color: 'text-indigo-400',
    bg: 'bg-indigo-950/40',
    border: 'border-indigo-500/40',
    desc: 'Positive fundamental tailwinds with Stage 2 Momentum Breakout confirmed.',
  },
  HOLD_WAIT: {
    label: 'HOLD & WAIT',
    color: 'text-amber-400',
    bg: 'bg-amber-950/40',
    border: 'border-amber-500/40',
    desc: 'Fairly valued or mixed fundamentals. Hold existing position and await catalysts.',
  },
  REDUCE: {
    label: 'REDUCE / EXIT',
    color: 'text-rose-400',
    bg: 'bg-rose-950/40',
    border: 'border-rose-500/40',
    desc: 'Weak fundamental forensics, deteriorating guidance, or valuation overextension.',
  },
  INSUFFICIENT_DATA: {
    label: 'INSUFFICIENT DATA',
    color: 'text-zinc-400',
    bg: 'bg-zinc-900/40',
    border: 'border-zinc-700/40',
    desc: 'Data coverage is below 60% threshold for institutional evaluation.',
  },
}

export const InstitutionalThesisCard: React.FC<InstitutionalThesisCardProps> = ({
  scorecard,
  piotroski,
  reverseDcf,
  tactical,
  insider,
  falsifiers,
  evidenceSnapshot,
}) => {
  if (!scorecard) return null

  const fallbackStance = STANCE_CONFIG.INSUFFICIENT_DATA || {
    label: 'INSUFFICIENT DATA',
    color: 'text-zinc-400',
    bg: 'bg-zinc-900/40',
    border: 'border-zinc-700/40',
    desc: 'Data coverage is below 60% threshold for institutional evaluation.',
  }
  const stance = STANCE_CONFIG[scorecard.action_stance] || fallbackStance

  return (
    <div className="rounded-2xl border border-zinc-800 bg-gradient-to-b from-zinc-900/90 to-zinc-950/90 p-6 shadow-2xl backdrop-blur-xl">
      {/* Header & Action Stance */}
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 border-b border-zinc-800/80 pb-5">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-bold uppercase tracking-wider text-indigo-400">Institutional Thesis</span>
            <span
              title={`Usable Coverage: ${scorecard.usable_coverage_pct ?? scorecard.coverage_pct}% | Verified Coverage: ${scorecard.verified_coverage_pct ?? scorecard.coverage_pct}%`}
              className="rounded-full bg-zinc-800 px-2 py-0.5 text-[10px] font-medium text-zinc-400"
            >
              Coverage: {scorecard.usable_coverage_pct ?? scorecard.coverage_pct}%
              {scorecard.verified_coverage_pct != null && scorecard.verified_coverage_pct !== (scorecard.usable_coverage_pct ?? scorecard.coverage_pct) && (
                <span className="text-zinc-500 ml-1">(Verified: {scorecard.verified_coverage_pct}%)</span>
              )}
            </span>
          </div>
          <h3 className="mt-1 text-xl font-bold text-zinc-100">4-Pillars Investment Committee Engine</h3>
        </div>

        {/* Action Stance Badge */}
        <div className={`rounded-xl border px-4 py-2.5 ${stance.bg} ${stance.border}`}>
          <div className="flex items-center gap-2">
            <div className={`h-2.5 w-2.5 rounded-full ${stance.color.replace('text-', 'bg-')} animate-pulse`} />
            <span className={`text-sm font-extrabold tracking-wide ${stance.color}`}>{stance.label}</span>
          </div>
          <p className="mt-1 text-[11px] text-zinc-400 max-w-xs leading-snug">{stance.desc}</p>
        </div>
      </div>

      {/* Dual Scores Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 my-6">
        {/* Core Conviction Score */}
        <div className="rounded-xl border border-zinc-800/80 bg-zinc-900/40 p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">Core Conviction Score</span>
            <span className="text-xs text-indigo-400 font-medium">Fundamental + Guidance + Valuation</span>
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="text-3xl font-extrabold text-zinc-100">{scorecard.core_conviction_score.toFixed(1)}</span>
            <span className="text-sm font-medium text-zinc-500">/ 10.0</span>
          </div>
          <div className="mt-3 h-2 w-full rounded-full bg-zinc-800 overflow-hidden">
            <div
              className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-emerald-400 transition-all duration-500"
              style={{ width: `${Math.min(100, scorecard.core_conviction_score * 10)}%` }}
            />
          </div>
          <div className="mt-3 grid grid-cols-3 gap-2 text-[11px] text-zinc-400">
            <div>Quality: <span className="font-semibold text-zinc-200">{scorecard.fundamental_quality_score}</span></div>
            <div>Guidance: <span className="font-semibold text-zinc-200">{scorecard.guidance_expectation_score}</span></div>
            <div>Valuation: <span className="font-semibold text-zinc-200">{scorecard.valuation_margin_score}</span></div>
          </div>
        </div>

        {/* Execution Readiness Score */}
        <div className="rounded-xl border border-zinc-800/80 bg-zinc-900/40 p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">Execution Readiness</span>
            <span className="text-xs text-sky-400 font-medium">Technicals + S/R + Form 4</span>
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="text-3xl font-extrabold text-zinc-100">{scorecard.execution_readiness_score.toFixed(1)}</span>
            <span className="text-sm font-medium text-zinc-500">/ 10.0</span>
          </div>
          <div className="mt-3 h-2 w-full rounded-full bg-zinc-800 overflow-hidden">
            <div
              className="h-full rounded-full bg-gradient-to-r from-sky-500 to-teal-400 transition-all duration-500"
              style={{ width: `${Math.min(100, scorecard.execution_readiness_score * 10)}%` }}
            />
          </div>
          <div className="mt-3 grid grid-cols-2 gap-2 text-[11px] text-zinc-400">
            <div>Stage: <span className="font-semibold text-zinc-200">{tactical?.price_stage || 'N/A'}</span></div>
            <div>Tactical R:R: <span className="font-semibold text-zinc-200">{tactical?.tactical_risk_reward_ratio ? `${tactical.tactical_risk_reward_ratio}:1` : 'N/A'}</span></div>
          </div>
        </div>
      </div>

      {/* 4 Pillars Summary Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 my-4">
        {/* Pillar 1: Piotroski */}
        <div className="rounded-xl border border-zinc-800/60 bg-zinc-900/30 p-3.5">
          <div className="text-[11px] font-medium text-zinc-400">Piotroski F-Score</div>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span className="text-xl font-bold text-zinc-100">
              {piotroski?.is_eligible && piotroski?.f_score !== null && piotroski?.f_score !== undefined ? `${piotroski.f_score} / 9` : 'N/A'}
            </span>
            <span className="text-[10px] uppercase text-zinc-500">
              {piotroski?.status === 'not_applicable' ? 'Exempt' : piotroski?.status}
            </span>
          </div>
          <p className="mt-1 text-[11px] text-zinc-400 leading-tight">
            {piotroski?.exclusion_reason || `Prof: ${piotroski?.profitability_points || 0}/4 | Lev: ${piotroski?.leverage_liquidity_points || 0}/3`}
          </p>
        </div>

        {/* Pillar 2: Reverse DCF */}
        <div className="rounded-xl border border-zinc-800/60 bg-zinc-900/30 p-3.5">
          <div className="text-[11px] font-medium text-zinc-400">12M Target & Implied Growth</div>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span className="text-xl font-bold text-zinc-100">
              {reverseDcf?.target_price_12m ? `$${reverseDcf.target_price_12m}` : 'N/A'}
            </span>
            {reverseDcf?.upside_12m_pct !== undefined && reverseDcf?.upside_12m_pct !== null && (
              <span className={`text-xs font-semibold ${reverseDcf.upside_12m_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                {reverseDcf.upside_12m_pct >= 0 ? `+${reverseDcf.upside_12m_pct}%` : `${reverseDcf.upside_12m_pct}%`}
              </span>
            )}
          </div>
          <p className="mt-1 text-[11px] text-zinc-400 leading-tight">
            Market Implied Growth: <span className="font-semibold text-zinc-200">{reverseDcf?.market_implied_growth_pct !== null && reverseDcf?.market_implied_growth_pct !== undefined ? `${reverseDcf.market_implied_growth_pct}%` : 'N/A'}</span>
          </p>
        </div>

        {/* Pillar 3: Tactical Setup */}
        <div className="rounded-xl border border-zinc-800/60 bg-zinc-900/30 p-3.5">
          <div className="text-[11px] font-medium text-zinc-400">Optimal Buy Zone</div>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span className="text-base font-bold text-zinc-100">
              {tactical?.buy_zone_min && tactical?.buy_zone_max ? `$${tactical.buy_zone_min} - $${tactical.buy_zone_max}` : 'N/A'}
            </span>
          </div>
          <p className="mt-1 text-[11px] text-zinc-400 leading-tight">
            Stop: <span className="font-semibold text-rose-400">${tactical?.invalidation_stop_loss || 'N/A'}</span> | Target: <span className="font-semibold text-emerald-400">${tactical?.tactical_target_price || 'N/A'}</span>
          </p>
        </div>

        {/* Pillar 4: Form 4 Insider */}
        <div className="rounded-xl border border-zinc-800/60 bg-zinc-900/30 p-3.5">
          <div className="text-[11px] font-medium text-zinc-400">SEC Form 4 Conviction</div>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span className="text-base font-bold text-zinc-100 uppercase">
              {insider?.status.replace('_', ' ') || 'N/A'}
            </span>
          </div>
          <p className="mt-1 text-[11px] text-zinc-400 leading-tight">
            Code P Buys: <span className="font-semibold text-zinc-200">{insider?.open_market_p_count_90d || 0}</span> (C-Suite: {insider?.c_suite_p_count || 0})
          </p>
        </div>
      </div>

      {/* Thesis Falsifiers (Kill-Switches) */}
      {falsifiers && falsifiers.length > 0 && (
        <div className="mt-6 rounded-xl border border-rose-900/30 bg-rose-950/10 p-4">
          <div className="flex items-center gap-2 text-rose-400 text-xs font-bold uppercase tracking-wider">
            <span>🛑 Thesis Falsifiers & Invalidation Criteria</span>
          </div>
          <div className="mt-3 space-y-2.5">
            {falsifiers.map((f) => (
              <div key={f.falsifier_id} className="flex flex-col sm:flex-row sm:items-baseline justify-between gap-2 border-b border-rose-900/20 pb-2 last:border-0 last:pb-0">
                <div>
                  <div className="text-xs font-semibold text-zinc-200">
                    {f.metric_name}: <span className="text-rose-300 font-normal">{f.condition}</span>
                  </div>
                  <div className="text-[11px] text-zinc-400 mt-0.5">{f.narrative_explanation}</div>
                </div>
                {f.source_ref && (
                  <div className="text-[10px] font-mono text-zinc-500 shrink-0">
                    Ref: {f.source_ref}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Manifest-Level Evidence Provenance (Phase 0-5) */}
      <EvidenceProvenanceDrawer snapshot={evidenceSnapshot} />
    </div>
  )
}
