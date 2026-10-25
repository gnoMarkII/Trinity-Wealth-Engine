import { useState } from 'react'
import type { CandidateEvaluationResultDTO } from '../../api/types'

interface Props {
  result: CandidateEvaluationResultDTO
  onAddToWatchlist: (result: CandidateEvaluationResultDTO) => void
  disabled?: boolean
}

export default function OpportunityCandidateCard({ result, onAddToWatchlist, disabled = false }: Props) {
  const [expanded, setExpanded] = useState(false)
  const { candidate } = result

  // Priority Styles
  const priorityBadge = () => {
    switch (result.research_priority) {
      case 1:
        return (
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-extrabold bg-gradient-to-r from-emerald-500 to-teal-600 text-white shadow-xs">
            <span>⭐</span> P1 Top Priority
          </span>
        )
      case 2:
        return (
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-bold bg-sky-100 text-sky-800 border border-sky-200">
            P2 Secondary
          </span>
        )
      case 3:
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-zinc-100 text-zinc-600">
            P3 Monitor
          </span>
        )
      default:
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-rose-50 text-rose-700 border border-rose-200">
            P4 Excluded / Review
          </span>
        )
    }
  }

  // 1. Macro Alignment Badge
  const macroBadge = () => {
    switch (result.macro_alignment) {
      case 'tailwind':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[11px] font-bold bg-emerald-50 text-emerald-700 border border-emerald-200">
            <span>🍃</span> Tailwind {result.macro_stage ? `(${result.macro_stage})` : ''}
          </span>
        )
      case 'headwind':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[11px] font-bold bg-rose-50 text-rose-700 border border-rose-200">
            <span>⚠️</span> Headwind {result.macro_stage ? `(${result.macro_stage})` : ''}
          </span>
        )
      case 'neutral':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[11px] font-medium bg-zinc-100 text-zinc-600">
            <span>⚪</span> Neutral
          </span>
        )
      default:
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[11px] font-medium bg-zinc-50 text-zinc-400">
            Macro N/A
          </span>
        )
    }
  }

  // 2. Axis Policy Fit Badge
  const policyBadge = () => {
    switch (result.policy_fit) {
      case 'pass':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[11px] font-bold bg-emerald-50 text-emerald-700 border border-emerald-200">
            <span>✓</span> Axis: Pass
          </span>
        )
      case 'needs_review':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[11px] font-bold bg-amber-50 text-amber-800 border border-amber-200">
            <span>🔍</span> Axis: Review
          </span>
        )
      default:
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[11px] font-bold bg-rose-50 text-rose-700 border border-rose-200">
            <span>✕</span> Axis: Excluded
          </span>
        )
    }
  }

  // 3. Capacity Status Badge
  const capacityBadge = () => {
    switch (result.capacity_status) {
      case 'available':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[11px] font-bold bg-emerald-50 text-emerald-800 border border-emerald-200">
            <span>💰</span> Room: {result.remaining_capacity_thb ? `฿${Math.round(result.remaining_capacity_thb).toLocaleString()}` : 'Available'}
          </span>
        )
      case 'constrained':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[11px] font-bold bg-rose-50 text-rose-700 border border-rose-200">
            <span>⛔</span> Capped / Full
          </span>
        )
      case 'unknown':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[11px] font-bold bg-amber-50 text-amber-800 border border-amber-200">
            <span>❓</span> Room Unknown
          </span>
        )
      default:
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[11px] font-medium bg-zinc-50 text-zinc-400">
            Room N/A
          </span>
        )
    }
  }

  const normalizedScore =
    result.composite_score !== null && result.composite_score !== undefined
      ? result.composite_score <= 10.0 && result.composite_score > 0
        ? result.composite_score * 10
        : result.composite_score
      : null

  return (
    <div className={`rounded-2xl border transition-all duration-200 bg-white ${
      result.is_actionable
        ? 'border-emerald-200 shadow-sm hover:shadow-md'
        : result.policy_fit === 'fail'
        ? 'border-zinc-200 bg-zinc-50/50 opacity-80'
        : 'border-zinc-200 shadow-xs hover:shadow-sm'
    }`}>
      <div className="p-4 sm:p-5 space-y-4">
        {/* Header Row */}
        <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-3">
          <div className="space-y-1">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="font-mono text-base font-extrabold text-zinc-900 tracking-tight">
                {candidate.symbol}
              </span>
              <span className="text-xs px-2 py-0.5 rounded bg-zinc-100 font-semibold text-zinc-600 font-mono">
                {candidate.market}
              </span>
              <span className="text-xs px-2 py-0.5 rounded bg-sky-50 font-semibold text-sky-700">
                {candidate.sector_name_th || candidate.sector_name}
              </span>
              {priorityBadge()}
            </div>
            <div className="text-xs text-zinc-500 font-medium truncate max-w-md">
              {candidate.company_name}
            </div>
          </div>

          {/* Action Button & Badges */}
          <div className="flex items-center gap-2 shrink-0 self-end sm:self-start">
            {candidate.already_directly_held && (
              <span className="px-2.5 py-1 rounded-lg text-xs font-bold bg-blue-50 text-blue-700 border border-blue-200">
                ✓ ถือตรงแล้ว
              </span>
            )}
            {candidate.already_watchlisted ? (
              <span className="px-2.5 py-1 rounded-lg text-xs font-bold bg-amber-50 text-amber-700 border border-amber-200">
                ★ ใน Watchlist
              </span>
            ) : (
              <button
                type="button"
                onClick={() => onAddToWatchlist(result)}
                disabled={disabled}
                className="px-3 py-1.5 rounded-xl text-xs font-bold bg-flow-blue text-white shadow-xs hover:bg-sky-600 active:scale-95 transition-all flex items-center gap-1.5 disabled:opacity-50"
              >
                <span>+</span> Add to Watchlist
              </button>
            )}
          </div>
        </div>

        {/* 3D Evaluation Badges */}
        <div className="flex flex-wrap items-center gap-2 pt-1 border-t border-zinc-100">
          <span className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider">3D Status:</span>
          {macroBadge()}
          {policyBadge()}
          {capacityBadge()}
          {result.suggested_bucket_id && (
            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[11px] font-bold bg-indigo-50 text-indigo-700 border border-indigo-200">
              🎯 Bucket: {result.suggested_bucket_id}
            </span>
          )}
        </div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 p-3 rounded-xl bg-zinc-50/80 border border-zinc-100 text-xs">
          <div>
            <div className="text-[11px] text-zinc-400">Micro Composite Score</div>
            <div
              className={`font-mono font-bold text-sm ${
                normalizedScore !== null
                  ? normalizedScore >= 70
                    ? 'text-emerald-600'
                    : normalizedScore >= 50
                    ? 'text-amber-600'
                    : 'text-rose-600'
                  : 'text-zinc-800'
              }`}
            >
              {normalizedScore !== null ? `${normalizedScore.toFixed(1)} / 100` : '-'}
            </div>
          </div>
          <div>
            <div className="text-[11px] text-zinc-400">Valuation Upside</div>
            <div className={`font-mono font-bold text-sm ${
              (result.valuation_upside_pct ?? 0) > 0 ? 'text-emerald-600' : 'text-zinc-700'
            }`}>
              {result.valuation_upside_pct !== null && result.valuation_upside_pct !== undefined
                ? `${result.valuation_upside_pct > 0 ? '+' : ''}${result.valuation_upside_pct.toFixed(1)}%`
                : '-'}
            </div>
          </div>
          <div>
            <div className="text-[11px] text-zinc-400">Target Price</div>
            <div className="font-mono font-bold text-zinc-800 text-sm">
              {result.target_price !== null && result.target_price !== undefined
                ? `${candidate.currency === 'USD' ? '$' : '฿'}${result.target_price.toFixed(2)}`
                : '-'}
            </div>
          </div>
          <div>
            <div className="text-[11px] text-zinc-400">Economic Moat</div>
            <div className="font-semibold text-zinc-700 text-sm">
              {result.moat_rating || 'None / N/A'}
            </div>
          </div>
        </div>

        {/* Current Exposure Summary (if held directly or indirectly) */}
        {(candidate.current_total_exposure_thb > 0) && (
          <div className="flex items-center justify-between text-xs px-3 py-2 rounded-lg bg-sky-50/60 border border-sky-100 text-zinc-700">
            <span className="flex items-center gap-1.5 font-medium">
              <span>🔍</span> ปัจจุบันมี exposure รวมในพอร์ต:
            </span>
            <span className="font-mono font-bold text-flow-blue">
              ฿{Math.round(candidate.current_total_exposure_thb).toLocaleString()} ({candidate.current_total_weight_pct.toFixed(2)}%)
              {candidate.current_indirect_value_thb > 0 && (
                <span className="text-zinc-500 font-normal ml-1 text-[11px]">
                  (ตรง ฿{Math.round(candidate.current_direct_value_thb).toLocaleString()} / กองทุน ฿{Math.round(candidate.current_indirect_value_thb).toLocaleString()})
                </span>
              )}
            </span>
          </div>
        )}

        {/* Expandable Reasons Drawer */}
        <div>
          <button
            type="button"
            onClick={() => setExpanded(!expanded)}
            className="text-xs font-semibold text-flow-blue hover:text-sky-700 flex items-center gap-1 transition-colors"
          >
            <span>{expanded ? '▼' : '▶'}</span>
            <span>{expanded ? 'ซ่อนรายละเอียดการวิเคราะห์' : 'ดูรายละเอียดเหตุผล & หลักฐานประเมิน 3 มิติ'}</span>
          </button>

          {expanded && (
            <div className="mt-3 p-3.5 rounded-xl bg-zinc-100/70 border border-zinc-200/80 space-y-2.5 text-xs text-zinc-700 animate-fade-in">
              {result.priority_reasons.length > 0 && (
                <div>
                  <div className="font-bold text-zinc-800 text-[11px] uppercase tracking-wide">🏆 Priority Rationale:</div>
                  <ul className="list-disc list-inside space-y-0.5 text-zinc-600 mt-0.5 pl-1">
                    {result.priority_reasons.map((r, i) => (
                      <li key={i}>{r}</li>
                    ))}
                  </ul>
                </div>
              )}

              {result.policy_reasons.length > 0 && (
                <div>
                  <div className="font-bold text-zinc-800 text-[11px] uppercase tracking-wide">📋 Investor Axis / Principles:</div>
                  <ul className="list-disc list-inside space-y-0.5 text-zinc-600 mt-0.5 pl-1">
                    {result.policy_reasons.map((r, i) => (
                      <li key={i}>{r}</li>
                    ))}
                  </ul>
                </div>
              )}

              {result.macro_reasons.length > 0 && (
                <div>
                  <div className="font-bold text-zinc-800 text-[11px] uppercase tracking-wide">🌐 Macro Alignment:</div>
                  <ul className="list-disc list-inside space-y-0.5 text-zinc-600 mt-0.5 pl-1">
                    {result.macro_reasons.map((r, i) => (
                      <li key={i}>{r}</li>
                    ))}
                  </ul>
                </div>
              )}

              {result.capacity_reasons.length > 0 && (
                <div>
                  <div className="font-bold text-zinc-800 text-[11px] uppercase tracking-wide">⚖️ Look-Through Capacity:</div>
                  <ul className="list-disc list-inside space-y-0.5 text-zinc-600 mt-0.5 pl-1">
                    {result.capacity_reasons.map((r, i) => (
                      <li key={i}>{r}</li>
                    ))}
                  </ul>
                </div>
              )}

              {candidate.source_refs.length > 0 && (
                <div className="pt-1 border-t border-zinc-200 text-[11px] text-zinc-500 font-mono">
                  Sources: {candidate.source_refs.join(', ')}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
