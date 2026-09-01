import React, { useState } from 'react'
import { parseDataQualityFlag, type FlagCategory, type FormattedFlag } from '../../lib/dataQualityFlags'

interface DataQualityFlagsCardProps {
  flags: string[]
}

const CATEGORY_DOT: Record<FlagCategory, string> = {
  critical_fallback: 'bg-rose-500 shadow-[0_0_6px_rgba(244,63,94,0.5)]',
  model_methodology: 'bg-sky-500 shadow-[0_0_6px_rgba(14,165,233,0.5)]',
  market_warning: 'bg-amber-500 shadow-[0_0_6px_rgba(245,158,11,0.5)]',
  low_impact_fallback: 'bg-zinc-400',
}

const CATEGORY_ACCENT: Record<
  FlagCategory,
  { borderL: string; bg: string; badgeClass: string; shortLabel: string }
> = {
  critical_fallback: {
    borderL: 'border-l-rose-500',
    bg: 'bg-rose-50/40 hover:bg-rose-50/70',
    badgeClass: 'bg-rose-50 text-rose-700 border-rose-200',
    shortLabel: 'Fallback',
  },
  model_methodology: {
    borderL: 'border-l-sky-500',
    bg: 'bg-sky-50/40 hover:bg-sky-50/70',
    badgeClass: 'bg-sky-50 text-sky-700 border-sky-200',
    shortLabel: 'Methodology',
  },
  market_warning: {
    borderL: 'border-l-amber-500',
    bg: 'bg-amber-50/40 hover:bg-amber-50/70',
    badgeClass: 'bg-amber-50 text-amber-700 border-amber-200',
    shortLabel: 'Market Context',
  },
  low_impact_fallback: {
    borderL: 'border-l-zinc-400',
    bg: 'bg-surface hover:bg-surface-strong',
    badgeClass: 'bg-zinc-100 text-zinc-700 border-zinc-200',
    shortLabel: 'Note',
  },
}

export const DataQualityFlagsCard: React.FC<DataQualityFlagsCardProps> = ({ flags }) => {
  const [isExpanded, setIsExpanded] = useState(true)

  if (!flags || flags.length === 0) return null

  const parsedFlags = flags.map(parseDataQualityFlag)

  // Group by category while preserving priority order
  const categoryOrder: FlagCategory[] = ['critical_fallback', 'model_methodology', 'market_warning', 'low_impact_fallback']
  const grouped = categoryOrder.reduce<Record<FlagCategory, FormattedFlag[]>>((acc, cat) => {
    acc[cat] = parsedFlags.filter((f) => f.category === cat)
    return acc
  }, { critical_fallback: [], model_methodology: [], market_warning: [], low_impact_fallback: [] })

  const activeCategories = categoryOrder.filter((cat) => grouped[cat].length > 0)

  return (
    <section className="flow-panel rounded-2xl border border-edge/80 p-4 shadow-sm transition-all animate-card-in">
      {/* Header bar with summary pill counters and collapse button */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2.5">
          <div className="flex items-center gap-2">
            <span className="text-base">🛡️</span>
            <div>
              <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-900 flex items-center gap-2">
                <span>Data Quality & Model Assumptions</span>
                <span className="px-2 py-0.2 rounded-full bg-surface border border-edge text-[10px] font-bold text-zinc-600">
                  {flags.length}
                </span>
              </h3>
            </div>
          </div>

          {/* Mini Status Chips */}
          <div className="flex flex-wrap items-center gap-1.5 ml-1">
            {activeCategories.map((cat) => {
              const count = grouped[cat].length
              const cfg = CATEGORY_ACCENT[cat]
              return (
                <span
                  key={cat}
                  className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full border text-[10px] font-semibold ${cfg.badgeClass}`}
                >
                  <span className={`w-1.5 h-1.5 rounded-full ${CATEGORY_DOT[cat]}`} />
                  <span>
                    {count} {cfg.shortLabel}
                  </span>
                </span>
              )
            })}
          </div>
        </div>

        {/* Expand / Collapse Button */}
        <button
          onClick={() => setIsExpanded((prev) => !prev)}
          className="text-xs font-semibold text-sky-700 hover:text-sky-800 bg-surface hover:bg-surface-strong px-2.5 py-1 rounded-xl border border-edge flex items-center gap-1.5 transition-colors shadow-2xs"
          aria-expanded={isExpanded}
        >
          <span>{isExpanded ? 'ย่อแถบ' : 'ขยายดูรายละเอียด'}</span>
          <span className="text-[10px]">{isExpanded ? '▲' : '▼'}</span>
        </button>
      </div>

      {/* Compact Grid of Quality Flags */}
      {isExpanded && (
        <div className="mt-3.5 pt-3 border-t border-edge/60 grid grid-cols-1 md:grid-cols-2 gap-2.5 animate-card-in">
          {categoryOrder.map((cat) => {
            const items = grouped[cat]
            if (items.length === 0) return null
            const accent = CATEGORY_ACCENT[cat]

            return items.map((item, idx) => (
              <div
                key={`${cat}-${idx}`}
                className={`rounded-xl border border-edge/60 border-l-[3px] ${accent.borderL} ${accent.bg} p-2.5 transition-all shadow-2xs flex flex-col justify-between`}
                title={item.rawCode}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex items-center gap-1.5">
                    <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${CATEGORY_DOT[cat]}`} />
                    <span className="text-xs font-semibold text-zinc-900 leading-snug">
                      {item.label}
                    </span>
                  </div>
                  <span className={`px-1.5 py-0.2 rounded text-[9px] font-bold border shrink-0 ${accent.badgeClass}`}>
                    {accent.shortLabel}
                  </span>
                </div>

                {item.subtext && (
                  <p className="text-[11px] text-zinc-500 mt-1 leading-relaxed pl-3">
                    {item.subtext}
                  </p>
                )}
              </div>
            ))
          })}
        </div>
      )}
    </section>
  )
}
