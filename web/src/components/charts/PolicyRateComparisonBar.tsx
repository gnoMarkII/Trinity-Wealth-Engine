import React from 'react'

export interface PolicyRateItemUI {
  country: string
  rateValue: number
  rateType: string
  effectiveDate: string
  previousRate?: number
  currency?: string
}

export interface PolicyRateComparisonBarProps {
  rates: PolicyRateItemUI[]
  spreadsVsBotRepo?: Record<string, number> // in bps
  botRepoRate?: number
  title?: string
  subtitle?: string
  className?: string
}

const COUNTRY_NAMES: Record<string, { name: string; flag: string }> = {
  US: { name: 'United States', flag: '🇺🇸' },
  XM: { name: 'Eurozone (ECB)', flag: '🇪🇺' },
  JP: { name: 'Japan', flag: '🇯🇵' },
  GB: { name: 'United Kingdom', flag: '🇬🇧' },
  TH: { name: 'Thailand (BOT)', flag: '🇹🇭' },
  CN: { name: 'China (PBOC)', flag: '🇨🇳' },
  AU: { name: 'Australia (RBA)', flag: '🇦🇺' },
  CA: { name: 'Canada (BOC)', flag: '🇨🇦' },
  CH: { name: 'Switzerland (SNB)', flag: '🇨🇭' },
  KR: { name: 'South Korea (BOK)', flag: '🇰🇷' },
  IN: { name: 'India (RBI)', flag: '🇮🇳' },
  BR: { name: 'Brazil (BCB)', flag: '🇧🇷' },
}

export const PolicyRateComparisonBar: React.FC<PolicyRateComparisonBarProps> = ({
  rates = [],
  spreadsVsBotRepo = {},
  botRepoRate = 2.5,
  title = 'Global Central Bank Policy Rates',
  subtitle = 'Bank for International Settlements (BIS) • 12 Global Jurisdictions',
  className = '',
}) => {
  // Normalize items to support both camelCase and snake_case from API
  const normalizedRates = React.useMemo(() => {
    return (rates || []).map((r: any) => {
      const val = typeof r.rateValue === 'number' ? r.rateValue : typeof r.rate_value === 'number' ? r.rate_value : 0
      return {
        country: r.country,
        rateValue: val,
        rateType: r.rateType || r.rate_type || 'Central Bank Policy Rate',
        effectiveDate: r.effectiveDate || r.effective_date || '',
        previousRate: r.previousRate ?? r.previous_rate,
        currency: r.currency,
      }
    })
  }, [rates])

  // Sort descending by rateValue
  const sorted = React.useMemo(() => {
    return [...normalizedRates].sort((a, b) => b.rateValue - a.rateValue)
  }, [normalizedRates])

  const maxRate = Math.max(...normalizedRates.map((r) => r.rateValue), 6.0)

  return (
    <div
      className={`rounded-2xl border border-sky-100 bg-white/80 p-5 shadow-[0_8px_25px_rgba(14,165,233,0.06)] backdrop-blur-md ${className}`}
    >
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-sky-100/60 pb-3">
        <div>
          <h3 className="font-semibold text-zinc-900 tracking-tight">{title}</h3>
          <p className="text-xs text-zinc-500 mt-0.5">{subtitle}</p>
        </div>

        <div className="rounded-lg bg-sky-50 px-3 py-1.5 text-right">
          <span className="block text-[10px] uppercase font-bold text-sky-700">BOT Benchmark</span>
          <span className="font-mono text-xs font-extrabold text-sky-800">
            {typeof botRepoRate === 'number' ? botRepoRate.toFixed(2) : '2.50'}% (1D Repo)
          </span>
        </div>
      </div>

      {/* Semantic Guidance on Rate Types & Effective Dates */}
      <div className="mt-3 flex items-start gap-2 rounded-lg bg-sky-50/50 p-2 text-[11px] text-sky-800 border border-sky-200/50">
        <span className="font-bold text-sky-700 shrink-0">ℹ BIS Contract Note:</span>
        <span>
          Policy rates are nation-specific benchmark instruments (e.g. US Fed Funds Target, ECB Deposit
          Facility, BOJ Call Rate). Effective date is when the rate took legal effect, which may differ
          from policy meeting dates.
        </span>
      </div>

      {/* Bars list */}
      <div className="mt-4 space-y-2.5">
        {sorted.map((item) => {
          const info = COUNTRY_NAMES[item.country] || { name: item.country, flag: '🌐' }
          const widthPct = Math.max(2, (item.rateValue / maxRate) * 100)
          const isThailand = item.country === 'TH'
          const rawSpread = spreadsVsBotRepo[item.country] ?? (item.rateValue - botRepoRate) * 100
          const spreadBps = typeof rawSpread === 'number' && !isNaN(rawSpread) ? rawSpread : 0

          return (
            <div
              key={item.country}
              className={`rounded-xl p-2.5 transition-all ${
                isThailand
                  ? 'bg-sky-50/90 ring-1 ring-sky-300 shadow-sm'
                  : 'bg-slate-50/70 hover:bg-slate-100/80'
              }`}
            >
              <div className="flex items-center justify-between text-xs mb-1">
                <div className="flex items-center gap-2">
                  <span className="text-base">{info.flag}</span>
                  <span className="font-semibold text-zinc-800">
                    {info.name} <span className="font-mono text-[11px] text-zinc-400">({item.country})</span>
                  </span>
                  {isThailand && (
                    <span className="rounded bg-sky-600 px-1.5 py-0.2 font-mono text-[10px] font-bold text-white">
                      LOCAL
                    </span>
                  )}
                </div>

                <div className="flex items-center gap-3">
                  {/* Spread vs BOT */}
                  {!isThailand && (
                    <span
                      className={`font-mono text-[11px] font-bold ${
                        spreadBps > 0 ? 'text-amber-600' : spreadBps < 0 ? 'text-blue-600' : 'text-zinc-500'
                      }`}
                    >
                      {spreadBps > 0 ? `+${spreadBps.toFixed(0)}` : spreadBps.toFixed(0)} bps vs BOT
                    </span>
                  )}

                  {/* Rate Value */}
                  <span className="font-mono text-sm font-extrabold text-zinc-900 w-14 text-right">
                    {item.rateValue.toFixed(2)}%
                  </span>
                </div>
              </div>

              {/* Progress Bar Track */}
              <div className="h-2 w-full overflow-hidden rounded-full bg-slate-200/70">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${
                    isThailand
                      ? 'bg-sky-600'
                      : spreadBps > 0
                      ? 'bg-amber-500'
                      : 'bg-indigo-500'
                  }`}
                  style={{ width: `${widthPct}%` }}
                />
              </div>

              {/* Metadata row: Rate type & Effective date */}
              <div className="mt-1 flex items-center justify-between text-[10px] text-zinc-400">
                <span className="truncate max-w-[280px]">{item.rateType}</span>
                <span className="font-mono">Effective: {item.effectiveDate}</span>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
