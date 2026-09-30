import React from 'react'

export interface EarningsSurpriseUI {
  fiscal_quarter_end: string
  date_reported: string
  eps?: number
  consensus_eps?: number
  surprise_pct?: number
}

export interface AnalystRatingsUI {
  consensus: string
  analyst_count: number
  target_price_mean?: number
  target_price_high?: number
  target_price_low?: number
  broker_names?: string[]
}

export interface UpcomingEarningsUI {
  earnings_date: string
  date_status: 'confirmed' | 'estimated' | 'unspecified'
  fiscal_quarter?: string
  time_of_day?: string
}

export interface NasdaqConsensusCardProps {
  symbol: string
  coverageStatus: 'full' | 'partial' | 'no_coverage'
  hasEarningsSurprise: boolean
  hasAnalystRatings: boolean
  upcomingEarnings?: UpcomingEarningsUI | null
  ratings?: AnalystRatingsUI | null
  surpriseHistory?: EarningsSurpriseUI[]
  className?: string
}

export const NasdaqConsensusCard: React.FC<NasdaqConsensusCardProps> = ({
  symbol,
  coverageStatus,
  hasEarningsSurprise,
  hasAnalystRatings,
  upcomingEarnings,
  ratings,
  surpriseHistory = [],
  className = '',
}) => {
  if (coverageStatus === 'no_coverage') {
    return (
      <div
        className={`rounded-2xl border border-slate-200/80 bg-white/70 p-5 shadow-sm backdrop-blur-md ${className}`}
      >
        <div className="flex items-center justify-between border-b border-slate-100 pb-3">
          <div className="flex items-center gap-2">
            <h3 className="font-semibold text-zinc-900">{symbol} Institutional Consensus</h3>
            <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[10px] font-medium text-zinc-500">
              No Coverage
            </span>
          </div>
          <span className="text-xs text-zinc-400">Nasdaq Intelligence</span>
        </div>
        <div className="py-6 text-center text-xs text-zinc-500">
          No sell-side analyst ratings or earnings surprise history currently reported by Nasdaq for{' '}
          <span className="font-semibold font-mono text-zinc-700">{symbol}</span>.
        </div>
      </div>
    )
  }

  return (
    <div
      className={`rounded-2xl border border-sky-100 bg-white/80 p-5 shadow-[0_8px_25px_rgba(14,165,233,0.06)] backdrop-blur-md ${className}`}
    >
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-sky-100/60 pb-3">
        <div className="flex items-center gap-2">
          <h3 className="font-semibold text-zinc-900 tracking-tight">
            {symbol} Sell-Side Consensus & Earnings
          </h3>
          <span
            className={`rounded-full px-2 py-0.5 font-mono text-[10px] font-bold ${
              coverageStatus === 'full'
                ? 'bg-emerald-50 text-emerald-700'
                : 'bg-amber-50 text-amber-700'
            }`}
          >
            {coverageStatus.toUpperCase()}
          </span>
        </div>
        <span className="font-mono text-xs text-zinc-400">Source: Nasdaq Institutional Feed</span>
      </div>

      {/* Upcoming Earnings Date Banner */}
      {upcomingEarnings && (
        <div className="mt-3 flex items-center justify-between rounded-xl bg-sky-50/80 p-3 border border-sky-100">
          <div>
            <span className="text-[11px] font-bold uppercase tracking-wider text-sky-800">
              Next Earnings Release
            </span>
            <div className="flex items-center gap-2 mt-0.5">
              <span className="font-mono text-sm font-extrabold text-sky-950">
                {upcomingEarnings.earnings_date}
              </span>
              <span
                className={`rounded px-1.5 py-0.2 font-mono text-[10px] font-bold ${
                  upcomingEarnings.date_status === 'confirmed'
                    ? 'bg-emerald-600 text-white'
                    : 'bg-amber-500 text-white'
                }`}
              >
                {upcomingEarnings.date_status.toUpperCase()}
              </span>
            </div>
          </div>
          {upcomingEarnings.time_of_day && (
            <span className="text-xs text-sky-700 font-medium">{upcomingEarnings.time_of_day}</span>
          )}
        </div>
      )}

      {/* Ratings Section */}
      {hasAnalystRatings && ratings && (
        <div className="mt-4 rounded-xl bg-slate-50/70 p-3.5 border border-slate-100">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <span className="text-[10px] uppercase font-bold text-zinc-400">Analyst Consensus</span>
              <div className="flex items-baseline gap-2 mt-0.5">
                <span className="font-mono text-lg font-black text-emerald-600">
                  {ratings.consensus}
                </span>
                <span className="text-xs text-zinc-500">
                  from <span className="font-mono font-semibold">{ratings.analyst_count}</span> analysts
                </span>
              </div>
            </div>

            {ratings.target_price_mean !== undefined && (
              <div className="text-right">
                <span className="text-[10px] uppercase font-bold text-zinc-400">Mean Target Price</span>
                <p className="font-mono text-base font-extrabold text-zinc-900">
                  ${ratings.target_price_mean.toFixed(2)}
                </p>
                {ratings.target_price_low !== undefined && ratings.target_price_high !== undefined && (
                  <span className="font-mono text-[10px] text-zinc-400">
                    Low ${ratings.target_price_low.toFixed(0)} - High ${ratings.target_price_high.toFixed(0)}
                  </span>
                )}
              </div>
            )}
          </div>

          {ratings.broker_names && ratings.broker_names.length > 0 && (
            <div className="mt-2 pt-2 border-t border-slate-200/60 flex flex-wrap gap-1">
              <span className="text-[10px] text-zinc-400 mr-1">Brokers:</span>
              {ratings.broker_names.slice(0, 4).map((name) => (
                <span
                  key={name}
                  className="rounded bg-white px-1.5 py-0.2 text-[10px] font-medium text-zinc-600 border border-slate-200/60"
                >
                  {name}
                </span>
              ))}
            </div>
          )}
        </div>
      )}

      {/* EPS Surprise History Table */}
      {hasEarningsSurprise && surpriseHistory.length > 0 && (
        <div className="mt-4">
          <h4 className="text-xs font-bold uppercase tracking-wider text-zinc-600 mb-2">
            Quarterly EPS Surprises
          </h4>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-slate-100 text-[10px] uppercase font-bold text-zinc-400">
                  <th className="pb-1.5 font-semibold">Quarter</th>
                  <th className="pb-1.5 font-semibold">Reported</th>
                  <th className="pb-1.5 font-semibold text-right">Actual EPS</th>
                  <th className="pb-1.5 font-semibold text-right">Consensus</th>
                  <th className="pb-1.5 font-semibold text-right">Surprise %</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-mono">
                {surpriseHistory.map((s, idx) => {
                  const surprise = s.surprise_pct ?? 0
                  const isPositive = surprise > 0
                  return (
                    <tr key={idx} className="hover:bg-slate-50/60 transition-colors">
                      <td className="py-1.5 text-zinc-700 font-medium">{s.fiscal_quarter_end}</td>
                      <td className="py-1.5 text-zinc-400 text-[11px]">{s.date_reported}</td>
                      <td className="py-1.5 text-right font-bold text-zinc-800">
                        {s.eps !== undefined ? `$${s.eps.toFixed(2)}` : '—'}
                      </td>
                      <td className="py-1.5 text-right text-zinc-500">
                        {s.consensus_eps !== undefined ? `$${s.consensus_eps.toFixed(2)}` : '—'}
                      </td>
                      <td
                        className={`py-1.5 text-right font-bold ${
                          isPositive ? 'text-emerald-600' : surprise < 0 ? 'text-rose-600' : 'text-zinc-500'
                        }`}
                      >
                        {isPositive ? `+${surprise.toFixed(1)}%` : `${surprise.toFixed(1)}%`}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
