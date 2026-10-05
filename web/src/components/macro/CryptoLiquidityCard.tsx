import React from 'react'
import type { CryptoMacroLiquidityDTO } from '../../api/types'

interface CryptoLiquidityCardProps {
  data: CryptoMacroLiquidityDTO | null
  loading?: boolean
  error?: string | null
  className?: string
}

export const CryptoLiquidityCard: React.FC<CryptoLiquidityCardProps> = ({
  data,
  loading = false,
  error = null,
  className = '',
}) => {
  if (loading) {
    return (
      <div className={`rounded-2xl border border-sky-100 bg-white/80 p-5 shadow-[0_8px_25px_rgba(14,165,233,0.06)] backdrop-blur-md animate-pulse ${className}`}>
        <div className="h-5 w-48 bg-sky-100/80 rounded mb-4" />
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="h-24 bg-slate-100/80 rounded-xl border border-slate-100" />
          ))}
        </div>
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className={`rounded-xl border border-amber-200 bg-amber-50 p-5 text-xs text-amber-900 ${className}`} role="status">
        Crypto Macro Liquidity: {error || 'ไม่มีข้อมูลสภาพคล่องสินทรัพย์ดิจิทัล'}
      </div>
    )
  }

  const isExpanding = data.liquidity_regime?.toLowerCase().includes('expan')
  const isContracting = data.liquidity_regime?.toLowerCase().includes('contract')

  const regimeBadge = isExpanding
    ? { bg: 'bg-emerald-50', text: 'text-emerald-700', border: 'border-emerald-200', label: data.liquidity_regime }
    : isContracting
    ? { bg: 'bg-rose-50', text: 'text-rose-700', border: 'border-rose-200', label: data.liquidity_regime }
    : { bg: 'bg-sky-50', text: 'text-sky-700', border: 'border-sky-200', label: data.liquidity_regime || 'Neutral' }

  const stableTotalB = data.stablecoin_total_usd ? (data.stablecoin_total_usd / 1e9).toFixed(1) : '—'
  const etfFlowM = data.etf_daily_net_inflow_usd != null ? (data.etf_daily_net_inflow_usd / 1e6).toFixed(1) : '—'
  const isEtfPositive = (data.etf_daily_net_inflow_usd ?? 0) >= 0

  return (
    <div className={`rounded-2xl border border-sky-100 bg-white/80 p-5 shadow-[0_8px_25px_rgba(14,165,233,0.06)] backdrop-blur-md ${className}`}>
      {/* Header */}
      <div className="mb-4 flex flex-wrap items-baseline justify-between gap-2 border-b border-sky-100/70 pb-3">
        <div>
          <h3 className="text-base font-bold text-zinc-900 tracking-tight flex items-center gap-2">
            <span>สภาพคล่องโลกและสินทรัพย์ดิจิทัล (Crypto Macro Liquidity Radar)</span>
            <span className="rounded-full bg-sky-100 text-sky-800 font-mono text-[10px] font-bold px-2.5 py-0.5">
              Level 1 Proxy
            </span>
          </h3>
          <p className="text-xs text-zinc-500">
            ตรวจวัดสภาพคล่องเงินสดในระบบ (Dry Powder) และความกล้าเสี่ยงของตลาดโลก ผ่าน Stablecoin Supply, BTC/Gold Ratio และ ETF Flow
          </p>
        </div>

        <div className="flex items-center gap-2">
          <span className={`rounded-md border px-2.5 py-1 text-xs font-bold uppercase tracking-wider ${regimeBadge.bg} ${regimeBadge.text} ${regimeBadge.border}`}>
            {regimeBadge.label}
          </span>
        </div>
      </div>

      <p className="text-[11px] text-zinc-500 mb-3">
        ข้อมูล ณ {data.as_of_date} • {data.source}
      </p>
      {data.is_stale && (
        <p role="status" className="text-xs text-amber-700 font-medium mb-3">
          ข้อมูลบางส่วนล้าสมัย: {data.stale_reason || 'รอข้อมูลล่าสุดจากผู้ให้บริการ'}
        </p>
      )}
      {data.limitations && <p className="text-[11px] text-zinc-400 mb-3">{data.limitations}</p>}

      {/* 4 Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
        {/* 1. Stablecoins Total */}
        <div className="rounded-xl border border-slate-100 bg-slate-50/80 p-4 transition-all hover:bg-white hover:shadow-xs hover:border-sky-200">
          <div className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">
            Total USD Stablecoins
          </div>
          <div className="mt-1 flex items-baseline gap-2">
            <span className="font-mono text-xl font-bold text-zinc-900">
              ${stableTotalB}B
            </span>
            {data.stablecoin_change_30d_pct !== null && data.stablecoin_change_30d_pct !== undefined && (
              <span className={`font-mono text-xs font-semibold ${data.stablecoin_change_30d_pct >= 0 ? 'text-emerald-600' : 'text-rose-600'}`}>
                {data.stablecoin_change_30d_pct >= 0 ? '+' : ''}{data.stablecoin_change_30d_pct.toFixed(2)}% (30d)
              </span>
            )}
          </div>
          <div className="mt-1 text-[11px] text-zinc-400">
            7d: {data.stablecoin_change_7d_pct !== null && data.stablecoin_change_7d_pct !== undefined ? `${data.stablecoin_change_7d_pct >= 0 ? '+' : ''}${data.stablecoin_change_7d_pct.toFixed(2)}%` : '—'}
          </div>
        </div>

        {/* 2. Bitcoin Spot Benchmark */}
        <div className="rounded-xl border border-slate-100 bg-slate-50/80 p-4 transition-all hover:bg-white hover:shadow-xs hover:border-sky-200">
          <div className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">
            Bitcoin Spot Benchmark
          </div>
          <div className="mt-1 flex items-baseline gap-2">
            <span className="font-mono text-xl font-bold text-zinc-900">
              ${data.btc_price_usd ? data.btc_price_usd.toLocaleString(undefined, { maximumFractionDigits: 0 }) : '—'}
            </span>
            {data.btc_change_24h_pct !== null && data.btc_change_24h_pct !== undefined && (
              <span className={`font-mono text-xs font-semibold ${data.btc_change_24h_pct >= 0 ? 'text-emerald-600' : 'text-rose-600'}`}>
                {data.btc_change_24h_pct >= 0 ? '+' : ''}{data.btc_change_24h_pct.toFixed(2)}%
              </span>
            )}
          </div>
          <div className="mt-1 text-[11px] text-zinc-400">
            7d Return: {data.btc_change_7d_pct !== null && data.btc_change_7d_pct !== undefined ? `${data.btc_change_7d_pct >= 0 ? '+' : ''}${data.btc_change_7d_pct.toFixed(2)}%` : '—'}
          </div>
        </div>

        {/* 3. BTC / Gold Valuation Ratio */}
        <div className="rounded-xl border border-slate-100 bg-slate-50/80 p-4 transition-all hover:bg-white hover:shadow-xs hover:border-sky-200">
          <div className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">
            BTC / Gold Ratio
          </div>
          <div className="mt-1 flex items-baseline gap-2">
            <span className="font-mono text-xl font-bold text-amber-600">
              {data.btc_gold_ratio ? `${data.btc_gold_ratio.toFixed(2)}x` : '—'}
            </span>
            <span className="text-[11px] text-zinc-400">
              (1 BTC in Oz)
            </span>
          </div>
          <div className="mt-1 text-[11px] text-zinc-500">
            Risk-On Appetite Barometer
          </div>
        </div>

        {/* 4. Spot BTC ETF Daily Net Inflow */}
        <div className="rounded-xl border border-slate-100 bg-slate-50/80 p-4 transition-all hover:bg-white hover:shadow-xs hover:border-sky-200">
          <div className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">
            US Spot BTC ETF Inflow
          </div>
          <div className="mt-1 flex items-baseline gap-2">
            <span className={`font-mono text-xl font-bold ${isEtfPositive ? 'text-emerald-600' : 'text-rose-600'}`}>
              {etfFlowM !== '—' ? `${isEtfPositive ? '+' : ''}$${etfFlowM}M` : '—'}
            </span>
          </div>
          <div className="mt-1 text-[11px] text-zinc-400">
            Institutional Net Daily Flow
          </div>
        </div>
      </div>

      {/* Top Stablecoins Share Bar */}
      {data.top_stablecoins && data.top_stablecoins.length > 0 && (
        <div className="mt-4 pt-3 border-t border-sky-100/70">
          <div className="flex items-center justify-between text-xs text-zinc-500 mb-2">
            <span>Market Share สินทรัพย์สภาพคล่อง (Top Stablecoins):</span>
            <span className="font-mono text-[11px] text-zinc-400">DeFiLlama Feed</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {data.top_stablecoins.slice(0, 5).map((s) => (
              <span
                key={s.symbol}
                className="inline-flex items-center gap-1.5 rounded-lg bg-white px-2.5 py-1 text-xs font-mono text-zinc-700 border border-slate-200 shadow-2xs"
              >
                <span className="font-bold text-sky-700">{s.symbol}</span>
                <span className="text-zinc-600">
                  {s.circulating_usd ? `$${(s.circulating_usd / 1e9).toFixed(1)}B` : ''}
                </span>
                {s.market_share_pct && (
                  <span className="text-zinc-400 text-[10px]">
                    ({s.market_share_pct}%)
                  </span>
                )}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
