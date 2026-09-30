import React, { useState } from 'react'
import { SourceProvenanceBadge } from './SourceProvenanceBadge'
import { PolicyRateComparisonBar } from '../../charts/PolicyRateComparisonBar'

interface RateItem {
  country: string
  rate_value?: number
  rateValue?: number
  rate_type?: string
  rateType?: string
  effective_date?: string
  effectiveDate?: string
  previous_rate?: number
  currency?: string
}

interface FedBotRatePairProps {
  ratesData: any
  loading?: boolean
  error?: string | null
  className?: string
}

export const FedBotRatePair: React.FC<FedBotRatePairProps> = ({
  ratesData,
  loading = false,
  error = null,
  className = '',
}) => {
  const [showAllGlobal, setShowAllGlobal] = useState(false)

  if (loading) {
    return (
      <div className={`rounded-2xl border border-sky-100 bg-white/90 p-5 shadow-sm animate-pulse ${className}`}>
        <div className="h-6 w-48 bg-slate-200 rounded mb-4" />
        <div className="h-32 bg-slate-100 rounded-xl" />
      </div>
    )
  }

  if (error || !ratesData || !Array.isArray(ratesData.rates)) {
    return (
      <div className={`rounded-2xl border border-amber-200 bg-amber-50/60 p-5 text-xs text-amber-900 ${className}`}>
        <div className="font-bold flex items-center gap-1.5">
          <span>⚠️</span>
          <span>Central Bank Policy Rates: ไม่สามารถโหลดข้อมูลได้</span>
        </div>
        <p className="mt-1 text-zinc-600">{error || 'ไม่มีข้อมูลอัตราดอกเบี้ยนโยบายจาก BIS'}</p>
      </div>
    )
  }

  const rates: RateItem[] = ratesData.rates || []
  const usRate = rates.find((r) => r.country === 'US')
  const thRate = rates.find((r) => r.country === 'TH')

  const usVal = typeof usRate?.rate_value === 'number' ? usRate.rate_value : typeof usRate?.rateValue === 'number' ? usRate.rateValue : null
  const thVal = typeof thRate?.rate_value === 'number' ? thRate.rate_value : typeof thRate?.rateValue === 'number' ? thRate.rateValue : null

  // Spread from backend dictionary (DO NOT recalculate in UI)
  const spreadVsBot = ratesData.spreads_vs_bot_repo?.['US']
  const effectiveDateUs = usRate?.effective_date || usRate?.effectiveDate || '—'
  const effectiveDateTh = thRate?.effective_date || thRate?.effectiveDate || '—'

  return (
    <div className={`rounded-2xl border border-sky-100 bg-white/90 p-5 shadow-sm ${className}`}>
      {/* Header & Provenance */}
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between border-b border-sky-100/70 pb-3">
        <div>
          <h3 className="font-bold text-zinc-900 text-sm tracking-tight flex items-center gap-2">
            <span>Fed Funds vs BoT 1D Repo (Policy Rate Spread)</span>
          </h3>
          <p className="text-xs text-zinc-500 mt-0.5">
            เปรียบเทียบอัตราดอกเบี้ยนโยบายสหรัฐฯ กับไทย พร้อมส่วนต่างคำนวณจากระบบ
          </p>
        </div>

        <SourceProvenanceBadge
          origin="provider"
          sourceName="BIS (Central Banks)"
          observedAt={ratesData.as_of_date}
          compact
        />
      </div>

      {/* Paired Two-Rate Comparison */}
      <div className="mt-4 grid grid-cols-1 sm:grid-cols-3 gap-3">
        {/* US Fed Rate */}
        <div className="rounded-xl bg-slate-50 p-3.5 border border-slate-100">
          <div className="flex items-center justify-between text-xs mb-1">
            <span className="font-semibold text-zinc-800 flex items-center gap-1.5">
              <span>🇺🇸</span>
              <span>Fed Funds Rate</span>
            </span>
            <span className="font-mono text-[10px] text-zinc-400">US Fed</span>
          </div>
          <div className="font-mono text-2xl font-extrabold text-zinc-900">
            {usVal !== null ? `${usVal.toFixed(2)}%` : 'ไม่มีข้อมูล'}
          </div>
          <div className="mt-1 text-[10px] text-zinc-400">
            {usRate?.rate_type || usRate?.rateType || 'Fed Funds Target Range'}
          </div>
          <div className="text-[10px] text-zinc-400 font-mono">
            มีผล: {effectiveDateUs}
          </div>
        </div>

        {/* Spread Box (Calculated by Python) */}
        <div className="rounded-xl bg-sky-50/80 p-3.5 border border-sky-200/80 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between text-xs mb-1">
              <span className="font-semibold text-sky-900">ส่วนต่าง (US – TH Spread)</span>
              <span className="rounded bg-sky-200/70 px-1.5 py-0.2 font-mono text-[9px] font-bold text-sky-950">
                PYTHON MATH
              </span>
            </div>
            <div className="font-mono text-2xl font-extrabold text-sky-900">
              {spreadVsBot !== undefined && spreadVsBot !== null
                ? `${spreadVsBot > 0 ? '+' : ''}${spreadVsBot.toFixed(0)} bps`
                : 'ไม่มีข้อมูล'}
            </div>
          </div>
          <div className="text-[10px] text-sky-800 mt-2">
            ส่วนต่างผลตอบแทนดึงดูด/กดดันเงินทุนเคลื่อนย้ายระหว่างประเทศ
          </div>
        </div>

        {/* TH BoT Rate */}
        <div className="rounded-xl bg-slate-50 p-3.5 border border-slate-100">
          <div className="flex items-center justify-between text-xs mb-1">
            <span className="font-semibold text-zinc-800 flex items-center gap-1.5">
              <span>🇹🇭</span>
              <span>BoT 1-Day Repo</span>
            </span>
            <span className="font-mono text-[10px] text-zinc-400">ธปท. (BOT)</span>
          </div>
          <div className="font-mono text-2xl font-extrabold text-sky-700">
            {thVal !== null ? `${thVal.toFixed(2)}%` : 'ไม่มีข้อมูล'}
          </div>
          <div className="mt-1 text-[10px] text-zinc-400">
            {thRate?.rate_type || thRate?.rateType || 'BOT 1-Day Bilateral Repo'}
          </div>
          <div className="text-[10px] text-zinc-400 font-mono">
            มีผล: {effectiveDateTh}
          </div>
        </div>
      </div>

      {/* Accordion Toggle for 12 Global Jurisdictions */}
      <div className="mt-4 border-t border-slate-100 pt-3">
        <button
          type="button"
          onClick={() => setShowAllGlobal(!showAllGlobal)}
          className="flex items-center justify-between w-full text-xs font-semibold text-zinc-700 hover:text-sky-700 transition-colors p-1"
        >
          <span className="flex items-center gap-1.5">
            <span>🌐</span>
            <span>{showAllGlobal ? 'ซ่อน' : 'ดู'} อัตราดอกเบี้ยธนาคารกลาง 12 ประเทศทั่วโลก (BIS Board)</span>
          </span>
          <span className="font-mono text-[11px] text-zinc-400">{showAllGlobal ? '▲' : '▼'}</span>
        </button>

        {showAllGlobal && (
          <div className="mt-3 pt-2">
            <PolicyRateComparisonBar
              rates={rates as any}
              spreadsVsBotRepo={ratesData.spreads_vs_bot_repo}
              botRepoRate={thVal ?? undefined}
            />
          </div>
        )}
      </div>
    </div>
  )
}
