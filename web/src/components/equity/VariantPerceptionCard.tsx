import React from 'react'
import type { VariantPerceptionSnapshotDTO } from '../../api/types'

export interface VariantPerceptionCardProps {
  data: VariantPerceptionSnapshotDTO
  isLoading?: boolean
}

export const VariantPerceptionCard: React.FC<VariantPerceptionCardProps> = ({
  data,
  isLoading = false,
}) => {
  if (isLoading) {
    return (
      <div
        data-testid="variant-perception-loading"
        className="bg-slate-900 border border-slate-800 rounded-xl p-6 animate-pulse text-slate-400 space-y-4"
      >
        <div className="h-6 bg-slate-800 rounded w-1/3" />
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="h-24 bg-slate-800 rounded" />
          <div className="h-24 bg-slate-800 rounded" />
          <div className="h-24 bg-slate-800 rounded" />
        </div>
      </div>
    )
  }

  const {
    ticker,
    period,
    metric_name,
    reported_value,
    consensus_value,
    fund_thesis_value,
    street_surprise,
    street_surprise_pct,
    variant_spread,
    variant_spread_pct,
    variant_realization,
    variant_realization_pct,
    alpha_verdict,
    unit,
  } = data

  const numStreetSurprise = Number(street_surprise) || 0
  const numVariantSpread = Number(variant_spread) || 0
  const numRealization = Number(variant_realization) || 0

  const getVerdictBadge = (verdict: string) => {
    switch (verdict) {
      case 'ALPHA_CONFIRMED':
        return (
          <div className="flex flex-col items-end gap-0.5">
            <span
              data-testid="verdict-alpha-confirmed"
              className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
            >
              <span>🎯</span>
              <span>ALPHA CONFIRMED</span>
            </span>
            <span className="text-[10px] text-emerald-400 font-sans">เรามองขาดกว่าตลาด</span>
          </div>
        )
      case 'CONSENSUS_CORRECT':
        return (
          <div className="flex flex-col items-end gap-0.5">
            <span
              data-testid="verdict-consensus-correct"
              className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-sky-500/20 text-sky-300 border border-sky-500/40"
            >
              <span>⚖️</span>
              <span>CONSENSUS ACCURATE</span>
            </span>
            <span className="text-[10px] text-sky-400 font-sans">ตลาดคาดได้แม่นยำกว่า</span>
          </div>
        )
      case 'DOUBLE_MISS':
        return (
          <div className="flex flex-col items-end gap-0.5">
            <span
              data-testid="verdict-double-miss"
              className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-rose-500/20 text-rose-300 border border-rose-500/40"
            >
              <span>🚨</span>
              <span>DOUBLE MISS</span>
            </span>
            <span className="text-[10px] text-rose-400 font-sans">หลุดเป้าทั้งเราและตลาด</span>
          </div>
        )
      default:
        return (
          <div className="flex flex-col items-end gap-0.5">
            <span
              data-testid="verdict-default"
              className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-amber-500/20 text-amber-300 border border-amber-500/40"
            >
              <span>🔎</span>
              <span>{verdict.replace(/_/g, ' ')}</span>
            </span>
            <span className="text-[10px] text-amber-400 font-sans">กำลังติดตามผล</span>
          </div>
        )
    }
  }

  const formatPct = (pctStr: string, num: number) => {
    const prefix = num > 0 ? '+' : ''
    return `${prefix}${pctStr}%`
  }

  return (
    <div
      data-testid="variant-perception-card"
      className="bg-slate-900 border border-slate-800 rounded-xl p-6 text-slate-200 shadow-xl space-y-6"
    >
      {/* 1. Header Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xl font-bold tracking-tight text-white">
              ${ticker}
            </span>
            <span className="text-xs font-mono px-2 py-0.5 rounded bg-slate-800 text-sky-400 border border-slate-700">
              {period}
            </span>
            <span className="text-xs text-slate-300 font-semibold">
              {metric_name.replace(/_/g, ' ').toUpperCase()}
            </span>
          </div>
          <p className="text-xs text-sky-400/90 mt-1 font-medium">
            เปรียบเทียบมุมมอง: ตลาดส่วนใหญ่คิดอย่างไร vs พอร์ตเราคิดอย่างไร (เพื่อหาโอกาสสร้างผลตอบแทนส่วนเกิน)
          </p>
          <p className="text-[11px] text-slate-400 mt-0.5">
            Institutional Two-Sided Estimate Ledger (Chapter 2 &amp; 5)
          </p>
        </div>

        <div className="flex items-center gap-2">
          {getVerdictBadge(alpha_verdict)}
        </div>
      </div>

      {/* 2. Three-Way Core Pillars */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {/* Consensus Pillar */}
        <div className="bg-slate-800/60 border border-slate-700/80 rounded-lg p-4 space-y-1.5">
          <div className="text-[11px] font-semibold text-slate-300 uppercase tracking-wider flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <span>Street Consensus</span>
              <span className="text-[10px] text-slate-400 font-normal normal-case">(มุมมองตลาด)</span>
            </div>
            <span className="text-slate-500 font-mono">C</span>
          </div>
          <div className="text-2xl font-bold font-mono text-white" data-testid="consensus-value">
            {consensus_value}
            <span className="text-xs font-normal text-slate-400 ml-1.5">{unit}</span>
          </div>
          <p className="text-[11px] text-slate-400 leading-snug">
            สิ่งที่นักวิเคราะห์วอลล์สตรีทส่วนใหญ่คาดการณ์ไว้ล่วงหน้า
          </p>
        </div>

        {/* Fund Thesis Pillar */}
        <div className="bg-sky-950/40 border border-sky-800/60 rounded-lg p-4 space-y-1.5">
          <div className="text-[11px] font-semibold text-sky-300 uppercase tracking-wider flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <span>Fund Thesis</span>
              <span className="text-[10px] text-sky-400 font-normal normal-case">(มุมมองพอร์ตเรา)</span>
            </div>
            <span className="text-sky-500 font-mono">E_fund</span>
          </div>
          <div className="text-2xl font-bold font-mono text-sky-200" data-testid="fund-thesis-value">
            {fund_thesis_value}
            <span className="text-xs font-normal text-sky-400 ml-1.5">{unit}</span>
          </div>
          <p className="text-[11px] text-sky-300/80 leading-snug">
            สิ่งที่เราวิเคราะห์เชิงลึกและเชื่อมั่นว่าจะทำได้จริง
          </p>
        </div>

        {/* Reported Actual Pillar */}
        <div className="bg-slate-800/90 border border-emerald-900/50 rounded-lg p-4 space-y-1.5">
          <div className="text-[11px] font-semibold text-emerald-300 uppercase tracking-wider flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <span>Reported Actual</span>
              <span className="text-[10px] text-emerald-400 font-normal normal-case">(ตัวเลขจริง)</span>
            </div>
            <span className="text-emerald-500 font-mono">A</span>
          </div>
          <div className="text-2xl font-bold font-mono text-emerald-300" data-testid="reported-value">
            {reported_value}
            <span className="text-xs font-normal text-emerald-400 ml-1.5">{unit}</span>
          </div>
          <p className="text-[11px] text-emerald-400/80 leading-snug">
            งบการเงินจริงที่บริษัทประกาศอย่างเป็นทางการ (SEC 10-Q / 8-K)
          </p>
        </div>
      </div>

      {/* 3. The 3 Spreads & Differentials */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-2">
        {/* Street Surprise */}
        <div className="rounded-lg border border-slate-800 bg-slate-950/60 p-3 space-y-1.5">
          <div className="flex items-center justify-between text-xs text-slate-300">
            <div>
              <span className="font-medium">Street Surprise</span>
              <span className="text-[10px] text-slate-400 ml-1 font-mono">(A - C)</span>
            </div>
            <span
              className={`font-mono font-bold text-xs ${
                numStreetSurprise >= 0 ? 'text-emerald-400' : 'text-rose-400'
              }`}
            >
              {formatPct(street_surprise_pct, numStreetSurprise)}
            </span>
          </div>
          <div
            data-testid="street-surprise"
            className="font-mono text-sm font-semibold text-slate-200"
          >
            {numStreetSurprise > 0 ? `+${street_surprise}` : street_surprise} {unit}
          </div>
          <p className="text-[11px] text-slate-400 leading-snug">
            {numStreetSurprise >= 0 ? 'ผลงานจริงดีกว่าที่ตลาดคาด' : 'ผลงานจริงต่ำกว่าที่ตลาดคาด'}
          </p>
        </div>

        {/* Variant Spread */}
        <div className="rounded-lg border border-sky-900/40 bg-sky-950/20 p-3 space-y-1.5">
          <div className="flex items-center justify-between text-xs text-sky-200">
            <div>
              <span className="font-medium">Variant Spread</span>
              <span className="text-[10px] text-sky-400 ml-1 font-mono">(E_fund - C)</span>
            </div>
            <span
              className={`font-mono font-bold text-xs ${
                numVariantSpread >= 0 ? 'text-sky-300' : 'text-amber-300'
              }`}
            >
              {formatPct(variant_spread_pct, numVariantSpread)}
            </span>
          </div>
          <div
            data-testid="variant-spread"
            className="font-mono text-sm font-semibold text-sky-200"
          >
            {numVariantSpread > 0 ? `+${variant_spread}` : variant_spread} {unit}
          </div>
          <p className="text-[11px] text-sky-300/80 leading-snug">
            ความมั่นใจของเราที่มองต่างจากตลาดเพื่อหาโอกาสทำกำไร
          </p>
        </div>

        {/* Variant Realization */}
        <div className="rounded-lg border border-slate-800 bg-slate-950/60 p-3 space-y-1.5">
          <div className="flex items-center justify-between text-xs text-slate-300">
            <div>
              <span className="font-medium">Realization</span>
              <span className="text-[10px] text-slate-400 ml-1 font-mono">(A - E_fund)</span>
            </div>
            <span
              className={`font-mono font-bold text-xs ${
                Math.abs(numRealization) < Math.abs(numStreetSurprise)
                  ? 'text-emerald-400'
                  : 'text-amber-400'
              }`}
            >
              {formatPct(variant_realization_pct, numRealization)}
            </span>
          </div>
          <div
            data-testid="variant-realization"
            className="font-mono text-sm font-semibold text-slate-200"
          >
            {numRealization > 0 ? `+${variant_realization}` : variant_realization} {unit}
          </div>
          <p className="text-[11px] text-slate-400 leading-snug">
            ส่วนต่างความแม่นยำระหว่างตัวเลขจริงกับที่เราประเมิน
          </p>
        </div>
      </div>

      {/* 4. Institutional Audit & Guardrail Footer */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-t border-slate-800/80 pt-3 text-[10px] font-mono text-slate-400">
        <div className="flex items-center gap-2">
          <span className="text-emerald-400">🔒</span>
          <span>คำนวณแม่นยำด้วยโปรแกรม ไม่ใช้การกะประมาณ (Deterministic Decimal Math)</span>
        </div>
        <div>
          <span>Street vs Thesis Verification Engine</span>
        </div>
      </div>
    </div>
  )
}


