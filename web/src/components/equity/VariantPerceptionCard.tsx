import React from 'react'
import type { VariantPerceptionSnapshotDTO } from '../../api/types'
import { AlertTriangleIcon, ScaleIcon, SearchIcon, ShieldCheckIcon, TargetIcon } from '../ui/AppIcons'

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
        className="flow-panel rounded-2xl border border-edge bg-panel p-6 animate-pulse text-slate-500 space-y-4 shadow-sm"
      >
        <div className="h-6 bg-slate-200/70 rounded w-1/3" />
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="h-24 bg-slate-200/60 rounded-xl" />
          <div className="h-24 bg-slate-200/60 rounded-xl" />
          <div className="h-24 bg-slate-200/60 rounded-xl" />
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
              className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-emerald-50 text-emerald-800 border border-emerald-200"
            >
              <TargetIcon className="h-3.5 w-3.5 text-emerald-600 shrink-0" />
              <span>ALPHA CONFIRMED</span>
            </span>
            <span className="text-[11px] text-emerald-700 font-sans">เรามองขาดกว่าตลาด</span>
          </div>
        )
      case 'CONSENSUS_CORRECT':
        return (
          <div className="flex flex-col items-end gap-0.5">
            <span
              data-testid="verdict-consensus-correct"
              className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-sky-50 text-sky-900 border border-sky-200"
            >
              <ScaleIcon className="h-3.5 w-3.5 text-sky-600 shrink-0" />
              <span>CONSENSUS ACCURATE</span>
            </span>
            <span className="text-[11px] text-sky-800 font-sans">ตลาดคาดได้แม่นยำกว่า</span>
          </div>
        )
      case 'DOUBLE_MISS':
        return (
          <div className="flex flex-col items-end gap-0.5">
            <span
              data-testid="verdict-double-miss"
              className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-rose-50 text-rose-800 border border-rose-200"
            >
              <AlertTriangleIcon className="h-3.5 w-3.5 text-rose-600 shrink-0" />
              <span>DOUBLE MISS</span>
            </span>
            <span className="text-[11px] text-rose-700 font-sans">หลุดเป้าทั้งเราและตลาด</span>
          </div>
        )
      default:
        return (
          <div className="flex flex-col items-end gap-0.5">
            <span
              data-testid="verdict-default"
              className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-amber-50 text-amber-800 border border-amber-200"
            >
              <SearchIcon className="h-3.5 w-3.5 text-amber-600 shrink-0" />
              <span>{verdict.replace(/_/g, ' ')}</span>
            </span>
            <span className="text-[11px] text-amber-700 font-sans">กำลังติดตามผล</span>
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
      className="flow-panel rounded-2xl border border-edge bg-panel p-6 text-slate-800 shadow-sm space-y-6"
    >
      {/* 1. Header Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-edge pb-4">
        <div>
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-xl font-bold tracking-tight text-slate-900">
              ${ticker}
            </span>
            <span className="text-xs font-mono font-semibold px-2 py-0.5 rounded-md bg-blue-50 text-blue-900 border border-blue-200">
              {period}
            </span>
            <span className="text-xs text-slate-700 font-semibold">
              {metric_name.replace(/_/g, ' ').toUpperCase()}
            </span>
          </div>
          <p className="text-xs text-section-accent mt-1 font-medium">
            เปรียบเทียบมุมมอง: ตลาดส่วนใหญ่คิดอย่างไร vs พอร์ตเราคิดอย่างไร (เพื่อหาโอกาสสร้างผลตอบแทนส่วนเกิน)
          </p>
          <p className="text-[11px] text-slate-500 mt-0.5">
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
        <div className="bg-surface border border-edge rounded-xl p-4 space-y-1.5 shadow-sm">
          <div className="text-[11px] font-semibold text-slate-700 uppercase tracking-wider flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <span>Street Consensus</span>
              <span className="text-[11px] text-slate-500 font-normal normal-case">(มุมมองตลาด)</span>
            </div>
            <span className="text-slate-600 font-mono">C</span>
          </div>
          <div className="text-2xl font-bold font-mono tabular-nums text-slate-900" data-testid="consensus-value">
            {consensus_value}
            <span className="text-xs font-normal text-slate-500 ml-1.5">{unit}</span>
          </div>
          <p className="text-[11px] text-slate-600 leading-snug">
            สิ่งที่นักวิเคราะห์วอลล์สตรีทส่วนใหญ่คาดการณ์ไว้ล่วงหน้า
          </p>
        </div>

        {/* Fund Thesis Pillar */}
        <div className="bg-sky-50/75 border border-sky-200 rounded-xl p-4 space-y-1.5 shadow-sm">
          <div className="text-[11px] font-semibold text-sky-900 uppercase tracking-wider flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <span>Fund Thesis</span>
              <span className="text-[11px] text-sky-700 font-normal normal-case">(มุมมองพอร์ตเรา)</span>
            </div>
            <span className="text-sky-700 font-mono">E_fund</span>
          </div>
          <div className="text-2xl font-bold font-mono tabular-nums text-sky-950" data-testid="fund-thesis-value">
            {fund_thesis_value}
            <span className="text-xs font-normal text-sky-700 ml-1.5">{unit}</span>
          </div>
          <p className="text-[11px] text-sky-900/80 leading-snug">
            สิ่งที่เราวิเคราะห์เชิงลึกและเชื่อมั่นว่าจะทำได้จริง
          </p>
        </div>

        {/* Reported Actual Pillar */}
        <div className="bg-emerald-50/70 border border-emerald-200 rounded-xl p-4 space-y-1.5 shadow-sm">
          <div className="text-[11px] font-semibold text-emerald-900 uppercase tracking-wider flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <span>Reported Actual</span>
              <span className="text-[11px] text-emerald-700 font-normal normal-case">(ตัวเลขจริง)</span>
            </div>
            <span className="text-emerald-700 font-mono">A</span>
          </div>
          <div className="text-2xl font-bold font-mono tabular-nums text-emerald-900" data-testid="reported-value">
            {reported_value}
            <span className="text-xs font-normal text-emerald-700 ml-1.5">{unit}</span>
          </div>
          <p className="text-[11px] text-emerald-800 leading-snug">
            งบการเงินจริงที่บริษัทประกาศอย่างเป็นทางการ (SEC 10-Q / 8-K)
          </p>
        </div>
      </div>

      {/* 3. The 3 Spreads & Differentials */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-2">
        {/* Street Surprise */}
        <div className="rounded-xl border border-edge bg-white/85 p-3 space-y-1.5">
          <div className="flex items-center justify-between text-xs text-slate-700">
            <div>
              <span className="font-medium">Street Surprise</span>
              <span className="text-[11px] text-slate-500 ml-1 font-mono">(A - C)</span>
            </div>
            <span
              className={`font-mono font-bold text-xs ${
                numStreetSurprise >= 0 ? 'text-emerald-700' : 'text-rose-700'
              }`}
            >
              {formatPct(street_surprise_pct, numStreetSurprise)}
            </span>
          </div>
          <div
            data-testid="street-surprise"
            className="font-mono tabular-nums text-sm font-semibold text-slate-900"
          >
            {numStreetSurprise > 0 ? `+${street_surprise}` : street_surprise} {unit}
          </div>
          <p className="text-[11px] text-slate-600 leading-snug">
            {numStreetSurprise >= 0 ? 'ผลงานจริงดีกว่าที่ตลาดคาด' : 'ผลงานจริงต่ำกว่าที่ตลาดคาด'}
          </p>
        </div>

        {/* Variant Spread */}
        <div className="rounded-xl border border-sky-200 bg-sky-50/45 p-3 space-y-1.5">
          <div className="flex items-center justify-between text-xs text-sky-900">
            <div>
              <span className="font-medium">Variant Spread</span>
              <span className="text-[11px] text-sky-700 ml-1 font-mono">(E_fund - C)</span>
            </div>
            <span
              className={`font-mono font-bold text-xs ${
                numVariantSpread >= 0 ? 'text-sky-800' : 'text-amber-700'
              }`}
            >
              {formatPct(variant_spread_pct, numVariantSpread)}
            </span>
          </div>
          <div
            data-testid="variant-spread"
            className="font-mono tabular-nums text-sm font-semibold text-sky-950"
          >
            {numVariantSpread > 0 ? `+${variant_spread}` : variant_spread} {unit}
          </div>
          <p className="text-[11px] text-sky-800 leading-snug">
            ความมั่นใจของเราที่มองต่างจากตลาดเพื่อหาโอกาสทำกำไร
          </p>
        </div>

        {/* Variant Realization */}
        <div className="rounded-xl border border-edge bg-white/85 p-3 space-y-1.5">
          <div className="flex items-center justify-between text-xs text-slate-700">
            <div>
              <span className="font-medium">Realization</span>
              <span className="text-[11px] text-slate-500 ml-1 font-mono">(A - E_fund)</span>
            </div>
            <span
              className={`font-mono font-bold text-xs ${
                Math.abs(numRealization) < Math.abs(numStreetSurprise)
                  ? 'text-emerald-700'
                  : 'text-amber-700'
              }`}
            >
              {formatPct(variant_realization_pct, numRealization)}
            </span>
          </div>
          <div
            data-testid="variant-realization"
            className="font-mono tabular-nums text-sm font-semibold text-slate-900"
          >
            {numRealization > 0 ? `+${variant_realization}` : variant_realization} {unit}
          </div>
          <p className="text-[11px] text-slate-600 leading-snug">
            ส่วนต่างความแม่นยำระหว่างตัวเลขจริงกับที่เราประเมิน
          </p>
        </div>
      </div>

      {/* 4. Institutional Audit & Guardrail Footer */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-t border-edge pt-3 text-[11px] font-mono text-slate-600">
        <div className="flex items-center gap-2">
          <ShieldCheckIcon className="h-3.5 w-3.5 text-emerald-600 shrink-0" />
          <span>คำนวณแม่นยำด้วยโปรแกรม ไม่ใช้การกะประมาณ (Deterministic Decimal Math)</span>
        </div>
        <div>
          <span>Street vs Thesis Verification Engine</span>
        </div>
      </div>
    </div>
  )
}


