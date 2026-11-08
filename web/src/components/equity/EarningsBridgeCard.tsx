import React from 'react'
import type { EarningsBridgeDTO, EarningsDriverDTO } from '../../api/types'

interface EarningsBridgeCardProps {
  data: EarningsBridgeDTO
  isLoading?: boolean
}

export const EarningsBridgeCard: React.FC<EarningsBridgeCardProps> = ({ data, isLoading }) => {
  if (isLoading) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 animate-pulse text-slate-400">
        <div className="h-6 bg-slate-800 rounded w-1/3 mb-4" />
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-6">
          <div className="h-20 bg-slate-800 rounded" />
          <div className="h-20 bg-slate-800 rounded" />
          <div className="h-20 bg-slate-800 rounded" />
        </div>
        <div className="h-32 bg-slate-800 rounded" />
      </div>
    )
  }

  const {
    ticker,
    period,
    revenue_surprise: rev,
    eps_surprise: eps,
    ebit_margin_reported,
    ebit_margin_consensus,
    revenue_drivers: drivers,
    sum_drivers,
    unallocated_delta,
    is_mathematically_balanced,
    thesis_reconciliation_verdict: verdict,
    thesis_reconciliation_narrative: verdictNarrative,
  } = data

  const getDirectionBadge = (dir: string, pct: string) => {
    if (dir === 'BEAT') {
      return (
        <span
          className="inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
          title={`ดีกว่าตลาดคาด +${pct}%`}
        >
          +{pct}% BEAT
        </span>
      )
    }
    if (dir === 'MISS') {
      return (
        <span
          className="inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold bg-rose-500/20 text-rose-400 border border-rose-500/30"
          title={`แย่กว่าตลาดคาด ${pct}%`}
        >
          {pct}% MISS
        </span>
      )
    }
    return (
      <span
        className="inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold bg-amber-500/20 text-amber-400 border border-amber-500/30"
        title={`ตรงตามตลาดคาด ${pct}%`}
      >
        {pct}% INLINE
      </span>
    )
  }

  const getVerdictBadge = (v: string) => {
    if (v === 'THESIS_CONFIRMED' || v === 'THESIS_ACCELERATING') {
      return (
        <div className="flex flex-col items-end gap-0.5">
          <span className="inline-flex items-center px-2.5 py-1 rounded text-xs font-bold uppercase tracking-wide bg-emerald-500/20 text-emerald-300 border border-emerald-500/40">
            ✅ {v.replace('_', ' ')}
          </span>
          <span className="text-[10px] text-emerald-400 font-sans">ตรงตามแผนลงทุน</span>
        </div>
      )
    }
    if (v === 'THESIS_IMPAIRED') {
      return (
        <div className="flex flex-col items-end gap-0.5">
          <span className="inline-flex items-center px-2.5 py-1 rounded text-xs font-bold uppercase tracking-wide bg-rose-500/20 text-rose-300 border border-rose-500/40">
            ⚠️ {v.replace('_', ' ')}
          </span>
          <span className="text-[10px] text-rose-400 font-sans">งบแย่กว่าคาด เสี่ยงผิดแผน</span>
        </div>
      )
    }
    return (
      <div className="flex flex-col items-end gap-0.5">
        <span className="inline-flex items-center px-2.5 py-1 rounded text-xs font-bold uppercase tracking-wide bg-amber-500/20 text-amber-300 border border-amber-500/40">
          🔍 {v.replace('_', ' ')}
        </span>
        <span className="text-[10px] text-amber-400 font-sans">กำลังติดตามผล</span>
      </div>
    )
  }

  const getCategoryColor = (cat: string) => {
    switch (cat) {
      case 'VOLUME':
        return 'text-blue-400 bg-blue-500/10 border-blue-500/30'
      case 'PRICE_MIX':
        return 'text-purple-400 bg-purple-500/10 border-purple-500/30'
      case 'FX':
        return 'text-amber-400 bg-amber-500/10 border-amber-500/30'
      case 'COST_PRODUCTIVITY':
        return 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30'
      default:
        return 'text-slate-400 bg-slate-500/10 border-slate-500/30'
    }
  }

  const getCategoryThaiLabel = (cat: string) => {
    switch (cat) {
      case 'VOLUME':
        return 'ปริมาณขาย'
      case 'PRICE_MIX':
        return 'ราคาขาย/กลุ่มสินค้า'
      case 'FX':
        return 'ผลกระทบอัตราแลกเปลี่ยน'
      case 'COST_PRODUCTIVITY':
        return 'ประสิทธิภาพต้นทุน'
      default:
        return cat
    }
  }

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 text-slate-200 shadow-xl" data-testid="earnings-bridge-card">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-800 gap-3">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-lg font-bold text-white tracking-wide">
              Reported vs Expected Earnings Bridge
            </h3>
            <span className="text-xs px-2 py-0.5 rounded bg-blue-500/20 text-blue-300 border border-blue-500/30 font-mono">
              {ticker} &bull; {period}
            </span>
          </div>
          <p className="text-xs text-sky-400/90 mt-0.5 font-medium">
            สะพานวิเคราะห์งบการเงิน: เจาะลึกผลประกอบการจริงเทียบกับที่ตลาดคาดการณ์ พร้อมแจกแจงสาเหตุ (Volume, Price, FX)
          </p>
          <p className="text-[11px] text-slate-400 mt-0.5">
            Deterministic causal decomposition: &Delta; Revenue = &Delta; Volume + &Delta; Price/Mix + &Delta; FX (Chapter 5)
          </p>
        </div>
        <div>{getVerdictBadge(verdict)}</div>
      </div>

      {/* Top 3 Surprise Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 my-5">
        {/* Revenue Surprise */}
        <div className="bg-slate-800/60 border border-slate-700/60 rounded-lg p-3.5 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <span className="text-xs font-medium text-slate-300 uppercase tracking-wider">Revenue</span>
              <span className="text-[10px] text-slate-400 font-normal">ยอดขาย/รายได้</span>
            </div>
            {getDirectionBadge(rev.direction, rev.surprise_pct)}
          </div>
          <div className="mt-2">
            <div className="text-xl font-bold font-mono text-white">
              {rev.reported} <span className="text-xs font-normal text-slate-400">{rev.unit}</span>
            </div>
            <div className="text-xs text-slate-400 mt-1 flex justify-between font-mono">
              <span>Consensus: {rev.consensus}</span>
              <span className={Number(rev.delta) >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                &Delta; {Number(rev.delta) >= 0 ? `+${rev.delta}` : rev.delta}
              </span>
            </div>
            <div className="text-[10px] text-slate-400 mt-1">
              {Number(rev.delta) >= 0 ? '🎉 ยอดขายจริงสูงกว่าที่ตลาดคาด' : '⚠️ ยอดขายจริงต่ำกว่าที่ตลาดคาด'}
            </div>
          </div>
        </div>

        {/* EPS Surprise */}
        <div className="bg-slate-800/60 border border-slate-700/60 rounded-lg p-3.5 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <span className="text-xs font-medium text-slate-300 uppercase tracking-wider">EPS</span>
              <span className="text-[10px] text-slate-400 font-normal">กำไรสุทธิต่อหุ้น</span>
            </div>
            {getDirectionBadge(eps.direction, eps.surprise_pct)}
          </div>
          <div className="mt-2">
            <div className="text-xl font-bold font-mono text-white">
              {eps.reported} <span className="text-xs font-normal text-slate-400">{eps.unit}</span>
            </div>
            <div className="text-xs text-slate-400 mt-1 flex justify-between font-mono">
              <span>Consensus: {eps.consensus}</span>
              <span className={Number(eps.delta) >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                &Delta; {Number(eps.delta) >= 0 ? `+${eps.delta}` : eps.delta}
              </span>
            </div>
            <div className="text-[10px] text-slate-400 mt-1">
              {Number(eps.delta) >= 0 ? '🎉 บริษัททำกำไรได้ดีกว่าคาด' : '⚠️ บริษัททำกำไรต่ำกว่าคาด'}
            </div>
          </div>
        </div>

        {/* Margin Walk */}
        <div className="bg-slate-800/60 border border-slate-700/60 rounded-lg p-3.5 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <span className="text-xs font-medium text-slate-300 uppercase tracking-wider">EBIT Margin</span>
              <span className="text-[10px] text-slate-400 font-normal">อัตรากำไรจากการดำเนินงาน</span>
            </div>
            <span className="text-xs font-mono text-slate-400">Actual vs Exp</span>
          </div>
          <div className="mt-2">
            <div className="text-xl font-bold font-mono text-white">
              {ebit_margin_reported ? `${ebit_margin_reported}%` : '—'}
            </div>
            <div className="text-xs text-slate-400 mt-1 flex justify-between font-mono">
              <span>Consensus: {ebit_margin_consensus ? `${ebit_margin_consensus}%` : '—'}</span>
              {ebit_margin_reported && ebit_margin_consensus && (
                <span className={Number(ebit_margin_reported) >= Number(ebit_margin_consensus) ? 'text-emerald-400' : 'text-rose-400'}>
                  {Number(ebit_margin_reported) >= Number(ebit_margin_consensus) ? '+' : ''}
                  {(Number(ebit_margin_reported) - Number(ebit_margin_consensus)).toFixed(1)}%
                </span>
              )}
            </div>
            <div className="text-[10px] text-slate-400 mt-1">
              ประสิทธิภาพแปลงยอดขายเป็นกำไรก่อนดอกเบี้ย/ภาษี
            </div>
          </div>
        </div>
      </div>

      {/* Mathematical Balance Gate */}
      <div className="mb-4">
        {is_mathematically_balanced ? (
          <div className="flex flex-col sm:flex-row sm:items-center justify-between px-3.5 py-2.5 rounded-lg bg-emerald-950/40 border border-emerald-500/30 text-xs text-emerald-300 gap-2">
            <div className="flex items-center gap-2">
              <span>🛡️</span>
              <div>
                <span className="font-semibold">Mathematical Sum Invariant Verified:</span>{' '}
                <span className="font-mono text-slate-300">
                  &Sigma; Drivers ({sum_drivers} {rev.unit}) == Net Delta ({rev.delta} {rev.unit})
                </span>
                <span className="text-[10px] text-emerald-400/80 block mt-0.5">
                  ✓ รวมทุกปัจจัยแล้วเท่ากับส่วนต่างยอดขายจริง 100% พอดี (โปร่งใสตรวจสอบได้)
                </span>
              </div>
            </div>
            <span className="font-mono text-[11px] text-emerald-400 font-bold bg-emerald-900/50 px-2 py-0.5 rounded self-start sm:self-auto">
              EXACT BALANCE
            </span>
          </div>
        ) : (
          <div className="flex flex-col sm:flex-row sm:items-center justify-between px-3.5 py-2.5 rounded-lg bg-rose-950/40 border border-rose-500/30 text-xs text-rose-300 gap-2">
            <div className="flex items-center gap-2">
              <span>⚠️</span>
              <div>
                <span className="font-semibold">Unbalanced Causal Variance:</span>{' '}
                <span className="font-mono text-slate-300">
                  Unallocated Delta: {unallocated_delta} {rev.unit}
                </span>
                <span className="text-[10px] text-rose-400/80 block mt-0.5">
                  ⚠️ มียอดส่วนต่างที่ยังไม่สามารถระบุสาเหตุแน่ชัดได้ {unallocated_delta} {rev.unit}
                </span>
              </div>
            </div>
            <span className="font-mono text-[11px] text-rose-400 font-bold bg-rose-900/50 px-2 py-0.5 rounded self-start sm:self-auto">
              UNBALANCED
            </span>
          </div>
        )}
      </div>

      {/* Driver Breakdown Table */}
      <div className="border border-slate-800 rounded-lg overflow-hidden mb-4">
        <div className="bg-slate-800/80 px-4 py-2.5 text-xs font-semibold uppercase tracking-wider text-slate-300 flex justify-between items-center">
          <div className="flex items-center gap-2">
            <span>Causal Revenue Drivers</span>
            <span className="text-[10px] text-slate-400 font-normal normal-case">(ปัจจัยขับเคลื่อนยอดขายที่เพิ่มขึ้นหรือลดลง)</span>
          </div>
          <span>Impact ({rev.unit})</span>
        </div>
        <div className="divide-y divide-slate-800/60">
          {drivers.map((d: EarningsDriverDTO, idx: number) => {
            const isPositive = Number(d.impact_value) >= 0
            return (
              <div key={idx} className="px-4 py-3 flex flex-col sm:flex-row sm:items-center justify-between hover:bg-slate-800/30 transition-colors gap-2">
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-semibold text-slate-100">{d.name}</span>
                    <span className={`text-[10px] px-2 py-0.5 rounded border font-semibold ${getCategoryColor(d.category)}`}>
                      {d.category}
                    </span>
                    <span className="text-[10px] text-slate-400 font-normal">({getCategoryThaiLabel(d.category)})</span>
                  </div>
                  {d.narrative && <p className="text-xs text-slate-400 mt-0.5">{d.narrative}</p>}
                </div>
                <div className="text-right font-mono font-bold text-sm">
                  <span className={isPositive ? 'text-emerald-400' : 'text-rose-400'}>
                    {isPositive ? `+${d.impact_value}` : d.impact_value}
                  </span>
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Thesis Reconciliation Footer */}
      {verdictNarrative && (
        <div className="p-3.5 rounded-lg bg-slate-800/40 border border-slate-700/50 text-xs space-y-1">
          <div className="flex items-center gap-1.5 font-bold text-slate-200 uppercase tracking-wide">
            <span>Thesis Impact Assessment:</span>
            <span className="text-[11px] text-sky-400 font-normal normal-case">(สรุปผลกระทบต่อนโยบายการลงทุน)</span>
          </div>
          <p className="text-slate-300 leading-relaxed">{verdictNarrative}</p>
        </div>
      )}
    </div>
  )
}


