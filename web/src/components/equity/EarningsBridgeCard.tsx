import React from 'react'
import type { EarningsBridgeDTO, EarningsDriverDTO } from '../../api/types'
import { AlertTriangleIcon, CheckCircleIcon, SearchIcon, ShieldCheckIcon } from '../ui/AppIcons'

interface EarningsBridgeCardProps {
  data: EarningsBridgeDTO
  isLoading?: boolean
}

export const EarningsBridgeCard: React.FC<EarningsBridgeCardProps> = ({ data, isLoading }) => {
  if (isLoading) {
    return (
      <div className="flow-panel rounded-2xl border border-edge bg-panel p-6 animate-pulse text-slate-500 shadow-sm">
        <div className="h-6 bg-slate-200/70 rounded w-1/3 mb-4" />
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-6">
          <div className="h-20 bg-slate-200/60 rounded-xl" />
          <div className="h-20 bg-slate-200/60 rounded-xl" />
          <div className="h-20 bg-slate-200/60 rounded-xl" />
        </div>
        <div className="h-32 bg-slate-200/60 rounded-xl" />
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
          className="inline-flex items-center px-2 py-0.5 rounded-md text-xs font-semibold bg-emerald-50 text-emerald-800 border border-emerald-200"
          title={`ดีกว่าตลาดคาด +${pct}%`}
        >
          +{pct}% BEAT
        </span>
      )
    }
    if (dir === 'MISS') {
      return (
        <span
          className="inline-flex items-center px-2 py-0.5 rounded-md text-xs font-semibold bg-rose-50 text-rose-800 border border-rose-200"
          title={`แย่กว่าตลาดคาด ${pct}%`}
        >
          {pct}% MISS
        </span>
      )
    }
    return (
      <span
        className="inline-flex items-center px-2 py-0.5 rounded-md text-xs font-semibold bg-amber-50 text-amber-800 border border-amber-200"
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
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-bold uppercase tracking-wide bg-emerald-50 text-emerald-800 border border-emerald-200">
            <CheckCircleIcon className="h-3.5 w-3.5 text-emerald-600 shrink-0" />
            <span>{v.replace('_', ' ')}</span>
          </span>
          <span className="text-[11px] text-emerald-700 font-sans">ตรงตามแผนลงทุน</span>
        </div>
      )
    }
    if (v === 'THESIS_IMPAIRED') {
      return (
        <div className="flex flex-col items-end gap-0.5">
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-bold uppercase tracking-wide bg-rose-50 text-rose-800 border border-rose-200">
            <AlertTriangleIcon className="h-3.5 w-3.5 text-rose-600 shrink-0" />
            <span>{v.replace('_', ' ')}</span>
          </span>
          <span className="text-[11px] text-rose-700 font-sans">งบแย่กว่าคาด เสี่ยงผิดแผน</span>
        </div>
      )
    }
    return (
      <div className="flex flex-col items-end gap-0.5">
        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-bold uppercase tracking-wide bg-amber-50 text-amber-800 border border-amber-200">
          <SearchIcon className="h-3.5 w-3.5 text-amber-600 shrink-0" />
          <span>{v.replace('_', ' ')}</span>
        </span>
        <span className="text-[11px] text-amber-700 font-sans">กำลังติดตามผล</span>
      </div>
    )
  }

  const getCategoryColor = (cat: string) => {
    switch (cat) {
      case 'VOLUME':
        return 'text-blue-800 bg-blue-50 border-blue-200'
      case 'PRICE_MIX':
        return 'text-sky-800 bg-sky-50 border-sky-200'
      case 'FX':
        return 'text-amber-800 bg-amber-50 border-amber-200'
      case 'COST_PRODUCTIVITY':
        return 'text-emerald-800 bg-emerald-50 border-emerald-200'
      default:
        return 'text-slate-700 bg-slate-100 border-slate-200'
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
    <div className="flow-panel rounded-2xl border border-edge bg-panel p-6 text-slate-800 shadow-sm" data-testid="earnings-bridge-card">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-edge gap-3">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="text-lg font-bold text-slate-900 tracking-wide">
              Reported vs Expected Earnings Bridge
            </h3>
            <span className="text-xs px-2 py-0.5 rounded-md bg-sky-50 text-sky-700 border border-sky-200 font-mono font-semibold">
              {ticker} &bull; {period}
            </span>
          </div>
          <p className="text-xs text-sky-700 mt-0.5 font-medium">
            สะพานวิเคราะห์งบการเงิน: เจาะลึกผลประกอบการจริงเทียบกับที่ตลาดคาดการณ์ พร้อมแจกแจงสาเหตุ (Volume, Price, FX)
          </p>
          <p className="text-[11px] text-slate-500 mt-0.5">
            Deterministic causal decomposition: &Delta; Revenue = &Delta; Volume + &Delta; Price/Mix + &Delta; FX (Chapter 5)
          </p>
        </div>
        <div>{getVerdictBadge(verdict)}</div>
      </div>

      {/* Top 3 Surprise Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 my-5">
        {/* Revenue Surprise */}
        <div className="bg-surface-2/75 border border-edge rounded-xl p-3.5 flex flex-col justify-between">
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-1.5">
              <span className="text-xs font-semibold text-slate-700 uppercase tracking-wider">Revenue</span>
              <span className="text-[11px] text-slate-500 font-normal">ยอดขาย/รายได้</span>
            </div>
            {getDirectionBadge(rev.direction, rev.surprise_pct)}
          </div>
          <div className="mt-2">
            <div className="text-xl font-bold font-mono text-slate-900">
              {rev.reported} <span className="text-xs font-normal text-slate-500">{rev.unit}</span>
            </div>
            <div className="text-xs text-slate-500 mt-1 flex justify-between font-mono">
              <span>Consensus: {rev.consensus}</span>
              <span className={Number(rev.delta) >= 0 ? 'text-emerald-700 font-semibold' : 'text-rose-700 font-semibold'}>
                &Delta; {Number(rev.delta) >= 0 ? `+${rev.delta}` : rev.delta}
              </span>
            </div>
            <div className="text-[11px] text-slate-600 mt-1.5 inline-flex items-center gap-1.5">
              {Number(rev.delta) >= 0 ? (
                <>
                  <CheckCircleIcon className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
                  <span>ยอดขายจริงสูงกว่าที่ตลาดคาด</span>
                </>
              ) : (
                <>
                  <AlertTriangleIcon className="w-3.5 h-3.5 text-amber-600 shrink-0" />
                  <span>ยอดขายจริงต่ำกว่าที่ตลาดคาด</span>
                </>
              )}
            </div>
          </div>
        </div>

        {/* EPS Surprise */}
        <div className="bg-surface-2/75 border border-edge rounded-xl p-3.5 flex flex-col justify-between">
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-1.5">
              <span className="text-xs font-semibold text-slate-700 uppercase tracking-wider">EPS</span>
              <span className="text-[11px] text-slate-500 font-normal">กำไรสุทธิต่อหุ้น</span>
            </div>
            {getDirectionBadge(eps.direction, eps.surprise_pct)}
          </div>
          <div className="mt-2">
            <div className="text-xl font-bold font-mono text-slate-900">
              {eps.reported} <span className="text-xs font-normal text-slate-500">{eps.unit}</span>
            </div>
            <div className="text-xs text-slate-500 mt-1 flex justify-between font-mono">
              <span>Consensus: {eps.consensus}</span>
              <span className={Number(eps.delta) >= 0 ? 'text-emerald-700 font-semibold' : 'text-rose-700 font-semibold'}>
                &Delta; {Number(eps.delta) >= 0 ? `+${eps.delta}` : eps.delta}
              </span>
            </div>
            <div className="text-[11px] text-slate-600 mt-1.5 inline-flex items-center gap-1.5">
              {Number(eps.delta) >= 0 ? (
                <>
                  <CheckCircleIcon className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
                  <span>บริษัททำกำไรได้ดีกว่าคาด</span>
                </>
              ) : (
                <>
                  <AlertTriangleIcon className="w-3.5 h-3.5 text-amber-600 shrink-0" />
                  <span>บริษัททำกำไรต่ำกว่าคาด</span>
                </>
              )}
            </div>
          </div>
        </div>

        {/* Margin Walk */}
        <div className="bg-surface-2/75 border border-edge rounded-xl p-3.5 flex flex-col justify-between">
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-1.5">
              <span className="text-xs font-semibold text-slate-700 uppercase tracking-wider">EBIT Margin</span>
              <span className="text-[11px] text-slate-500 font-normal">อัตรากำไรจากการดำเนินงาน</span>
            </div>
            <span className="text-xs font-mono text-slate-500">Actual vs Exp</span>
          </div>
          <div className="mt-2">
            <div className="text-xl font-bold font-mono text-slate-900">
              {ebit_margin_reported ? `${ebit_margin_reported}%` : '—'}
            </div>
            <div className="text-xs text-slate-500 mt-1 flex justify-between font-mono">
              <span>Consensus: {ebit_margin_consensus ? `${ebit_margin_consensus}%` : '—'}</span>
              {ebit_margin_reported && ebit_margin_consensus && (
                <span className={Number(ebit_margin_reported) >= Number(ebit_margin_consensus) ? 'text-emerald-700 font-semibold' : 'text-rose-700 font-semibold'}>
                  {Number(ebit_margin_reported) >= Number(ebit_margin_consensus) ? '+' : ''}
                  {(Number(ebit_margin_reported) - Number(ebit_margin_consensus)).toFixed(1)}%
                </span>
              )}
            </div>
            <div className="text-[11px] text-slate-600 mt-1.5">
              ประสิทธิภาพแปลงยอดขายเป็นกำไรก่อนดอกเบี้ย/ภาษี
            </div>
          </div>
        </div>
      </div>

      {/* Mathematical Balance Gate */}
      <div className="mb-4">
        {is_mathematically_balanced ? (
          <div className="flex flex-col sm:flex-row sm:items-center justify-between px-3.5 py-2.5 rounded-xl bg-emerald-50 border border-emerald-200 text-xs text-emerald-800 gap-2">
            <div className="flex items-center gap-2.5">
              <ShieldCheckIcon className="w-4 h-4 text-emerald-600 shrink-0" />
              <div>
                <span className="font-semibold">Mathematical Sum Invariant Verified:</span>{' '}
                <span className="font-mono text-slate-700">
                  &Sigma; Drivers ({sum_drivers} {rev.unit}) == Net Delta ({rev.delta} {rev.unit})
                </span>
                <span className="text-[11px] text-emerald-700 block mt-0.5">
                  รวมทุกปัจจัยแล้วเท่ากับส่วนต่างยอดขายจริง 100% พอดี (โปร่งใสตรวจสอบได้)
                </span>
              </div>
            </div>
            <span className="font-mono text-[11px] text-emerald-700 font-bold bg-emerald-100 border border-emerald-200 px-2.5 py-0.5 rounded-md self-start sm:self-auto shrink-0">
              EXACT BALANCE
            </span>
          </div>
        ) : (
          <div className="flex flex-col sm:flex-row sm:items-center justify-between px-3.5 py-2.5 rounded-xl bg-rose-50 border border-rose-200 text-xs text-rose-800 gap-2">
            <div className="flex items-center gap-2.5">
              <AlertTriangleIcon className="w-4 h-4 text-rose-600 shrink-0" />
              <div>
                <span className="font-semibold">Unbalanced Causal Variance:</span>{' '}
                <span className="font-mono text-slate-700">
                  Unallocated Delta: {unallocated_delta} {rev.unit}
                </span>
                <span className="text-[11px] text-rose-700 block mt-0.5">
                  มียอดส่วนต่างที่ยังไม่สามารถระบุสาเหตุแน่ชัดได้ {unallocated_delta} {rev.unit}
                </span>
              </div>
            </div>
            <span className="font-mono text-[11px] text-rose-700 font-bold bg-rose-100 border border-rose-200 px-2.5 py-0.5 rounded-md self-start sm:self-auto shrink-0">
              UNBALANCED
            </span>
          </div>
        )}
      </div>

      {/* Driver Breakdown Table */}
      <div className="border border-edge rounded-xl overflow-hidden mb-4 bg-white/75">
        <div className="bg-surface-2/90 px-4 py-2.5 text-xs font-semibold uppercase tracking-wider text-slate-700 flex justify-between items-center border-b border-edge">
          <div className="flex items-center gap-2">
            <span>Causal Revenue Drivers</span>
            <span className="text-[11px] text-slate-500 font-normal normal-case">(ปัจจัยขับเคลื่อนยอดขายที่เพิ่มขึ้นหรือลดลง)</span>
          </div>
          <span>Impact ({rev.unit})</span>
        </div>
        <div className="divide-y divide-edge/80">
          {drivers.map((d: EarningsDriverDTO, idx: number) => {
            const isPositive = Number(d.impact_value) >= 0
            return (
              <div key={idx} className="px-4 py-3 flex flex-col sm:flex-row sm:items-center justify-between hover:bg-sky-50/40 transition-colors gap-2">
                <div className="flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-sm font-semibold text-slate-900">{d.name}</span>
                    <span className={`text-[11px] px-2 py-0.5 rounded border font-semibold ${getCategoryColor(d.category)}`}>
                      {d.category}
                    </span>
                    <span className="text-[11px] text-slate-500 font-normal">({getCategoryThaiLabel(d.category)})</span>
                  </div>
                  {d.narrative && <p className="text-xs text-slate-600 mt-0.5">{d.narrative}</p>}
                </div>
                <div className="text-right font-mono font-bold text-sm shrink-0">
                  <span className={isPositive ? 'text-emerald-700' : 'text-rose-700'}>
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
        <div className="p-3.5 rounded-xl bg-surface-2/75 border border-edge text-xs space-y-1">
          <div className="flex items-center gap-1.5 font-bold text-slate-800 uppercase tracking-wide">
            <span>Thesis Impact Assessment:</span>
            <span className="text-[11px] text-sky-700 font-medium normal-case">(สรุปผลกระทบต่อนโยบายการลงทุน)</span>
          </div>
          <p className="text-slate-600 leading-relaxed">{verdictNarrative}</p>
        </div>
      )}
    </div>
  )
}


