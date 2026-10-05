import React from 'react'
import { FedBotRatePair } from './FedBotRatePair'
import { SourceProvenanceBadge } from './SourceProvenanceBadge'
import type { ThaiFundFlowDTO, TreasuryYieldCurveDTO, MacroDashboardDTO } from '../../../api/types'

interface CrossBorderSectionProps {
  globalPolicyRates: any | null
  yieldCurve: TreasuryYieldCurveDTO | null
  flow: ThaiFundFlowDTO | null
  aiData?: MacroDashboardDTO | null
  loadingPolicy?: boolean
  errorPolicy?: string | null
}

export const CrossBorderSection: React.FC<CrossBorderSectionProps> = ({
  globalPolicyRates,
  yieldCurve,
  flow,
  loadingPolicy = false,
  errorPolicy = null,
}) => {
  const foreignRow = flow?.investors?.find(
    (i) => i.investor_type.toLowerCase().includes('foreign') || i.investor_type.includes('ต่างชาติ')
  )
  const foreignNet = foreignRow?.net_value
  const hasForeignNet = foreignNet != null && Number.isFinite(foreignNet)
  const foreignNetMil = hasForeignNet ? (foreignNet / 1e6).toFixed(1) : null
  const isForeignBuy = hasForeignNet && foreignNet > 0

  const us10y = yieldCurve?.yields?.find((y) => y.maturity === '10 Yr')?.yield_percent
  const spread10y2y = yieldCurve?.spread_10y_2y_bps

  return (
    <div className="space-y-6">
      {/* 1. Policy Rate Spread (Fed vs BoT) */}
      <div className="space-y-3">
        <div className="flex items-center justify-between border-b border-sky-100/70 pb-2">
          <div>
            <h2 className="text-base font-bold text-zinc-900 tracking-tight flex items-center gap-2">
              <span>อัตราดอกเบี้ยนโยบายและส่วนต่าง (Policy Rate Differential)</span>
            </h2>
            <p className="text-xs text-zinc-500">
              เปรียบเทียบ Fed Funds กับ BoT 1D Repo เพื่อประเมินทิศทางแรงกดดันเงินทุนเคลื่อนย้ายระหว่างประเทศ
            </p>
          </div>
        </div>

        <FedBotRatePair
          ratesData={globalPolicyRates}
          loading={loadingPolicy}
          error={errorPolicy}
        />
      </div>

      {/* 2. Capital Flow Transmission & Cross-Border Indicators */}
      <div className="space-y-3">
        <div className="flex items-center justify-between border-b border-sky-100/70 pb-2">
          <div>
            <h2 className="text-base font-bold text-zinc-900 tracking-tight flex items-center gap-2">
              <span>ช่องทางส่งผ่านเงินทุนและค่าเงิน (Transmission Channel)</span>
            </h2>
            <p className="text-xs text-zinc-500">
              ความสัมพันธ์ระหว่างพันธบัตรสหรัฐฯ กับแรงซื้อขายต่างชาติในตลาดหุ้นไทย
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 items-start">
          {/* Card 1: US Yield vs Thai Foreign Flow */}
          <div className="rounded-2xl border border-sky-100 bg-white/90 p-5 shadow-sm space-y-3">
            <div className="flex items-center justify-between border-b border-slate-100 pb-2.5">
              <h3 className="text-sm font-semibold text-zinc-900">
                US 10Y Yield vs Foreign SET Flow
              </h3>
              <div className="flex items-center gap-1.5">
                <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[10px] font-medium text-zinc-600">
                  คำอธิบายกลไกทั่วไป
                </span>
                <SourceProvenanceBadge origin="deterministic" compact />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3 text-xs">
              <div className="rounded-xl bg-slate-50 p-3 border border-slate-100">
                <span className="text-[10px] text-zinc-500 block">US 10Y Benchmark</span>
                <span className="font-mono text-xl font-extrabold text-zinc-900 mt-0.5 block">
                  {us10y !== undefined && us10y !== null ? `${us10y.toFixed(2)}%` : '—'}
                </span>
                <span className="text-[10px] text-zinc-400 mt-1 block">
                  {spread10y2y !== null && spread10y2y !== undefined ? `10Y-2Y: ${spread10y2y} bps` : ''}
                </span>
              </div>

              <div className="rounded-xl bg-slate-50 p-3 border border-slate-100">
                <span className="text-[10px] text-zinc-500 block">SET Foreign Net Flow</span>
                <span
                  className={`font-mono text-xl font-extrabold mt-0.5 block ${
                    isForeignBuy ? 'text-emerald-700' : hasForeignNet && foreignNet < 0 ? 'text-rose-600' : 'text-zinc-500'
                  }`}
                >
                  {foreignNetMil !== null ? `${isForeignBuy ? '+' : ''}${foreignNetMil} M` : '—'}
                </span>
                <span className="text-[10px] text-zinc-400 mt-1 block">ยอดซื้อขายต่างชาติ</span>
              </div>
            </div>

            <p className="text-xs text-zinc-600 leading-relaxed bg-sky-50/50 p-2.5 rounded-xl border border-sky-100">
              💡 <strong>กลไกส่งผ่าน:</strong> เมื่อ Yield พันธบัตรสหรัฐฯ ปรับตัวสูงขึ้น ดอลลาร์มีแนวโน้มแข็งค่า ส่งผลให้เกิดแรงกดดันเงินทุนไหลออกจากตลาดเกิดใหม่ (EM Outflow) รวมถึงตลาดหุ้นไทย
            </p>
          </div>

          {/* Card 2: CME Gold vs GTA Gold & FX */}
          <div className="rounded-2xl border border-sky-100 bg-white/90 p-5 shadow-sm space-y-3">
            <div className="flex items-center justify-between border-b border-slate-100 pb-2.5">
              <h3 className="text-sm font-semibold text-zinc-900">
                ทองคำโลก vs ทองคำแท่งไทย (FX Mechanism)
              </h3>
              <div className="flex items-center gap-1.5">
                <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[10px] font-medium text-zinc-600">
                  คำอธิบายกลไกทั่วไป
                </span>
                <SourceProvenanceBadge origin="deterministic" compact />
              </div>
            </div>

            <div className="text-xs text-zinc-600 space-y-2">
              <div className="rounded-xl bg-amber-50/60 p-3 border border-amber-200/80">
                <span className="font-semibold text-amber-950 block text-[11px] mb-1">
                  สมการคำนวณราคาทองคำแท่งในประเทศ:
                </span>
                <code className="font-mono text-[10px] text-amber-900 block bg-white/80 p-2 rounded border border-amber-200">
                  ราคาไทย = (Gold Spot USD/oz × USD/THB × 0.965 × 15.244 / 31.1035) + Premium
                </code>
              </div>
              <p className="text-[11px] leading-relaxed text-zinc-600">
                ราคาทองคำแท่ง 96.5% ในไทยไม่ได้ขึ้นกับราคาทองคำโลก (CME Gold) เพียงอย่างเดียว แต่ขึ้นกับอัตราแลกเปลี่ยน USD/THB หากค่าเงินบาทอ่อนค่า จะช่วยพยุงราคาทองคำในประเทศแม้ราคาทองคำโลกจะย่อตัว
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
