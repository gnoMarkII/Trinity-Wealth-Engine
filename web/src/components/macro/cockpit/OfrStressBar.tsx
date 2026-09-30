import React from 'react'
import { SourceProvenanceBadge } from './SourceProvenanceBadge'

interface CategoryItem {
  label: string
  value: number
}

interface OfrStressBarProps {
  fsiValue: number | null | undefined
  asOfDate?: string
  regime?: string
  categories?: CategoryItem[]
  loading?: boolean
  error?: string | null
  className?: string
}

export const OfrStressBar: React.FC<OfrStressBarProps> = ({
  fsiValue,
  asOfDate,
  regime,
  categories = [],
  loading = false,
  error = null,
  className = '',
}) => {
  if (loading) {
    return (
      <div className={`rounded-2xl border border-sky-100 bg-white/90 p-5 shadow-sm animate-pulse ${className}`}>
        <div className="h-6 w-48 bg-slate-200 rounded mb-4" />
        <div className="h-44 bg-slate-100 rounded-xl" />
      </div>
    )
  }

  if (error || fsiValue === null || fsiValue === undefined) {
    return (
      <div className={`rounded-2xl border border-amber-200 bg-amber-50/60 p-5 text-xs text-amber-900 ${className}`}>
        <div className="font-bold flex items-center gap-1.5">
          <span>⚠️</span>
          <span>US Financial Stress Index: ไม่สามารถโหลดข้อมูลได้</span>
        </div>
        <p className="mt-1 text-zinc-600">{error || 'ไม่มีข้อมูลจาก Office of Financial Research (OFR)'}</p>
      </div>
    )
  }

  const isStressPositive = fsiValue > 0
  const maxAbs = Math.max(1.5, ...categories.map((c) => Math.abs(c.value)), Math.abs(fsiValue))

  return (
    <div className={`rounded-2xl border border-sky-100 bg-white/90 p-5 shadow-sm ${className}`}>
      {/* Header & Provenance */}
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between border-b border-sky-100/70 pb-3">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="font-bold text-zinc-900 text-sm tracking-tight">
              US Financial Stress Index (OFR FSI)
            </h3>
            <span
              className={`rounded-md border px-2 py-0.5 text-[10px] font-bold uppercase ${
                isStressPositive
                  ? 'border-rose-200 bg-rose-50 text-rose-700'
                  : 'border-emerald-200 bg-emerald-50 text-emerald-700'
              }`}
            >
              {regime || (isStressPositive ? 'Elevated Stress' : 'Below Average Stress')}
            </span>
          </div>
          <p className="text-xs text-zinc-500 mt-0.5">
            ดัชนีความตึงตัวในตลาดการเงินสหรัฐฯ 5 หมวด (ค่า 0 = ระดับความเครียดเฉลี่ยในอดีต)
          </p>
        </div>

        <SourceProvenanceBadge
          origin="provider"
          sourceName="OFR (US Treasury)"
          observedAt={asOfDate}
          publishedAt="T-2 Business Days Lag"
          compact
        />
      </div>

      {/* Aggregate Score Display */}
      <div className="mt-4 flex flex-wrap items-baseline justify-between gap-2 rounded-xl bg-slate-50 p-3 border border-slate-100">
        <div>
          <span className="text-[11px] font-semibold text-zinc-600">คะแนนความตึงตัวรวม (FSI Aggregate):</span>
          <p className="text-[10px] text-zinc-400">หน่วยเป็นส่วนเบี่ยงเบนมาตรฐาน (σ) รอบค่าเฉลี่ย 0</p>
        </div>
        <div className="flex items-baseline gap-1.5 font-mono">
          <span
            className={`text-2xl font-extrabold ${
              isStressPositive ? 'text-rose-600' : 'text-emerald-700'
            }`}
          >
            {fsiValue > 0 ? `+${fsiValue.toFixed(2)}` : fsiValue.toFixed(2)}
          </span>
          <span className="text-xs text-zinc-500 font-semibold">σ</span>
        </div>
      </div>

      {/* 5 Categories Diverging Bars around Zero Baseline */}
      {categories.length > 0 && (
        <div className="mt-4 space-y-2 text-xs">
          <div className="flex items-center justify-between text-[10px] text-zinc-400 font-semibold uppercase px-1">
            <span>หมวดความเสี่ยง</span>
            <div className="flex items-center gap-16 mr-2">
              <span>← สงบกว่าปกติ</span>
              <span>ตึงตัวกว่าปกติ →</span>
            </div>
          </div>

          <div className="space-y-2">
            {categories.map((cat) => {
              const val = cat.value
              const isPos = val > 0
              const pct = Math.min(50, (Math.abs(val) / maxAbs) * 50)

              return (
                <div key={cat.label} className="rounded-lg bg-slate-50/60 p-2 border border-slate-100/70">
                  <div className="flex items-center justify-between mb-1 text-[11px]">
                    <span className="font-semibold text-zinc-800">{cat.label}</span>
                    <span
                      className={`font-mono font-bold ${
                        isPos ? 'text-amber-600' : 'text-emerald-600'
                      }`}
                    >
                      {isPos ? `+${val.toFixed(2)}` : val.toFixed(2)} σ
                    </span>
                  </div>

                  {/* Dual-sided Diverging Bar */}
                  <div className="relative h-2 w-full overflow-hidden rounded-full bg-slate-200 flex">
                    {/* Left half (Negative: Below avg stress) */}
                    <div className="relative w-1/2 h-full flex justify-end">
                      {!isPos && (
                        <div
                          className="h-full bg-emerald-500 rounded-l-full transition-all duration-500"
                          style={{ width: `${pct * 2}%` }}
                        />
                      )}
                    </div>
                    {/* Zero divider */}
                    <div className="w-[1px] h-full bg-zinc-400 z-10" />
                    {/* Right half (Positive: Elevated stress) */}
                    <div className="relative w-1/2 h-full flex justify-start">
                      {isPos && (
                        <div
                          className="h-full bg-amber-500 rounded-r-full transition-all duration-500"
                          style={{ width: `${pct * 2}%` }}
                        />
                      )}
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      <p className="mt-3 text-[11px] text-zinc-400">
        ℹ️ ใช้ Diverging Bars รอบแกน 0 แทนการใช้กรอบสมมติ ข้อมูลประกาศย้อนหลัง 2 วันทำการตามธรรมชาติของรายงาน OFR
      </p>
    </div>
  )
}
