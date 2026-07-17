import { useState } from 'react'
import type { AllocationTargetDTO, BucketAllocationSummaryDTO, ActualPortfolioStateDTO } from '../../api/types'
import { formatTHB } from './PortfolioSummaryCards'
import BucketTargetModal from './Modals/BucketTargetModal'
import { TradeIcon } from './icons/PortfolioIcons'

interface Props {
  targets: AllocationTargetDTO[]
  summaries: BucketAllocationSummaryDTO[]
  warning: string | null
  onSelectBucket: (bucketId: string) => void
  onSuccess?: (state: ActualPortfolioStateDTO) => void
  onOpenTradeModal?: () => void
}

const DEFAULT_COLORS = [
  '#0284c7', // sky-600
  '#4f46e5', // indigo-600
  '#059669', // emerald-600
  '#d97706', // amber-600
  '#db2777', // pink-600
  '#7c3aed', // violet-600
  '#0d9488', // teal-600
  '#475569', // slate-600
]

function getBucketColor(index: number, explicitColor?: string | null): string {
  if (explicitColor) return explicitColor
  return DEFAULT_COLORS[index % DEFAULT_COLORS.length] ?? '#0284c7'
}

export default function PortfolioOverviewTab({ targets, summaries, warning, onSelectBucket, onSuccess, onOpenTradeModal }: Props) {
  const [hoveredBucket, setHoveredBucket] = useState<string | null>(null)
  const [targetModalOpen, setTargetModalOpen] = useState(false)

  // คำนวณเส้นรอบวงสำหรับ SVG Donut Chart
  // Inner Ring: Target (radius 60)
  // Outer Ring: Actual (radius 95)
  const innerRadius = 60
  const outerRadius = 95
  const strokeWidthInner = 18
  const strokeWidthOuter = 22

  const innerCircumference = 2 * Math.PI * innerRadius
  const outerCircumference = 2 * Math.PI * outerRadius

  // คำนวณ segments สำหรับ Inner (Targets)
  let innerOffset = 0
  const innerSegments = targets.map((t, idx) => {
    const pct = Math.max(0, Math.min(100, t.target_percent))
    const strokeDasharray = `${(pct / 100) * innerCircumference} ${innerCircumference}`
    const strokeDashoffset = -innerOffset
    innerOffset += (pct / 100) * innerCircumference
    return {
      bucketId: t.bucket_id,
      name: t.name,
      pct: t.target_percent,
      strokeDasharray,
      strokeDashoffset,
      color: getBucketColor(idx, t.color),
    }
  })

  // คำนวณ segments สำหรับ Outer (Actuals)
  let outerOffset = 0
  const outerSegments = summaries.map((s, idx) => {
    const pct = Math.max(0, Math.min(100, s.actual_percent))
    const strokeDasharray = `${(pct / 100) * outerCircumference} ${outerCircumference}`
    const strokeDashoffset = -outerOffset
    outerOffset += (pct / 100) * outerCircumference
    const targetIdx = targets.findIndex((t) => t.bucket_id === s.bucket_id)
    return {
      bucketId: s.bucket_id,
      name: s.name,
      pct: s.actual_percent,
      value: s.actual_value_thb,
      variance: s.variance,
      strokeDasharray,
      strokeDashoffset,
      color: getBucketColor(targetIdx >= 0 ? targetIdx : idx, s.color),
    }
  })

  const activeSummary = hoveredBucket ? summaries.find((s) => s.bucket_id === hoveredBucket) : null
  const activeTarget = hoveredBucket ? targets.find((t) => t.bucket_id === hoveredBucket) : null

  // คำนวณ Total Absolute Variance ของพอร์ต
  const totalVariance = summaries.reduce((acc, s) => acc + Math.abs(s.variance), 0)

  return (
    <div className="space-y-6">
      {/* Target Sum Warning Badge */}
      {warning && (
        <div className="flex items-center gap-3 rounded-2xl border border-amber-200 bg-amber-50/80 p-4 text-amber-900 shadow-sm">
          <svg className="h-5 w-5 shrink-0 text-amber-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
          </svg>
          <div className="text-sm font-medium">
            <span className="font-bold">แจ้งเตือนสัดส่วนเป้าหมาย:</span> {warning}
          </div>
        </div>
      )}

      {/* Grid: Donut Chart + Table */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        {/* Concentric Double-Ring Donut Chart Panel */}
        <div className="rounded-2xl border border-sky-100 bg-panel p-6 shadow-sm lg:col-span-5 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between">
              <h3 className="text-base font-bold text-zinc-900">Allocation Rings (Target vs Actual)</h3>
              <span className="rounded-full bg-sky-50 px-2.5 py-1 text-xs font-medium text-sky-700">
                วงใน: Target | วงนอก: Actual
              </span>
            </div>
            <p className="mt-1 text-xs text-zinc-500">
              เปรียบเทียบสัดส่วนเป้าหมายตามแผนกลยุทธ์กับมูลค่าพอร์ตจริงในปัจจุบัน
            </p>
          </div>

          <div className="relative my-6 flex items-center justify-center">
            <svg className="h-64 w-64 -rotate-90 transform" viewBox="0 0 240 240">
              {/* Background Tracks */}
              <circle cx="120" cy="120" r={innerRadius} fill="transparent" stroke="#f1f5f9" strokeWidth={strokeWidthInner} />
              <circle cx="120" cy="120" r={outerRadius} fill="transparent" stroke="#f1f5f9" strokeWidth={strokeWidthOuter} />

              {/* Inner Ring (Target) */}
              {innerSegments.map((seg) => {
                const isHovered = hoveredBucket === seg.bucketId
                const isDimmed = hoveredBucket !== null && hoveredBucket !== seg.bucketId
                return (
                  <circle
                    key={`inner-${seg.bucketId}`}
                    cx="120"
                    cy="120"
                    r={innerRadius}
                    fill="transparent"
                    stroke={seg.color}
                    strokeWidth={isHovered ? strokeWidthInner + 3 : strokeWidthInner}
                    strokeDasharray={seg.strokeDasharray}
                    strokeDashoffset={seg.strokeDashoffset}
                    opacity={isDimmed ? 0.35 : 1}
                    className="transition-all duration-300 cursor-pointer"
                    onMouseEnter={() => setHoveredBucket(seg.bucketId)}
                    onMouseLeave={() => setHoveredBucket(null)}
                    onClick={() => onSelectBucket(seg.bucketId)}
                  >
                    <title>{`${seg.name} Target: ${seg.pct.toFixed(1)}%`}</title>
                  </circle>
                )
              })}

              {/* Outer Ring (Actual) */}
              {outerSegments.map((seg) => {
                const isHovered = hoveredBucket === seg.bucketId
                const isDimmed = hoveredBucket !== null && hoveredBucket !== seg.bucketId
                return (
                  <circle
                    key={`outer-${seg.bucketId}`}
                    cx="120"
                    cy="120"
                    r={outerRadius}
                    fill="transparent"
                    stroke={seg.color}
                    strokeWidth={isHovered ? strokeWidthOuter + 4 : strokeWidthOuter}
                    strokeDasharray={seg.strokeDasharray}
                    strokeDashoffset={seg.strokeDashoffset}
                    opacity={isDimmed ? 0.35 : 1}
                    className="transition-all duration-300 cursor-pointer"
                    onMouseEnter={() => setHoveredBucket(seg.bucketId)}
                    onMouseLeave={() => setHoveredBucket(null)}
                    onClick={() => onSelectBucket(seg.bucketId)}
                  >
                    <title>{`${seg.name} Actual: ${seg.pct.toFixed(1)}% (${formatTHB(seg.value)})`}</title>
                  </circle>
                )
              })}
            </svg>

            {/* Center Label (Interactive Hover / Total Variance Summary) */}
            <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none text-center p-4">
              {activeSummary ? (
                <div className="space-y-0.5 animate-fade-in">
                  <span className="text-xs font-bold text-zinc-500 uppercase tracking-wider">
                    {activeSummary.name}
                  </span>
                  <div className="text-xl font-extrabold font-mono tabular-nums text-zinc-900">
                    {activeSummary.actual_percent.toFixed(1)}%
                  </div>
                  <div className="text-[11px] font-mono tabular-nums text-zinc-500">
                    Target: {activeTarget ? `${activeTarget.target_percent.toFixed(1)}%` : 'N/A'}
                  </div>
                  <div className={`text-[11px] font-mono tabular-nums font-bold ${activeSummary.variance > 0 ? 'text-amber-600' : activeSummary.variance < 0 ? 'text-rose-600' : 'text-emerald-600'}`}>
                    Diff: {activeSummary.variance > 0 ? '+' : ''}{activeSummary.variance.toFixed(1)}%
                  </div>
                </div>
              ) : (
                <div className="space-y-0.5 animate-fade-in">
                  <span className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">Total Variance</span>
                  <div className={`text-2xl font-extrabold font-mono tabular-nums ${totalVariance <= 10 ? 'text-emerald-600' : 'text-amber-600'}`}>
                    {totalVariance.toFixed(1)}%
                  </div>
                  <span className="text-[11px] text-zinc-400">{summaries.length} Buckets (ชี้เพื่อดูรายละเอียด)</span>
                </div>
              )}
            </div>
          </div>

          <div className="border-t border-sky-100 pt-3 text-center">
            <span className="text-xs text-flow-blue font-medium">
              💡 คลิกที่วงหรือแถวตารางเพื่อกรองรายการ Holding ตาม Strategy Bucket
            </span>
          </div>
        </div>

        {/* Strategy Buckets Table Panel */}
        <div className="rounded-2xl border border-sky-100 bg-panel shadow-sm lg:col-span-7 flex flex-col overflow-hidden">
          <div className="border-b border-sky-100 bg-sky-50/40 px-6 py-4 flex items-center justify-between">
            <div>
              <h3 className="text-base font-bold text-zinc-900">Strategy Buckets Breakdown</h3>
              <p className="text-xs text-zinc-500">
                เปรียบเทียบมูลค่าจริงกับเป้าหมายแต่ละกลุ่ม พร้อมค่าความเบี่ยงเบน (Variance)
              </p>
            </div>
            <div className="flex items-center gap-2">
              <span className="rounded-xl border border-sky-200 bg-white px-3 py-1 text-xs font-semibold text-zinc-700 shadow-sm">
                {summaries.length} Buckets
              </span>
              {onSuccess && (
                <button
                  type="button"
                  onClick={() => setTargetModalOpen(true)}
                  className="rounded-xl bg-flow-blue px-3 py-1 text-xs font-bold text-white shadow-sm hover:bg-sky-600 transition-colors"
                >
                  ⚙️ ตั้งค่าเป้าหมาย
                </button>
              )}
            </div>
          </div>

          <div className="flex-1 overflow-x-auto p-2">
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="border-b border-sky-100 text-xs font-semibold uppercase tracking-wider text-zinc-500">
                  <th className="px-4 py-3">Bucket Name</th>
                  <th className="px-4 py-3 text-right">Target %</th>
                  <th className="px-4 py-3 text-right">Actual Value</th>
                  <th className="px-4 py-3 text-right">Actual %</th>
                  <th className="px-4 py-3 text-right">Variance</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-sky-50">
                {summaries.map((s, idx) => {
                  const color = getBucketColor(idx, s.color)
                  const isHovered = hoveredBucket === s.bucket_id
                  const varColorClass =
                    Math.abs(s.variance) <= 2.0
                      ? 'text-emerald-700 bg-emerald-50 border-emerald-200'
                      : s.variance > 0
                        ? 'text-amber-700 bg-amber-50 border-amber-200'
                        : 'text-rose-700 bg-rose-50 border-rose-200'

                  // คำนวณความยาว mini variance bar (สูงสุด 50%)
                  const barPercent = Math.min(50, (Math.abs(s.variance) / 20) * 50)

                  return (
                    <tr
                      key={s.bucket_id}
                      onClick={() => onSelectBucket(s.bucket_id)}
                      onMouseEnter={() => setHoveredBucket(s.bucket_id)}
                      onMouseLeave={() => setHoveredBucket(null)}
                      className={`cursor-pointer transition-all ${
                        isHovered ? 'bg-flow-cyan/15 scale-[1.005]' : 'hover:bg-sky-50/60'
                      }`}
                    >
                      <td className="px-4 py-3.5 font-medium text-zinc-900 flex items-center gap-2.5">
                        <span
                          className="h-3 w-3 shrink-0 rounded-full shadow-sm"
                          style={{ backgroundColor: color }}
                        />
                        <div>
                          <div className="font-bold text-zinc-800 group-hover:text-flow-blue">{s.name}</div>
                          <div className="text-[11px] font-mono tabular-nums text-zinc-400">{s.bucket_id}</div>
                        </div>
                      </td>
                      <td className="px-4 py-3.5 text-right font-mono tabular-nums text-zinc-700">
                        {s.target_percent.toFixed(1)}%
                      </td>
                      <td className="px-4 py-3.5 text-right font-mono tabular-nums font-semibold text-zinc-900">
                        {formatTHB(s.actual_value_thb)}
                      </td>
                      <td className="px-4 py-3.5 text-right font-mono tabular-nums font-bold text-zinc-900">
                        {s.actual_percent.toFixed(1)}%
                      </td>
                      <td className="px-4 py-3.5 text-right align-middle">
                        <div className="flex flex-col items-end gap-1">
                          <span
                            className={`inline-block rounded-lg border px-2 py-0.5 text-xs font-mono tabular-nums font-bold ${varColorClass}`}
                          >
                            {s.variance > 0 ? '+' : ''}
                            {s.variance.toFixed(1)}%
                          </span>
                          {/* Mini Variance Divergence Bar */}
                          <div className="w-24 h-1.5 bg-zinc-100 rounded-full overflow-hidden relative flex items-center">
                            <div className="absolute left-1/2 top-0 bottom-0 w-[1px] bg-zinc-300 z-10" />
                            {s.variance > 0 ? (
                              <div
                                className="h-full bg-amber-500 rounded-r-full transition-all duration-300 absolute left-1/2"
                                style={{ width: `${barPercent}%` }}
                              />
                            ) : s.variance < 0 ? (
                              <div
                                className="h-full bg-rose-500 rounded-l-full transition-all duration-300 absolute right-1/2"
                                style={{ width: `${barPercent}%` }}
                              />
                            ) : null}
                          </div>
                        </div>
                      </td>
                    </tr>
                  )
                })}
                {summaries.length === 0 && (
                  <tr>
                    <td colSpan={5} className="py-16 text-center">
                      <div className="mx-auto max-w-md flex flex-col items-center justify-center space-y-4 animate-fade-in">
                        <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-sky-50 border border-sky-100 text-flow-blue shadow-sm">
                          <TradeIcon className="h-8 w-8" />
                        </div>
                        <div className="space-y-1">
                          <h4 className="text-base font-extrabold text-zinc-900">ยังไม่มีสินทรัพย์ในพอร์ตการลงทุน</h4>
                          <p className="text-xs text-zinc-500 max-w-sm">
                            เริ่มต้นสร้างพอร์ตของคุณด้วยการบันทึกรายการเทรด หรือซื้อหุ้น/กองทุนแรกเข้าสู่ระบบเพื่อติดตามสัดส่วนตามกลยุทธ์
                          </p>
                        </div>
                        {onOpenTradeModal && (
                          <button
                            type="button"
                            onClick={onOpenTradeModal}
                            className="inline-flex items-center gap-2 rounded-xl bg-flow-blue px-4 py-2.5 text-xs font-bold text-white shadow-md hover:bg-sky-600 transition-all hover:scale-105"
                          >
                            <TradeIcon className="h-4 w-4" />
                            <span>บันทึกเทรดแรกของคุณทันที</span>
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {targetModalOpen && onSuccess && (
        <BucketTargetModal
          initialTargets={targets}
          onClose={() => setTargetModalOpen(false)}
          onSuccess={onSuccess}
        />
      )}
    </div>
  )
}
