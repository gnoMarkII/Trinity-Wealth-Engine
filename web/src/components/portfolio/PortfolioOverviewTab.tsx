import { useState, useMemo } from 'react'
import type { AllocationTargetDTO, BucketAllocationSummaryDTO, ActualPortfolioStateDTO, ActualHoldingDTO, PerformanceSnapshotDTO } from '../../api/types'
import { formatTHB } from '../../utils/formatters'
import BucketTargetModal from './Modals/BucketTargetModal'
import AiBucketAssignModal from './Modals/AiBucketAssignModal'
import { TradeIcon } from './icons/PortfolioIcons'
import PortfolioPerformanceCard from './PortfolioPerformanceCard'
import { PortfolioOverviewSkeleton } from '../ui/FlowSkeleton'

interface Props {
  portfolioId: string
  targets: AllocationTargetDTO[]
  summaries: BucketAllocationSummaryDTO[]
  warning: string | null
  loading?: boolean
  holdings?: ActualHoldingDTO[]
  allocationPolicy?: ActualPortfolioStateDTO['allocation_policy']
  allocationPlanOrigin?: ActualPortfolioStateDTO['allocation_plan_origin']
  onSelectBucket: (bucketId: string) => void
  onSuccess?: (state: ActualPortfolioStateDTO) => void
  onOpenTradeModal?: () => void
  onNavigateToPrinciples?: () => void
  performanceRows?: PerformanceSnapshotDTO[]
  daysRange?: number | undefined
  onChangeDaysRange?: (days: number | undefined) => void
  onSnapshotTriggered?: () => void
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

export default function PortfolioOverviewTab({
  portfolioId,
  targets,
  summaries,
  warning,
  loading = false,
  holdings,
  allocationPolicy,
  allocationPlanOrigin,
  onSelectBucket,
  onSuccess,
  onOpenTradeModal,
  onNavigateToPrinciples,
  performanceRows,
  daysRange,
  onChangeDaysRange,
  onSnapshotTriggered,
}: Props) {
  const [hoveredBucket, setHoveredBucket] = useState<string | null>(null)
  const [targetModalOpen, setTargetModalOpen] = useState(false)
  const [aiAssignModalOpen, setAiAssignModalOpen] = useState(false)

  const unassignedCount = useMemo(() => {
    if (!holdings) return 0
    return holdings.filter(
      (h) => !h.bucket_id || h.bucket_id.toLowerCase() === 'unassigned'
    ).length
  }, [holdings])


  // Geometry calculations for Donut Chart
  const innerRadius = 96
  const outerRadius = 132
  const strokeWidthInner = 18
  const strokeWidthOuter = 24
  const innerCircumference = 2 * Math.PI * innerRadius
  const outerCircumference = 2 * Math.PI * outerRadius


  let innerOffset = 0
  const innerSegments = targets.map((t, idx) => {
    const pct = Math.max(0, Math.min(100, t.target_percent))
    const rawDash = (pct / 100) * innerCircumference
    const gap = targets.length > 1 && rawDash > 16 ? 10 : 0
    const strokeDasharray = `${Math.max(0.1, rawDash - gap)} ${innerCircumference - rawDash + gap}`
    const strokeDashoffset = -innerOffset - gap / 2
    innerOffset += rawDash
    return {
      bucketId: t.bucket_id,
      name: t.name,
      pct: t.target_percent,
      strokeDasharray,
      strokeDashoffset,
      color: getBucketColor(idx, t.color),
    }
  })

  let outerOffset = 0
  const outerSegments = summaries.map((s, idx) => {
    const pct = Math.max(0, Math.min(100, s.actual_percent))
    const rawDash = (pct / 100) * outerCircumference
    const gap = summaries.length > 1 && rawDash > 16 ? 12 : 0
    const strokeDasharray = `${Math.max(0.1, rawDash - gap)} ${outerCircumference - rawDash + gap}`
    const strokeDashoffset = -outerOffset - gap / 2
    outerOffset += rawDash
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

  // Total Absolute Variance
  const totalVariance = summaries.reduce((acc, s) => acc + Math.abs(s.variance), 0)

  // Rebalance Check (Threshold ±5.0%)
  const rebalanceCandidates = useMemo(() => {
    return summaries
      .filter((s) => Math.abs(s.variance) >= 5.0)
      .map((s) => ({
        ...s,
        direction: s.variance > 0 ? ('over' as const) : ('under' as const),
      }))
  }, [summaries])

  if (loading && summaries.length === 0) {
    return <PortfolioOverviewSkeleton />
  }

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

      {/* Applied Policy & Lineage Banner (WP13 Task 8) */}
      {allocationPolicy || allocationPlanOrigin ? (
        <div className="rounded-2xl border border-sky-200 bg-gradient-to-r from-sky-50/90 via-white to-blue-50/60 p-5 shadow-xs flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-start gap-3.5">
            <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-sky-100 text-sky-700 font-bold text-xl shadow-2xs">
              📜
            </span>
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <h3 className="text-base font-extrabold text-zinc-900">
                  นโยบายกลยุทธ์ที่ใช้งานอยู่ (Active Allocation Policy)
                </h3>
                <span className="rounded-full bg-emerald-100 px-2.5 py-0.5 text-[11px] font-bold text-emerald-800">
                  ✓ ยืนยันและนำมาใช้แล้ว (Applied)
                </span>
                {allocationPlanOrigin?.is_modified_since_apply && (
                  <span className="rounded-full bg-amber-100 px-2.5 py-0.5 text-[11px] font-bold text-amber-900 flex items-center gap-1">
                    <span>⚠️</span>
                    <span>มีการแก้ไข Targets เพิ่มเติม</span>
                  </span>
                )}
              </div>

              <p className="text-xs text-zinc-600 mt-1">
                พอร์ตนี้ควบคุมด้วยแผนการจัดสรร Purpose Buckets
                {allocationPolicy?.applied_at_iso
                  ? ` (มีผลเมื่อ ${new Date(allocationPolicy.applied_at_iso).toLocaleString('th-TH')})`
                  : ''}
              </p>

              {/* Provenance source badges */}
              <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px] font-mono text-zinc-500">
                {allocationPlanOrigin?.essence_ref && (
                  <span className="rounded bg-sky-100/70 px-2 py-0.5 text-sky-800">
                    Essence: {allocationPlanOrigin.essence_ref.slice(0, 14)}...
                  </span>
                )}
                {allocationPlanOrigin?.axis_ref && (
                  <span className="rounded bg-indigo-100/70 px-2 py-0.5 text-indigo-800">
                    Axis: {allocationPlanOrigin.axis_ref.slice(0, 14)}...
                  </span>
                )}
                {allocationPlanOrigin?.applied_draft_id && (
                  <span className="rounded bg-zinc-100 px-2 py-0.5 text-zinc-700">
                    Draft: {allocationPlanOrigin.applied_draft_id}
                  </span>
                )}
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            {allocationPlanOrigin?.is_modified_since_apply && onNavigateToPrinciples && (
              <button
                type="button"
                onClick={onNavigateToPrinciples}
                className="rounded-xl border border-amber-300 bg-amber-50 px-3.5 py-2 text-xs font-bold text-amber-900 shadow-2xs hover:bg-amber-100 transition-colors"
              >
                ทบทวนแผนใหม่ →
              </button>
            )}
            {onNavigateToPrinciples && (
              <button
                type="button"
                onClick={onNavigateToPrinciples}
                className="rounded-xl bg-sky-600 px-4 py-2 text-xs font-bold text-white shadow-2xs hover:bg-sky-700 transition-colors"
              >
                ดูหลักการลงทุน (Principles) →
              </button>
            )}
          </div>
        </div>
      ) : (
        /* Guidance Card if no policy applied yet */
        <div className="rounded-2xl border border-sky-100 bg-gradient-to-r from-sky-50/80 via-white to-blue-50/50 p-4 sm:p-5 shadow-xs flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-sky-100 text-sky-700 font-bold text-lg shadow-xs">
              🧭
            </span>
            <div>
              <h3 className="text-sm sm:text-base font-bold text-zinc-900 flex items-center gap-2">
                <span>หลักการลงทุน (Investor Principles)</span>
                <span className="rounded-full bg-sky-100 px-2 py-0.5 text-[10px] font-bold text-sky-700">
                  ยังไม่ได้ Apply แผน
                </span>
              </h3>
              <p className="text-xs text-zinc-500 mt-0.5">
                ค้นหาแก่นแท้ในการลงทุน กำหนดแกนหลัก 8 หัวข้อ และสร้าง Purpose Buckets สำหรับพอร์ตนี้
              </p>
            </div>
          </div>

          {onNavigateToPrinciples && (
            <button
              type="button"
              onClick={onNavigateToPrinciples}
              className="shrink-0 rounded-xl bg-gradient-to-r from-sky-600 to-blue-600 px-4 py-2 text-xs sm:text-sm font-bold text-white shadow-xs hover:from-sky-700 hover:to-blue-700 active:scale-98 transition-all flex items-center gap-1.5 justify-center"
            >
              <span>ดูหลักการลงทุน / ค้นหาแก่นแท้ →</span>
            </button>
          )}
        </div>
      )}

      {/* Rebalance Review Indicator (WP13 Task 8) */}
      {rebalanceCandidates.length > 0 ? (
        <div className="rounded-2xl border border-amber-300 bg-gradient-to-r from-amber-50/80 via-white to-orange-50/60 p-4 sm:p-5 shadow-xs">
          <div className="flex items-start gap-3">
            <span className="text-2xl shrink-0 mt-0.5">⚖️</span>
            <div className="space-y-1 flex-1">
              <div className="flex items-center gap-2">
                <h4 className="text-sm sm:text-base font-extrabold text-amber-950">
                  แจ้งเตือนการปรับสมดุลพอร์ต (Rebalance Review Alert)
                </h4>
                <span className="rounded-full bg-amber-200/90 px-2.5 py-0.5 text-[11px] font-bold text-amber-900">
                  {rebalanceCandidates.length} Buckets เบี่ยงเบนเกิน ±5%
                </span>
              </div>
              <p className="text-xs text-amber-900/90 leading-relaxed">
                สัดส่วนมูลค่าจริงในปัจจุบันเบี่ยงเบนจากเป้าหมายกลยุทธ์เกินเกณฑ์แนะนำ (±5.0%)
                คุณสามารถจัดสรรเงินลงทุนใหม่ (DCA) ตามตารางด้านล่างเพื่อลดช่องว่าง หรือปรับสัดส่วนผ่านการเทรด
              </p>
              <div className="pt-2 flex flex-wrap gap-2">
                {rebalanceCandidates.map((c) => (
                  <span
                    key={c.bucket_id}
                    className={`inline-flex items-center gap-1.5 rounded-lg border px-2.5 py-1 text-xs font-bold font-sans tabular-nums ${
                      c.direction === 'over'
                        ? 'border-amber-300 bg-amber-100/80 text-amber-900'
                        : 'border-rose-300 bg-rose-100/80 text-rose-900'
                    }`}
                  >
                    <span>{c.name}:</span>
                    <span>
                      {c.variance > 0 ? '+' : ''}
                      {c.variance.toFixed(1)}%
                    </span>
                    <span className="text-[10px] font-normal opacity-75">
                      ({c.direction === 'over' ? 'Overweight' : 'Underweight'})
                    </span>
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>
      ) : summaries.length > 0 ? (
        <div className="rounded-xl border border-emerald-200 bg-emerald-50/70 px-4 py-2.5 shadow-2xs flex items-center justify-between">
          <div className="flex items-center gap-2 text-xs font-semibold text-emerald-900">
            <span>✅</span>
            <span>พอร์ตอยู่ในเกณฑ์สมดุลตามเป้าหมาย (ทุก Bucket อยู่ในกรอบเบี่ยงเบน ±5.0%)</span>
          </div>
          <span className="text-xs font-mono font-bold text-emerald-800 tabular-nums">
            Total Variance: {totalVariance.toFixed(1)}%
          </span>
        </div>
      ) : null}

      {/* Grid: Donut Chart + Table */}
      <div className="grid grid-cols-1 gap-6 2xl:grid-cols-12">
        {/* Concentric Double-Ring Donut Chart Panel */}
        <div className="rounded-2xl border border-sky-100 bg-panel p-5 sm:p-6 shadow-sm 2xl:col-span-5 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between gap-2">
              <h3 className="text-base font-bold text-zinc-900 truncate">Allocation Rings (Target vs Actual)</h3>
              <span className="rounded-full bg-sky-50 px-2.5 py-1 text-[11px] sm:text-xs font-medium text-sky-700 whitespace-nowrap shrink-0">
                วงใน: Target | วงนอก: Actual
              </span>
            </div>
            <p className="mt-1 text-xs text-zinc-500">
              เปรียบเทียบสัดส่วนเป้าหมาย (วงใน) กับมูลค่าจริงของพอร์ต (วงนอก)
            </p>

            <div className="mt-5 flex flex-col items-center justify-center">
              <div className="relative flex items-center justify-center">
                <svg width="280" height="280" viewBox="0 0 300 300" className="transform -rotate-90 max-w-full">
                  {/* Background Track Rings */}
                  <circle
                    cx="150"
                    cy="150"
                    r={innerRadius}
                    fill="none"
                    stroke="#f1f5f9"
                    strokeWidth={strokeWidthInner}
                  />
                  <circle
                    cx="150"
                    cy="150"
                    r={outerRadius}
                    fill="none"
                    stroke="#f8fafc"
                    strokeWidth={strokeWidthOuter}
                  />

                  {/* Inner Ring (Targets) */}
                  {innerSegments.map((s) => (
                    <circle
                      key={`inner-${s.bucketId}`}
                      cx="150"
                      cy="150"
                      r={innerRadius}
                      fill="none"
                      stroke={s.color}
                      strokeWidth={strokeWidthInner}
                      strokeDasharray={s.strokeDasharray}
                      strokeDashoffset={s.strokeDashoffset}
                      strokeLinecap="round"
                      className="transition-all duration-500 opacity-85 hover:opacity-100 cursor-pointer"
                      onClick={() => onSelectBucket(s.bucketId)}
                      onMouseEnter={() => setHoveredBucket(s.bucketId)}
                      onMouseLeave={() => setHoveredBucket(null)}
                    />
                  ))}

                  {/* Outer Ring (Actuals) */}
                  {outerSegments.map((s) => (
                    <circle
                      key={`outer-${s.bucketId}`}
                      cx="150"
                      cy="150"
                      r={outerRadius}
                      fill="none"
                      stroke={s.color}
                      strokeWidth={strokeWidthOuter}
                      strokeDasharray={s.strokeDasharray}
                      strokeDashoffset={s.strokeDashoffset}
                      strokeLinecap="round"
                      className="transition-all duration-500 hover:stroke-[28px] cursor-pointer"
                      onClick={() => onSelectBucket(s.bucketId)}
                      onMouseEnter={() => setHoveredBucket(s.bucketId)}
                      onMouseLeave={() => setHoveredBucket(null)}
                    />
                  ))}
                </svg>

                {/* Center Stats Display */}
                <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none text-center px-4">
                  {activeSummary ? (
                    <div className="space-y-1 animate-fade-in flex flex-col items-center justify-center">
                      <span className="text-xs font-bold text-zinc-500 uppercase tracking-wider line-clamp-1 max-w-[150px]">
                        {activeSummary.name}
                      </span>
                      <div className="text-2xl font-black font-sans tabular-nums text-zinc-900 tracking-tight leading-none my-0.5">
                        {activeSummary.actual_percent.toFixed(1)}%
                      </div>
                      <div className="text-xs font-semibold text-zinc-500">
                        Target: {activeTarget ? `${activeTarget.target_percent.toFixed(1)}%` : 'N/A'}
                      </div>
                      <div className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-md text-xs font-sans tabular-nums font-bold ${activeSummary.variance > 0 ? 'bg-amber-50 text-amber-700 border border-amber-200' : activeSummary.variance < 0 ? 'bg-rose-50 text-rose-700 border border-rose-200' : 'bg-emerald-50 text-emerald-700 border border-emerald-200'}`}>
                        <span>Diff: {activeSummary.variance > 0 ? '+' : ''}{activeSummary.variance.toFixed(1)}%</span>
                      </div>
                    </div>
                  ) : (
                    <div className="space-y-1 animate-fade-in flex flex-col items-center justify-center">
                      <span className="text-xs font-extrabold text-zinc-400 uppercase tracking-wider">Total Variance</span>
                      <div className={`text-2xl sm:text-3xl font-black font-sans tabular-nums tracking-tight leading-none my-1 ${totalVariance <= 10 ? 'text-emerald-600' : 'text-amber-600'}`}>
                        {totalVariance.toFixed(1)}%
                      </div>
                      <div className="text-xs font-semibold text-zinc-400 max-w-[140px] leading-tight text-center">
                        {summaries.length} Buckets <br /> (ชี้เพื่อดูรายละเอียด)
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>

            <div className="border-t border-sky-100 pt-3 text-center">
              <span className="text-xs sm:text-sm text-flow-blue font-bold">
                💡 คลิกที่วงหรือแถวตารางเพื่อกรองรายการ Holding ตาม Strategy Bucket
              </span>
            </div>
          </div>
        </div>

        {/* Strategy Buckets Table Panel */}
        <div className="rounded-2xl border border-sky-100 bg-panel shadow-sm 2xl:col-span-7 flex flex-col overflow-hidden">
          <div className="border-b border-sky-100 bg-sky-50/40 px-4 sm:px-6 py-3.5 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div>
              <h3 className="text-base sm:text-lg font-bold text-zinc-900">Strategy Buckets Breakdown</h3>
              <p className="text-xs text-zinc-500 font-medium">
                เปรียบเทียบมูลค่าจริงกับเป้าหมายแต่ละกลุ่ม พร้อมค่าความเบี่ยงเบน (Variance)
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-2 shrink-0">
              <span className="rounded-xl border border-sky-200 bg-white px-2.5 py-1 text-xs font-bold text-zinc-700 shadow-xs whitespace-nowrap">
                {summaries.length} Buckets
              </span>
              {onSuccess && (
                <>
                  <button
                    type="button"
                    aria-label="จัดสินทรัพย์เข้า Bucket ด้วย AI"
                    onClick={() => setAiAssignModalOpen(true)}
                    className="inline-flex items-center gap-1.5 rounded-xl border border-indigo-200 bg-gradient-to-r from-indigo-50 to-sky-50 px-3 py-1.5 text-xs font-bold text-indigo-900 shadow-xs hover:from-indigo-100 hover:to-sky-100 transition-all whitespace-nowrap"
                  >
                    <span>✨ จัดเข้า Bucket ด้วย AI</span>
                    {unassignedCount > 0 && (
                      <span className="rounded-full bg-amber-500 px-1.5 py-0.5 text-[10px] font-black text-white leading-none">
                        {unassignedCount}
                      </span>
                    )}
                  </button>
                  <button
                    type="button"
                    onClick={() => setTargetModalOpen(true)}
                    className="rounded-xl bg-flow-blue px-3 py-1.5 text-xs font-bold text-white shadow-xs hover:bg-sky-600 transition-colors whitespace-nowrap"
                  >
                    ⚙️ ตั้งค่าเป้าหมาย
                  </button>
                </>
              )}
            </div>
          </div>

          <div className="flex-1 overflow-x-auto p-2 custom-scrollbar">
            <table className="w-full text-left text-sm min-w-[560px]">
              <thead>
                <tr className="border-b border-sky-100 bg-sky-50/60 text-xs font-bold uppercase tracking-wider text-zinc-700">
                  <th className="px-3.5 py-3 whitespace-nowrap">Bucket Name</th>
                  <th className="px-3.5 py-3 text-right whitespace-nowrap">Target %</th>
                  <th className="px-3.5 py-3 text-right whitespace-nowrap">Actual Value</th>
                  <th className="px-3.5 py-3 text-right whitespace-nowrap">Actual %</th>
                  <th className="px-3.5 py-3 text-right whitespace-nowrap">Variance</th>
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
                      <td className="px-3.5 py-3.5 font-medium text-zinc-900 flex items-center gap-2.5" aria-label="Bucket Details">
                        <span
                          className="h-3 w-3 shrink-0 rounded-full shadow-xs"
                          style={{ backgroundColor: color }}
                        />
                        <div className="min-w-0">
                          <div className="font-extrabold text-zinc-900 text-sm sm:text-base group-hover:text-flow-blue truncate max-w-[180px] sm:max-w-none">{s.name}</div>
                          <div className="text-[11px] font-mono tabular-nums text-zinc-400 font-semibold truncate">{s.bucket_id}</div>
                        </div>
                      </td>
                      <td className="px-3.5 py-3.5 text-right font-sans tabular-nums font-bold text-zinc-800 text-xs sm:text-sm whitespace-nowrap">
                        {s.target_percent.toFixed(1)}%
                      </td>
                      <td className="px-3.5 py-3.5 text-right font-sans tabular-nums font-extrabold text-zinc-900 text-xs sm:text-sm whitespace-nowrap">
                        {formatTHB(s.actual_value_thb)}
                      </td>
                      <td className="px-3.5 py-3.5 text-right font-sans tabular-nums font-extrabold text-zinc-900 text-xs sm:text-sm whitespace-nowrap">
                        {s.actual_percent.toFixed(1)}%
                      </td>
                      <td className="px-3.5 py-3.5 text-right align-middle whitespace-nowrap">
                        <div className="flex flex-col items-end gap-1">
                          <span
                            className={`inline-block rounded-lg border px-2 py-0.5 text-xs font-sans tabular-nums font-bold ${varColorClass}`}
                          >
                            {s.variance > 0 ? '+' : ''}
                            {s.variance.toFixed(1)}%
                          </span>
                          <div className="w-16 sm:w-20 h-1.5 bg-zinc-100 rounded-full overflow-hidden relative flex items-center">
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

      {/* Portfolio Performance & NAV Growth Card (Streamlined Overview Integration) */}
      <PortfolioPerformanceCard
        performanceRows={performanceRows ?? []}
        daysRange={daysRange}
        onChangeDaysRange={onChangeDaysRange ?? (() => {})}
        portfolioId={portfolioId}
        onSnapshotTriggered={onSnapshotTriggered}
      />

      {targetModalOpen && onSuccess && (
        <BucketTargetModal
          portfolioId={portfolioId}
          initialTargets={targets}
          onClose={() => setTargetModalOpen(false)}
          onSuccess={onSuccess}
        />
      )}

      {aiAssignModalOpen && onSuccess && (
        <AiBucketAssignModal
          portfolioId={portfolioId}
          targets={targets}
          onClose={() => setAiAssignModalOpen(false)}
          onSuccess={(state) => {
            onSuccess(state)
            setAiAssignModalOpen(false)
          }}
        />
      )}
    </div>
  )
}
