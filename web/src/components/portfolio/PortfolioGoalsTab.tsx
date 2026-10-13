import { useState } from 'react'
import type { ActualGoalItemDTO, ActualGoalsResponseDTO } from '../../api/types'
import { formatTHB } from '../../utils/formatters'
import { api } from '../../api/client'
import GoalModal from './Modals/GoalModal'

interface Props {
  goals: ActualGoalItemDTO[]
  generatedAt: string | null
  portfolios?: import('../../api/types').PortfolioMetaDTO[]
  selectedPortfolioId?: string
  allocationPolicy?: import('../../api/types').ActualPortfolioStateDTO['allocation_policy']
  bucketSummaries?: import('../../api/types').BucketAllocationSummaryDTO[]
  onSuccess?: (res: ActualGoalsResponseDTO) => void
}

export default function PortfolioGoalsTab({
  goals,
  generatedAt,
  portfolios = [],
  selectedPortfolioId = 'default',
  allocationPolicy,
  bucketSummaries = [],
  onSuccess,
}: Props) {
  const [modalOpen, setModalOpen] = useState(false)
  const [editingGoal, setEditingGoal] = useState<ActualGoalItemDTO | null>(null)

  const handleDelete = async (name: string) => {
    if (!onSuccess) return
    if (!window.confirm(`คุณต้องการลบเป้าหมาย "${name}" หรือไม่?`)) return
    try {
      // ไม่กรองตามพอร์ต — ให้ response ที่คืนมายังคงมีเป้าหมายของทุกพอร์ตครบ ไม่ใช่แค่พอร์ตที่เลือกอยู่
      const res = await api.removeGoal(name)
      onSuccess(res)
    } catch (err: any) {
      alert(err?.message || 'ลบเป้าหมายไม่สำเร็จ')
    }
  }

  const getGoalTypeBadge = (type: string, bucketId?: string | null) => {
    switch (type) {
      case 'nav_target':
        return <span className="rounded-full bg-indigo-50 px-2.5 py-0.5 text-xs font-semibold text-indigo-700">เป้าหมาย NAV รวม</span>
      case 'cash_target':
        return <span className="rounded-full bg-emerald-50 px-2.5 py-0.5 text-xs font-semibold text-emerald-700">เป้าหมายเงินสด/สำรอง</span>
      case 'passive_income_ytd':
        return <span className="rounded-full bg-amber-50 px-2.5 py-0.5 text-xs font-semibold text-amber-700">เป้าหมายเงินปันผล/ปี</span>
      case 'bucket_target':
        return <span className="rounded-full bg-purple-50 px-2.5 py-0.5 text-xs font-semibold text-purple-700">Bucket: {bucketId || 'N/A'}</span>
      default:
        return <span className="rounded-full bg-sky-50 px-2.5 py-0.5 text-xs font-semibold text-sky-700">{type}</span>
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-center">
        <div>
          <h3 className="text-base font-bold text-zinc-900">Portfolio Goals ({goals.length} เป้าหมาย)</h3>
          <p className="text-xs text-zinc-500">
            ติดตามความคืบหน้าของเป้าหมายการลงทุน (Financial Goals) เทียบกับสถานะจริงในแต่ละพอร์ต
          </p>
        </div>
        <div className="flex items-center gap-3">
          {generatedAt && (
            <span className="text-xs text-zinc-400">
              อัปเดต: {new Date(generatedAt).toLocaleString('th-TH')}
            </span>
          )}
          {onSuccess && (
            <button
              type="button"
              onClick={() => {
                setEditingGoal(null)
                setModalOpen(true)
              }}
              className="rounded-xl bg-flow-blue px-4 py-2 text-xs font-bold text-white shadow-md hover:bg-sky-600 transition-colors"
            >
              + Add Goal
            </button>
          )}
        </div>
      </div>

      {/* Goals Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 2xl:grid-cols-3 gap-5">
        {goals.map((goal) => {
          const progress = Math.max(0, Math.min(100, goal.progress_pct))
          const isCompleted = progress >= 100
          const linkedPortName = portfolios.find(p => p.id === (goal.portfolio_id || 'default'))?.name || goal.portfolio_id || 'default'

          return (
            <div
              key={goal.name}
              className="flex flex-col justify-between rounded-2xl border border-sky-100 bg-panel p-5 sm:p-6 shadow-sm transition-all hover:shadow-md relative group"
            >
              <div>
                {/* Header Top Bar: Badges & Actions */}
                <div className="flex items-center justify-between gap-2 pb-2.5 border-b border-sky-100/70">
                  <div className="flex items-center gap-1.5 flex-wrap min-w-0">
                    {getGoalTypeBadge(goal.goal_type, goal.bucket_id)}
                    <span className="inline-flex items-center gap-1 rounded-full bg-zinc-100/80 px-2 py-0.5 text-[11px] font-medium text-zinc-600 truncate max-w-[180px]">
                      📁 {linkedPortName}
                    </span>
                  </div>
                  {onSuccess && (
                    <div className="flex items-center gap-1 shrink-0 opacity-80 group-hover:opacity-100 transition-opacity">
                      <button
                        type="button"
                        onClick={() => setEditingGoal(goal)}
                        className="rounded-lg p-1.5 text-zinc-400 hover:bg-sky-50 hover:text-flow-blue transition-colors"
                        title="แก้ไขเป้าหมาย"
                      >
                        ✏️
                      </button>
                      <button
                        type="button"
                        onClick={() => handleDelete(goal.name)}
                        className="rounded-lg p-1.5 text-zinc-400 hover:bg-rose-50 hover:text-rose-600 transition-colors"
                        title="ลบเป้าหมาย"
                      >
                        🗑️
                      </button>
                    </div>
                  )}
                </div>

                {/* Header Title: Full Width */}
                <div className="mt-3">
                  <h4 className="font-extrabold text-zinc-900 text-base sm:text-lg leading-snug tracking-tight">
                    {goal.name}
                  </h4>
                </div>

                {/* Progress Bar & Percentage */}
                <div className="mt-4 space-y-1.5">
                  <div className="flex items-baseline justify-between text-xs sm:text-sm">
                    <span className="text-zinc-500 font-medium">ความคืบหน้า</span>
                    <span className="font-extrabold text-zinc-900 font-mono text-sm sm:text-base">{progress.toFixed(1)}%</span>
                  </div>
                  {/* Progress Bar */}
                  <div className="h-2.5 w-full overflow-hidden rounded-full bg-sky-100/60 p-0.5">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${
                        isCompleted
                          ? 'bg-emerald-500'
                          : progress > 70
                            ? 'bg-flow-blue'
                            : progress > 30
                              ? 'bg-amber-500'
                              : 'bg-rose-400'
                      }`}
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                </div>

                {/* Current & Target Amount Box */}
                <div className="mt-4 grid grid-cols-2 gap-3 rounded-xl bg-sky-50/60 p-3.5 border border-sky-100/60">
                  <div className="min-w-0">
                    <span className="text-zinc-500 font-medium block text-[11px] mb-0.5">Current Amount</span>
                    <span className="font-bold font-sans tabular-nums text-sm sm:text-base text-zinc-900 block truncate" title={formatTHB(goal.current_amount_thb)}>
                      {formatTHB(goal.current_amount_thb)}
                    </span>
                  </div>
                  <div className="min-w-0 text-right">
                    <span className="text-zinc-500 font-medium block text-[11px] mb-0.5">Target Amount</span>
                    <span className="font-extrabold font-sans tabular-nums text-sm sm:text-base text-flow-blue block truncate" title={formatTHB(goal.target_amount_thb)}>
                      {formatTHB(goal.target_amount_thb)}
                    </span>
                  </div>
                </div>

                {/* Linked Purpose Bucket & Funding Attribution (WP13 Task 9) */}
                {goal.bucket_id && (
                  <div className="mt-3.5 rounded-xl border border-purple-100 bg-purple-50/50 p-3 text-xs">
                    {(() => {
                      const linkedBucket = allocationPolicy?.purpose_buckets?.find(
                        (pb) => pb.bucket_id === goal.bucket_id
                      )
                      const bucketSummary = bucketSummaries?.find(
                        (bs) => bs.bucket_id === goal.bucket_id
                      )
                      return (
                        <>
                          <div className="flex items-center justify-between gap-2">
                            <div className="flex items-center gap-1.5 min-w-0">
                              <span className="text-sm">🎯</span>
                              <span className="font-bold text-purple-950 truncate">
                                {linkedBucket?.name || goal.bucket_id}
                              </span>
                              {linkedBucket?.role && (
                                <span className="rounded bg-purple-100 px-1.5 py-0.5 text-[10px] text-purple-800">
                                  {linkedBucket.role}
                                </span>
                              )}
                            </div>
                            {bucketSummary && (
                              <span className="font-mono font-bold text-purple-900 tabular-nums shrink-0">
                                {formatTHB(bucketSummary.actual_value_thb)}
                              </span>
                            )}
                          </div>
                          {bucketSummary && (
                            <div className="mt-1 flex items-center justify-between text-[11px] text-purple-700/90 font-medium">
                              <span>สัดส่วนในพอร์ต: {bucketSummary.actual_percent.toFixed(1)}%</span>
                              {goal.target_amount_thb > 0 && (
                                <span>
                                  ทุนสนับสนุนเป้าหมาย:{' '}
                                  {((bucketSummary.actual_value_thb / goal.target_amount_thb) * 100).toFixed(1)}%
                                </span>
                              )}
                            </div>
                          )}
                        </>
                      )
                    })()}
                  </div>
                )}

                {goal.notes && (
                  <p className="mt-3 text-xs text-zinc-500 italic line-clamp-2">“{goal.notes}”</p>
                )}
              </div>


              <div className="mt-5 border-t border-sky-100 pt-3 flex items-center justify-between text-xs">
                <div>
                  {goal.deadline ? (
                    <span className="text-zinc-600 font-medium">
                      📅 เป้าหมาย: <strong className="text-zinc-800 font-semibold">{goal.deadline}</strong>
                    </span>
                  ) : (
                    <span className="text-zinc-400">ไม่มีกำหนดระยะเวลา</span>
                  )}
                </div>
                <div>
                  {goal.deadline_days_left !== null && goal.deadline_days_left !== undefined ? (
                    <span
                      className={`rounded-lg px-2.5 py-1 font-sans tabular-nums font-bold text-xs ${
                        goal.deadline_days_left < 30
                          ? 'bg-rose-50 text-rose-700 border border-rose-200/60'
                          : 'bg-sky-50 text-sky-700 border border-sky-200/60'
                      }`}
                    >
                      เหลือ {goal.deadline_days_left} วัน
                    </span>
                  ) : isCompleted ? (
                    <span className="rounded-lg bg-emerald-50 px-2.5 py-1 font-bold text-emerald-700 border border-emerald-200/60 text-xs">สำเร็จแล้ว 🎉</span>
                  ) : null}
                </div>
              </div>
            </div>
          )
        })}

        {goals.length === 0 && (
          <div className="col-span-full rounded-2xl border border-dashed border-sky-200 p-12 text-center text-zinc-400">
            ยังไม่มีเป้าหมายการลงทุน (Goals) ในขณะนี้
          </div>
        )}
      </div>

      {(modalOpen || editingGoal) && onSuccess && (
        <GoalModal
          initialGoal={editingGoal}
          existingNames={goals.map((g) => g.name)}
          portfolios={portfolios}
          selectedPortfolioId={selectedPortfolioId}
          onClose={() => {
            setModalOpen(false)
            setEditingGoal(null)
          }}
          onSuccess={(res) => {
            onSuccess(res)
          }}
        />
      )}
    </div>
  )
}
