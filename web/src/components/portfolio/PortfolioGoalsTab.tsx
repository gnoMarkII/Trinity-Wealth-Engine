import { useState } from 'react'
import type { ActualGoalItemDTO, ActualGoalsResponseDTO } from '../../api/types'
import { formatTHB } from './PortfolioSummaryCards'
import { api } from '../../api/client'
import GoalModal from './Modals/GoalModal'

interface Props {
  goals: ActualGoalItemDTO[]
  generatedAt: string | null
  onSuccess?: (res: ActualGoalsResponseDTO) => void
}

export default function PortfolioGoalsTab({ goals, generatedAt, onSuccess }: Props) {
  const [modalOpen, setModalOpen] = useState(false)
  const [editingGoal, setEditingGoal] = useState<ActualGoalItemDTO | null>(null)

  const handleDelete = async (name: string) => {
    if (!onSuccess) return
    if (!window.confirm(`คุณต้องการลบเป้าหมาย "${name}" หรือไม่?`)) return
    try {
      const res = await api.removeGoal(name)
      onSuccess(res)
    } catch (err: any) {
      alert(err?.message || 'ลบเป้าหมายไม่สำเร็จ')
    }
  }

  const getGoalTypeBadge = (type: string) => {
    switch (type) {
      case 'nav_target':
        return <span className="rounded-full bg-indigo-50 px-2.5 py-0.5 text-xs font-semibold text-indigo-700">เป้าหมาย NAV รวม</span>
      case 'cash_target':
        return <span className="rounded-full bg-emerald-50 px-2.5 py-0.5 text-xs font-semibold text-emerald-700">เป้าหมายเงินสด/สำรอง</span>
      case 'passive_income_ytd':
        return <span className="rounded-full bg-amber-50 px-2.5 py-0.5 text-xs font-semibold text-amber-700">เป้าหมายเงินปันผล/ปี</span>
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
            ติดตามความคืบหน้าของเป้าหมายการลงทุน (Financial Goals) เทียบกับสถานะจริงในปัจจุบัน
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
      <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
        {goals.map((goal) => {
          const progress = Math.max(0, Math.min(100, goal.progress_pct))
          const isCompleted = progress >= 100

          return (
            <div
              key={goal.name}
              className="flex flex-col justify-between rounded-2xl border border-sky-100 bg-panel p-6 shadow-sm transition-all hover:shadow-md relative group"
            >
              <div>
                <div className="flex items-start justify-between gap-2">
                  <h4 className="font-bold text-zinc-900 text-base">{goal.name}</h4>
                  <div className="flex items-center gap-1">
                    {getGoalTypeBadge(goal.goal_type)}
                    {onSuccess && (
                      <div className="flex items-center gap-1 ml-1 opacity-80 group-hover:opacity-100 transition-opacity">
                        <button
                          type="button"
                          onClick={() => setEditingGoal(goal)}
                          className="rounded p-1 text-zinc-400 hover:bg-sky-50 hover:text-flow-blue"
                          title="แก้ไขเป้าหมาย"
                        >
                          ✏️
                        </button>
                        <button
                          type="button"
                          onClick={() => handleDelete(goal.name)}
                          className="rounded p-1 text-zinc-400 hover:bg-rose-50 hover:text-rose-600"
                          title="ลบเป้าหมาย"
                        >
                          🗑️
                        </button>
                      </div>
                    )}
                  </div>
                </div>

                <div className="mt-5 space-y-2">
                  <div className="flex items-baseline justify-between text-sm">
                    <span className="text-zinc-500">ความคืบหน้า</span>
                    <span className="font-extrabold text-zinc-900">{progress.toFixed(1)}%</span>
                  </div>
                  {/* Progress Bar */}
                  <div className="h-3 w-full overflow-hidden rounded-full bg-sky-100/60 p-0.5">
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

                <div className="mt-4 grid grid-cols-2 gap-2 rounded-xl bg-sky-50/40 p-3 text-xs">
                  <div>
                    <span className="text-zinc-400 block">Current Amount</span>
                    <span className="font-bold font-mono tabular-nums text-zinc-800">{formatTHB(goal.current_amount_thb)}</span>
                  </div>
                  <div className="text-right">
                    <span className="text-zinc-400 block">Target Amount</span>
                    <span className="font-bold font-mono tabular-nums text-flow-blue">{formatTHB(goal.target_amount_thb)}</span>
                  </div>
                </div>

                {goal.notes && (
                  <p className="mt-3 text-xs text-zinc-500 italic line-clamp-2">“{goal.notes}”</p>
                )}
              </div>

              <div className="mt-5 border-t border-sky-100 pt-3 flex items-center justify-between text-xs">
                <div>
                  {goal.deadline ? (
                    <span className="text-zinc-500">
                      📅 เป้าหมาย: <strong className="text-zinc-700">{goal.deadline}</strong>
                    </span>
                  ) : (
                    <span className="text-zinc-400">ไม่มีกำหนดระยะเวลา</span>
                  )}
                </div>
                <div>
                  {goal.deadline_days_left !== null && goal.deadline_days_left !== undefined ? (
                    <span
                      className={`rounded-lg px-2 py-0.5 font-mono tabular-nums font-bold ${
                        goal.deadline_days_left < 30
                          ? 'bg-rose-50 text-rose-700'
                          : 'bg-sky-50 text-sky-700'
                      }`}
                    >
                      เหลือ {goal.deadline_days_left} วัน
                    </span>
                  ) : isCompleted ? (
                    <span className="rounded-lg bg-emerald-50 px-2 py-0.5 font-bold text-emerald-700">สำเร็จแล้ว 🎉</span>
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
