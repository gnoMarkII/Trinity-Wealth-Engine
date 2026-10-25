import React from 'react'
import type { CohortTrendItemDTO } from '../../../api/types'

interface PromiseCohortTrendChartProps {
  trend: {
    verdict: string
    older_half_hit_rate?: number | null
    newer_half_hit_rate?: number | null
    cohorts?: CohortTrendItemDTO[]
  }
}

export const PromiseCohortTrendChart: React.FC<PromiseCohortTrendChartProps> = ({ trend }) => {
  const cohorts = trend.cohorts || []

  const defaultVerdict = { text: 'ℹ️ ข้อมูลยังอยู่ในช่วงเริ่มต้น (Insufficient Data)', color: 'text-zinc-700 dark:text-zinc-300', bg: 'bg-zinc-500/10' }
  const verdictLabel: Record<string, { text: string; color: string; bg: string }> = {
    improving: { text: '📈 ความน่าเชื่อถือมีแนวโน้มพัฒนาดีขึ้น (Improving)', color: 'text-emerald-700 dark:text-emerald-300', bg: 'bg-emerald-500/10' },
    consistent: { text: '⚖️ ส่งมอบงานได้สม่ำเสมอตามมาตรฐาน (Consistent)', color: 'text-sky-700 dark:text-sky-300', bg: 'bg-sky-500/10' },
    stable: { text: '⚖️ ส่งมอบงานได้สม่ำเสมอตามมาตรฐาน (Stable)', color: 'text-sky-700 dark:text-sky-300', bg: 'bg-sky-500/10' },
    declining: { text: '📉 สถิติการทำตามสัญญาลดลงในระยะหลัง (Declining)', color: 'text-rose-700 dark:text-rose-300', bg: 'bg-rose-500/10' },
    deteriorating: { text: '📉 สถิติการทำตามสัญญาลดลงในระยะหลัง (Deteriorating)', color: 'text-rose-700 dark:text-rose-300', bg: 'bg-rose-500/10' },
    'insufficient data': defaultVerdict,
    'insufficient_data': defaultVerdict,
  }

  const verdictConfig = (trend.verdict && verdictLabel[trend.verdict]) ? verdictLabel[trend.verdict]! : defaultVerdict

  return (
    <div className="rounded-2xl border border-edge bg-surface p-5 shadow-sm space-y-4">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-edge/60 pb-3">
        <div>
          <h4 className="font-bold text-zinc-900 dark:text-zinc-100 text-sm flex items-center gap-2">
            <span>📊</span>
            <span>แนวโน้มการส่งมอบตามรุ่นสัญญา 8 ไตรมาส (8-Quarter Cohort Trend)</span>
          </h4>
          <p className="text-xs text-zinc-500 mt-0.5">
            อัตราความสำเร็จ (Hit Rate) ของสัญญาที่ให้ไว้ในแต่ละไตรมาส เปรียบเทียบครึ่งแรก vs ครึ่งหลัง
          </p>
        </div>
        <div className={`px-2.5 py-1 rounded-xl text-xs font-semibold ${verdictConfig.bg} ${verdictConfig.color} border border-current/20`}>
          {verdictConfig.text}
        </div>
      </div>

      {/* 8 Cohort Bars */}
      <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-8 gap-2 pt-2">
        {cohorts.map((c) => {
          const hitRate = c.hit_rate != null ? Math.round(c.hit_rate * 100) : null
          return (
            <div
              key={c.quarter}
              className="p-3 rounded-xl border border-edge/60 bg-surface-muted/30 flex flex-col justify-between space-y-2 text-center"
            >
              <div>
                <span className="text-[11px] font-bold text-zinc-900 dark:text-zinc-100 block">
                  {c.label || c.quarter}
                </span>
                <span className="text-[10px] text-zinc-400 block">
                  {c.made} สัญญา
                </span>
              </div>

              {/* Progress visual */}
              <div className="py-2">
                {hitRate != null ? (
                  <div className="space-y-1">
                    <span className={`text-base font-extrabold ${
                      hitRate >= 80 ? 'text-emerald-600 dark:text-emerald-400' :
                      hitRate >= 50 ? 'text-amber-600 dark:text-amber-400' :
                      'text-rose-600 dark:text-rose-400'
                    }`}>
                      {hitRate}%
                    </span>
                    <div className="w-full bg-zinc-200 dark:bg-zinc-800 h-1.5 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full ${
                          hitRate >= 80 ? 'bg-emerald-500' : hitRate >= 50 ? 'bg-amber-500' : 'bg-rose-500'
                        }`}
                        style={{ width: `${hitRate}%` }}
                      />
                    </div>
                  </div>
                ) : c.pending > 0 ? (
                  <div className="py-1">
                    <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold bg-blue-100 text-blue-800 dark:bg-blue-950 dark:text-blue-300">
                      รอผล {c.pending}
                    </span>
                  </div>
                ) : (
                  <span className="text-xs text-zinc-400 italic block py-1">—</span>
                )}
              </div>

              <div className="text-[10px] text-zinc-500 border-t border-edge/30 pt-1">
                {c.resolved === 0 ? (
                  <span className="text-zinc-400">ยังไม่ถึงกำหนด</span>
                ) : (
                  <span>สำเร็จ {c.delivered}/{c.resolved}</span>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
