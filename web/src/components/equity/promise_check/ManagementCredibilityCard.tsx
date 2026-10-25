import React from 'react'
import type { PromiseScoresDTO } from '../../../api/types'

interface ManagementCredibilityCardProps {
  scores: PromiseScoresDTO
}

export const ManagementCredibilityCard: React.FC<ManagementCredibilityCardProps> = ({ scores }) => {
  const credibility = scores.credibility ?? 0
  const grade = scores.grade || 'N/A'
  const confidence = scores.confidence || 'Low'
  const delivery = scores.delivery ?? 0
  const transparency = scores.transparency ?? 100

  // Color scheme based on letter grade
  const defaultColor = { bg: 'bg-amber-500/10', text: 'text-amber-600 dark:text-amber-400', border: 'border-amber-500/30', ring: 'ring-amber-500/20' }
  const gradeColors: Record<string, { bg: string; text: string; border: string; ring: string }> = {
    A: { bg: 'bg-emerald-500/10', text: 'text-emerald-600 dark:text-emerald-400', border: 'border-emerald-500/30', ring: 'ring-emerald-500/20' },
    B: { bg: 'bg-sky-500/10', text: 'text-sky-600 dark:text-sky-400', border: 'border-sky-500/30', ring: 'ring-sky-500/20' },
    C: defaultColor,
    D: { bg: 'bg-orange-500/10', text: 'text-orange-600 dark:text-orange-400', border: 'border-orange-500/30', ring: 'ring-orange-500/20' },
    F: { bg: 'bg-rose-500/10', text: 'text-rose-600 dark:text-rose-400', border: 'border-rose-500/30', ring: 'ring-rose-500/20' },
  }

  const color = (grade && gradeColors[grade]) ? gradeColors[grade]! : defaultColor

  return (
    <div className="rounded-2xl border border-edge bg-surface p-5 shadow-sm space-y-4">
      {/* Title & Rolling Window */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-edge/60 pb-3">
        <div className="flex items-center gap-2">
          <span className="text-xl">🏆</span>
          <div>
            <h4 className="font-bold text-zinc-900 dark:text-zinc-100 text-sm">
              Management Credibility Score
            </h4>
            <p className="text-xs text-zinc-500">
              ความน่าเชื่อถือของผู้บริหาร ย้อนหลัง 8 ไตรมาส ({scores.window.from} – {scores.window.to})
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-zinc-500">ความเชื่อมั่นทางสถิติ:</span>
          <span className={`px-2 py-0.5 rounded text-[11px] font-semibold ${
            confidence === 'High' ? 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300' :
            confidence === 'Medium' ? 'bg-sky-100 text-sky-800 dark:bg-sky-950 dark:text-sky-300' :
            'bg-zinc-100 text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300'
          }`}>
            {confidence} ({scores.n_resolved} สัญญาที่ปิดผล)
          </span>
        </div>
      </div>

      {/* Main Score & Sub-Pillars */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-center">
        {/* Overall Score & Grade Badge */}
        <div className={`flex items-center justify-between p-4 rounded-xl border ${color.border} ${color.bg} shadow-inner`}>
          <div>
            <span className="text-xs font-semibold text-zinc-600 dark:text-zinc-400 uppercase tracking-wider block">
              คะแนนรวม (0–100)
            </span>
            <div className="flex items-baseline gap-1 mt-1">
              <span className={`text-4xl font-extrabold ${color.text}`}>
                {credibility.toFixed(1)}
              </span>
              <span className="text-sm font-medium text-zinc-500">/ 100</span>
            </div>
          </div>
          <div className={`w-14 h-14 rounded-2xl flex flex-col items-center justify-center font-black text-2xl border-2 ${color.border} ${color.text} bg-surface shadow-sm`}>
            {grade}
            <span className="text-[9px] font-semibold text-zinc-400 -mt-1">GRADE</span>
          </div>
        </div>

        {/* Pillar 1: Delivery Score */}
        <div className="p-4 rounded-xl border border-edge bg-surface-muted/30 space-y-2">
          <div className="flex justify-between items-center text-xs">
            <span className="font-semibold text-zinc-700 dark:text-zinc-300">
              Delivery Rate (น้ำหนัก 80%)
            </span>
            <span className="font-bold text-zinc-900 dark:text-zinc-100">
              {delivery.toFixed(1)}%
            </span>
          </div>
          <div className="w-full bg-zinc-200 dark:bg-zinc-800 h-2 rounded-full overflow-hidden">
            <div
              className="bg-emerald-500 h-full rounded-full transition-all duration-500"
              style={{ width: `${Math.min(100, Math.max(0, delivery))}%` }}
            />
          </div>
          <p className="text-[11px] text-zinc-500 leading-tight">
            ทำได้จริงตามตัวเลข Guidance และส่งมอบ Milestone ตามกำหนด
          </p>
        </div>

        {/* Pillar 2: Transparency Score */}
        <div className="p-4 rounded-xl border border-edge bg-surface-muted/30 space-y-2">
          <div className="flex justify-between items-center text-xs">
            <span className="font-semibold text-zinc-700 dark:text-zinc-300">
              Transparency Score (น้ำหนัก 20%)
            </span>
            <span className="font-bold text-zinc-900 dark:text-zinc-100">
              {transparency.toFixed(1)}%
            </span>
          </div>
          <div className="w-full bg-zinc-200 dark:bg-zinc-800 h-2 rounded-full overflow-hidden">
            <div
              className="bg-sky-500 h-full rounded-full transition-all duration-500"
              style={{ width: `${Math.min(100, Math.max(0, transparency))}%` }}
            />
          </div>
          <p className="text-[11px] text-zinc-500 leading-tight">
            กล้ายอมรับข้อผิดพลาดเมื่อทำไม่ได้ (หักคะแนนหนักเมื่อทิ้งสัญญาเงียบๆ)
          </p>
        </div>
      </div>

      {/* Status Counters Strip */}
      <div className="pt-2 border-t border-edge/40 flex flex-wrap items-center gap-2 text-xs">
        <span className="text-zinc-500 font-medium">ผลการส่งมอบ:</span>
        <span className="px-2.5 py-1 rounded-lg bg-emerald-500/10 text-emerald-700 dark:text-emerald-300 font-medium border border-emerald-500/20">
          ✅ สำเร็จ {scores.counts['Delivered'] || 0}
        </span>
        <span className="px-2.5 py-1 rounded-lg bg-amber-500/10 text-amber-700 dark:text-amber-300 font-medium border border-amber-500/20">
          ⚠️ สำเร็จบางส่วน {scores.counts['Partial'] || 0}
        </span>
        <span className="px-2.5 py-1 rounded-lg bg-rose-500/10 text-rose-700 dark:text-rose-300 font-medium border border-rose-500/20">
          ❌ พลาดเป้า {scores.counts['Missed'] || 0}
        </span>
        <span className="px-2.5 py-1 rounded-lg bg-purple-500/10 text-purple-700 dark:text-purple-300 font-medium border border-purple-500/20">
          🔄 ปรับเป้าลง {scores.counts['Restated (lowered)'] || 0}
        </span>
        <span className="px-2.5 py-1 rounded-lg bg-zinc-500/10 text-zinc-700 dark:text-zinc-300 font-medium border border-zinc-500/20">
          👻 ทิ้งสัญญาเงียบๆ {scores.counts['Quietly dropped'] || 0}
        </span>
        <span className="px-2.5 py-1 rounded-lg bg-blue-500/10 text-blue-700 dark:text-blue-300 font-medium border border-blue-500/20 ml-auto">
          ⏳ รอสรุปผล {scores.counts['Pending'] || 0}
        </span>
      </div>
    </div>
  )
}
