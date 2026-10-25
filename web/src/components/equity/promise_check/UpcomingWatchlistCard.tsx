import React, { useMemo } from 'react'
import type { PromiseItemDTO } from '../../../api/types'

interface UpcomingWatchlistCardProps {
  promises: PromiseItemDTO[]
  onSelectPromise?: (promise: PromiseItemDTO) => void
}

export const UpcomingWatchlistCard: React.FC<UpcomingWatchlistCardProps> = ({
  promises,
  onSelectPromise,
}) => {
  const pendingPromises = useMemo(() => {
    return promises
      .filter((p) => p.status === 'Pending')
      .sort((a, b) => a.due.localeCompare(b.due))
  }, [promises])

  if (pendingPromises.length === 0) {
    return null
  }

  const targetDescription = (p: PromiseItemDTO) => {
    const u = p.unit ? ` ${p.unit}` : ''
    switch (p.target_kind) {
      case 'range':
        return `${p.target_low ?? ''}–${p.target_high ?? ''}${u}`
      case 'point':
        return `เป้าหมายประมาณ ${p.target_point ?? ''}${u}`
      case 'floor':
        return `อย่างน้อย ${p.target_low ?? ''}${u}`
      case 'ceiling':
        return `ไม่เกิน ${p.target_high ?? ''}${u}`
      case 'direction':
        return `ทิศทาง${p.direction === 'up' ? 'เพิ่มขึ้น' : p.direction === 'down' ? 'ลดลง' : 'ทรงตัว'}`
      case 'milestone':
      default:
        return 'ส่งมอบความคืบหน้าตามหมุดหมาย'
    }
  }

  return (
    <div className="rounded-2xl border border-edge bg-surface p-5 shadow-sm space-y-4">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-edge/60 pb-3">
        <div>
          <h4 className="font-bold text-zinc-900 dark:text-zinc-100 text-sm flex items-center gap-2">
            <span>🎯</span>
            <span>สิ่งที่ต้องฟังในการประชุมครั้งถัดไป (Upcoming Commitments Watchlist)</span>
          </h4>
          <p className="text-xs text-zinc-500 mt-0.5">
            คำสัญญาที่ยังไม่ถึงกำหนดส่งมอบ ({pendingPromises.length} ข้อ) — สิ่งที่นักลงทุนควรเตรียมติดตามตรวจการบ้านในไตรมาสถัดไป
          </p>
        </div>
        <span className="px-2.5 py-1 rounded-xl text-xs font-semibold bg-blue-100 text-blue-800 dark:bg-blue-950 dark:text-blue-300 border border-blue-500/20 w-fit">
          รอถึงกำหนด {pendingPromises.length} รายการ
        </span>
      </div>

      {/* Watchlist Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
        {pendingPromises.slice(0, 6).map((p) => (
          <div
            key={p.id}
            onClick={() => onSelectPromise?.(p)}
            className="rounded-xl border border-edge/60 bg-surface-muted/30 p-4 space-y-2.5 hover:bg-surface-muted/60 hover:border-sky-500/50 transition-all cursor-pointer group"
          >
            {/* Top info line */}
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <span className="px-2 py-0.5 rounded text-[11px] font-bold bg-sky-100 text-sky-800 dark:bg-sky-950 dark:text-sky-300 font-mono">
                  กำหนด: {p.due}
                </span>
                <span className="text-[10px] text-zinc-400">
                  {p.due_precision === 'explicit' ? 'ระบุตรงชัดเจน' : 'อนุมาน'}
                </span>
              </div>
              <span className="text-[10px] font-mono text-zinc-400">{p.id}</span>
            </div>

            {/* Metric Headline */}
            <div>
              <h5 className="font-bold text-xs text-zinc-900 dark:text-zinc-100 group-hover:text-sky-600 dark:group-hover:text-sky-400 transition-colors">
                {p.metric_th || p.metric}
              </h5>
              <span className="text-[11px] text-sky-700 dark:text-sky-400 font-medium">
                เป้าหมาย: {targetDescription(p)}
              </span>
            </div>

            {/* Verbatim Quote Excerpt */}
            <blockquote className="text-xs text-zinc-600 dark:text-zinc-300 italic font-serif bg-surface/80 p-2.5 rounded-lg border border-edge/40 line-clamp-3">
              "{p.quote}"
            </blockquote>

            {/* Speaker & Quarter made */}
            <div className="flex items-center justify-between text-[10px] text-zinc-400 border-t border-edge/30 pt-1.5">
              <span>{p.speaker.split(',')[0]}</span>
              <span>แถลงใน {p.made_in} {p.section === 'qa' ? '(Q&A)' : '(Remarks)'}</span>
            </div>
          </div>
        ))}
      </div>

      {pendingPromises.length > 6 && (
        <div className="text-center pt-1 text-xs text-zinc-500">
          ยังมีคำสัญญาที่รอถึงกำหนดอีก {pendingPromises.length - 6} รายการในตารางด้านล่าง
        </div>
      )}
    </div>
  )
}
