import React, { useMemo } from 'react'
import type { PromiseItemDTO, PromiseScoresDTO } from '../../../api/types'

interface StandoutPromisesCardProps {
  scores: PromiseScoresDTO
  promises: PromiseItemDTO[]
  onSelectPromise?: (promise: PromiseItemDTO) => void
}

export const StandoutPromisesCard: React.FC<StandoutPromisesCardProps> = ({
  scores,
  promises,
  onSelectPromise,
}) => {
  const promiseMap = useMemo(() => {
    const map = new Map<string, PromiseItemDTO>()
    for (const p of promises) {
      map.set(p.id, p)
    }
    return map
  }, [promises])

  const topKept = useMemo(() => {
    return (scores.top_kept || [])
      .map((id) => promiseMap.get(id))
      .filter((p): p is PromiseItemDTO => p !== undefined)
  }, [scores.top_kept, promiseMap])

  const topBroken = useMemo(() => {
    return (scores.top_broken || [])
      .map((id) => promiseMap.get(id))
      .filter((p): p is PromiseItemDTO => p !== undefined)
  }, [scores.top_broken, promiseMap])

  // Don't render the section if there are no kept and no broken items
  if (topKept.length === 0 && topBroken.length === 0) {
    return null
  }

  const formatDelta = (delta: number | null | undefined) => {
    if (delta == null) return null
    const sign = delta > 0 ? '+' : ''
    const val = (delta * 100).toFixed(1)
    const color = delta >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400'
    return <span className={`font-bold ${color}`}>{sign}{val}% เทียบเป้า</span>
  }

  return (
    <div className="rounded-2xl border border-edge bg-surface p-5 shadow-sm space-y-4">
      {/* Header */}
      <div className="border-b border-edge/60 pb-3">
        <h4 className="font-bold text-zinc-900 dark:text-zinc-100 text-sm flex items-center gap-2">
          <span>🌟</span>
          <span>คำสัญญาที่น่าสนใจที่สุด (Standout Promises Highlights)</span>
        </h4>
        <p className="text-xs text-zinc-500 mt-0.5">
          เปรียบเทียบคำพูดตอนให้สัญญา vs สิ่งที่เกิดขึ้นจริงตอนครบกำหนด (ถอดความจากบทสนทนาจริง)
        </p>
      </div>

      {/* 2-Column Grid: Kept vs Broken */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* Left: Top Kept Promises */}
        <div className="space-y-3">
          <div className="flex items-center gap-2 text-xs font-bold text-emerald-700 dark:text-emerald-300">
            <span>✅</span>
            <span>ทำได้ตามที่พูดไว้ (Standout Kept Promises)</span>
            <span className="px-1.5 py-0.2 rounded text-[10px] bg-emerald-100 dark:bg-emerald-950 font-semibold">
              {topKept.length} รายการ
            </span>
          </div>

          {topKept.length === 0 ? (
            <div className="rounded-xl border border-dashed border-edge/60 p-5 text-center text-xs text-zinc-500 italic">
              ยังไม่มีคำสัญญาที่ปิดผลและทำได้
            </div>
          ) : (
            <div className="space-y-3">
              {topKept.map((p) => (
                <div
                  key={p.id}
                  onClick={() => onSelectPromise?.(p)}
                  className="rounded-xl border border-emerald-500/20 bg-emerald-500/5 dark:bg-emerald-500/10 border-l-4 border-l-emerald-500 p-4 space-y-2.5 transition-all hover:shadow-sm hover:border-emerald-500/50 cursor-pointer group"
                >
                  <div className="flex items-center justify-between gap-2 text-[11px]">
                    <span className="font-mono text-zinc-400 text-[10px]">{p.id}</span>
                    <span className="text-zinc-500 dark:text-zinc-400 truncate max-w-[200px]">
                      {p.speaker.split(',')[0]} · {p.made_in}
                    </span>
                  </div>

                  <h5 className="font-bold text-xs text-zinc-900 dark:text-zinc-100 group-hover:text-sky-600 dark:group-hover:text-sky-400 transition-colors">
                    {p.metric_th || p.metric}
                  </h5>

                  {/* Original Quote */}
                  <div className="space-y-1">
                    <span className="text-[10px] font-semibold text-zinc-500 block uppercase tracking-wider">
                      สิ่งที่พูดตอนให้สัญญา:
                    </span>
                    <blockquote className="text-xs text-zinc-700 dark:text-zinc-300 italic font-serif bg-surface/60 p-2 rounded-lg border border-edge/40">
                      "{p.quote}"
                    </blockquote>
                  </div>

                  {/* Resolving Quote & Outcome */}
                  <div className="space-y-1 pt-1">
                    <div className="flex items-center justify-between text-[10px]">
                      <span className="font-semibold text-emerald-700 dark:text-emerald-300">
                        สิ่งที่เกิดขึ้นจริงใน {p.resolved_in || p.due}:
                      </span>
                      {formatDelta(p.delta_pct)}
                    </div>
                    {p.evidence_quote ? (
                      <blockquote className="text-xs text-emerald-900 dark:text-emerald-200 italic font-serif bg-emerald-500/10 p-2 rounded-lg border border-emerald-500/20">
                        "{p.evidence_quote}"
                      </blockquote>
                    ) : (
                      <div className="text-[11px] text-zinc-500 italic">
                        ได้รับการยืนยันตามกำหนด
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Right: Top Broken / Dropped Promises */}
        <div className="space-y-3">
          <div className="flex items-center gap-2 text-xs font-bold text-rose-700 dark:text-rose-300">
            <span>⚠️</span>
            <span>พลาดเป้าหรือทิ้งสัญญาเงียบๆ (Broken or Dropped)</span>
            <span className="px-1.5 py-0.2 rounded text-[10px] bg-rose-100 dark:bg-rose-950 font-semibold">
              {topBroken.length} รายการ
            </span>
          </div>

          {topBroken.length === 0 ? (
            <div className="rounded-xl border border-emerald-200/60 bg-emerald-50/40 dark:bg-emerald-950/20 p-5 text-center text-xs text-emerald-800 dark:text-emerald-300 space-y-1">
              <span className="text-xl block">🎉</span>
              <span className="font-bold block">ไม่มีคำสัญญาที่พลาดเป้าหรือทิ้งเงียบๆ</span>
              <p className="text-[11px] text-zinc-500">
                ผู้บริหารชุดนี้ส่งมอบงานตามสัญญาครบถ้วนทุกรายการที่ครบกำหนดในรอบ 8 ไตรมาส
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              {topBroken.map((p) => {
                const isDropped = p.status === 'Quietly dropped'
                return (
                  <div
                    key={p.id}
                    onClick={() => onSelectPromise?.(p)}
                    className="rounded-xl border border-rose-500/20 bg-rose-500/5 dark:bg-rose-500/10 border-l-4 border-l-rose-500 p-4 space-y-2.5 transition-all hover:shadow-sm hover:border-rose-500/50 cursor-pointer group"
                  >
                    <div className="flex items-center justify-between gap-2 text-[11px]">
                      <span className="font-mono text-zinc-400 text-[10px]">{p.id}</span>
                      <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-rose-100 text-rose-800 dark:bg-rose-950 dark:text-rose-300">
                        {isDropped ? '👻 ละทิ้งเงียบๆ' : p.status === 'Restated' ? '🔄 ปรับลดเป้า' : '❌ พลาดเป้า'}
                      </span>
                    </div>

                    <h5 className="font-bold text-xs text-zinc-900 dark:text-zinc-100 group-hover:text-rose-600 dark:group-hover:text-rose-400 transition-colors">
                      {p.metric_th || p.metric}
                    </h5>

                    {/* Original Quote */}
                    <div className="space-y-1">
                      <span className="text-[10px] font-semibold text-zinc-500 block uppercase tracking-wider">
                        สิ่งที่พูดตอนให้สัญญา ({p.made_in}):
                      </span>
                      <blockquote className="text-xs text-zinc-700 dark:text-zinc-300 italic font-serif bg-surface/60 p-2 rounded-lg border border-edge/40">
                        "{p.quote}"
                      </blockquote>
                    </div>

                    {/* What Happened */}
                    <div className="space-y-1 pt-1">
                      <span className="text-[10px] font-semibold text-rose-700 dark:text-rose-300 block">
                        สิ่งที่เกิดขึ้นจริง:
                      </span>
                      {p.evidence_quote ? (
                        <blockquote className="text-xs text-rose-900 dark:text-rose-200 italic font-serif bg-rose-500/10 p-2 rounded-lg border border-rose-500/20">
                          "{p.evidence_quote}"
                        </blockquote>
                      ) : isDropped ? (
                        <div className="text-xs text-rose-800 dark:text-rose-300 bg-rose-500/10 p-2 rounded-lg border border-rose-500/20 italic">
                          ไม่ถูกพูดถึงอีกเลยในการประชุมครั้งต่อๆ มา และไม่มีการแถลงปรับเป้าหมายต่อสาธารณะ
                        </div>
                      ) : (
                        <div className="text-xs text-zinc-500 italic">
                          {p.notes_th || p.notes || 'ส่งมอบไม่เป็นไปตามเป้าหมาย'}
                        </div>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
