import { useState } from 'react'
import type { NewsYoutubeApprovalPayload } from '../api/types'

interface Props {
  payload: NewsYoutubeApprovalPayload
  onApprove: (approvedNewsLinks: string[], approvedYoutubeLinks: string[]) => void
  submitting?: boolean
}

export default function ApprovalPanel({ payload, onApprove, submitting }: Props) {
  const [selectedNews, setSelectedNews] = useState<Set<string>>(new Set())
  const [selectedYoutube, setSelectedYoutube] = useState<Set<string>>(new Set())
  const [showFetched, setShowFetched] = useState(false)

  function toggle(set: Set<string>, setSet: (s: Set<string>) => void, link: string) {
    const next = new Set(set)
    if (next.has(link)) next.delete(link)
    else next.add(link)
    setSet(next)
  }

  function toggleAll(items: { link: string }[], set: Set<string>, setSet: (s: Set<string>) => void) {
    const allSelected = items.length > 0 && items.every((i) => set.has(i.link))
    setSet(allSelected ? new Set() : new Set(items.map((i) => i.link)))
  }

  const totalSelected = selectedNews.size + selectedYoutube.size

  const fetchedNewsCount = payload.news_candidates.filter((n) => n.is_fetched).length
  const fetchedYoutubeCount = payload.youtube_candidates.filter((v) => v.is_fetched).length
  const fetchedCount = fetchedNewsCount + fetchedYoutubeCount

  const visibleNews = showFetched ? payload.news_candidates : payload.news_candidates.filter((n) => !n.is_fetched)
  const visibleYoutube = showFetched
    ? payload.youtube_candidates
    : payload.youtube_candidates.filter((v) => !v.is_fetched)

  return (
    <div className="space-y-4 rounded-xl border border-amber-200 bg-amber-50 p-4 shadow-sm shadow-black/5">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-amber-500" />
          <h3 className="text-sm font-semibold text-amber-800">รอการอนุมัติ — เลือกรายการที่ต้องการเจาะลึก</h3>
        </div>
        {fetchedCount > 0 && (
          <button
            type="button"
            onClick={() => setShowFetched((v) => !v)}
            className="shrink-0 text-xs font-medium text-amber-800 hover:underline"
          >
            {showFetched ? 'ซ่อนรายการที่อ่านแล้ว' : `แสดงรายการที่อ่านแล้ว (${fetchedCount})`}
          </button>
        )}
      </div>

      {visibleNews.length > 0 && (
        <div>
          <div className="mb-2 flex items-center justify-between">
            <h4 className="text-xs font-semibold uppercase tracking-wide text-zinc-500">
              ข่าว ({visibleNews.length})
            </h4>
            <button
              type="button"
              onClick={() => toggleAll(visibleNews, selectedNews, setSelectedNews)}
              className="text-xs font-medium text-sky-700 hover:underline"
            >
              {visibleNews.every((n) => selectedNews.has(n.link)) ? 'ยกเลิกทั้งหมด' : 'เลือกทั้งหมด'}
            </button>
          </div>
          <ul className="space-y-1.5">
            {visibleNews.map((n) => (
              <li key={n.link}>
                <label className="flex cursor-pointer items-start gap-2 rounded-lg border border-edge bg-panel p-2 text-xs text-zinc-700 hover:border-zinc-300">
                  <input
                    type="checkbox"
                    checked={selectedNews.has(n.link)}
                    onChange={() => toggle(selectedNews, setSelectedNews, n.link)}
                    className="mt-0.5 accent-sky-500"
                  />
                  <span>
                    <span className={n.is_fetched ? 'text-zinc-400 line-through' : 'text-zinc-800'}>{n.title}</span>{' '}
                    {n.is_fetched && (
                      <span className="rounded border border-edge bg-surface px-1 py-0.5 text-[10px] text-zinc-500">
                        อ่านแล้ว
                      </span>
                    )}{' '}
                    <span className="text-zinc-500">
                      · {n.source} · {n.age_hours}h{n.is_stale ? ' ⚠️' : ''}
                    </span>
                  </span>
                </label>
              </li>
            ))}
          </ul>
        </div>
      )}

      {visibleYoutube.length > 0 && (
        <div>
          <div className="mb-2 flex items-center justify-between">
            <h4 className="text-xs font-semibold uppercase tracking-wide text-zinc-500">
              YouTube ({visibleYoutube.length})
            </h4>
            <button
              type="button"
              onClick={() => toggleAll(visibleYoutube, selectedYoutube, setSelectedYoutube)}
              className="text-xs font-medium text-sky-700 hover:underline"
            >
              {visibleYoutube.every((v) => selectedYoutube.has(v.link)) ? 'ยกเลิกทั้งหมด' : 'เลือกทั้งหมด'}
            </button>
          </div>
          <ul className="space-y-1.5">
            {visibleYoutube.map((v) => (
              <li key={v.link}>
                <label className="flex cursor-pointer items-start gap-2 rounded-lg border border-edge bg-panel p-2 text-xs text-zinc-700 hover:border-zinc-300">
                  <input
                    type="checkbox"
                    checked={selectedYoutube.has(v.link)}
                    onChange={() => toggle(selectedYoutube, setSelectedYoutube, v.link)}
                    className="mt-0.5 accent-sky-500"
                  />
                  <span>
                    <span className={v.is_fetched ? 'text-zinc-400 line-through' : 'text-zinc-800'}>{v.title}</span>{' '}
                    {v.is_fetched && (
                      <span className="rounded border border-edge bg-surface px-1 py-0.5 text-[10px] text-zinc-500">
                        อ่านแล้ว
                      </span>
                    )}{' '}
                    <span className="text-zinc-500">
                      · {v.channel} · {v.published}
                    </span>
                  </span>
                </label>
              </li>
            ))}
          </ul>
        </div>
      )}

      {visibleNews.length === 0 && visibleYoutube.length === 0 && (
        <p className="text-xs text-zinc-500">
          {fetchedCount > 0 && !showFetched
            ? `ไม่มีรายการใหม่ — มี ${fetchedCount} รายการที่เคยอ่านแล้ว (กด "แสดงรายการที่อ่านแล้ว" ด้านบน)`
            : 'ไม่มีรายการใหม่ให้เลือก'}
        </p>
      )}

      <button
        onClick={() => onApprove(Array.from(selectedNews), Array.from(selectedYoutube))}
        disabled={submitting}
        className="rounded-lg bg-sky-500 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-sky-600 disabled:opacity-50"
      >
        {submitting ? 'กำลังส่ง...' : `อนุมัติและดำเนินการต่อ (${totalSelected} รายการ)`}
      </button>
    </div>
  )
}
