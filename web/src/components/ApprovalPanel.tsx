import { useState } from 'react'
import type { ApprovalPayload, NewsFunnelApprovalPayload, NewsYoutubeApprovalPayload } from '../api/types'

interface Props {
  payload: ApprovalPayload
  onApprove: (approvedNewsLinks: string[], approvedYoutubeLinks: string[], approvedEventIds?: string[]) => void
  submitting?: boolean
}

export default function ApprovalPanel({ payload, onApprove, submitting }: Props) {
  if (payload.type === 'news_funnel_approval') {
    return <NewsFunnelApprovalView payload={payload} onApprove={onApprove} submitting={submitting} />
  }
  return <NewsYoutubeApprovalView payload={payload} onApprove={onApprove} submitting={submitting} />
}

function NewsFunnelApprovalView({
  payload,
  onApprove,
  submitting,
}: {
  payload: NewsFunnelApprovalPayload
  onApprove: Props['onApprove']
  submitting?: boolean
}) {
  const [selectedEventIds, setSelectedEventIds] = useState<Set<string>>(new Set())

  const visibleCandidates = payload.candidates
    .slice()
    .sort((a, b) => (a.triage_source === 'heuristic_fallback' ? 1 : 0) - (b.triage_source === 'heuristic_fallback' ? 1 : 0))

  function toggle(id: string) {
    const next = new Set(selectedEventIds)
    if (next.has(id)) next.delete(id)
    else next.add(id)
    setSelectedEventIds(next)
  }

  function toggleAll() {
    if (selectedEventIds.size === visibleCandidates.length && visibleCandidates.length > 0) {
      setSelectedEventIds(new Set())
    } else {
      setSelectedEventIds(new Set(visibleCandidates.map((c) => c.event_id)))
    }
  }

  const allSelected = visibleCandidates.length > 0 && selectedEventIds.size === visibleCandidates.length

  return (
    <div className="space-y-4 rounded-xl border border-amber-200 bg-amber-50 p-4 shadow-sm shadow-black/5">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-amber-500" />
          <h3 className="text-sm font-semibold text-amber-800">รอการอนุมัติ — เลือกรายการข่าว High-Impact ที่ต้องการสังเคราะห์</h3>
        </div>
        {visibleCandidates.length > 0 && (
          <button
            type="button"
            onClick={toggleAll}
            className="text-xs font-medium text-sky-700 hover:underline"
          >
            {allSelected ? 'ยกเลิกทั้งหมด' : 'เลือกทั้งหมด'}
          </button>
        )}
      </div>

      {visibleCandidates.length > 0 ? (
        <ul className="space-y-2">
          {visibleCandidates.map((c) => {
            const maxScore = Math.max(c.macro_impact_score || 0, c.asset_impact_score || 0)
            return (
              <li key={c.event_id}>
                <label className="relative flex cursor-pointer items-start gap-2.5 rounded-lg border border-edge bg-panel p-3 text-xs text-zinc-700 transition-colors hover:border-zinc-300">
                  <input
                    type="checkbox"
                    checked={selectedEventIds.has(c.event_id)}
                    onChange={() => toggle(c.event_id)}
                    className="mt-1 accent-sky-500"
                  />
                  <div className="flex-1 space-y-1.5 pr-20">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="rounded bg-amber-100 px-1.5 py-0.5 text-[11px] font-semibold text-amber-800">
                        Score: {maxScore}/10
                      </span>
                      {c.triage_source === 'heuristic_fallback' && (
                        <span
                          title="คะแนนจาก heuristic fallback (LLM triage ล้มเหลวรอบ ingest) — โปรดตรวจสอบเนื้อหาก่อนอนุมัติ"
                          className="rounded border border-red-200 bg-red-50 px-1.5 py-0.5 text-[10px] font-semibold text-red-700"
                        >
                          ⚠️ Heuristic
                        </span>
                      )}
                      <span className="font-semibold text-zinc-900">{c.canonical_title}</span>
                    </div>
                    {c.comprehensive_summary && (
                      <p className="text-zinc-600">{c.comprehensive_summary}</p>
                    )}
                    <div className="flex flex-wrap gap-1.5 pt-0.5">
                      {c.extracted_tickers?.map((ticker) => {
                        const t = ticker.replace(/^\[\[|\]\]$/g, '').split('|')[0]?.trim() || ''
                        if (!t) return null
                        return (
                          <span
                            key={ticker}
                            className="rounded border border-sky-200 bg-sky-50 px-1.5 py-0.5 text-[10px] font-medium text-sky-800"
                          >
                            {t}
                          </span>
                        )
                      })}
                      {c.extracted_themes?.map((theme) => {
                        const th = theme.replace(/^\[\[|\]\]$/g, '').split('|')[0]?.trim() || ''
                        if (!th) return null
                        return (
                          <span
                            key={theme}
                            className="rounded border border-purple-200 bg-purple-50 px-1.5 py-0.5 text-[10px] font-medium text-purple-800"
                          >
                            {th}
                          </span>
                        )
                      })}
                    </div>
                  </div>
                </label>
              </li>
            )
          })}
        </ul>
      ) : (
        <p className="text-xs text-zinc-500">ไม่มีรายการข่าว High-Impact รออนุมัติ</p>
      )}

      <button
        onClick={() => onApprove([], [], Array.from(selectedEventIds))}
        disabled={submitting}
        className={`rounded-lg px-4 py-2 text-sm font-medium text-white transition-colors disabled:opacity-50 ${
          selectedEventIds.size === 0 ? 'bg-zinc-600 hover:bg-zinc-700' : 'bg-sky-500 hover:bg-sky-600'
        }`}
      >
        {submitting
          ? 'กำลังส่ง...'
          : selectedEventIds.size === 0
            ? 'ข้ามรอบนี้ (0 รายการ) — ไม่สังเคราะห์และไม่ปฏิเสธข่าว'
            : `อนุมัติและดำเนินการต่อ (${selectedEventIds.size} รายการ)`}
      </button>
    </div>
  )
}

function NewsYoutubeApprovalView({
  payload,
  onApprove,
  submitting,
}: {
  payload: NewsYoutubeApprovalPayload
  onApprove: Props['onApprove']
  submitting?: boolean
}) {
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
        className={`rounded-lg px-4 py-2 text-sm font-medium text-white transition-colors disabled:opacity-50 ${
          totalSelected === 0 ? 'bg-zinc-600 hover:bg-zinc-700' : 'bg-sky-500 hover:bg-sky-600'
        }`}
      >
        {submitting
          ? 'กำลังส่ง...'
          : totalSelected === 0
            ? 'ข้ามรอบนี้ (0 รายการ) — ไม่บันทึก'
            : `อนุมัติและดำเนินการต่อ (${totalSelected} รายการ)`}
      </button>
    </div>
  )
}
