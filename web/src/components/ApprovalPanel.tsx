import { useState } from 'react'
import type {
  ApprovalPayload,
  NewsFunnelApprovalPayload,
  NewsYoutubeApprovalPayload,
  YoutubePitchApprovalPayload,
} from '../api/types'

interface Props {
  payload: ApprovalPayload
  onApprove: (
    approvedNewsLinks: string[],
    approvedYoutubeLinks: string[],
    approvedEventIds?: string[],
    approvedPitchIds?: string[],
    action?: 'approve' | 'refresh_sources',
    unverifiedDraftSelections?: import('../api/types').UnverifiedDraftSelection[],
    pitchPresentationStyles?: Record<string, string>
  ) => void
  submitting?: boolean
}

export default function ApprovalPanel({ payload, onApprove, submitting }: Props) {
  if (payload.type === 'news_funnel_approval') {
    return <NewsFunnelApprovalView payload={payload} onApprove={onApprove} submitting={submitting} />
  }
  if (payload.type === 'youtube_pitch_approval') {
    return <YoutubePitchApprovalView payload={payload} onApprove={onApprove} submitting={submitting} />
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
                          title={`คะแนนจาก heuristic fallback (${c.triage_fallback_reason ? `สาเหตุ: ${c.triage_fallback_reason}` : 'LLM triage ล้มเหลวรอบ ingest'}) — โปรดตรวจสอบเนื้อหาก่อนอนุมัติ`}
                          className="rounded border border-red-200 bg-red-50 px-1.5 py-0.5 text-[10px] font-semibold text-red-700"
                        >
                          ⚠️ Heuristic{c.triage_fallback_reason ? ` (${c.triage_fallback_reason})` : ''}
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
  const [youtubeSort, setYoutubeSort] = useState<'date_desc' | 'date_asc' | 'channel'>('date_desc')

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

  const sortedYoutube = visibleYoutube.slice().sort((a, b) => {
    if (youtubeSort === 'date_desc') {
      const cmp = (b.published || '').localeCompare(a.published || '')
      if (cmp !== 0) return cmp
      const chCmp = (a.channel || '').localeCompare(b.channel || '')
      return chCmp !== 0 ? chCmp : (a.title || '').localeCompare(b.title || '')
    }
    if (youtubeSort === 'date_asc') {
      const cmp = (a.published || '').localeCompare(b.published || '')
      if (cmp !== 0) return cmp
      const chCmp = (a.channel || '').localeCompare(b.channel || '')
      return chCmp !== 0 ? chCmp : (a.title || '').localeCompare(b.title || '')
    }
    // channel (default / original grouping by channel)
    const chCmp = (a.channel || '').localeCompare(b.channel || '')
    if (chCmp !== 0) return chCmp
    return (b.published || '').localeCompare(a.published || '')
  })

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

      {sortedYoutube.length > 0 && (
        <div>
          <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
            <div className="flex flex-wrap items-center gap-2">
              <h4 className="text-xs font-semibold uppercase tracking-wide text-zinc-500">
                YouTube ({sortedYoutube.length})
              </h4>
              <div className="flex items-center gap-1 rounded-md border border-edge bg-white px-1.5 py-0.5 text-[11px] shadow-2xs">
                <span className="text-zinc-400">เรียงตาม:</span>
                <button
                  type="button"
                  onClick={() => setYoutubeSort('date_desc')}
                  className={`rounded px-1.5 py-0.5 font-medium transition-colors ${
                    youtubeSort === 'date_desc'
                      ? 'bg-sky-500 text-white shadow-2xs'
                      : 'text-zinc-600 hover:bg-zinc-100 hover:text-zinc-900'
                  }`}
                  title="เรียงตามวันที่ใหม่สุดไปเก่าสุด"
                >
                  วันที่ล่าสุด ↓
                </button>
                <button
                  type="button"
                  onClick={() => setYoutubeSort('date_asc')}
                  className={`rounded px-1.5 py-0.5 font-medium transition-colors ${
                    youtubeSort === 'date_asc'
                      ? 'bg-sky-500 text-white shadow-2xs'
                      : 'text-zinc-600 hover:bg-zinc-100 hover:text-zinc-900'
                  }`}
                  title="เรียงตามวันที่เก่าสุดไปใหม่สุด"
                >
                  วันที่เก่าสุด ↑
                </button>
                <button
                  type="button"
                  onClick={() => setYoutubeSort('channel')}
                  className={`rounded px-1.5 py-0.5 font-medium transition-colors ${
                    youtubeSort === 'channel'
                      ? 'bg-sky-500 text-white shadow-2xs'
                      : 'text-zinc-600 hover:bg-zinc-100 hover:text-zinc-900'
                  }`}
                  title="จัดกลุ่มเรียงตามชื่อช่อง"
                >
                  ช่อง
                </button>
              </div>
            </div>
            <button
              type="button"
              onClick={() => toggleAll(sortedYoutube, selectedYoutube, setSelectedYoutube)}
              className="text-xs font-medium text-sky-700 hover:underline"
            >
              {sortedYoutube.every((v) => selectedYoutube.has(v.link)) ? 'ยกเลิกทั้งหมด' : 'เลือกทั้งหมด'}
            </button>
          </div>
          <ul className="space-y-1.5">
            {sortedYoutube.map((v) => (
              <li key={v.link}>
                <label className="flex cursor-pointer items-start gap-2 rounded-lg border border-edge bg-panel p-2 text-xs text-zinc-700 hover:border-zinc-300">
                  <input
                    type="checkbox"
                    checked={selectedYoutube.has(v.link)}
                    onChange={() => toggle(selectedYoutube, setSelectedYoutube, v.link)}
                    className="mt-0.5 accent-sky-500"
                  />
                  <span>
                    <span className={v.is_fetched ? 'text-zinc-400 line-through' : 'text-zinc-800 font-medium'}>{v.title}</span>{' '}
                    {v.is_fetched && (
                      <span className="rounded border border-edge bg-surface px-1 py-0.5 text-[10px] text-zinc-500">
                        อ่านแล้ว
                      </span>
                    )}{' '}
                    <span className="text-zinc-500">
                      · <span className="font-medium text-sky-700">{v.channel}</span> · <span className="inline-flex items-center rounded border border-zinc-200/60 bg-zinc-100 px-1 py-0.2 font-mono text-[10px] text-zinc-600">📅 {v.published}</span>
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

function YoutubePitchApprovalView({
  payload,
  onApprove,
  submitting,
}: {
  payload: YoutubePitchApprovalPayload
  onApprove: Props['onApprove']
  submitting?: boolean
}) {
  const [selectedPitchIds, setSelectedPitchIds] = useState<Set<string>>(new Set())
  const [draftSelections, setDraftSelections] = useState<import('../api/types').UnverifiedDraftSelection[]>([])
  const [draftModalTarget, setDraftModalTarget] = useState<(typeof payload.pitches)[0] | null>(null)
  const [draftAcknowledge, setDraftAcknowledge] = useState(false)
  const [pitchStyles, setPitchStyles] = useState<Record<string, string>>({})

  const pitches = payload.pitches || []
  const selectablePitches = pitches.filter((pitch) => pitch.source_readiness === 'ready')

  function canApprove(pitch: (typeof pitches)[number]) {
    return pitch.source_readiness === 'ready'
  }

  function toggle(id: string) {
    const pitch = pitches.find((item) => item.pitch_id === id)
    if (!pitch || !canApprove(pitch)) return
    const next = new Set(selectedPitchIds)
    if (next.has(id)) next.delete(id)
    else next.add(id)
    setSelectedPitchIds(next)
  }

  function toggleAll() {
    if (selectedPitchIds.size === selectablePitches.length && selectablePitches.length > 0) {
      setSelectedPitchIds(new Set())
    } else {
      setSelectedPitchIds(new Set(selectablePitches.map((p) => p.pitch_id)))
    }
  }

  function openDraftModal(pitch: (typeof pitches)[number]) {
    setDraftModalTarget(pitch)
    setDraftAcknowledge(false)
  }

  function confirmDraft() {
    if (!draftModalTarget || !draftAcknowledge || !draftModalTarget.unverified_draft_eligibility_token) return
    setDraftSelections((prev) => [
      ...prev.filter(d => d.pitch_id !== draftModalTarget.pitch_id),
      {
        pitch_id: draftModalTarget.pitch_id,
        ack: {
          acknowledged: true,
          policy_version: 'unverified-draft-v1',
          eligibility_token: draftModalTarget.unverified_draft_eligibility_token!
        }
      }
    ])
    setDraftModalTarget(null)
  }

  function removeDraft(pitch_id: string) {
    setDraftSelections((prev) => prev.filter(d => d.pitch_id !== pitch_id))
  }

  const allSelected = selectablePitches.length > 0 && selectedPitchIds.size === selectablePitches.length

  return (
    <div className="space-y-4 rounded-xl border border-sky-200 bg-sky-50/50 p-4 shadow-sm shadow-black/5 relative">
      {draftModalTarget && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
          <div className="w-full max-w-md rounded-xl bg-white p-5 shadow-lg relative z-50">
            <h4 className="text-lg font-bold text-red-700">⚠️ สร้าง Unverified Draft?</h4>
            <p className="mt-2 text-sm text-zinc-600">
              การสร้าง Draft สำหรับหัวข้อ <strong>{draftModalTarget.working_titles[0]}</strong> จะข้ามขั้นตอนตรวจสอบแหล่งข่าว โดยพบปัญหาดังนี้:
            </p>
            {draftModalTarget.source_readiness_issues && draftModalTarget.source_readiness_issues.length > 0 && (
              <ul className="mt-2 list-inside list-disc text-sm text-red-600 font-medium">
                {draftModalTarget.source_readiness_issues.map((issue, idx) => (
                  <li key={idx}>{issue}</li>
                ))}
              </ul>
            )}
            <p className="mt-2 text-sm text-zinc-600">
              ทำให้ Briefing Book ที่ได้อาจมีข้อมูลที่ไม่สามารถตรวจสอบความถูกต้องได้
            </p>
            <label className="mt-4 flex cursor-pointer items-start gap-2 rounded-lg bg-red-50 p-3 text-sm text-red-900 border border-red-200">
              <input
                type="checkbox"
                checked={draftAcknowledge}
                onChange={(e) => setDraftAcknowledge(e.target.checked)}
                className="mt-0.5 accent-red-600"
              />
              <span>ข้าพเจ้ารับทราบความเสี่ยง และยืนยันให้สร้าง Unverified Draft</span>
            </label>
            <div className="mt-5 flex justify-end gap-3">
              <button
                onClick={() => setDraftModalTarget(null)}
                className="rounded-lg px-4 py-2 text-sm font-medium text-zinc-600 hover:bg-zinc-100"
              >
                ยกเลิก
              </button>
              <button
                onClick={confirmDraft}
                disabled={!draftAcknowledge}
                className="rounded-lg bg-red-600 px-4 py-2 text-sm font-medium text-white hover:bg-red-700 disabled:opacity-50"
              >
                ยืนยันการสร้าง Draft
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-flow-cyan" />
          <h3 className="text-sm font-semibold text-sky-900">รอการอนุมัติ — เลือกไอเดียคลิป YouTube ที่ต้องการสร้าง Research-Grade Briefing Book</h3>
        </div>
        {pitches.length > 0 && (
          <button
            type="button"
            onClick={toggleAll}
            className="text-xs font-medium text-sky-700 hover:underline"
          >
            {allSelected ? 'ยกเลิกทั้งหมด' : 'เลือกทั้งหมด'}
          </button>
        )}
      </div>

      {(payload.source_refresh_attempts ?? 0) < 1 && pitches.length > 0 && selectablePitches.length < pitches.length && (
        <div role="alert" className="rounded-lg border border-amber-300 bg-amber-50 p-3 text-xs text-amber-950">
          <p className="font-semibold">มีบางหัวข้อที่ข้อมูลแหล่งข่าวไม่พร้อม</p>
          <p className="mt-1">คุณสามารถสร้าง Unverified Draft หากหัวข้อนั้นมีสิทธิ์ (ปุ่มแดงในรายการ) หรือค้นหาแหล่งข่าวใหม่ได้อีก {1 - (payload.source_refresh_attempts ?? 0)} ครั้ง</p>
          <button
            type="button"
            onClick={() => onApprove([], [], [], [], 'refresh_sources')}
            disabled={submitting}
            className="mt-2 rounded-md bg-amber-700 px-3 py-1.5 text-xs font-medium text-white hover:bg-amber-800 disabled:opacity-50"
          >
            ค้นหาแหล่งข่าวใหม่
          </button>
        </div>
      )}

      {pitches.length > 0 ? (
        <ul className="space-y-3">
          {pitches.map((p) => {
            const approved = canApprove(p)
            return (
            <li key={p.pitch_id}>
              <label className={`relative flex items-start gap-3 rounded-xl border border-edge bg-panel p-3.5 text-xs text-zinc-700 transition-colors ${approved ? 'cursor-pointer hover:border-sky-300' : 'cursor-not-allowed border-red-200 bg-red-50/40'}`}>
                <input
                  type="checkbox"
                  checked={selectedPitchIds.has(p.pitch_id)}
                  onChange={() => toggle(p.pitch_id)}
                  disabled={!approved}
                  className="mt-1 accent-sky-500"
                />
                <div className="flex-1 space-y-2 pr-2">
                  <div className="flex flex-wrap items-center gap-2">
                    {p.investigation_mode && (
                      <span className="rounded bg-indigo-50 px-2 py-0.5 text-[11px] font-semibold text-indigo-800 border border-indigo-200/60 uppercase">
                        🔍 Mode: {p.investigation_mode}
                      </span>
                    )}
                    {p.recommended_format && (
                      <span className="rounded bg-flow-cyan/10 px-2 py-0.5 text-[11px] font-semibold text-sky-800 border border-sky-200/60">
                        🎬 {p.recommended_format}
                      </span>
                    )}
                    {p.target_audience && (
                      <span className="rounded bg-purple-50 px-2 py-0.5 text-[11px] font-medium text-purple-800 border border-purple-200/60">
                        👥 {p.target_audience}
                      </span>
                    )}
                    {p.estimated_impact && (
                      <span className="rounded bg-amber-50 px-2 py-0.5 text-[11px] font-medium text-amber-800 border border-amber-200/60">
                        ⚡ Impact: {p.estimated_impact}
                      </span>
                    )}
                    <div className="flex items-center gap-1.5 ml-auto">
                      <span className="text-[11px] font-medium text-zinc-500">Style:</span>
                      <select
                        value={pitchStyles[p.pitch_id] || p.presentation_style || 'narrative'}
                        onChange={(e) => setPitchStyles(prev => ({ ...prev, [p.pitch_id]: e.target.value }))}
                        onClick={(e) => e.stopPropagation()}
                        className="rounded border border-zinc-200 bg-white px-2 py-0.5 text-[11px] font-medium text-zinc-700 shadow-2xs hover:border-sky-300 focus:border-sky-500 focus:outline-none focus:ring-1 focus:ring-sky-500"
                      >
                        <option value="narrative">บทบรรยาย (Narrative)</option>
                        <option value="interview_qa">สัมภาษณ์ (Interview Q&A)</option>
                      </select>
                    </div>
                  </div>

                  <div className="space-y-1">
                    <p className="text-[11px] font-semibold text-zinc-500 uppercase tracking-wide">🎯 ตัวเลือก Working Titles:</p>
                    <ul className="list-inside list-disc space-y-0.5 font-semibold text-zinc-900">
                      {(p.working_titles || []).map((title, idx) => (
                        <li key={idx} className="text-sm">{title}</li>
                      ))}
                    </ul>
                  </div>

                  {(p.primary_anchor_title || p.primary_anchor_event_id) && (
                    <div className="rounded-lg bg-blue-50/70 p-2 border border-blue-200/60 text-blue-950">
                      <span className="font-semibold text-blue-800">📌 Primary Anchor:</span> {p.primary_anchor_title || p.primary_anchor_event_id}
                    </div>
                  )}

                  {(p.core_thesis || p.core_hook) && (
                    <div className="rounded-lg bg-surface/70 p-2 border border-edge/60">
                      <span className="font-semibold text-zinc-800">🎯 Core Thesis:</span> {p.core_thesis || p.core_hook}
                    </div>
                  )}

                  {(p.parking_lot_ideas?.length ?? 0) > 0 && (
                    <div className="rounded-lg bg-purple-50/70 p-2 border border-purple-200/60 text-purple-950">
                      <span className="font-semibold text-purple-800">📦 Parking Lot ({p.parking_lot_ideas?.length} ideas):</span>
                      <ul className="list-disc list-inside mt-1 space-y-0.5 text-xs text-purple-900">
                        {p.parking_lot_ideas?.map((idea, idx) => (
                          <li key={idx}>{idea}</li>
                        ))}
                      </ul>
                    </div>
                  )}


                  {p.counter_intuitive_lead && (
                    <div className="rounded-lg bg-red-50/70 p-2 border border-red-200/60 text-red-950">
                      <span className="font-semibold text-red-800">⚡ Counter-Intuitive Lead:</span> {p.counter_intuitive_lead}
                    </div>
                  )}

                  {p.analogy_generator && (
                    <div className="rounded-lg bg-amber-50/70 p-2 border border-amber-200/60 text-amber-950">
                      <span className="font-semibold text-amber-800">💡 Analogy Generator:</span> {p.analogy_generator}
                    </div>
                  )}

                  {p.thumbnail_concept && (
                    <div className="rounded-lg bg-sky-50/70 p-2 border border-sky-200/60 text-sky-950">
                      <span className="font-semibold text-sky-800">🖼️ Thumbnail Concept:</span> {p.thumbnail_concept}
                    </div>
                  )}

                  {p.audience_takeaway && (
                    <div className="rounded-lg bg-emerald-50/70 p-2 border border-emerald-200/60 text-emerald-950">
                      <span className="font-semibold text-emerald-800">🎁 Audience Takeaway:</span> {p.audience_takeaway}
                    </div>
                  )}

                  {p.source_readiness && (
                    <div className={`rounded-lg border p-2 ${p.source_readiness === 'ready' ? 'border-emerald-200 bg-emerald-50/70 text-emerald-900' : p.source_readiness === 'unknown' ? 'border-amber-200 bg-amber-50/70 text-amber-900' : 'border-red-200 bg-red-50/70 text-red-900'}`}>
                      <div className="flex items-center justify-between gap-2">
                        <span className="font-semibold">Source readiness: {p.source_readiness === 'unknown' ? 'สถานะไม่ชัดเจน (Unknown)' : p.source_readiness}</span>
                        {p.unverified_draft_eligibility_token && !approved && (
                          <div className="flex gap-2">
                            {draftSelections.some(d => d.pitch_id === p.pitch_id) ? (
                              <button
                                type="button"
                                onClick={(e) => { e.preventDefault(); removeDraft(p.pitch_id); }}
                                className="rounded bg-red-100 px-2 py-1 text-[11px] font-bold text-red-700 hover:bg-red-200"
                              >
                                ยกเลิก Draft
                              </button>
                            ) : (
                              <button
                                type="button"
                                onClick={(e) => { e.preventDefault(); openDraftModal(p); }}
                                className="rounded bg-red-600 px-2 py-1 text-[11px] font-bold text-white hover:bg-red-700 shadow-sm"
                              >
                                สร้าง Unverified Draft
                              </button>
                            )}
                          </div>
                        )}
                      </div>
                      {draftSelections.some(d => d.pitch_id === p.pitch_id) && (
                        <p className="mt-1 text-xs font-bold text-red-600 bg-red-100 p-1.5 rounded inline-block">
                          ⚠️ รออนุมัติสร้างเป็น Unverified Draft
                        </p>
                      )}
                      {p.source_readiness_issues && p.source_readiness_issues.length > 0 && (
                        <ul className="mt-1 list-inside list-disc space-y-0.5">
                          {p.source_readiness_issues.map((issue, idx) => <li key={idx}>{issue}</li>)}
                        </ul>
                      )}
                      {p.source_readiness === 'unknown' && (
                        <p className="mt-1 text-xs text-amber-700">ไม่สามารถระบุได้ว่าแหล่งข่าวพร้อมหรือไม่ กรุณาตรวจสอบด้วยตนเองก่อนดำเนินการต่อ หรือเลือกสร้างเป็น Unverified Draft หากทำได้</p>
                      )}
                    </div>
                  )}

                  {(p.key_questions_to_answer?.length > 0 || p.research_hypotheses?.length > 0) && (
                    <details className="mt-1 text-xs text-zinc-600">
                      <summary className="cursor-pointer font-medium text-sky-700 hover:underline">
                        ดู Key Questions & Research Hypotheses ({((p.key_questions_to_answer?.length || 0) + (p.research_hypotheses?.length || 0))} รายการ)
                      </summary>
                      <div className="mt-2 space-y-2 pl-2 border-l-2 border-sky-200">
                        {p.key_questions_to_answer?.length > 0 && (
                          <div>
                            <p className="font-semibold text-zinc-700">❓ Key Questions to Answer:</p>
                            <ul className="list-inside list-disc space-y-0.5 text-zinc-600">
                              {p.key_questions_to_answer.map((q, idx) => (
                                <li key={idx}>{q}</li>
                              ))}
                            </ul>
                          </div>
                        )}
                        {p.research_hypotheses?.length > 0 && (
                          <div>
                            <p className="font-semibold text-zinc-700">💡 Research Hypotheses:</p>
                            <ul className="list-inside list-disc space-y-0.5 text-zinc-600">
                              {p.research_hypotheses.map((h, idx) => (
                                <li key={idx}>{h}</li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    </details>
                  )}
                </div>
              </label>
            </li>
            )
          })}
        </ul>
      ) : (
        <p className="text-xs text-zinc-500">ไม่มีไอเดียคลิป YouTube รออนุมัติ</p>
      )}

      {(selectablePitches.length > 0 || draftSelections.length > 0) && (
        <button
          onClick={() => onApprove([], [], [], Array.from(selectedPitchIds), 'approve', draftSelections, pitchStyles)}
          disabled={submitting}
          className={`rounded-lg px-4 py-2 text-sm font-medium text-white transition-colors disabled:opacity-50 ${
            (selectedPitchIds.size === 0 && draftSelections.length === 0) ? 'bg-zinc-600 hover:bg-zinc-700' : 'bg-sky-500 hover:bg-sky-600'
          }`}
        >
          {submitting
            ? 'กำลังส่ง...'
            : (selectedPitchIds.size === 0 && draftSelections.length === 0)
              ? 'ข้ามรอบนี้ (0 รายการ) — ไม่สร้าง Briefing Book'
              : `อนุมัติและสร้าง Briefing Book (ปกติ ${selectedPitchIds.size}, Draft ${draftSelections.length} รายการ)`}
        </button>
      )}
    </div>
  )
}
