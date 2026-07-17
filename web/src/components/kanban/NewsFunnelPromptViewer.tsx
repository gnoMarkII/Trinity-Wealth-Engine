import { useState, useEffect } from 'react'
import { api } from '../../api/client'
import type { NewsFunnelPendingItem, NewsFunnelFilteredItem } from '../../api/types'
import ReactMarkdown from 'react-markdown'

interface Props {
  prompt: string
  onItemDeleted?: () => void
}

export default function NewsFunnelPromptViewer({ prompt }: Props) {
  const [items, setItems] = useState<NewsFunnelPendingItem[] | null>(null)
  const [filteredItems, setFilteredItems] = useState<NewsFunnelFilteredItem[] | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [showFiltered, setShowFiltered] = useState(false)

  useEffect(() => {
    let mounted = true
    setLoading(true)
    setError(null)
    api
      .getNewsFunnelPending()
      .then((data) => {
        if (mounted) setItems(data)
      })
      .catch((err) => {
        if (mounted) setError(err?.message || 'ไม่สามารถโหลดรายการข่าวรอสังเคราะห์ได้')
      })
      .finally(() => {
        if (mounted) setLoading(false)
      })

    api
      .getNewsFunnelFiltered()
      .then((data) => {
        if (mounted) setFilteredItems(data)
      })
      .catch((err) => {
        console.error('Failed to load filtered items:', err)
      })

    return () => {
      mounted = false
    }
  }, [prompt])

  // ถ้าโหลดไม่สำเร็จหรือไม่มีรายการ pending จาก API ให้ fallback เป็น ReactMarkdown เดิม
  if (!loading && (!items || items.length === 0) && error) {
    return (
      <div className="mt-3.5 rounded-xl border border-edge bg-surface p-3.5 text-xs text-zinc-700 leading-relaxed shadow-sm">
        <ReactMarkdown
          components={{
            a: ({ children, href }) => (
              <a href={href} target="_blank" rel="noreferrer" className="font-medium text-sky-700 underline underline-offset-2 hover:text-sky-900">
                {children}
              </a>
            ),
            code: ({ children }) => <code className="rounded bg-surface-strong px-1 py-0.5 font-mono text-[0.84em] text-zinc-700">{children}</code>,
            h1: ({ children }) => <h4 className="mt-4 text-sm font-bold text-zinc-900 first:mt-0">{children}</h4>,
            h2: ({ children }) => <h4 className="mt-4 text-sm font-bold text-zinc-900 first:mt-0">{children}</h4>,
            h3: ({ children }) => <h5 className="mt-3 text-xs font-bold text-sky-900 first:mt-0">{children}</h5>,
            h4: ({ children }) => <h6 className="mt-3 text-xs font-bold text-zinc-800 first:mt-0">{children}</h6>,
            li: ({ children }) => <li className="ml-4 list-disc pl-1 mt-0.5">{children}</li>,
            p: ({ children }) => <p className="mt-2 first:mt-0 leading-relaxed">{children}</p>,
            ul: ({ children }) => <ul className="mt-1.5 space-y-1">{children}</ul>,
            hr: () => <hr className="my-3 border-edge" />,
          }}
        >
          {prompt.replace(/\[\[([^|\]]+)(?:\|[^\]]+)?\]\]/g, '$1')}
        </ReactMarkdown>
      </div>
    )
  }

  return (
    <div className="mt-3.5 space-y-4">
      <div className="flex items-center justify-between rounded-xl border border-sky-200 bg-sky-50 px-3.5 py-2.5 text-xs text-sky-900 shadow-sm">
        <div className="flex items-center gap-2">
          <span className="text-base">📰</span>
          <span className="font-semibold">
            รายการข่าว High-Impact ที่รอสังเคราะห์ในรอบนี้ ({loading ? 'กำลังโหลด...' : items ? `${items.length} รายการ` : '0 รายการ'})
          </span>
        </div>
        {!loading && items && items.length > 0 && (
          <span className="text-[11px] font-medium text-sky-700">สามารถเลือกอนุมัติข่าวที่ต้องการได้ในการ์ด</span>
        )}
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-6 text-xs text-zinc-400">
          กำลังโหลดรายการข่าว High-Impact...
        </div>
      ) : !items || items.length === 0 ? (
        <div className="rounded-xl border border-edge bg-surface p-4 text-center text-xs text-zinc-500">
          ไม่มีข่าวค้างรอในรอบนี้แล้ว
        </div>
      ) : (
        <div className="space-y-3">
          {items
            .slice()
            .sort((a, b) => (a.triage_source === 'heuristic_fallback' ? 1 : 0) - (b.triage_source === 'heuristic_fallback' ? 1 : 0))
            .map((item, idx) => {
              const maxScore = Math.max(item.macro_impact_score || 0, item.asset_impact_score || 0)
              const firstLink = item.links && item.links.length > 0 ? item.links[0] : null
              const cleanTag = (s: string) => (s.replace(/^\[\[|\]\]$/g, '').split('|')[0]?.trim() || '')
              const allTags = [
                ...(item.extracted_tickers || []).map(cleanTag),
                ...(item.extracted_themes || []).map(cleanTag),
              ].filter(Boolean)

              return (
                <div
                  key={item.event_id}
                  className="relative flex flex-col gap-2 rounded-xl border border-edge bg-surface p-4 text-xs text-zinc-700 shadow-sm transition-colors hover:border-zinc-300"
                >
                  <div className="flex flex-wrap items-center gap-2 font-semibold text-zinc-900">
                    <span className="rounded bg-sky-100 px-2 py-0.5 text-[11px] font-bold text-sky-800">
                      #{idx + 1}
                    </span>
                    <span className="rounded bg-amber-100 px-1.5 py-0.5 text-[11px] font-semibold text-amber-800">
                      Score: {maxScore}/10
                    </span>
                    {item.triage_source === 'heuristic_fallback' && (
                      <span
                        title="คะแนนจาก heuristic fallback (LLM triage ล้มเหลวรอบ ingest) — โปรดตรวจสอบเนื้อหาก่อนอนุมัติ"
                        className="rounded border border-red-200 bg-red-50 px-1.5 py-0.5 text-[10px] font-semibold text-red-700"
                      >
                        ⚠️ Heuristic
                      </span>
                    )}
                    <span className="text-sm leading-snug">{item.canonical_title}</span>
                  </div>

                  <div className="mt-1 flex flex-wrap gap-4 text-zinc-600">
                    <span>
                      <b>Macro Impact:</b> {item.macro_impact_score}/10
                    </span>
                    <span>
                      <b>Asset Impact:</b> {item.asset_impact_score}/10
                    </span>
                  </div>

                  {item.comprehensive_summary && (
                    <p className="mt-1 leading-relaxed text-zinc-600">
                      <b>สรุปเนื้อหา:</b> {item.comprehensive_summary}
                    </p>
                  )}

                  {allTags.length > 0 && (
                    <div className="mt-1 flex flex-wrap items-center gap-1.5">
                      <b className="mr-1 text-zinc-700">แท็กที่เกี่ยวข้อง:</b>
                      {allTags.map((tag, tIdx) => (
                        <span
                          key={tIdx}
                          className="rounded border border-zinc-200 bg-zinc-50 px-1.5 py-0.5 font-mono text-[10px] font-medium text-zinc-700"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  )}

                  {firstLink && (
                    <div className="mt-1">
                      <a
                        href={firstLink}
                        target="_blank"
                        rel="noreferrer"
                        className="inline-flex items-center gap-1 font-medium text-sky-700 underline underline-offset-2 hover:text-sky-900"
                      >
                        🔗 อ่านข่าวต้นฉบับ
                      </a>
                    </div>
                  )}
                </div>
              )
            })}
        </div>
      )}

      {filteredItems && filteredItems.length > 0 && (
        <div className="mt-6 rounded-xl border border-zinc-200 bg-zinc-50/50 overflow-hidden shadow-sm">
          <button
            type="button"
            onClick={() => setShowFiltered(!showFiltered)}
            className="flex w-full items-center justify-between px-4 py-3 text-left transition-colors hover:bg-zinc-100/80 focus:outline-none"
          >
            <div className="flex items-center gap-2">
              <span className="text-base">🗄️</span>
              <span className="font-semibold text-zinc-800 text-xs">
                ข่าวที่ไม่ผ่านเกณฑ์หรือข้ามการบันทึก ({filteredItems.length} รายการ)
              </span>
            </div>
            <span className="text-xs font-medium text-zinc-500">
              {showFiltered ? '▲ ซ่อนรายการ' : '▼ แสดงรายการ'}
            </span>
          </button>

          {showFiltered && (
            <div className="border-t border-zinc-200 p-4 space-y-3 bg-surface">
              <div className="text-[11px] text-zinc-500 mb-2">
                รายการเหล่านี้ตกเกณฑ์คะแนน (Score &lt; 7), ถูกปฏิเสธ (Rejected) หรือข้ามเนื่องจากดึงข้อมูลไม่สำเร็จ (Skipped Error)
              </div>
              {filteredItems
                .slice()
                .sort((a, b) => (b.ingested_at || '').localeCompare(a.ingested_at || ''))
                .map((item, idx) => {
                  const maxScore = Math.max(item.macro_impact_score || 0, item.asset_impact_score || 0)
                  const firstLink = item.links && item.links.length > 0 ? item.links[0] : null
                  const badgeInfo =
                    item.status === 'skipped_error'
                      ? { text: 'Skipped: ดึงข้อมูลไม่สำเร็จ', cls: 'bg-red-100 text-red-800 border-red-200' }
                      : item.status === 'rejected'
                      ? { text: 'Rejected: ไม่ผ่านคัดเลือก', cls: 'bg-zinc-200 text-zinc-700 border-zinc-300' }
                      : { text: 'Low Impact (Score < 7)', cls: 'bg-amber-100 text-amber-800 border-amber-200' }

                  const reasonOrError = item.error_msg || item.triage_reasoning || null

                  return (
                    <div
                      key={item.event_id || idx}
                      className="flex flex-col gap-1.5 rounded-lg border border-edge bg-surface p-3.5 text-xs text-zinc-700 shadow-2xs opacity-90"
                    >
                      <div className="flex flex-wrap items-center gap-2 font-medium text-zinc-800">
                        <span className={`rounded border px-2 py-0.5 text-[10px] font-bold ${badgeInfo.cls}`}>
                          {badgeInfo.text}
                        </span>
                        <span className="rounded bg-zinc-100 px-1.5 py-0.5 text-[10px] font-semibold text-zinc-600">
                          Score: {maxScore}/10
                        </span>
                        <span className="font-semibold text-zinc-900 leading-snug">{item.canonical_title}</span>
                      </div>

                      {reasonOrError && (
                        <div className="mt-0.5 rounded bg-zinc-50 px-2 py-1 text-[11px] text-zinc-600 border border-zinc-200/60">
                          <b>เหตุผล/รายละเอียด:</b> {reasonOrError}
                        </div>
                      )}

                      {item.comprehensive_summary && (
                        <p className="text-[11px] text-zinc-500 leading-relaxed">
                          {item.comprehensive_summary}
                        </p>
                      )}

                      {firstLink && (
                        <div className="mt-0.5">
                          <a
                            href={firstLink}
                            target="_blank"
                            rel="noreferrer"
                            className="inline-flex items-center gap-1 text-[11px] font-medium text-sky-700 underline underline-offset-2 hover:text-sky-900"
                          >
                            🔗 ข่าวต้นฉบับ
                          </a>
                        </div>
                      )}
                    </div>
                  )
                })}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
