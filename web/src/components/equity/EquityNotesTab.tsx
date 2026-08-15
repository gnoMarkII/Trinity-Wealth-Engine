import React, { useEffect, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { api } from '../../api/client'
import type { EquityNoteContentDTO, EquityNoteItemDTO, EquityNotesDTO } from '../../api/types'

interface ParsedNote {
  frontmatter: { key: string; value: string }[] | null
  body: string
}

function parseNoteContent(raw: string): ParsedNote {
  if (!raw || !raw.startsWith('---')) {
    return { frontmatter: null, body: raw }
  }
  const match = raw.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$/)
  if (!match) {
    return { frontmatter: null, body: raw }
  }

  const yamlBlock = match[1] ?? ''
  const body = match[2] ?? ''

  const frontmatter: { key: string; value: string }[] = []
  for (const line of yamlBlock.split('\n')) {
    const trimmed = line.trim()
    if (!trimmed || trimmed.startsWith('#')) continue
    const colonIdx = trimmed.indexOf(':')
    if (colonIdx > 0) {
      const key = trimmed.slice(0, colonIdx).trim()
      let val = trimmed.slice(colonIdx + 1).trim()
      if ((val.startsWith('"') && val.endsWith('"')) || (val.startsWith("'") && val.endsWith("'"))) {
        val = val.slice(1, -1)
      }
      frontmatter.push({ key, value: val })
    }
  }

  return { frontmatter: frontmatter.length > 0 ? frontmatter : null, body }
}

function ObsidianNoteViewer({ content }: { content: string }) {
  const { frontmatter, body } = parseNoteContent(content)

  const formattedBody = body.replace(/\[\[([^\]|]+)(?:\|([^\]]+))?\]\]/g, (_m, target, label) => {
    const text = (label || target).trim()
    return `\`[[${text}]]\``
  })

  return (
    <div className="space-y-4">
      {frontmatter && frontmatter.length > 0 && (
        <div className="rounded-xl border border-edge bg-surface/70 p-3.5 space-y-2 text-xs">
          <div className="flex items-center gap-1.5 font-semibold text-zinc-700 uppercase tracking-wider text-[11px] border-b border-edge/60 pb-2">
            <span>⚙️</span> Obsidian Properties
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-4 gap-y-2 pt-1">
            {frontmatter.map((item, idx) => (
              <div key={idx} className="flex items-start gap-2 overflow-hidden">
                <span className="font-mono text-zinc-500 min-w-[80px] shrink-0 font-medium">{item.key}:</span>
                <span className="text-zinc-800 break-words font-sans">
                  {item.key === 'tags' || item.key === 'tickers' || item.value.startsWith('[') ? (
                    <span className="inline-flex flex-wrap gap-1">
                      {item.value
                        .replace(/[[\]"']/g, '')
                        .split(',')
                        .map((t) => t.trim())
                        .filter(Boolean)
                        .map((tagText, i) => (
                          <span key={i} className="px-1.5 py-0.5 rounded bg-sky-50 text-sky-700 border border-sky-200 text-[10px] font-medium">
                            #{tagText.replace(/^#/, '')}
                          </span>
                        ))}
                    </span>
                  ) : item.value.startsWith('http') ? (
                    <a href={item.value} target="_blank" rel="noreferrer" className="text-sky-600 underline hover:text-sky-800">
                      {item.value}
                    </a>
                  ) : (
                    item.value.replace(/[[\]]/g, '')
                  )}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="prose prose-sm max-w-none text-zinc-800 bg-surface/40 p-5 rounded-xl border border-edge/60 shadow-inner leading-relaxed">
        <ReactMarkdown
          components={{
            a: ({ children, href }) => (
              <a href={href} target="_blank" rel="noreferrer" className="font-medium text-sky-600 underline underline-offset-2 hover:text-sky-800">
                {children}
              </a>
            ),
            code: ({ children, className }) => {
              const str = String(children)
              if (str.startsWith('[[') && str.endsWith(']]')) {
                const cleanTag = str.slice(2, -2).trim()
                return (
                  <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-md bg-purple-50 text-purple-700 border border-purple-200 font-sans text-xs font-medium">
                    📌 {cleanTag}
                  </span>
                )
              }
              return <code className={className || "rounded bg-surface-strong px-1.5 py-0.5 font-mono text-[0.84em] text-purple-700"}>{children}</code>
            },
            h1: ({ children }) => <h1 className="text-xl font-bold text-zinc-900 border-b border-edge pb-2 mt-6 mb-3 first:mt-0">{children}</h1>,
            h2: ({ children }) => <h2 className="text-lg font-semibold text-zinc-900 border-b border-edge/60 pb-1.5 mt-5 mb-2.5 first:mt-0">{children}</h2>,
            h3: ({ children }) => <h3 className="text-base font-semibold text-zinc-800 mt-4 mb-2 first:mt-0">{children}</h3>,
            h4: ({ children }) => <h4 className="text-sm font-semibold text-zinc-800 mt-3 mb-1.5 first:mt-0">{children}</h4>,
            p: ({ children }) => <p className="my-2.5 leading-relaxed text-zinc-700">{children}</p>,
            ul: ({ children }) => <ul className="my-2 space-y-1.5 list-disc pl-5 text-zinc-700">{children}</ul>,
            ol: ({ children }) => <ol className="my-2 space-y-1.5 list-decimal pl-5 text-zinc-700">{children}</ol>,
            li: ({ children }) => <li className="pl-0.5">{children}</li>,
            blockquote: ({ children }) => (
              <blockquote className="border-l-4 border-sky-400 bg-sky-50/50 italic px-4 py-2 my-3 rounded-r-lg text-zinc-700">
                {children}
              </blockquote>
            ),
            table: ({ children }) => (
              <div className="overflow-x-auto my-4 rounded-lg border border-edge">
                <table className="min-w-full divide-y divide-edge text-xs">{children}</table>
              </div>
            ),
            thead: ({ children }) => <thead className="bg-surface">{children}</thead>,
            th: ({ children }) => <th className="px-3 py-2 text-left font-semibold text-zinc-700">{children}</th>,
            td: ({ children }) => <td className="px-3 py-2 border-t border-edge/50 text-zinc-600">{children}</td>,
            hr: () => <hr className="my-4 border-edge" />
          }}
        >
          {formattedBody}
        </ReactMarkdown>
      </div>
    </div>
  )
}

interface EquityNotesTabProps {
  ticker: string
}

export const EquityNotesTab: React.FC<EquityNotesTabProps> = ({ ticker }) => {
  const [notesData, setNotesData] = useState<EquityNotesDTO | null>(null)
  const [loading, setLoading] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)

  // Preview Modal state
  const [selectedNote, setSelectedNote] = useState<EquityNoteItemDTO | null>(null)
  const [contentData, setContentData] = useState<EquityNoteContentDTO | null>(null)
  const [contentLoading, setContentLoading] = useState<boolean>(false)
  const [contentError, setContentError] = useState<string | null>(null)

  useEffect(() => {
    let isMounted = true
    const fetchNotes = async () => {
      setLoading(true)
      setError(null)
      try {
        const data = await api.getEquityNotes(ticker)
        if (isMounted) {
          setNotesData(data)
        }
      } catch (err: any) {
        if (isMounted) {
          setError(err.message || 'ไม่สามารถโหลดข้อมูล Obsidian Notes ได้')
        }
      } finally {
        if (isMounted) {
          setLoading(false)
        }
      }
    }

    fetchNotes()
    return () => {
      isMounted = false
    }
  }, [ticker])

  const handleOpenPreview = async (item: EquityNoteItemDTO) => {
    setSelectedNote(item)
    setContentData(null)
    setContentLoading(true)
    setContentError(null)

    try {
      const data = await api.getEquityNoteContent(item.relative_path)
      setContentData(data)
    } catch (err: any) {
      setContentError(err.message || 'ไม่สามารถอ่านเนื้อหาโน้ตได้')
    } finally {
      setContentLoading(false)
    }
  }

  const handleCloseModal = () => {
    setSelectedNote(null)
    setContentData(null)
    setContentError(null)
  }

  const getMatchedBadge = (matchedBy: string) => {
    switch (matchedBy) {
      case 'watchlist':
        return (
          <span className="px-2 py-0.5 rounded-md text-[11px] font-medium bg-amber-50 text-amber-700 border border-amber-200">
            📌 Watchlist
          </span>
        )
      case 'stock_note':
        return (
          <span className="px-2 py-0.5 rounded-md text-[11px] font-medium bg-sky-50 text-sky-700 border border-sky-200">
            📝 Stock Note
          </span>
        )
      case 'journal':
        return (
          <span className="px-2 py-0.5 rounded-md text-[11px] font-medium bg-purple-50 text-purple-700 border border-purple-200">
            📓 Trading Journal
          </span>
        )
      default:
        return (
          <span className="px-2 py-0.5 rounded-md text-[11px] font-medium bg-zinc-100 text-zinc-600 border border-zinc-200">
            📄 {matchedBy}
          </span>
        )
    }
  }

  if (loading) {
    return (
      <div className="flex justify-center items-center h-48 text-zinc-500" aria-live="polite">
        กำลังโหลด Obsidian Notes...
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex flex-col justify-center items-center h-48 bg-red-50 rounded-xl border border-red-200 p-6 text-center text-red-600" role="alert">
        <h3 className="text-base font-medium mb-1">เกิดข้อผิดพลาด</h3>
        <p className="text-sm">{error}</p>
      </div>
    )
  }

  if (!notesData || notesData.items.length === 0) {
    return (
      <div className="flex flex-col justify-center items-center h-64 bg-surface rounded-xl border border-dashed border-edge p-8 text-center">
        <svg className="w-12 h-12 text-zinc-400 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
        </svg>
        <h3 className="text-lg font-medium text-zinc-900 mb-1">ไม่พบ Obsidian Notes สำหรับหุ้น {ticker}</h3>
        <p className="text-zinc-500 max-w-md text-sm">
          สามารถเพิ่มโน้ตเกี่ยวกับหุ้นตัวนี้ใน Obsidian Vault ได้ที่โฟลเดอร์ <code className="bg-surface-strong px-1.5 py-0.5 rounded text-xs">Stocks/{ticker}</code> หรือใส่ Tag <code className="bg-surface-strong px-1.5 py-0.5 rounded text-xs">#{ticker}</code> ใน Trading Journal
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between border-b border-edge pb-3">
        <div className="text-xs font-semibold uppercase tracking-wider text-sky-600">
          Obsidian Notes ({notesData.total_count})
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4">
        {notesData.items.map((item, idx) => (
          <div
            key={idx}
            className="flex flex-col justify-between rounded-xl border border-edge bg-panel p-4 shadow-sm shadow-black/5 hover:border-sky-200 transition-colors"
          >
            <div className="space-y-2">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <h4 className="font-semibold text-zinc-900 text-base leading-snug">
                    {item.title}
                  </h4>
                  <p className="text-xs text-zinc-400 mt-0.5">📁 {item.folder}</p>
                </div>
                {getMatchedBadge(item.matched_by)}
              </div>

              <p className="text-xs leading-relaxed text-zinc-600 bg-surface/50 p-2.5 rounded-lg border border-edge/50 line-clamp-3">
                {item.snippet || 'ไม่มีข้อความตัวอย่าง'}
              </p>

              <div className="text-[11px] text-zinc-400">
                แก้ไขเมื่อ {new Date(item.modified_at).toLocaleString('th-TH')}
              </div>
            </div>

            <div className="mt-3 pt-3 border-t border-edge flex items-center justify-end gap-3">
              <button
                onClick={() => handleOpenPreview(item)}
                className="inline-flex items-center gap-1 text-xs font-medium text-zinc-700 bg-surface hover:bg-surface-strong px-3 py-1.5 rounded-lg border border-edge transition-colors"
              >
                📄 ดูเนื้อหา
              </button>

              <a
                href={item.obsidian_uri}
                className="inline-flex items-center gap-1 text-xs font-medium text-sky-600 bg-sky-50 hover:bg-sky-100 px-3 py-1.5 rounded-lg border border-sky-200 transition-colors"
              >
                🔗 เปิดใน Obsidian
              </a>
            </div>
          </div>
        ))}
      </div>

      {/* Content Preview Modal */}
      {selectedNote && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4 animate-fade-in">
          <div className="bg-panel border border-edge rounded-2xl shadow-2xl max-w-3xl w-full max-h-[85vh] flex flex-col overflow-hidden">
            {/* Modal Header */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-edge">
              <div>
                <h3 className="font-semibold text-lg text-zinc-900">{selectedNote.title}</h3>
                <p className="text-xs text-zinc-500">{selectedNote.relative_path}</p>
              </div>
              <button
                onClick={handleCloseModal}
                className="p-1 rounded-lg text-zinc-400 hover:text-zinc-600 hover:bg-surface transition-colors"
                aria-label="Close"
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Modal Body */}
            <div className="p-6 overflow-y-auto flex-1">
              {contentLoading && (
                <div className="flex justify-center items-center h-48 text-zinc-500">
                  กำลังโหลดเนื้อหาโน้ต...
                </div>
              )}

              {contentError && (
                <div className="bg-red-50 border border-red-200 text-red-600 p-4 rounded-xl text-sm">
                  {contentError}
                </div>
              )}

              {contentData && (
                <ObsidianNoteViewer content={contentData.content} />
              )}
            </div>

            {/* Modal Footer */}
            <div className="flex items-center justify-between px-6 py-4 border-t border-edge bg-surface/30">
              <span className="text-xs text-zinc-400">
                {contentData?.modified_at
                  ? `แก้ไขล่าสุด: ${new Date(contentData.modified_at).toLocaleString('th-TH')}`
                  : ''}
              </span>
              <div className="flex items-center gap-3">
                <a
                  href={selectedNote.obsidian_uri}
                  className="inline-flex items-center gap-1.5 text-xs font-medium text-sky-600 bg-sky-50 hover:bg-sky-100 px-3.5 py-2 rounded-lg border border-sky-200 transition-colors"
                >
                  🔗 เปิดใน Obsidian
                </a>
                <button
                  onClick={handleCloseModal}
                  className="text-xs font-medium text-zinc-700 bg-surface hover:bg-surface-strong px-4 py-2 rounded-lg border border-edge transition-colors"
                >
                  ปิด
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
