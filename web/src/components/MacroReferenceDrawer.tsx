import { useState, useEffect, useRef } from 'react'
import type { MacroDashboardDTO } from '../api/types'

interface Props {
  data: MacroDashboardDTO
  isOpen: boolean
  onClose: () => void
}

type TabKey = 'source_files' | 'observables' | 'items'

export default function MacroReferenceDrawer({ data, isOpen, onClose }: Props) {
  const [activeTab, setActiveTab] = useState<TabKey>('source_files')
  const closeButtonRef = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    if (!isOpen) return
    closeButtonRef.current?.focus()
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [isOpen, onClose])

  if (!isOpen) return null

  // Extract source files
  const sourceFiles = data.source_files || []

  // Collect all observable refs across items
  const observableSet = new Set<string>()
  data.asset_allocation?.forEach((a) => {
    a.observable_refs?.forEach((r) => observableSet.add(r))
  })
  data.pair_trades?.forEach((pt) => {
    pt.observable_refs?.forEach((r) => observableSet.add(r))
  })
  data.regime_evidence?.forEach((re) => {
    re.observable_refs?.forEach((r) => observableSet.add(r))
  })
  const allObservables = Array.from(observableSet)

  return (
    <div className="fixed inset-0 z-50 overflow-hidden">
      {/* Backdrop */}
      <div
        className="animate-fade-in fixed inset-0 bg-black/40 backdrop-blur-sm transition-opacity"
        onClick={onClose}
      />

      {/* Slide-over Panel */}
      <div className="fixed inset-y-0 right-0 flex max-w-full pl-10">
        <div
          role="dialog"
          aria-modal="true"
          aria-labelledby="macro-reference-drawer-title"
          className="animate-drawer-in flex w-screen max-w-lg flex-col bg-white shadow-2xl"
        >
          {/* Drawer Header */}
          <div className="flex items-center justify-between border-b border-zinc-200 bg-zinc-900 px-6 py-4 text-white">
            <div className="flex items-center gap-2.5">
              <span className="text-xl">📚</span>
              <div>
                <h2 id="macro-reference-drawer-title" className="text-base font-semibold">
                  แหล่งอ้างอิงและข้อมูลฐาน (References)
                </h2>
                <p className="text-xs text-zinc-400">
                  ที่มาข้อมูล รายการไฟล์สแนปชอต และตัวชี้วัดเศรษฐกิจ
                </p>
              </div>
            </div>
            <button
              ref={closeButtonRef}
              onClick={onClose}
              className="rounded-lg bg-zinc-800 px-3 py-1.5 text-xs font-medium text-zinc-300 hover:bg-zinc-700 hover:text-white"
            >
              ✕ ปิดหน้าต่าง
            </button>
          </div>

          {/* Sub-Tabs Selector */}
          <div className="flex border-b border-zinc-200 bg-zinc-50/80 px-6 pt-3">
            <button
              onClick={() => setActiveTab('source_files')}
              aria-pressed={activeTab === 'source_files'}
              className={`mr-6 border-b-2 pb-3 text-xs font-semibold transition-colors ${
                activeTab === 'source_files'
                  ? 'border-zinc-900 text-zinc-900'
                  : 'border-transparent text-zinc-500 hover:text-zinc-800'
              }`}
            >
              📑 ไฟล์ต้นทาง ({sourceFiles.length})
            </button>
            <button
              onClick={() => setActiveTab('observables')}
              aria-pressed={activeTab === 'observables'}
              className={`mr-6 border-b-2 pb-3 text-xs font-semibold transition-colors ${
                activeTab === 'observables'
                  ? 'border-zinc-900 text-zinc-900'
                  : 'border-transparent text-zinc-500 hover:text-zinc-800'
              }`}
            >
              📊 ตัวชี้วัดตลาด ({allObservables.length})
            </button>
            <button
              onClick={() => setActiveTab('items')}
              aria-pressed={activeTab === 'items'}
              className={`border-b-2 pb-3 text-xs font-semibold transition-colors ${
                activeTab === 'items'
                  ? 'border-zinc-900 text-zinc-900'
                  : 'border-transparent text-zinc-500 hover:text-zinc-800'
              }`}
            >
              🔍 แยกรายกลยุทธ์
            </button>
          </div>

          {/* Tab Contents */}
          <div className="custom-scrollbar flex-1 overflow-y-auto overscroll-contain p-6">
            {/* TAB 1: Source Files */}
            {activeTab === 'source_files' && (
              <div className="space-y-4">
                <div className="rounded-lg bg-zinc-100 p-3 text-xs text-zinc-600">
                  <span className="font-semibold text-zinc-800">ระบบประเมินผล:</span>{' '}
                  {data.generated_by || 'Strategic Allocator / Macro Core Engine'}{' '}
                  <span className="text-zinc-400">({data.evaluated_at})</span>
                </div>

                <h3 className="text-sm font-semibold text-zinc-900">
                  รายการไฟล์ในฐานข้อมูล (Obsidian Vault Knowledge Base)
                </h3>

                {sourceFiles.length === 0 ? (
                  <p className="text-xs italic text-zinc-400">ไม่มีรายการไฟล์ต้นทางระบุไว้</p>
                ) : (
                  <ul className="space-y-2">
                    {sourceFiles.map((file, idx) => {
                      const isPython = file.endsWith('.py')
                      return (
                        <li
                          key={idx}
                          className="flex items-center justify-between rounded-xl border border-zinc-200 bg-white p-3 shadow-sm transition-colors hover:border-zinc-300"
                        >
                          <div className="flex items-center gap-2.5">
                            <span className="text-base">{isPython ? '⚙️' : '📄'}</span>
                            <div>
                              <div className="font-mono text-xs font-semibold text-zinc-800">{file}</div>
                              <div className="text-[11px] text-zinc-400">
                                {isPython ? 'Quantitative Analytics Script' : 'Obsidian PKM Snapshot / Report'}
                              </div>
                            </div>
                          </div>
                          <span className="rounded-md bg-zinc-100 px-2 py-0.5 text-[10px] font-medium text-zinc-600">
                            {isPython ? 'Code Module' : 'Markdown'}
                          </span>
                        </li>
                      )
                    })}
                  </ul>
                )}
              </div>
            )}

            {/* TAB 2: Observables */}
            {activeTab === 'observables' && (
              <div className="space-y-4">
                <div className="rounded-lg bg-blue-50/60 p-3 text-xs text-blue-900">
                  รายการรหัสซีรีส์และตัวชี้วัดเศรษฐกิจมหภาค (Observable Metrics) จาก FRED,
                  ตลาดการเงิน และระบบดัชนีชี้วัดภายใน
                </div>

                {allObservables.length === 0 ? (
                  <p className="text-xs italic text-zinc-400">ไม่มีตัวชี้วัดระบุไว้</p>
                ) : (
                  <div className="grid grid-cols-1 gap-2.5">
                    {allObservables.map((obs, idx) => (
                      <div
                        key={idx}
                        className="flex items-center justify-between rounded-xl border border-zinc-200 bg-zinc-50/60 p-3"
                      >
                        <div className="flex items-center gap-2">
                          <span className="h-2 w-2 rounded-full bg-blue-600" />
                          <span className="font-mono text-xs font-semibold text-zinc-900">{obs}</span>
                        </div>
                        <span className="text-[11px] font-medium text-zinc-500">Economic Indicator</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* TAB 3: By Strategy Item */}
            {activeTab === 'items' && (
              <div className="space-y-5">
                {/* Asset Allocations */}
                <div>
                  <h3 className="mb-2.5 text-xs font-bold uppercase tracking-wider text-zinc-500">
                    Cross-Asset Allocations
                  </h3>
                  <div className="space-y-2">
                    {data.asset_allocation?.map((a, idx) => (
                      <div key={idx} className="rounded-xl border border-zinc-200 bg-white p-3 text-xs">
                        <div className="flex items-center justify-between font-semibold text-zinc-900">
                          <span>{a.asset_class}</span>
                          <span className="text-zinc-500">{a.stance}</span>
                        </div>
                        {(a.source_refs && a.source_refs.length > 0) ||
                        (a.observable_refs && a.observable_refs.length > 0) ? (
                          <div className="mt-2 space-y-1 text-zinc-600">
                            {a.source_refs && a.source_refs.length > 0 && (
                              <div>
                                <span className="font-semibold text-zinc-700">Sources: </span>
                                {a.source_refs.join(', ')}
                              </div>
                            )}
                            {a.observable_refs && a.observable_refs.length > 0 && (
                              <div>
                                <span className="font-semibold text-zinc-700">Observables: </span>
                                {a.observable_refs.join(', ')}
                              </div>
                            )}
                          </div>
                        ) : (
                          <div className="mt-1 italic text-zinc-400">อ้างอิงจากรายงานหลัก</div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>

                {/* Pair Trades */}
                {data.pair_trades && data.pair_trades.length > 0 && (
                  <div>
                    <h3 className="mb-2.5 text-xs font-bold uppercase tracking-wider text-zinc-500">
                      Tactical Pair Trades
                    </h3>
                    <div className="space-y-2">
                      {data.pair_trades.map((pt, idx) => (
                        <div key={idx} className="rounded-xl border border-zinc-200 bg-white p-3 text-xs">
                          <div className="font-semibold text-zinc-900">
                            Long {pt.long_leg} / Short {pt.short_leg}
                          </div>
                          {(pt.source_refs && pt.source_refs.length > 0) ||
                          (pt.observable_refs && pt.observable_refs.length > 0) ? (
                            <div className="mt-2 space-y-1 text-zinc-600">
                              {pt.source_refs && pt.source_refs.length > 0 && (
                                <div>
                                  <span className="font-semibold text-zinc-700">Sources: </span>
                                  {pt.source_refs.join(', ')}
                                </div>
                              )}
                              {pt.observable_refs && pt.observable_refs.length > 0 && (
                                <div>
                                  <span className="font-semibold text-zinc-700">Observables: </span>
                                  {pt.observable_refs.join(', ')}
                                </div>
                              )}
                            </div>
                          ) : (
                            <div className="mt-1 italic text-zinc-400">อ้างอิงจากรายงานหลัก</div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Drawer Footer */}
          <div className="border-t border-zinc-200 bg-zinc-50 px-6 py-3 text-right">
            <button
              onClick={onClose}
              className="rounded-lg bg-zinc-900 px-4 py-2 text-xs font-semibold text-white hover:bg-zinc-800"
            >
              ปิดหน้าต่างอ้างอิง
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
