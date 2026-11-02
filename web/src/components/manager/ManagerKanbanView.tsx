import { useEffect, useRef, useState } from 'react'
import { api, ApiError } from '../../api/client'
import type { KanbanCardDTO } from '../../api/types'
import { columnForStatus, type TerminalStatus } from '../../lib/agentStatus'
import LiveTerminal from '../LiveTerminal'
import KanbanColumn from '../kanban/KanbanColumn'
import KanbanDetailDrawer from '../kanban/KanbanDetailDrawer'
import KanbanCardModal from '../kanban/KanbanCardModal'
import AddCardDropdown from '../kanban/AddCardDropdown'
import EditTemplateModal from '../kanban/EditTemplateModal'
import TextInput from '../ui/TextInput'
import type { ColumnDef, FlowFilter, StatusFilter, WorkspacePreview } from '../kanban/types'
import { FLOW_OPTIONS } from '../../lib/flows'
import { loadQuickTemplates, saveQuickTemplates, type QuickTemplate } from '../../lib/quickTemplateStorage'

export const KANBAN_COLUMNS: ColumnDef[] = [
  { key: 'backlog', label: 'Backlog', icon: '📥' },
  { key: 'approval', label: 'Request Approve', icon: '🙋' },
  { key: 'executing', label: 'Workers Executing', icon: '🔍' },
  { key: 'done', label: 'Done', icon: '✅' },
]

const STATUS_COLUMN_MAP: Record<StatusFilter, string[]> = {
  active: ['approval', 'executing'],
  all: ['backlog', 'approval', 'executing', 'done'],
  backlog: ['backlog'],
  done: ['done'],
}

const STATUS_TABS: { key: StatusFilter; label: string }[] = [
  { key: 'all', label: 'ทั้งหมด' },
  { key: 'active', label: 'กำลังดำเนินการ' },
  { key: 'backlog', label: 'Backlog' },
  { key: 'done', label: 'เสร็จแล้ว' },
]

const DONE_HIDE_AFTER_DAYS = 3
const NOTICE_DISMISS_MS = 2500
const NOTICE_LEAVE_MS = 150
const DELETE_ANIM_MS = 150

type ModalState = { mode: 'create' } | { mode: 'edit'; card: KanbanCardDTO } | null

interface ActiveDispatch {
  jobId: string
  dispatchedAt: number
}

function daysSince(unixSeconds: number): number {
  return (Date.now() / 1000 - unixSeconds) / 86400
}

export interface ManagerKanbanViewProps {
  layoutMode?: 'compact' | 'wide' | 'full'
  refreshTrigger?: number
  selectedJobId?: string | null
  onAskAiWithCard?: (prompt: string, card: KanbanCardDTO) => void
  onWatchPipeline?: (jobId: string) => void
  onApprovalCountChange?: (count: number) => void
}

export default function ManagerKanbanView({
  layoutMode = 'compact',
  refreshTrigger,
  selectedJobId,
  onAskAiWithCard,
  onWatchPipeline,
  onApprovalCountChange,
}: ManagerKanbanViewProps) {
  const [cards, setCards] = useState<KanbanCardDTO[]>([])
  const [modalState, setModalState] = useState<ModalState>(null)
  const [modalError, setModalError] = useState<string | null>(null)
  const [quickTemplates, setQuickTemplates] = useState<QuickTemplate[]>(loadQuickTemplates)
  const [editingTemplateIndex, setEditingTemplateIndex] = useState<number | null>(null)
  const [activeDispatches, setActiveDispatches] = useState<Record<string, ActiveDispatch>>({})
  const [liveNode, setLiveNode] = useState<Record<string, string | null>>({})
  const [liveLogCount, setLiveLogCount] = useState<Record<string, number>>({})
  const [nowTick, setNowTick] = useState(Date.now())
  const [dispatching, setDispatching] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [noticeLeaving, setNoticeLeaving] = useState(false)
  const [removingIds, setRemovingIds] = useState<Set<string>>(new Set())
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('all')
  const [flowFilter, setFlowFilter] = useState<FlowFilter>('all')
  const [search, setSearch] = useState('')
  const [selectedCardId, setSelectedCardId] = useState<string | null>(null)
  const [compactActiveTab, setCompactActiveTab] = useState<string>('backlog')

  const tickTimer = useRef<number | null>(null)
  const noticeDismissTimer = useRef<number | null>(null)
  const noticeLeaveTimer = useRef<number | null>(null)
  const deleteTimers = useRef<Set<number>>(new Set())
  const hasLoadedOnceRef = useRef(false)
  const cardsRef = useRef<KanbanCardDTO[]>([])
  const dispatchingRef = useRef(false)
  const handledJobIdRef = useRef<string | null>(null)
  const activeCount = Object.keys(activeDispatches).length

  useEffect(() => {
    cardsRef.current = cards
    const approvalCount = cards.filter((c) => c.column_name === 'approval').length
    onApprovalCountChange?.(approvalCount)
  }, [cards, onApprovalCountChange])

  function removeActiveDispatch(cardId: string) {
    setActiveDispatches((prev) => {
      if (!(cardId in prev)) return prev
      const next = { ...prev }
      delete next[cardId]
      return next
    })
    setLiveNode((prev) => {
      const next = { ...prev }
      delete next[cardId]
      return next
    })
    setLiveLogCount((prev) => {
      const next = { ...prev }
      delete next[cardId]
      return next
    })
  }

  async function refresh() {
    const list = await api.listKanbanCards()
    setCards(list)
  }

  useEffect(() => {
    refresh().catch((e) => setError(e instanceof ApiError ? e.message : 'โหลดการ์ดไม่สำเร็จ'))
  }, [])

  useEffect(() => {
    if (refreshTrigger !== undefined && refreshTrigger > 0) {
      refresh().catch(() => {})
    }
  }, [refreshTrigger])

  useEffect(() => {
    // Periodically sync cards every 5 seconds so background runs show up live
    const pollInterval = window.setInterval(() => {
      refresh().catch(() => {})
    }, 5000)
    return () => {
      window.clearInterval(pollInterval)
    }
  }, [])

  useEffect(() => {
    if (cards.length > 0) hasLoadedOnceRef.current = true
  }, [cards])

  // Auto-select card matching selectedJobId ONCE when a job runs
  useEffect(() => {
    if (!selectedJobId) {
      handledJobIdRef.current = null
      return
    }

    // If this job has already been auto-selected, do NOT re-select on subsequent polls/renders
    if (handledJobIdRef.current === selectedJobId) {
      return
    }

    const targetJobId = selectedJobId

    function selectMatching(items: KanbanCardDTO[]) {
      const match = items.find(
        (c) => c.job_id === targetJobId || Boolean(c.prompt && c.prompt.includes(targetJobId))
      )
      if (match) {
        // Mark as handled immediately so subsequent polls or user card changes are never overridden
        handledJobIdRef.current = targetJobId
        setSelectedCardId(match.card_id)
        if (layoutMode === 'compact') {
          setCompactActiveTab(match.column_name)
        }
        return true
      }
      return false
    }

    if (selectMatching(cards)) return

    // Retry fetching cards in case the card is being created asynchronously in background
    let active = true
    const timer1 = window.setTimeout(async () => {
      if (!active || handledJobIdRef.current === targetJobId) return
      try {
        const list = await api.listKanbanCards()
        setCards(list)
        selectMatching(list)
      } catch {}
    }, 600)

    const timer2 = window.setTimeout(async () => {
      if (!active || handledJobIdRef.current === targetJobId) return
      try {
        const list = await api.listKanbanCards()
        setCards(list)
        if (!selectMatching(list)) {
          // If still not matched after retries, mark as handled so we stop querying
          handledJobIdRef.current = targetJobId
        }
      } catch {
        handledJobIdRef.current = targetJobId
      }
    }, 1600)

    return () => {
      active = false
      window.clearTimeout(timer1)
      window.clearTimeout(timer2)
    }
  }, [selectedJobId, cards, layoutMode])

  useEffect(() => {
    if (activeCount > 0) {
      tickTimer.current = window.setInterval(() => setNowTick(Date.now()), 1000)
    }
    return () => {
      if (tickTimer.current) window.clearInterval(tickTimer.current)
    }
  }, [activeCount])

  useEffect(() => {
    const timers = deleteTimers.current
    return () => {
      if (noticeDismissTimer.current) window.clearTimeout(noticeDismissTimer.current)
      if (noticeLeaveTimer.current) window.clearTimeout(noticeLeaveTimer.current)
      timers.forEach((id) => window.clearTimeout(id))
    }
  }, [])

  function flashNotice(message: string) {
    if (noticeDismissTimer.current) window.clearTimeout(noticeDismissTimer.current)
    if (noticeLeaveTimer.current) window.clearTimeout(noticeLeaveTimer.current)
    setNotice(message)
    setNoticeLeaving(false)
    noticeDismissTimer.current = window.setTimeout(() => {
      setNoticeLeaving(true)
      noticeLeaveTimer.current = window.setTimeout(() => {
        setNotice(null)
        setNoticeLeaving(false)
      }, NOTICE_LEAVE_MS)
    }, NOTICE_DISMISS_MS)
  }

  async function createCard(title: string, flow: string = 'manager', prompt?: string, scope: string = 'both') {
    const trimmed = title.trim()
    if (!trimmed) return
    const { created } = await api.createKanbanCard(trimmed, flow, prompt, scope)
    if (!created) {
      flashNotice(`มีการ์ดนี้อยู่ใน Backlog แล้ว — ไม่เพิ่มซ้ำ`)
    }
    await refresh()
  }

  async function updateCard(cardId: string, title: string, prompt: string, flow: string, scope: string) {
    await api.updateKanbanCard(cardId, title, prompt, flow, scope)
    await refresh()
  }

  async function handleModalSubmit(values: { title: string; prompt: string; flow: string; scope: string }) {
    setModalError(null)
    try {
      if (modalState?.mode === 'create') {
        await createCard(values.title, values.flow, values.prompt, values.scope)
      } else if (modalState?.mode === 'edit') {
        await updateCard(modalState.card.card_id, values.title, values.prompt, values.flow, values.scope)
      }
      setModalState(null)
    } catch (e) {
      setModalError(e instanceof ApiError ? e.message : 'บันทึกการ์ดไม่สำเร็จ')
    }
  }

  function openCreateModal() {
    setModalError(null)
    setModalState({ mode: 'create' })
  }

  function openEditModal(card: KanbanCardDTO) {
    setModalError(null)
    setModalState({ mode: 'edit', card })
  }

  function closeModal() {
    setModalError(null)
    setModalState(null)
  }

  async function handleQuickCreate(label: string, instruction: string, flow: string, scope: string) {
    try {
      await createCard(label, flow, instruction, scope)
    } catch (e) {
      setError(e instanceof ApiError ? e.message : 'เพิ่มการ์ดไม่สำเร็จ')
    }
  }

  function saveEditedTemplate(index: number, next: QuickTemplate) {
    setQuickTemplates((prev) => {
      const updated = prev.map((t, i) => (i === index ? next : t))
      saveQuickTemplates(updated)
      return updated
    })
    setEditingTemplateIndex(null)
  }

  async function deleteCard(cardId: string) {
    setRemovingIds((prev) => new Set(prev).add(cardId))
    const timerId = window.setTimeout(async () => {
      deleteTimers.current.delete(timerId)
      try {
        await api.deleteKanbanCard(cardId)
        removeActiveDispatch(cardId)
        if (cardId === selectedCardId) setSelectedCardId(null)
        await refresh()
      } catch (e) {
        setError(e instanceof ApiError ? e.message : 'ลบการ์ดไม่สำเร็จ')
      } finally {
        setRemovingIds((prev) => {
          const next = new Set(prev)
          next.delete(cardId)
          return next
        })
      }
    }, DELETE_ANIM_MS)
    deleteTimers.current.add(timerId)
  }

  async function dispatchCard(card: KanbanCardDTO, flow: string) {
    if (dispatchingRef.current) return
    dispatchingRef.current = true
    setDispatching(true)
    try {
      const instruction = card.prompt?.trim() || card.title
      const job = await api.dispatchJob(instruction, card.card_id, flow, card.scope ?? 'both')
      setActiveDispatches((prev) => ({
        ...prev,
        [card.card_id]: { jobId: job.job_id, dispatchedAt: Date.now() },
      }))
      await refresh()
    } catch (e) {
      setError(e instanceof ApiError ? e.message : 'สั่งงานไม่สำเร็จ')
    } finally {
      dispatchingRef.current = false
      setDispatching(false)
    }
  }

  function handleTerminalStatusChange(cardId: string, status: TerminalStatus) {
    const targetColumn = columnForStatus(status)
    if (targetColumn) {
      const card = cardsRef.current.find((c) => c.card_id === cardId)
      if (card && card.column_name !== targetColumn) {
        api
          .moveKanbanCard(cardId, targetColumn)
          .then(refresh)
          .catch((e) => setError(e instanceof ApiError ? e.message : 'อัปเดตสถานะการ์ดไม่สำเร็จ'))
      }
      const activeDispatch = activeDispatches[cardId]
      removeActiveDispatch(cardId)

      // Auto-resume-all เฉพาะเมื่อกด Play จากบอร์ดโดยตรง (ไม่ได้เปิด Drawer สำหรับ cardId นี้)
      // และเป็น flow news_funnel ที่ได้ interrupt_payload เป็น news_funnel_approval
      if (status === 'awaiting_approval' && cardId !== selectedCardId && card?.flow === 'news_funnel' && activeDispatch?.jobId) {
        api.getJobStatus(activeDispatch.jobId).then(async (job) => {
          if (job.interrupt_payload?.type === 'news_funnel_approval' && Array.isArray(job.interrupt_payload.candidates)) {
            const allEventIds = job.interrupt_payload.candidates.map((c: any) => c.event_id).filter(Boolean)
            await api.resumeJob(job.job_id, [], [], allEventIds)
            await refresh()
          }
        }).catch((e) => console.error('Failed to auto-resume news_funnel job from board:', e))
      }
    }
  }

  function workspacePreviewFor(card: KanbanCardDTO): WorkspacePreview | undefined {
    const d = activeDispatches[card.card_id]
    if (!d) return undefined
    return {
      node: liveNode[card.card_id] ?? null,
      logCount: liveLogCount[card.card_id] ?? 0,
      elapsedSeconds: (nowTick - d.dispatchedAt) / 1000,
    }
  }

  const visibleColumns = KANBAN_COLUMNS.filter((col) => STATUS_COLUMN_MAP[statusFilter].includes(col.key))
  const searchLower = search.trim().toLowerCase()
  const editingTemplate = editingTemplateIndex !== null ? quickTemplates[editingTemplateIndex] : undefined

  function cardsForColumn(colKey: string): KanbanCardDTO[] {
    const filtered = cards.filter((c) => {
      if (c.column_name !== colKey) return false
      if (flowFilter !== 'all' && c.flow !== flowFilter) return false
      if (searchLower && !c.title.toLowerCase().includes(searchLower)) return false
      if (colKey === 'done' && daysSince(c.updated_at) > DONE_HIDE_AFTER_DAYS) return false
      return true
    })
    if (colKey === 'done') {
      return [...filtered].sort((a, b) => b.updated_at - a.updated_at)
    }
    return filtered
  }

  function handleAskAi(card: KanbanCardDTO) {
    if (!onAskAiWithCard) return
    const prompt = `กรุณาช่วยวิเคราะห์และให้คำแนะนำสำหรับงานนี้:\n- ID: #AG-${card.display_seq ?? card.card_id}\n- Flow: ${card.flow}\n- หัวข้องาน: ${card.title}\n- คำสั่ง: ${card.prompt?.trim() || '(ไม่มีคำสั่งเพิ่มเติม)'}\n- สถานะปัจจุบัน: ${card.column_name}`
    onAskAiWithCard(prompt, card)
  }

  const selectedCard = cards.find((c) => c.card_id === selectedCardId) ?? null

  return (
    <div className="flex h-full w-full min-h-0 min-w-0 flex-col overflow-hidden bg-slate-50/50">
      {/* Top Filter and Controls Bar */}
      <div className="shrink-0 space-y-2 border-b border-sky-100 bg-white/80 p-3 backdrop-blur-md">
        <div className="flex items-center gap-2">
          <TextInput
            uiSize="sm"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="ค้นหาชื่องาน..."
            className="flex-1 min-w-[140px]"
          />
          <AddCardDropdown
            quickTemplates={quickTemplates}
            onQuickCreate={handleQuickCreate}
            onOpenCustomModal={openCreateModal}
            onEditTemplate={setEditingTemplateIndex}
          />
        </div>

        {/* Filters Row */}
        <div className="flex flex-wrap items-center justify-between gap-1.5 text-xs">
          {/* Status Tabs */}
          <div className="flex items-center rounded-lg bg-sky-100/60 p-0.5">
            {STATUS_TABS.map((t) => (
              <button
                key={t.key}
                type="button"
                onClick={() => setStatusFilter(t.key)}
                className={`rounded-md px-2 py-1 text-[11px] font-semibold transition-all ${
                  statusFilter === t.key
                    ? 'bg-white text-sky-950 shadow-sm'
                    : 'text-zinc-600 hover:text-sky-900'
                }`}
              >
                {t.label}
              </button>
            ))}
          </div>

          {/* Flow Dropdown / Filter & Manual Refresh */}
          <div className="flex items-center gap-1.5">
            <select
              value={flowFilter}
              onChange={(e) => setFlowFilter(e.target.value as FlowFilter)}
              className="rounded-lg border border-sky-100 bg-white px-2 py-1 text-[11px] font-medium text-sky-950 shadow-sm focus:outline-none focus:ring-1 focus:ring-sky-300"
            >
              <option value="all">ทุกกระบวนงาน (Flows)</option>
              {FLOW_OPTIONS.map((f) => (
                <option key={f.key} value={f.key}>
                  {f.label}
                </option>
              ))}
            </select>
            <button
              type="button"
              onClick={() => refresh().catch(() => {})}
              title="รีเฟรชข้อมูลบอร์ดงาน"
              className="flex h-7 w-7 items-center justify-center rounded-lg border border-sky-100 bg-white text-xs text-sky-700 shadow-sm hover:bg-sky-50 transition-colors"
            >
              🔄
            </button>
          </div>
        </div>
      </div>

      {notice && (
        <div
          className={`mx-3 mt-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-1.5 text-xs text-amber-800 ${
            noticeLeaving ? 'animate-notice-out' : 'animate-notice-in'
          }`}
        >
          {notice}
        </div>
      )}
      {error && (
        <div className="mx-3 mt-2 animate-notice-in rounded-lg border border-rose-200 bg-rose-50 px-3 py-1.5 text-xs text-rose-700">
          {error}
        </div>
      )}

      {/* Main Board Container */}
      <div className="flex-1 min-h-0 overflow-hidden relative">
        {layoutMode === 'compact' ? (
          /* COMPACT MODE: 1 Column at a time with quick Swimlane tabs or full-screen card drawer */
          selectedCard ? (
            <div className="flex h-full flex-col">
              <div className="flex shrink-0 items-center justify-between border-b border-sky-100 bg-sky-50/70 px-3 py-2">
                <button
                  type="button"
                  onClick={() => setSelectedCardId(null)}
                  className="flex items-center gap-1 text-xs font-semibold text-sky-800 hover:text-sky-950 transition-colors"
                >
                  <span>←</span>
                  <span>กลับไปที่บอร์ด</span>
                </button>
                <span className="font-mono text-[11px] text-zinc-400">#AG-{selectedCard.display_seq}</span>
              </div>
              <div className="flex-1 min-h-0">
                <KanbanDetailDrawer
                  card={selectedCard}
                  embedded={true}
                  onClose={() => setSelectedCardId(null)}
                  onCardTransition={refresh}
                  onDispatchCard={(c, flow) => dispatchCard(c, flow)}
                  onAskAi={handleAskAi}
                  onWatchPipeline={onWatchPipeline}
                />
              </div>
            </div>
          ) : (
            <div className="flex h-full flex-col p-2.5">
              {/* Compact Column Tab Selectors */}
              <div className="mb-2 flex items-center justify-between gap-1 overflow-x-auto rounded-xl bg-sky-100/40 p-1">
                {visibleColumns.map((col) => {
                  const count = cardsForColumn(col.key).length
                  const isActive = compactActiveTab === col.key
                  return (
                    <button
                      key={col.key}
                      type="button"
                      onClick={() => setCompactActiveTab(col.key)}
                      className={`flex flex-1 min-w-[70px] items-center justify-center gap-1 rounded-lg px-2 py-1.5 text-xs font-semibold transition-all ${
                        isActive
                          ? 'bg-white text-sky-950 shadow-sm'
                          : 'text-zinc-600 hover:text-sky-900'
                      }`}
                    >
                      <span>{col.icon}</span>
                      <span className="truncate">{col.label}</span>
                      <span className="rounded-full bg-sky-200/60 px-1 py-0.2 text-[9px] font-mono">
                        {count}
                      </span>
                    </button>
                  )
                })}
              </div>

              {/* Render Active Column in Compact Mode */}
              <div className="flex-1 min-h-0">
                {visibleColumns
                  .filter((col) => col.key === compactActiveTab)
                  .map((col) => (
                    <KanbanColumn
                      key={col.key}
                      column={col}
                      cards={cardsForColumn(col.key)}
                      isBacklogColumn={col.key === 'backlog'}
                      removingIds={removingIds}
                      selectedCardId={selectedCardId}
                      workspacePreviewFor={workspacePreviewFor}
                      staggerCards={!hasLoadedOnceRef.current}
                      dispatchDisabled={dispatching}
                      onDeleteCard={deleteCard}
                      onCardClick={(c) => setSelectedCardId(c.card_id)}
                      onEditCard={openEditModal}
                      onDispatchCard={(c) => {
                        setError(null)
                        dispatchCard(c, c.flow ?? 'manager')
                      }}
                      onAskAiCard={handleAskAi}
                    />
                  ))}
              </div>
            </div>
          )
        ) : (
          /* WIDE & FULL MODE: 4 Columns side-by-side with optional Right-Hand Drawer */
          <div className="flex h-full w-full min-h-0 overflow-hidden">
            {/* Columns Area */}
            <div className="flex-1 min-h-0 min-w-0 overflow-x-auto p-3">
              <div
                className="grid h-full min-h-0 gap-3 pb-2"
                style={{
                  gridTemplateColumns: `repeat(${visibleColumns.length}, minmax(240px, 1fr))`,
                }}
              >
                {visibleColumns.map((col) => (
                  <KanbanColumn
                    key={col.key}
                    column={col}
                    cards={cardsForColumn(col.key)}
                    isBacklogColumn={col.key === 'backlog'}
                    removingIds={removingIds}
                    selectedCardId={selectedCardId}
                    workspacePreviewFor={workspacePreviewFor}
                    staggerCards={!hasLoadedOnceRef.current}
                    dispatchDisabled={dispatching}
                    onDeleteCard={deleteCard}
                    onCardClick={(c) => setSelectedCardId(c.card_id)}
                    onEditCard={openEditModal}
                    onDispatchCard={(c) => {
                      setError(null)
                      dispatchCard(c, c.flow ?? 'manager')
                    }}
                    onAskAiCard={handleAskAi}
                  />
                ))}
              </div>
            </div>

            {/* Split Drawer in Wide/Full Mode */}
            {selectedCard && (
              <div className="w-[380px] shrink-0 border-l border-sky-100 bg-white">
                <KanbanDetailDrawer
                  card={selectedCard}
                  embedded={true}
                  onClose={() => setSelectedCardId(null)}
                  onCardTransition={refresh}
                  onDispatchCard={(c, flow) => dispatchCard(c, flow)}
                  onAskAi={handleAskAi}
                  onWatchPipeline={onWatchPipeline}
                />
              </div>
            )}
          </div>
        )}
      </div>

      {/* Background terminal driver for auto column transitions */}
      {Object.entries(activeDispatches).map(([cardId, d]) => (
        <LiveTerminal
          key={d.jobId}
          jobId={d.jobId}
          onStatusChange={(status) => handleTerminalStatusChange(cardId, status)}
          onNodeUpdate={(node) => setLiveNode((prev) => ({ ...prev, [cardId]: node }))}
          onLineCountChange={(count) => setLiveLogCount((prev) => ({ ...prev, [cardId]: count }))}
          hideUi
        />
      ))}

      {modalState && (
        <KanbanCardModal
          mode={modalState.mode}
          initialTitle={modalState.mode === 'edit' ? modalState.card.title : undefined}
          initialPrompt={modalState.mode === 'edit' ? (modalState.card.prompt ?? '') : undefined}
          initialFlow={modalState.mode === 'edit' ? modalState.card.flow : undefined}
          initialScope={modalState.mode === 'edit' ? modalState.card.scope : undefined}
          errorMessage={modalError}
          onClose={closeModal}
          onSubmit={handleModalSubmit}
        />
      )}

      {editingTemplateIndex !== null && editingTemplate && (
        <EditTemplateModal
          template={editingTemplate}
          onClose={() => setEditingTemplateIndex(null)}
          onSave={(next) => saveEditedTemplate(editingTemplateIndex, next)}
        />
      )}
    </div>
  )
}

