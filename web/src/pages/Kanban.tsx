import { useEffect, useRef, useState } from 'react'
import { api, ApiError } from '../api/client'
import type { KanbanCardDTO } from '../api/types'
import { columnForStatus, type TerminalStatus } from '../lib/agentStatus'
import LiveTerminal from '../components/LiveTerminal'
import KanbanColumn from '../components/kanban/KanbanColumn'
import KanbanHeader from '../components/kanban/KanbanHeader'
import KanbanDetailDrawer from '../components/kanban/KanbanDetailDrawer'
import KanbanCardModal from '../components/kanban/KanbanCardModal'
import AddCardDropdown from '../components/kanban/AddCardDropdown'
import EditTemplateModal from '../components/kanban/EditTemplateModal'
import TextInput from '../components/ui/TextInput'
import type { ColumnDef, FlowFilter, StatusFilter, WorkspacePreview } from '../components/kanban/types'
import { loadQuickTemplates, saveQuickTemplates, type QuickTemplate } from '../lib/quickTemplateStorage'

const COLUMNS: ColumnDef[] = [
  { key: 'backlog', label: 'Backlog', icon: '📥' },
  { key: 'approval', label: 'Request Approve', icon: '🙋' },
  { key: 'executing', label: 'Workers Executing', icon: '🔍' },
  { key: 'done', label: 'Done', icon: '✅' },
]

// Status Tabs กรอง "คอลัมน์ไหนโชว์บ้าง" ไม่ใช่กรองการ์ดรายตัว — ตรงกับความหมายของ Kanban
// ที่จัดกลุ่มด้วยคอลัมน์อยู่แล้ว (ดู Rev.2 §3.1)
const STATUS_COLUMN_MAP: Record<StatusFilter, string[]> = {
  active: ['approval', 'executing'],
  all: ['backlog', 'approval', 'executing', 'done'],
  backlog: ['backlog'],
  done: ['done'],
}

const DONE_FADE_AFTER_DAYS = 7
const NOTICE_DISMISS_MS = 2500
const NOTICE_LEAVE_MS = 150
const DELETE_ANIM_MS = 150

type ModalState = { mode: 'create' } | { mode: 'edit'; card: KanbanCardDTO } | null

// การ์ดที่ dispatch แล้วแต่ยังไม่ถึงสถานะปลายทาง (done/error/awaiting_approval) — เก็บเป็น
// map คีย์ด้วย card_id แทนที่จะเป็นค่าเดี่ยว เพราะ backend รับหลาย job ได้พร้อมกัน (queue)
// ถ้า dispatch การ์ด B ก่อนการ์ด A จะเสร็จ ค่าเดี่ยวเดิมจะทำให้ A หลุด tracking ค้างคอลัมน์
// "Workers Executing" ตลอดไปแม้ backend รันเสร็จจริงแล้ว (เจอจริงจาก review)
interface ActiveDispatch {
  jobId: string
  dispatchedAt: number
}

function daysSince(unixSeconds: number): number {
  return (Date.now() / 1000 - unixSeconds) / 86400
}

export default function Kanban() {
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
  const tickTimer = useRef<number | null>(null)
  const noticeDismissTimer = useRef<number | null>(null)
  const noticeLeaveTimer = useRef<number | null>(null)
  const deleteTimers = useRef<Set<number>>(new Set())
  const hasLoadedOnceRef = useRef(false)
  const activeCount = Object.keys(activeDispatches).length

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
    setCards(await api.listKanbanCards())
  }

  useEffect(() => {
    refresh().catch((e) => setError(e instanceof ApiError ? e.message : 'โหลดการ์ดไม่สำเร็จ'))
  }, [])

  useEffect(() => {
    // ref (ไม่ใช่ state) เพราะต้องอ่านค่า "ก่อนแฟล็กถูกตั้ง" ในรอบ render เดียวกับที่การ์ดจริง
    // ปรากฏครั้งแรก ถ้าใช้ state ทั้งคู่จะถูก batch เข้า render เดียวกันแล้วธงจะกลายเป็น true
    // ไปแล้วตั้งแต่ก่อนการ์ดจะ render จริง ทำให้ stagger ไม่ทำงานเลย
    if (cards.length > 0) hasLoadedOnceRef.current = true
  }, [cards])

  // เดินนาฬิกาทุก 1s เฉพาะตอนมีงานกำลังรันอยู่อย่างน้อย 1 ตัว — ให้ elapsed time ใน
  // workspace preview ขยับจริง (ใช้ activeCount ไม่ใช่ตัว object เอง กัน interval
  // restart ทุกครั้งที่ node/logCount ของ job ใดๆ อัปเดต)
  useEffect(() => {
    if (activeCount > 0) {
      tickTimer.current = window.setInterval(() => setNowTick(Date.now()), 1000)
    }
    return () => {
      if (tickTimer.current) window.clearInterval(tickTimer.current)
    }
  }, [activeCount])

  useEffect(() => {
    // จับ Set instance ไว้ในตัวแปร local — ref.current ตอน cleanup อาจเป็นคนละค่ากับตอน
    // effect รัน (react-hooks/exhaustive-deps) แม้ในเคสนี้ Set ไม่เคยถูกแทนที่ก็ตาม
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
      // เก็บ modal เปิดค้างไว้ให้ผู้ใช้แก้แล้วลองใหม่ — เดิมไม่มี catch เลย ทำให้ error
      // หายเงียบและโมดัลไม่ปิดโดยไม่บอกเหตุผลอะไรเลย
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
        // เดิมไม่มี catch — ถ้า delete fail การ์ดจะค้าง opacity เฟดครึ่งเดียวตลอดไป
        // เพราะ removingIds ไม่เคยถูกลบออก ตอนนี้ finally ด้านล่างลบให้เสมอไม่ว่าจะสำเร็จหรือพัง
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
      setDispatching(false)
    }
  }

  // background driver (hideUi) ขับแค่ dispatch → running → (done หรือ awaiting_approval)
  // เท่านั้น — พอถึง awaiting_approval มันปิด connection ตัวเองแล้ว (LiveTerminal design)
  // ส่วนที่เหลือ (approve → resume → done) เป็นหน้าที่ของ terminal ใน KanbanDetailDrawer
  // ที่ผู้ใช้ต้องเปิดดูเพื่อกด approve อยู่แล้ว ผูกกับ card_id ของมันเอง ไม่ชนกัน
  // รับ cardId ตรงๆ (ผูกกับ closure ของ card นั้นตอน render แต่ละ LiveTerminal instance)
  // แทนการอ้าง activeCardId ตัวเดียวส่วนกลาง เพื่อให้ track ได้หลาย job พร้อมกัน
  function handleTerminalStatusChange(cardId: string, status: TerminalStatus) {
    const targetColumn = columnForStatus(status)
    if (targetColumn) {
      // guard: ถ้าการ์ดอยู่คอลัมน์เป้าหมายอยู่แล้ว ไม่ยิง API ซ้ำ — กัน race กับ
      // KanbanDetailDrawer ที่ผูกกับ card_id เดียวกันตอนเปิด drawer ดูงานที่กำลังรันอยู่
      const card = cards.find((c) => c.card_id === cardId)
      if (card && card.column_name !== targetColumn) {
        api
          .moveKanbanCard(cardId, targetColumn)
          .then(refresh)
          .catch((e) => setError(e instanceof ApiError ? e.message : 'อัปเดตสถานะการ์ดไม่สำเร็จ'))
      }
      removeActiveDispatch(cardId)
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

  const visibleColumns = COLUMNS.filter((col) => STATUS_COLUMN_MAP[statusFilter].includes(col.key))
  const searchLower = search.trim().toLowerCase()
  const editingTemplate = editingTemplateIndex !== null ? quickTemplates[editingTemplateIndex] : undefined

  function cardsForColumn(colKey: string): KanbanCardDTO[] {
    return cards.filter((c) => {
      if (c.column_name !== colKey) return false
      if (flowFilter !== 'all' && c.flow !== flowFilter) return false
      if (searchLower && !c.title.toLowerCase().includes(searchLower)) return false
      return true
    })
  }

  return (
    <div className="flex h-full items-start gap-4">
      <div className="flex h-full min-w-0 flex-1 animate-page-in flex-col gap-6">
        <div className="flex flex-wrap items-center gap-3">
          <h1 className="text-xl font-semibold text-zinc-900">Agent Kanban Board</h1>
          <TextInput
            uiSize="sm"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="ค้นหางาน..."
            className="ml-auto w-56"
          />
          <AddCardDropdown
            quickTemplates={quickTemplates}
            onQuickCreate={handleQuickCreate}
            onOpenCustomModal={openCreateModal}
            onEditTemplate={setEditingTemplateIndex}
          />
        </div>

        <KanbanHeader
          statusFilter={statusFilter}
          onStatusFilterChange={setStatusFilter}
          flowFilter={flowFilter}
          onFlowFilterChange={setFlowFilter}
        />

        {notice && (
          <p
            className={`rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800 ${
              noticeLeaving ? 'animate-notice-out' : 'animate-notice-in'
            }`}
          >
            {notice}
          </p>
        )}
        {error && <p className="animate-notice-in text-sm text-red-600">{error}</p>}

        <div
          className="grid min-h-0 flex-1 gap-3 overflow-x-auto pb-2"
          style={{ gridTemplateColumns: `repeat(${visibleColumns.length}, minmax(140px, 1fr))` }}
        >
          {visibleColumns.map((col) => (
            <KanbanColumn
              key={col.key}
              column={col}
              cards={cardsForColumn(col.key)}
              isBacklogColumn={col.key === 'backlog'}
              isCardFaded={(c) => col.key === 'done' && daysSince(c.updated_at) > DONE_FADE_AFTER_DAYS}
              removingIds={removingIds}
              selectedCardId={selectedCardId}
              workspacePreviewFor={workspacePreviewFor}
              staggerCards={!hasLoadedOnceRef.current}
              onDeleteCard={deleteCard}
              onCardClick={(c) => setSelectedCardId(c.card_id)}
              onEditCard={openEditModal}
              onDispatchCard={(c) => {
                if (!dispatching) {
                  setError(null)
                  dispatchCard(c, c.flow ?? 'manager')
                }
              }}
            />
          ))}
        </div>

        {/* background driver: ไม่โชว์ UI — แค่ขับ auto column-transition + workspace preview
            บนการ์ดที่กำลัง active อยู่ (ดูรายละเอียดจริงต้องคลิกการ์ดเปิด Drawer) — render
            1 instance ต่อ 1 job ที่ยังไม่ถึงสถานะปลายทาง เพื่อไม่ให้การ์ดที่ dispatch ก่อนหน้า
            หลุด tracking เมื่อมีการ dispatch การ์ดใหม่ซ้อนเข้ามา */}
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
      </div>

      <KanbanDetailDrawer
        card={cards.find((c) => c.card_id === selectedCardId) ?? null}
        onClose={() => setSelectedCardId(null)}
        onCardTransition={refresh}
      />

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
