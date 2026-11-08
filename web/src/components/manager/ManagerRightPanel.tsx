import { useState, useEffect, useCallback, useRef } from 'react'
import type { ActiveAgentStatusDTO, KanbanCardDTO } from '../../api/types'
import ManagerLiveExecutionView from './ManagerLiveExecutionView'
import ManagerSwarmView from './ManagerSwarmView'
import ManagerKanbanView from './ManagerKanbanView'
import MorningDeltaCard from './MorningDeltaCard'

export type ManagerRightPanelTab = 'kanban' | 'execution' | 'swarm' | 'review'
export type KanbanWidthMode = 'compact' | 'wide' | 'full'

const STORAGE_KEY_PANEL_WIDTH = 'trinity_manager_right_panel_width'
const DEFAULT_PANEL_WIDTH = 420
const MIN_PANEL_WIDTH = 340
const MAX_PANEL_WIDTH = 1200
const WIDE_KANBAN_THRESHOLD = 680

export interface ManagerRightPanelProps {
  isOpen: boolean
  onClose: () => void
  activeTab: ManagerRightPanelTab
  onTabChange: (tab: ManagerRightPanelTab) => void
  selectedJobId: string | null
  onSelectJob: (jobId: string) => void
  detectedJobIds: string[]
  activeAgentStatus?: ActiveAgentStatusDTO | null
  refreshTrigger?: number
  onTriggerQuickJob?: (prompt: string) => void
  onAskAiWithCard?: (prompt: string, card: KanbanCardDTO) => void
}

function loadInitialWidth(): number {
  try {
    const raw = localStorage.getItem(STORAGE_KEY_PANEL_WIDTH)
    if (!raw) return DEFAULT_PANEL_WIDTH
    const parsed = Number(raw)
    if (Number.isFinite(parsed) && parsed >= MIN_PANEL_WIDTH && parsed <= MAX_PANEL_WIDTH) {
      return parsed
    }
  } catch {
    // ignore storage errors
  }
  return DEFAULT_PANEL_WIDTH
}

export default function ManagerRightPanel({
  isOpen,
  onClose,
  activeTab,
  onTabChange,
  selectedJobId,
  onSelectJob,
  detectedJobIds,
  activeAgentStatus,
  refreshTrigger,
  onTriggerQuickJob,
  onAskAiWithCard,
}: ManagerRightPanelProps) {
  const [kanbanWidthMode, setKanbanWidthMode] = useState<KanbanWidthMode>('compact')
  const [panelWidth, setPanelWidth] = useState<number>(loadInitialWidth)
  const [isResizing, setIsResizing] = useState(false)
  const [approvalCount, setApprovalCount] = useState<number>(0)
  const dragStartRef = useRef<{ startX: number; startWidth: number } | null>(null)

  const clampWidth = useCallback((w: number) => {
    const maxAllowed =
      typeof window !== 'undefined' && window.innerWidth > 768
        ? Math.min(MAX_PANEL_WIDTH, Math.max(MIN_PANEL_WIDTH, window.innerWidth - 280))
        : MAX_PANEL_WIDTH
    return Math.round(Math.min(maxAllowed, Math.max(MIN_PANEL_WIDTH, w)))
  }, [])

  const updateWidth = useCallback(
    (nextWidth: number) => {
      const clamped = clampWidth(nextWidth)
      setPanelWidth(clamped)
      setKanbanWidthMode(clamped >= WIDE_KANBAN_THRESHOLD ? 'wide' : 'compact')
      try {
        localStorage.setItem(STORAGE_KEY_PANEL_WIDTH, String(clamped))
      } catch {
        // ignore storage errors
      }
    },
    [clampWidth]
  )

  useEffect(() => {
    if (!isResizing) return

    function handleMouseMove(e: MouseEvent) {
      if (!dragStartRef.current) return
      // Dragging left increases right panel width; dragging right decreases it
      const deltaX = dragStartRef.current.startX - e.clientX
      updateWidth(dragStartRef.current.startWidth + deltaX)
    }

    function handleMouseUp() {
      setIsResizing(false)
      dragStartRef.current = null
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
    }

    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
    window.addEventListener('mousemove', handleMouseMove)
    window.addEventListener('mouseup', handleMouseUp)
    return () => {
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
      window.removeEventListener('mousemove', handleMouseMove)
      window.removeEventListener('mouseup', handleMouseUp)
    }
  }, [isResizing, updateWidth])

  if (!isOpen) return null

  const isFullMode = activeTab === 'kanban' && kanbanWidthMode === 'full'
  const effectiveKanbanLayout: KanbanWidthMode = isFullMode
    ? 'full'
    : panelWidth >= WIDE_KANBAN_THRESHOLD || kanbanWidthMode === 'wide'
      ? 'wide'
      : 'compact'

  // Dynamic width classes and inline width style for smooth left-right resizing
  const widthClasses = isFullMode
    ? 'w-full max-w-none xl:w-full xl:max-w-none xl:flex-1'
    : 'w-full max-w-[92vw] xl:shrink-0'

  const inlinePanelStyle = isFullMode ? undefined : { width: `${panelWidth}px` }

  function handleResizeStart(e: React.MouseEvent<HTMLDivElement>) {
    e.preventDefault()
    if (isFullMode) {
      const fallbackStart =
        typeof window !== 'undefined' ? Math.min(880, window.innerWidth - 400) : 760
      setKanbanWidthMode('wide')
      setPanelWidth(fallbackStart)
      dragStartRef.current = { startX: e.clientX, startWidth: fallbackStart }
    } else {
      dragStartRef.current = { startX: e.clientX, startWidth: panelWidth }
    }
    setIsResizing(true)
  }

  function handleResizeDoubleClick() {
    setKanbanWidthMode('compact')
    setPanelWidth(DEFAULT_PANEL_WIDTH)
    try {
      localStorage.setItem(STORAGE_KEY_PANEL_WIDTH, String(DEFAULT_PANEL_WIDTH))
    } catch {
      // ignore
    }
  }

  function handleResizeKeyDown(e: React.KeyboardEvent<HTMLDivElement>) {
    if (e.key === 'ArrowLeft') {
      e.preventDefault()
      updateWidth(panelWidth + 40)
    } else if (e.key === 'ArrowRight') {
      e.preventDefault()
      updateWidth(panelWidth - 40)
    }
  }

  return (
    <>
      {/* Mobile Backdrop Overlay (only on < xl screens) */}
      <div
        className="fixed inset-0 z-40 bg-sky-950/20 backdrop-blur-sm xl:hidden"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Main Panel Container */}
      <aside
        style={inlinePanelStyle}
        data-testid="manager-right-panel"
        className={`fixed inset-y-0 right-0 z-50 flex flex-col border-l border-sky-100/80 bg-panel/95 shadow-[-12px_0_32px_rgba(14,165,233,0.05)] backdrop-blur-xl ${
          isResizing ? 'transition-none' : 'transition-[width] duration-200 ease-out'
        } xl:static xl:z-auto xl:h-full ${widthClasses}`}
        aria-label="Agent Work, Kanban and Swarm Panel"
      >
        {/* Interactive Drag-to-Resize Handle on Left Edge */}
        <div
          role="separator"
          tabIndex={0}
          aria-orientation="vertical"
          aria-label="ปรับขนาดแถบด้านข้าง"
          aria-valuenow={panelWidth}
          aria-valuemin={MIN_PANEL_WIDTH}
          aria-valuemax={MAX_PANEL_WIDTH}
          title="ลากซ้าย-ขวาเพื่อปรับขนาด (ดับเบิลคลิกเพื่อรีเซ็ต)"
          onMouseDown={handleResizeStart}
          onDoubleClick={handleResizeDoubleClick}
          onKeyDown={handleResizeKeyDown}
          className={`group absolute inset-y-0 -left-1.5 z-30 hidden w-3 cursor-col-resize items-center justify-center xl:flex ${
            isResizing ? 'bg-sky-400/20' : 'hover:bg-sky-400/15'
          }`}
        >
          {/* Subtle central grip pill */}
          <div
            className={`h-10 w-1 rounded-full transition-all ${
              isResizing
                ? 'h-16 bg-sky-500 shadow-[0_0_8px_rgba(14,165,233,0.6)]'
                : 'bg-sky-200/80 group-hover:h-14 group-hover:bg-sky-400'
            }`}
          />
          {/* Floating width indicator badge while actively dragging */}
          {isResizing && (
            <span className="pointer-events-none absolute -left-16 top-1/2 -translate-y-1/2 rounded-lg bg-sky-950/90 px-2 py-0.5 font-mono text-xs font-semibold text-white shadow-md">
              {panelWidth}px
            </span>
          )}
        </div>

        {/* Top Header with Tab Switcher & Action Buttons */}
        <div className="flex h-14 shrink-0 items-center justify-between border-b border-sky-100/80 px-3 bg-white/70 backdrop-blur-md gap-2">
          {/* Segmented Tab Controls */}
          <div className="flex items-center rounded-xl bg-sky-100/50 p-0.5 overflow-x-auto">
            {/* Morning Review Tab */}
            <button
              type="button"
              onClick={() => onTabChange('review')}
              className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-semibold transition-all ${
                activeTab === 'review'
                  ? 'bg-white text-sky-950 shadow-sm'
                  : 'text-zinc-600 hover:text-sky-900'
              }`}
            >
              <span>🏛️</span>
              <span>Morning Delta</span>
            </button>

            {/* Kanban Tab */}
            <button
              type="button"
              onClick={() => onTabChange('kanban')}
              className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-semibold transition-all ${
                activeTab === 'kanban'
                  ? 'bg-white text-sky-950 shadow-sm'
                  : 'text-zinc-600 hover:text-sky-900'
              }`}
            >
              <span>📋</span>
              <span>บอร์ดงาน</span>
              {approvalCount > 0 && (
                <span className="rounded-full bg-purple-100 px-1.5 py-0.5 text-xs font-mono font-bold text-purple-700 animate-pulse">
                  {approvalCount}
                </span>
              )}
            </button>

            {/* Swarm Tab */}
            <button
              type="button"
              onClick={() => onTabChange('swarm')}
              className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-semibold transition-all ${
                activeTab === 'swarm'
                  ? 'bg-white text-sky-950 shadow-sm'
                  : 'text-zinc-600 hover:text-sky-900'
              }`}
            >
              <span>🐝</span>
              <span>Trinity Swarm</span>
              <span className="rounded-full bg-sky-200/80 px-1.5 py-0.5 text-xs font-mono font-bold text-sky-900">
                5
              </span>
            </button>
          </div>

          {/* Right Header Controls (Width Mode Toggle for Kanban & Close Button) */}
          <div className="flex items-center gap-1">
            {activeTab === 'kanban' && (
              <div className="hidden sm:flex items-center rounded-lg border border-sky-100 bg-white/80 p-0.5">
                <button
                  type="button"
                  onClick={() => {
                    setKanbanWidthMode('compact')
                    setPanelWidth(DEFAULT_PANEL_WIDTH)
                  }}
                  title="มุมมองกะทัดรัด (คอลัมน์เดียว)"
                  className={`rounded px-1.5 py-1 text-[10px] font-medium transition-all ${
                    effectiveKanbanLayout === 'compact'
                      ? 'bg-sky-100 text-sky-900 font-bold'
                      : 'text-zinc-400 hover:text-zinc-700'
                  }`}
                >
                  ▍
                </button>
                <button
                  type="button"
                  onClick={() => setKanbanWidthMode('full')}
                  title="มุมมองขยายเต็มจอ"
                  className={`rounded px-1.5 py-1 text-[10px] font-medium transition-all ${
                    effectiveKanbanLayout === 'full'
                      ? 'bg-sky-100 text-sky-900 font-bold'
                      : 'text-zinc-400 hover:text-zinc-700'
                  }`}
                >
                  ⛶
                </button>
              </div>
            )}

            {/* Close Panel Button */}
            <button
              type="button"
              onClick={onClose}
              title="ย่อแถบด้านข้าง"
              className="flex h-8 w-8 items-center justify-center rounded-xl border border-sky-100 text-zinc-400 hover:bg-sky-50 hover:text-sky-800 transition-colors"
            >
              <span className="text-sm">✕</span>
            </button>
          </div>
        </div>

        {/* Tab Content Body */}
        <div className="flex-1 min-h-0 overflow-hidden relative flex flex-col">
          {activeTab === 'review' ? (
            <div className="h-full overflow-y-auto p-3.5 custom-scrollbar">
              <MorningDeltaCard defaultExpanded={true} />
            </div>
          ) : activeTab === 'kanban' ? (
            <ManagerKanbanView
              layoutMode={effectiveKanbanLayout}
              refreshTrigger={refreshTrigger}
              selectedJobId={selectedJobId}
              onAskAiWithCard={onAskAiWithCard}
              onWatchPipeline={(jobId) => {
                onSelectJob(jobId)
              }}
              onApprovalCountChange={setApprovalCount}
            />
          ) : activeTab === 'execution' ? (
            <ManagerLiveExecutionView
              jobId={selectedJobId}
              allJobIds={detectedJobIds}
              onSelectJob={onSelectJob}
              onTriggerQuickJob={onTriggerQuickJob}
              onSwitchToSwarm={() => onTabChange('swarm')}
            />
          ) : (
            <ManagerSwarmView
              activeStatus={activeAgentStatus}
              onTriggerResearch={onTriggerQuickJob}
            />
          )}
        </div>
      </aside>
    </>
  )
}
