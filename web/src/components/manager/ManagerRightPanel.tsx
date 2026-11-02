import { useState } from 'react'
import type { ActiveAgentStatusDTO, KanbanCardDTO } from '../../api/types'
import ManagerLiveExecutionView from './ManagerLiveExecutionView'
import ManagerSwarmView from './ManagerSwarmView'
import ManagerKanbanView from './ManagerKanbanView'

export type ManagerRightPanelTab = 'kanban' | 'execution' | 'swarm'
export type KanbanWidthMode = 'compact' | 'wide' | 'full'

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
  const [approvalCount, setApprovalCount] = useState<number>(0)

  if (!isOpen) return null

  // Dynamic width classes based on active tab and width mode
  const widthClasses =
    activeTab === 'kanban'
      ? kanbanWidthMode === 'full'
        ? 'w-full max-w-none xl:w-full xl:max-w-none xl:flex-1'
        : kanbanWidthMode === 'wide'
          ? 'w-full max-w-[880px] xl:w-[880px] xl:max-w-[880px] xl:shrink-0'
          : 'w-full max-w-[420px] xl:w-[420px] xl:max-w-[420px] xl:shrink-0'
      : 'w-full max-w-[420px] xl:w-[410px] xl:max-w-[410px] xl:shrink-0 xl:min-w-[360px]'

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
        className={`fixed inset-y-0 right-0 z-50 flex flex-col border-l border-sky-100 bg-panel/95 shadow-[-15px_0_40px_rgba(14,165,233,0.06)] backdrop-blur-xl transition-all duration-300 xl:static xl:z-auto xl:h-full ${widthClasses}`}
        aria-label="Agent Work, Kanban and Swarm Panel"
      >
        {/* Top Header with Tab Switcher & Action Buttons */}
        <div className="flex h-14 shrink-0 items-center justify-between border-b border-sky-100 px-3 bg-white/70 backdrop-blur-md gap-2">
          {/* Segmented Tab Controls */}
          <div className="flex items-center rounded-xl bg-sky-100/60 p-0.5 overflow-x-auto">
            {/* Kanban Tab */}
            <button
              type="button"
              onClick={() => onTabChange('kanban')}
              className={`flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-bold transition-all ${
                activeTab === 'kanban'
                  ? 'bg-white text-sky-950 shadow-sm'
                  : 'text-zinc-600 hover:text-sky-900'
              }`}
            >
              <span>📋</span>
              <span>บอร์ดงาน</span>
              {approvalCount > 0 && (
                <span className="rounded-full bg-purple-100 px-1.5 py-0.2 text-[9px] font-mono text-purple-700 animate-pulse">
                  {approvalCount}
                </span>
              )}
            </button>

            {/* Swarm Tab */}
            <button
              type="button"
              onClick={() => onTabChange('swarm')}
              className={`flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-bold transition-all ${
                activeTab === 'swarm'
                  ? 'bg-white text-sky-950 shadow-sm'
                  : 'text-zinc-600 hover:text-sky-900'
              }`}
            >
              <span>🐝</span>
              <span>Trinity Swarm</span>
              <span className="rounded-full bg-sky-200/80 px-1.5 py-0.2 text-[9px] font-mono text-sky-900">
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
                  onClick={() => setKanbanWidthMode('compact')}
                  title="มุมมองกะทัดรัด (คอลัมน์เดียว)"
                  className={`rounded px-1.5 py-1 text-[10px] font-medium transition-all ${
                    kanbanWidthMode === 'compact'
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
                    kanbanWidthMode === 'full'
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
        <div className="flex-1 min-h-0 overflow-hidden relative">
          {activeTab === 'kanban' ? (
            <ManagerKanbanView
              layoutMode={kanbanWidthMode}
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
