import type { CSSProperties, KeyboardEvent } from 'react'
import type { KanbanCardDTO } from '../../api/types'
import { nodeDisplayName } from '../../lib/nodeDisplayNames'
import { FLOW_TAG } from '../../lib/flowTags'
import type { WorkspacePreview } from './types'

interface Props {
  card: KanbanCardDTO
  faded: boolean
  removing: boolean
  selected?: boolean
  workspacePreview?: WorkspacePreview
  onDelete: () => void
  onClick?: () => void
  editable?: boolean
  onEdit?: () => void
  onDispatch?: () => void
  style?: CSSProperties
}

function formatElapsed(seconds: number): string {
  if (seconds < 60) return `${Math.max(0, Math.round(seconds))}s`
  return `${Math.round(seconds / 60)}m`
}

function handleCardKeyDown(e: KeyboardEvent<HTMLDivElement>, onClick?: () => void) {
  if (!onClick) return
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault()
    onClick()
  }
}

export default function KanbanCard({
  card,
  faded,
  removing,
  selected,
  workspacePreview,
  onDelete,
  onClick,
  editable,
  onEdit,
  onDispatch,
  style,
}: Props) {
  return (
    <div
      onClick={onClick}
      onKeyDown={(e) => handleCardKeyDown(e, onClick)}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
      style={style}
      className={`group relative rounded-lg border bg-white p-3 pr-8 text-xs text-zinc-800 shadow-[0_1px_2px_rgba(0,0,0,0.03)] transition-all duration-150 hover:-translate-y-0.5 hover:shadow-md focus-visible:outline focus-visible:outline-2 focus-visible:outline-terra ${
        selected || workspacePreview ? 'border-2 border-terra-light' : 'border-zinc-200 hover:border-zinc-300'
      } ${onClick ? 'cursor-pointer' : 'cursor-default'} ${
        removing ? 'animate-card-out' : 'animate-card-in'
      } ${faded ? 'opacity-40' : ''}`}
    >
      {editable && onEdit && (
        <button
          onClick={(e) => {
            e.stopPropagation()
            onEdit()
          }}
          onKeyDown={(e) => e.stopPropagation()}
          title="แก้ไขการ์ด"
          aria-label="แก้ไขการ์ด"
          className="absolute right-6 top-1 rounded p-0.5 text-zinc-400 opacity-0 transition-opacity hover:bg-zinc-100 hover:text-terra focus:opacity-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-terra group-hover:opacity-100"
        >
          ✎
        </button>
      )}
      <button
        onClick={(e) => {
          e.stopPropagation()
          onDelete()
        }}
        onKeyDown={(e) => e.stopPropagation()}
        title="ลบการ์ด"
        aria-label="ลบการ์ด"
        className="absolute right-1 top-1 rounded p-0.5 text-zinc-400 opacity-0 transition-opacity hover:bg-zinc-100 hover:text-red-600 focus:opacity-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-red-500 group-hover:opacity-100"
      >
        ✕
      </button>
      <div className="mb-1 flex items-center justify-between gap-2 pr-4">
        <div className="flex items-center gap-1.5">
          {editable && onDispatch && (
            <button
              onClick={(e) => {
                e.stopPropagation()
                onDispatch()
              }}
              onKeyDown={(e) => e.stopPropagation()}
              title="ส่งงานให้ Manager"
              aria-label="ส่งงานให้ Manager"
              className="rounded p-0.5 text-zinc-400 transition-colors hover:bg-zinc-100 hover:text-terra focus-visible:outline focus-visible:outline-2 focus-visible:outline-terra"
            >
              ▶
            </button>
          )}
          {card.display_seq != null && (
            <span className="font-mono text-[11px] font-medium text-zinc-400">#AG-{card.display_seq}</span>
          )}
        </div>
        <span className="rounded border border-purple-200/60 bg-purple-50 px-1.5 py-0.5 text-[10px] text-purple-700">
          {FLOW_TAG[card.flow] ?? `#${card.flow}`}
        </span>
      </div>
      <p className="break-words text-sm font-semibold leading-snug text-zinc-800">{card.title}</p>
      {workspacePreview && (
        <div className="mt-2 flex items-center gap-1.5 rounded-md border border-zinc-200/80 bg-surface px-1.5 py-1 text-[10px] font-medium text-zinc-700">
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-500" />
          <span>
            {nodeDisplayName(workspacePreview.node)} • {workspacePreview.logCount} log lines •{' '}
            {formatElapsed(workspacePreview.elapsedSeconds)}
          </span>
        </div>
      )}
    </div>
  )
}
