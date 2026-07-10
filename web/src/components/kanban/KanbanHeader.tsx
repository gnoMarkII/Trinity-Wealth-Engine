import type { FlowFilter, StatusFilter } from './types'

interface Props {
  statusFilter: StatusFilter
  onStatusFilterChange: (f: StatusFilter) => void
  flowFilter: FlowFilter
  onFlowFilterChange: (f: FlowFilter) => void
}

const STATUS_TABS: { key: StatusFilter; label: string }[] = [
  { key: 'active', label: 'Active' },
  { key: 'all', label: 'All' },
  { key: 'backlog', label: 'Backlog' },
  { key: 'done', label: 'Done' },
]

const FLOW_TABS: { key: FlowFilter; label: string }[] = [
  { key: 'all', label: 'All Flows' },
  { key: 'manager', label: 'Macro' },
  { key: 'news_youtube', label: 'News/YouTube' },
]

export default function KanbanHeader({
  statusFilter,
  onStatusFilterChange,
  flowFilter,
  onFlowFilterChange,
}: Props) {
  return (
    <div className="flex flex-wrap items-center gap-3">
      <div className="flex gap-1 rounded-lg border border-zinc-200 bg-white p-1">
        {STATUS_TABS.map((t) => (
          <button
            key={t.key}
            onClick={() => onStatusFilterChange(t.key)}
            aria-pressed={statusFilter === t.key}
            className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
              statusFilter === t.key ? 'bg-surface text-zinc-900' : 'text-zinc-500 hover:text-zinc-800'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>
      <div className="flex gap-1 rounded-lg border border-zinc-200 bg-white p-1">
        {FLOW_TABS.map((t) => (
          <button
            key={t.key}
            onClick={() => onFlowFilterChange(t.key)}
            aria-pressed={flowFilter === t.key}
            className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
              flowFilter === t.key ? 'bg-terra/10 text-terra' : 'text-zinc-500 hover:text-zinc-800'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>
    </div>
  )
}
