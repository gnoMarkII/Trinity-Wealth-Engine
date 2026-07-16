import type { FlowFilter, StatusFilter } from './types'
import { FLOW_LABEL } from '../../lib/flows'

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
  { key: 'manager', label: FLOW_LABEL.manager },
  { key: 'news_youtube', label: FLOW_LABEL.news_youtube },
  { key: 'news_funnel', label: FLOW_LABEL.news_funnel },
]

export default function KanbanHeader({
  statusFilter,
  onStatusFilterChange,
  flowFilter,
  onFlowFilterChange,
}: Props) {
  return (
    <div className="flex flex-wrap items-center gap-3">
      <div className="flex gap-1 rounded-xl border border-sky-200 bg-panel p-1 shadow-sm shadow-sky-100/60">
        {STATUS_TABS.map((t) => (
          <button
            key={t.key}
            onClick={() => onStatusFilterChange(t.key)}
            aria-pressed={statusFilter === t.key}
            className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
              statusFilter === t.key ? 'bg-sky-50 text-sky-800 shadow-sm' : 'text-zinc-500 hover:bg-sky-50 hover:text-zinc-800'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>
      <div className="flex gap-1 rounded-xl border border-sky-200 bg-panel p-1 shadow-sm shadow-sky-100/60">
        {FLOW_TABS.map((t) => (
          <button
            key={t.key}
            onClick={() => onFlowFilterChange(t.key)}
            aria-pressed={flowFilter === t.key}
            className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
              flowFilter === t.key ? 'bg-flow-cyan/10 text-sky-700 shadow-sm' : 'text-zinc-500 hover:bg-sky-50 hover:text-zinc-800'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>
    </div>
  )
}
