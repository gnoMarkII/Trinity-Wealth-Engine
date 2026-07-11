import type { KanbanCardDTO } from '../../api/types'
import KanbanCard from './KanbanCard'
import type { ColumnDef, WorkspacePreview } from './types'

const STATUS_DOT: Record<string, string> = {
  backlog: 'bg-blue-500',
  approval: 'bg-purple-500',
  executing: 'bg-emerald-500 animate-pulse',
  done: 'bg-emerald-600',
}

const STAGGER_STEP_MS = 30
const STAGGER_CAP_MS = 240

interface Props {
  column: ColumnDef
  cards: KanbanCardDTO[]
  isBacklogColumn: boolean
  isCardFaded: (card: KanbanCardDTO) => boolean
  removingIds: Set<string>
  selectedCardId?: string | null
  workspacePreviewFor: (card: KanbanCardDTO) => WorkspacePreview | undefined
  staggerCards?: boolean
  onDeleteCard: (cardId: string) => void
  onCardClick?: (card: KanbanCardDTO) => void
  onEditCard?: (card: KanbanCardDTO) => void
  onDispatchCard?: (card: KanbanCardDTO) => void
}

export default function KanbanColumn({
  column,
  cards,
  isBacklogColumn,
  isCardFaded,
  removingIds,
  selectedCardId,
  workspacePreviewFor,
  staggerCards,
  onDeleteCard,
  onCardClick,
  onEditCard,
  onDispatchCard,
}: Props) {
  return (
    <div className="flex h-full flex-col rounded-2xl border border-sky-100 bg-white/70 p-2.5 shadow-[0_10px_35px_rgba(14,165,233,0.06)] backdrop-blur-md transition-colors duration-150">
      <h3 className="mb-3 flex shrink-0 items-center gap-1.5 px-1 text-xs font-semibold text-sky-950">
        <span className={`h-1.5 w-1.5 rounded-full ${STATUS_DOT[column.key] ?? 'bg-zinc-400'}`} />
        {column.label} <span className="text-zinc-400">({cards.length})</span>
      </h3>
      <div className="min-h-0 flex-1 space-y-2.5 overflow-y-auto pt-2.5 px-0.5 pb-2">
        {cards.map((c, i) => (
          <KanbanCard
            key={c.card_id}
            card={c}
            faded={isCardFaded(c)}
            removing={removingIds.has(c.card_id)}
            selected={c.card_id === selectedCardId}
            workspacePreview={workspacePreviewFor(c)}
            onDelete={() => onDeleteCard(c.card_id)}
            onClick={onCardClick ? () => onCardClick(c) : undefined}
            editable={isBacklogColumn}
            onEdit={onEditCard ? () => onEditCard(c) : undefined}
            onDispatch={onDispatchCard ? () => onDispatchCard(c) : undefined}
            style={staggerCards ? { animationDelay: `${Math.min(i * STAGGER_STEP_MS, STAGGER_CAP_MS)}ms` } : undefined}
          />
        ))}
      </div>
    </div>
  )
}
