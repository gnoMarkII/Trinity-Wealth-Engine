import { useEffect, useRef, useState } from 'react'
import type { QuickTemplate } from '../../lib/quickTemplateStorage'
import Button from '../ui/Button'
import { flowLabel } from '../../lib/flows'

interface Props {
  quickTemplates: QuickTemplate[]
  onQuickCreate: (label: string, instruction: string, flow: string, scope: string) => void
  onOpenCustomModal: () => void
  onEditTemplate: (index: number) => void
}

interface IndexedTemplate extends QuickTemplate {
  index: number
}

function groupByFlow(templates: QuickTemplate[]): { flow: string; label: string; items: IndexedTemplate[] }[] {
  const groups: { flow: string; label: string; items: IndexedTemplate[] }[] = []
  templates.forEach((t, index) => {
    let group = groups.find((g) => g.flow === t.flow)
    if (!group) {
      group = { flow: t.flow, label: flowLabel(t.flow), items: [] }
      groups.push(group)
    }
    group.items.push({ ...t, index })
  })
  return groups
}

export default function AddCardDropdown({
  quickTemplates,
  onQuickCreate,
  onOpenCustomModal,
  onEditTemplate,
}: Props) {
  const [open, setOpen] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === 'Escape') setOpen(false)
    }
    function onClickOutside(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) setOpen(false)
    }
    window.addEventListener('keydown', onKeyDown)
    window.addEventListener('mousedown', onClickOutside)
    return () => {
      window.removeEventListener('keydown', onKeyDown)
      window.removeEventListener('mousedown', onClickOutside)
    }
  }, [open])

  const groups = groupByFlow(quickTemplates)

  return (
    <div ref={containerRef} className="relative">
      <Button
        size="sm"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-haspopup="menu"
      >
        + เพิ่มการ์ด
      </Button>

      {open && (
        <div
          role="menu"
          className="animate-dropdown-in absolute right-0 top-full z-40 mt-1 w-64 rounded-lg border border-edge bg-panel p-1.5 shadow-lg shadow-black/5"
        >
          {groups.map((group) => (
            <div key={group.flow} className="mb-1 last:mb-0">
              <p className="px-2 py-1 text-[10px] font-semibold uppercase tracking-wide text-zinc-400">
                {group.label}
              </p>
              {group.items.map((t) => (
                <div key={t.label} className="group/item flex items-center rounded-md hover:bg-sky-500/10">
                  <button
                    role="menuitem"
                    onClick={() => {
                      onQuickCreate(t.label, t.instruction, t.flow, t.scope)
                      setOpen(false)
                    }}
                    className="block flex-1 truncate px-2 py-1.5 text-left text-xs text-zinc-700 group-hover/item:text-sky-500"
                  >
                    {t.label}
                  </button>
                  <button
                    type="button"
                    title="แก้ไขปุ่มลัดนี้"
                    aria-label="แก้ไขปุ่มลัดนี้"
                    onClick={() => {
                      onEditTemplate(t.index)
                      setOpen(false)
                    }}
                    className="shrink-0 rounded px-1.5 py-1.5 text-zinc-400 opacity-0 transition-opacity hover:text-sky-800 focus:opacity-100 focus-visible:opacity-100 group-hover/item:opacity-100"
                  >
                    ✎
                  </button>
                </div>
              ))}
            </div>
          ))}

          <div className="mt-1 border-t border-edge pt-1">
            <button
              role="menuitem"
              onClick={() => {
                onOpenCustomModal()
                setOpen(false)
              }}
              className="block w-full rounded-md px-2 py-1.5 text-left text-xs font-medium text-sky-700 transition-colors hover:bg-sky-500/10"
            >
              + กำหนดเอง...
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
