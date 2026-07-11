import { useState } from 'react'
import Modal from '../ui/Modal'
import SegmentedControl from '../ui/SegmentedControl'
import Button from '../ui/Button'
import TextInput from '../ui/TextInput'
import { FLOW_OPTIONS, SCOPE_OPTIONS } from '../../lib/flows'

interface Props {
  mode: 'create' | 'edit'
  initialTitle?: string
  initialPrompt?: string
  initialFlow?: string
  initialScope?: string
  errorMessage?: string | null
  onClose: () => void
  onSubmit: (values: { title: string; prompt: string; flow: string; scope: string }) => Promise<void>
}

export default function KanbanCardModal({
  mode,
  initialTitle = '',
  initialPrompt = '',
  initialFlow = 'manager',
  initialScope = 'both',
  errorMessage,
  onClose,
  onSubmit,
}: Props) {
  const [title, setTitle] = useState(initialTitle)
  const [prompt, setPrompt] = useState(initialPrompt)
  const [flow, setFlow] = useState(initialFlow)
  const [scope, setScope] = useState(initialScope)
  const [submitting, setSubmitting] = useState(false)

  const trimmedTitle = title.trim()

  async function handleSubmit() {
    if (!trimmedTitle || submitting) return
    setSubmitting(true)
    try {
      await onSubmit({ title: trimmedTitle, prompt: prompt.trim(), flow, scope })
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <Modal titleId="kanban-card-modal-title" onClose={onClose}>
      <h2 id="kanban-card-modal-title" className="mb-4 text-sm font-semibold text-zinc-900">
        {mode === 'create' ? 'เพิ่มการ์ดใหม่' : 'แก้ไขการ์ด'}
      </h2>

      <div className="space-y-4">
        <div>
          <label htmlFor="kanban-card-title" className="mb-1 block text-xs font-medium text-zinc-600">
            ชื่อการ์ด
          </label>
          <TextInput
            id="kanban-card-title"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSubmit()}
            placeholder="ชื่อการ์ดสั้นๆ เช่น 'วิเคราะห์พอร์ตวันนี้'"
            className="w-full"
          />
        </div>

        <div>
          {/* span + role="group" แทน <label> — SegmentedControl เป็นกลุ่มปุ่ม ไม่ใช่ form control */}
          <span id="kanban-card-flow-label" className="mb-1 block text-xs font-medium text-zinc-600">ประเภทงาน</span>
          <SegmentedControl options={FLOW_OPTIONS} value={flow} onChange={setFlow} ariaLabelledby="kanban-card-flow-label" />
        </div>

        {flow === 'news_youtube' && (
          <div>
            <span id="kanban-card-scope-label" className="mb-1 block text-xs font-medium text-zinc-600">ขอบเขต</span>
            <SegmentedControl options={SCOPE_OPTIONS} value={scope} onChange={setScope} ariaLabelledby="kanban-card-scope-label" />
          </div>
        )}

        <div>
          <label htmlFor="kanban-card-prompt" className="mb-1 block text-xs font-medium text-zinc-600">
            Prompt สำหรับ Manager (ไม่บังคับ)
          </label>
          <textarea
            id="kanban-card-prompt"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => {
              if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') handleSubmit()
            }}
            rows={7}
            placeholder="อธิบายรายละเอียดงานให้ agent เข้าใจชัดเจน เช่น ขอบเขตการวิเคราะห์ กรอบเวลา สินทรัพย์ที่สนใจ ข้อมูลอ้างอิงพิเศษ ฯลฯ"
            className="w-full resize-none rounded-lg border border-edge bg-panel px-3 py-2 text-sm text-zinc-900 outline-none transition-colors placeholder-zinc-400 focus:border-sky-500 focus:ring-1 focus:ring-sky-500/30"
          />
          <p className="mt-1 text-xs text-zinc-500">ถ้าเว้นว่างไว้ ระบบจะใช้ชื่อการ์ดเป็นคำสั่งแทน</p>
        </div>
      </div>

      {errorMessage && (
        <p className="mt-4 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">{errorMessage}</p>
      )}

      <div className="mt-5 flex justify-end gap-2">
        <Button type="button" variant="secondary" onClick={onClose}>
          ยกเลิก
        </Button>
        <Button type="button" onClick={handleSubmit} disabled={!trimmedTitle || submitting}>
          {mode === 'create' ? 'เพิ่มการ์ด' : 'บันทึก'}
        </Button>
      </div>
    </Modal>
  )
}
