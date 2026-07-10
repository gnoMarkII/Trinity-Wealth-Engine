import { useState } from 'react'
import Modal from '../ui/Modal'
import SegmentedControl from '../ui/SegmentedControl'
import Button from '../ui/Button'
import TextInput from '../ui/TextInput'
import type { QuickTemplate } from '../../lib/quickTemplateStorage'

interface Props {
  template: QuickTemplate
  onClose: () => void
  onSave: (template: QuickTemplate) => void
}

const FLOW_OPTIONS: { key: string; label: string }[] = [
  { key: 'manager', label: 'Macro' },
  { key: 'news_youtube', label: 'News/YouTube' },
]

const SCOPE_OPTIONS: { key: string; label: string }[] = [
  { key: 'news', label: 'ข่าวเท่านั้น' },
  { key: 'youtube', label: 'YouTube เท่านั้น' },
  { key: 'both', label: 'ทั้งคู่' },
]

export default function EditTemplateModal({ template, onClose, onSave }: Props) {
  const [label, setLabel] = useState(template.label)
  const [instruction, setInstruction] = useState(template.instruction)
  const [flow, setFlow] = useState(template.flow)
  const [scope, setScope] = useState(template.scope)

  const trimmedLabel = label.trim()
  const trimmedInstruction = instruction.trim()
  const canSave = trimmedLabel.length > 0 && trimmedInstruction.length > 0

  function handleSave() {
    if (!canSave) return
    onSave({ label: trimmedLabel, instruction: trimmedInstruction, flow, scope })
  }

  return (
    <Modal titleId="edit-template-modal-title" onClose={onClose}>
      <h2 id="edit-template-modal-title" className="mb-4 text-sm font-semibold text-zinc-900">
        แก้ไขปุ่มลัด
      </h2>

      <div className="space-y-4">
        <div>
          <label htmlFor="edit-template-label" className="mb-1 block text-xs font-medium text-zinc-600">
            ชื่อปุ่มลัด
          </label>
          <TextInput
            id="edit-template-label"
            value={label}
            onChange={(e) => setLabel(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSave()}
            placeholder="เช่น 'วิเคราะห์เศรษฐกิจมหภาค'"
            className="w-full"
          />
        </div>

        <div>
          <label className="mb-1 block text-xs font-medium text-zinc-600">ประเภทงาน</label>
          <SegmentedControl options={FLOW_OPTIONS} value={flow} onChange={setFlow} />
        </div>

        {flow === 'news_youtube' && (
          <div>
            <label className="mb-1 block text-xs font-medium text-zinc-600">ขอบเขต</label>
            <SegmentedControl options={SCOPE_OPTIONS} value={scope} onChange={setScope} />
          </div>
        )}

        <div>
          <label htmlFor="edit-template-instruction" className="mb-1 block text-xs font-medium text-zinc-600">
            คำสั่งเต็มที่จะส่งให้ Manager
          </label>
          <textarea
            id="edit-template-instruction"
            value={instruction}
            onChange={(e) => setInstruction(e.target.value)}
            onKeyDown={(e) => {
              if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') handleSave()
            }}
            rows={5}
            placeholder="คำสั่งที่จะถูกส่งให้ agent ทันทีเมื่อกดปุ่มลัดนี้"
            className="w-full resize-none rounded-lg border border-zinc-200 bg-white px-3 py-2 text-sm text-zinc-900 outline-none transition-colors placeholder-zinc-400 focus:border-terra focus:ring-1 focus:ring-terra/30"
          />
        </div>
      </div>

      <div className="mt-5 flex justify-end gap-2">
        <Button type="button" variant="secondary" onClick={onClose}>
          ยกเลิก
        </Button>
        <Button type="button" onClick={handleSave} disabled={!canSave}>
          บันทึก
        </Button>
      </div>
    </Modal>
  )
}
