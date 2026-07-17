import { useState } from 'react'
import FormModal, { FormField, FormTextarea } from './FormModal'
import { JournalIcon } from '../icons/PortfolioIcons'
import { api } from '../../../api/client'
import type { JournalEntryDTO } from '../../../api/types'

interface Props {
  initialSymbol?: string
  onClose: () => void
  onSuccess: (entries: JournalEntryDTO[]) => void
  zIndexClassName?: string
}

export default function JournalModal({ initialSymbol, onClose, onSuccess, zIndexClassName = 'z-[60]' }: Props) {
  const [entry, setEntry] = useState(initialSymbol ? `**[NOTE] ${initialSymbol}**\n` : '')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!entry.trim()) {
      setError('กรุณาเขียนบันทึก')
      return
    }
    setLoading(true)
    setError(null)
    try {
      const rows = await api.appendJournal({ entry: entry.trim() })
      onSuccess(rows)
      onClose()
    } catch (err: any) {
      setError(err?.message || 'บันทึก Journal ไม่สำเร็จ')
    } finally {
      setLoading(false)
    }
  }

  return (
    <FormModal
      titleId="journal-modal-title"
      title="เขียนบันทึกการลงทุนใหม่ (Add Trading Journal Entry)"
      icon={<JournalIcon className="w-5 h-5 text-flow-blue" />}
      onClose={onClose}
      onSubmit={handleSubmit}
      error={error}
      loading={loading}
      submitText="📑 บันทึกเข้า Trading Journal"
      submitClassName="bg-flow-blue hover:bg-sky-600"
      zIndexClassName={zIndexClassName}
      panelClassName="max-w-lg rounded-2xl border border-sky-100 bg-white p-6 shadow-2xl"
    >
      <FormField
        label="บันทึกเหตุผล สภาพตลาด บทเรียน หรือข้อผิดพลาด (Qualitative Notes)"
        hint="💡 เคล็ดลับ: หากพิมพ์ **[BUY] AAPL** ระบบจะสร้าง Wikilink เชื่อมโยงไปยังหน้าหุ้น AAPL ใน Obsidian ให้อัตโนมัติ"
        required
      >
        <FormTextarea
          rows={5}
          value={entry}
          onChange={(e) => setEntry(e.target.value)}
          placeholder={`**[BUY] AAPL**\nซื้อหุ้น Apple เพิ่มที่แนวรับ 180 USD เนื่องจากงบ Q3 เติบโตดีกว่าคาด...`}
          className="font-mono"
          required
        />
      </FormField>
    </FormModal>
  )
}
