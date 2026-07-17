import { type ReactNode } from 'react'
import Modal from './Modal'

interface Props {
  isOpen: boolean
  title: string
  description?: ReactNode
  confirmText?: string
  cancelText?: string
  isDanger?: boolean
  loading?: boolean
  onConfirm: () => void
  onClose: () => void
}

export default function ConfirmDialog({
  isOpen,
  title,
  description,
  confirmText = 'ยืนยัน',
  cancelText = 'ยกเลิก',
  isDanger = true,
  loading = false,
  onConfirm,
  onClose,
}: Props) {
  if (!isOpen) return null

  return (
    <Modal titleId="confirm-dialog-title" onClose={onClose} zIndexClassName="z-[60]">
      <div className="space-y-4 text-left">
        <div className="flex items-start gap-3.5">
          <div
            className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-full ${
              isDanger ? 'bg-red-100 text-red-600' : 'bg-sky-100 text-sky-600'
            }`}
          >
            {isDanger ? <span className="text-lg">🗑️</span> : <span className="text-lg">❓</span>}
          </div>
          <div className="flex-1 space-y-1 pt-0.5">
            <h3 id="confirm-dialog-title" className="text-sm font-semibold text-zinc-900">
              {title}
            </h3>
            {description && (
              <div className="text-xs leading-relaxed text-zinc-600">
                {description}
              </div>
            )}
          </div>
        </div>

        <div className="flex items-center justify-end gap-2 border-t border-edge pt-3">
          <button
            type="button"
            onClick={onClose}
            disabled={loading}
            className="rounded-xl border border-edge bg-surface px-3.5 py-1.5 text-xs font-semibold text-zinc-700 shadow-sm transition-colors hover:bg-surface-strong disabled:opacity-50"
          >
            {cancelText}
          </button>
          <button
            type="button"
            onClick={onConfirm}
            disabled={loading}
            className={`rounded-xl px-3.5 py-1.5 text-xs font-semibold text-white shadow-sm transition-colors disabled:opacity-50 ${
              isDanger ? 'bg-red-600 hover:bg-red-700' : 'bg-sky-600 hover:bg-sky-700'
            }`}
          >
            {loading ? 'กำลังดำเนินการ...' : confirmText}
          </button>
        </div>
      </div>
    </Modal>
  )
}
