import { useState } from 'react'
import Modal from '../../ui/Modal'
import { api } from '../../../api/client'
import type { AllocationTargetDTO, ActualPortfolioStateDTO } from '../../../api/types'

interface Props {
  symbols: string[]
  targets: AllocationTargetDTO[]
  onClose: () => void
  onSuccess: (state: ActualPortfolioStateDTO) => void
}

export default function BatchAssignBucketModal({ symbols, targets, onClose, onSuccess }: Props) {
  const [selectedBucketId, setSelectedBucketId] = useState<string>(targets[0]?.bucket_id || '')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    try {
      const bucketPayload = selectedBucketId || null
      let state: ActualPortfolioStateDTO
      const firstSymbol = symbols[0]
      if (symbols.length === 1 && firstSymbol) {
        state = await api.assignHoldingBucket(firstSymbol, { bucket_id: bucketPayload })
      } else {
        state = await api.batchAssignHoldingBuckets({ symbols, bucket_id: bucketPayload })
      }
      onSuccess(state)
      onClose()
    } catch (err: any) {
      setError(err?.message || 'เปลี่ยน Bucket ไม่สำเร็จ')
    } finally {
      setLoading(false)
    }
  }

  return (
    <Modal titleId="batch-bucket-modal-title" onClose={onClose} panelClassName="max-w-md rounded-2xl border border-sky-100 bg-white p-6 shadow-2xl">
      <div className="flex items-center justify-between border-b border-zinc-100 pb-3">
        <h3 id="batch-bucket-modal-title" className="text-base font-bold text-zinc-900">
          📁 กำหนดกลุ่มกลยุทธ์ (Bucket Assign)
        </h3>
        <button type="button" onClick={onClose} className="text-zinc-400 hover:text-zinc-600">
          ✕
        </button>
      </div>

      <form onSubmit={handleSubmit} className="mt-4 space-y-4">
        {error && (
          <div className="rounded-xl bg-rose-50 p-3 text-xs font-semibold text-rose-800 border border-rose-200">
            ⚠️ {error}
          </div>
        )}

        <div className="rounded-xl bg-zinc-50 p-3 text-xs text-zinc-600">
          <span className="font-bold text-zinc-800">รายการที่เลือก ({symbols.length} ตัว): </span>
          <span className="font-mono text-flow-blue font-semibold">{symbols.slice(0, 10).join(', ')}{symbols.length > 10 ? ` ... และอีก ${symbols.length - 10} ตัว` : ''}</span>
        </div>

        <div>
          <label className="block text-xs font-bold text-zinc-700 mb-1.5">เลือก Bucket เป้าหมาย</label>
          <select
            value={selectedBucketId}
            onChange={(e) => setSelectedBucketId(e.target.value)}
            className="w-full rounded-xl border border-zinc-300 bg-white px-3 py-2 text-sm font-medium text-zinc-800 focus:border-flow-blue focus:outline-none"
          >
            <option value="">-- ไม่ระบุกลุ่ม (Unassigned) --</option>
            {targets.map((t) => (
              <option key={t.bucket_id} value={t.bucket_id}>
                {t.name} ({t.bucket_id}) - เป้าหมาย {t.target_percent}%
              </option>
            ))}
          </select>
        </div>

        <div className="flex justify-end gap-2 border-t border-zinc-100 pt-4">
          <button
            type="button"
            onClick={onClose}
            className="rounded-xl border border-zinc-300 bg-white px-4 py-2 text-xs font-semibold text-zinc-700 hover:bg-zinc-50"
          >
            ยกเลิก
          </button>
          <button
            type="submit"
            disabled={loading}
            className="rounded-xl bg-flow-blue px-5 py-2 text-xs font-bold text-white shadow-md hover:bg-sky-600 disabled:opacity-50"
          >
            {loading ? 'กำลังบันทึก...' : 'ยืนยันการเปลี่ยน Bucket'}
          </button>
        </div>
      </form>
    </Modal>
  )
}
