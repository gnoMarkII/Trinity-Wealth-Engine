import { useState } from 'react'
import Modal from '../../ui/Modal'
import { api } from '../../../api/client'
import type { AllocationTargetDTO, ActualPortfolioStateDTO } from '../../../api/types'

interface Props {
  initialTargets: AllocationTargetDTO[]
  onClose: () => void
  onSuccess: (state: ActualPortfolioStateDTO) => void
}

export default function BucketTargetModal({ initialTargets, onClose, onSuccess }: Props) {
  const [targets, setTargets] = useState<AllocationTargetDTO[]>(() =>
    initialTargets.length > 0
      ? initialTargets.map((t) => ({ ...t }))
      : [
          { bucket_id: 'core_equities', name: 'Core Equities', target_percent: 60, color: '#10B981' },
          { bucket_id: 'defensive', name: 'Defensive Assets', target_percent: 25, color: '#F59E0B' },
          { bucket_id: 'cash', name: 'Cash Reserves', target_percent: 15, color: '#6B7280' },
        ]
  )
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const totalPercent = targets.reduce((sum, t) => sum + (Number(t.target_percent) || 0), 0)

  const handleAdd = () => {
    const newId = `bucket_${Date.now().toString().slice(-4)}`
    setTargets([...targets, { bucket_id: newId, name: 'New Bucket', target_percent: 0, color: '#3B82F6' }])
  }

  const handleRemove = (index: number) => {
    setTargets(targets.filter((_, i) => i !== index))
  }

  const handleChange = (index: number, field: keyof AllocationTargetDTO, val: any) => {
    const next = [...targets]
    const current = next[index]
    if (current) {
      next[index] = { ...current, [field]: val }
    }
    setTargets(next)
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (Math.abs(totalPercent - 100.0) > 0.01) {
      setError(`ผลรวมสัดส่วนเป้าหมายต้องเท่ากับ 100% (ปัจจุบันได้ ${totalPercent.toFixed(1)}%)`)
      return
    }
    setLoading(true)
    setError(null)
    try {
      const formatted: AllocationTargetDTO[] = targets.map((t) => ({
        bucket_id: t.bucket_id || '',
        name: t.name || '',
        target_percent: Number(t.target_percent) || 0,
        color: t.color ?? null,
      }))
      const state = await api.upsertAllocationTargets({ targets: formatted })
      onSuccess(state)
      onClose()
    } catch (err: any) {
      setError(err?.message || 'บันทึก Target ไม่สำเร็จ')
    } finally {
      setLoading(false)
    }
  }

  return (
    <Modal titleId="bucket-target-modal-title" onClose={onClose} panelClassName="max-w-xl rounded-2xl border border-sky-100 bg-white p-6 shadow-2xl">
      <div className="flex items-center justify-between border-b border-zinc-100 pb-4">
        <h3 id="bucket-target-modal-title" className="text-lg font-bold text-zinc-900">
          ⚙️ ตั้งค่าสัดส่วนกลยุทธ์ (Allocation Targets)
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

        <div className="space-y-3 max-h-80 overflow-y-auto pr-1">
          {targets.map((t, idx) => (
            <div key={t.bucket_id || idx} className="flex items-center gap-2 rounded-xl border border-zinc-200 bg-zinc-50/50 p-2.5 text-xs">
              <input
                type="color"
                value={t.color || '#3B82F6'}
                onChange={(e) => handleChange(idx, 'color', e.target.value)}
                className="h-8 w-8 cursor-pointer rounded border-0 bg-transparent p-0"
                title="เลือกสี"
              />
              <div className="flex-1 space-y-1">
                <input
                  type="text"
                  value={t.name}
                  onChange={(e) => handleChange(idx, 'name', e.target.value)}
                  placeholder="ชื่อ Bucket"
                  className="w-full rounded-lg border border-zinc-300 px-2.5 py-1.5 font-medium text-zinc-800 focus:border-flow-blue focus:outline-none"
                  required
                />
                <div className="flex items-center gap-2 text-[11px] text-zinc-500">
                  <span>ID:</span>
                  <input
                    type="text"
                    value={t.bucket_id}
                    onChange={(e) => handleChange(idx, 'bucket_id', e.target.value)}
                    className="w-28 rounded border border-zinc-200 px-1.5 py-0.5 font-mono text-zinc-600 focus:border-flow-blue focus:outline-none"
                    required
                  />
                </div>
              </div>
              <div className="flex items-center gap-1">
                <input
                  type="number"
                  step="0.1"
                  min="0"
                  max="100"
                  value={t.target_percent}
                  onChange={(e) => handleChange(idx, 'target_percent', parseFloat(e.target.value) || 0)}
                  className="w-16 rounded-lg border border-zinc-300 px-2 py-1.5 text-right font-bold text-zinc-900 focus:border-flow-blue focus:outline-none"
                  required
                />
                <span className="font-semibold text-zinc-600">%</span>
              </div>
              <button
                type="button"
                onClick={() => handleRemove(idx)}
                className="rounded-lg p-1.5 text-zinc-400 hover:bg-rose-50 hover:text-rose-600"
                title="ลบ Bucket"
              >
                ✕
              </button>
            </div>
          ))}
        </div>

        <button
          type="button"
          onClick={handleAdd}
          className="w-full rounded-xl border border-dashed border-sky-300 bg-sky-50/50 py-2 text-xs font-bold text-flow-blue hover:bg-sky-50"
        >
          + เพิ่ม Bucket ใหม่
        </button>

        <div className="flex items-center justify-between rounded-xl bg-zinc-100 p-3 text-xs font-bold">
          <span className="text-zinc-600">ผลรวมเป้าหมายทั้งหมด:</span>
          <span className={`text-sm ${Math.abs(totalPercent - 100) > 0.1 ? 'text-amber-600 font-extrabold' : 'text-emerald-600'}`}>
            {totalPercent.toFixed(1)}% {Math.abs(totalPercent - 100) > 0.1 && '(ไม่ครบ 100%)'}
          </span>
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
            {loading ? 'กำลังบันทึก...' : 'บันทึกสัดส่วนเป้าหมาย'}
          </button>
        </div>
      </form>
    </Modal>
  )
}
