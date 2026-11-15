import { useState } from 'react'
import Modal from '../../ui/Modal'
import { api } from '../../../api/client'
import { DEFAULT_ALLOCATION_TARGETS, type AllocationTargetDTO, type ActualPortfolioStateDTO } from '../../../api/types'
import {
  getUniqueBucketColor,
  getRandomizedBucketColor,
  randomizeAllBucketColors,
} from '../../../lib/bucketColors'

interface Props {
  portfolioId: string
  initialTargets: AllocationTargetDTO[]
  onClose: () => void
  onSuccess: (state: ActualPortfolioStateDTO) => void
}

export default function BucketTargetModal({ portfolioId, initialTargets, onClose, onSuccess }: Props) {
  const [targets, setTargets] = useState<AllocationTargetDTO[]>(() =>
    initialTargets.length > 0
      ? initialTargets.map((t) => ({ ...t }))
      : DEFAULT_ALLOCATION_TARGETS.map((t) => ({ ...t }))
  )
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const totalPercent = targets.reduce((sum, t) => sum + (Number(t.target_percent) || 0), 0)

  const handleAdd = () => {
    const newId = `bucket_${Date.now().toString().slice(-4)}`
    const existingColors = targets.map((t) => t.color).filter(Boolean) as string[]
    const autoColor = getUniqueBucketColor(existingColors)
    setTargets([...targets, { bucket_id: newId, name: 'New Bucket', target_percent: 0, color: autoColor }])
  }

  const handleRandomizeRow = (index: number) => {
    const next = [...targets]
    const current = next[index]
    if (!current) return
    const existingColors = targets.map((t) => t.color).filter(Boolean) as string[]
    const randomColor = getRandomizedBucketColor(existingColors, current.color)
    next[index] = { ...current, color: randomColor }
    setTargets(next)
  }

  const handleRandomizeAll = () => {
    setTargets(randomizeAllBucketColors(targets))
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
      const state = await api.upsertAllocationTargets({ targets: formatted }, portfolioId)
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
      <div className="flex items-center justify-between border-b border-slate-100 pb-4">
        <h3 id="bucket-target-modal-title" className="text-lg font-bold text-slate-900">
          ⚙️ ตั้งค่าสัดส่วนกลยุทธ์ (Allocation Targets)
        </h3>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={handleRandomizeAll}
            className="flex items-center gap-1 rounded-xl border border-sky-200 bg-sky-50 px-2.5 py-1 text-xs font-semibold text-blue-700 transition-colors hover:bg-sky-100 hover:border-sky-300"
            title="สุ่มสีใหม่ทุก Bucket ให้ไม่ซ้ำกัน"
          >
            🎨 สุ่มสีทั้งหมด
          </button>
          <button type="button" onClick={onClose} className="text-slate-500 hover:text-slate-600">
            ✕
          </button>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="mt-4 space-y-4">
        {error && (
          <div className="rounded-xl bg-rose-50 p-3 text-xs font-semibold text-rose-800 border border-rose-200">
            ⚠️ {error}
          </div>
        )}

        <div className="space-y-3 max-h-80 overflow-y-auto pr-1">
          {targets.map((t, idx) => (
            <div key={t.bucket_id || idx} className="flex items-center gap-2 rounded-xl border border-slate-200 bg-slate-50/50 p-2.5 text-xs">
              <div className="flex items-center gap-1">
                <input
                  type="color"
                  value={t.color || '#3B82F6'}
                  onChange={(e) => handleChange(idx, 'color', e.target.value)}
                  className="h-8 w-8 cursor-pointer rounded border-0 bg-transparent p-0"
                  title="เลือกสี"
                />
                <button
                  type="button"
                  onClick={() => handleRandomizeRow(idx)}
                  className="rounded-lg p-1 text-slate-500 transition-colors hover:bg-sky-50 hover:text-blue-600"
                  title="สุ่มสีใหม่สำหรับ Bucket นี้"
                >
                  🎲
                </button>
              </div>
              <div className="flex-1 space-y-1">
                <input
                  type="text"
                  value={t.name}
                  onChange={(e) => handleChange(idx, 'name', e.target.value)}
                  placeholder="ชื่อ Bucket"
                  className="w-full rounded-lg border border-slate-300 px-2.5 py-1.5 font-medium text-slate-800 focus:border-flow-blue focus:outline-none"
                  required
                />
                <div className="flex items-center gap-2 text-[11px] text-slate-500">
                  <span>ID:</span>
                  <input
                    type="text"
                    value={t.bucket_id}
                    onChange={(e) => handleChange(idx, 'bucket_id', e.target.value)}
                    className="w-28 rounded border border-slate-200 px-1.5 py-0.5 font-mono text-slate-600 focus:border-flow-blue focus:outline-none"
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
                  className="w-16 rounded-lg border border-slate-300 px-2 py-1.5 text-right font-bold text-slate-900 focus:border-flow-blue focus:outline-none"
                  required
                />
                <span className="font-semibold text-slate-600">%</span>
              </div>
              <button
                type="button"
                onClick={() => handleRemove(idx)}
                className="rounded-lg p-1.5 text-slate-500 hover:bg-rose-50 hover:text-rose-600"
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

        <div className="flex items-center justify-between rounded-xl bg-slate-100 p-3 text-xs font-bold">
          <span className="text-slate-600">ผลรวมเป้าหมายทั้งหมด:</span>
          <span className={`text-sm ${Math.abs(totalPercent - 100) > 0.1 ? 'text-amber-600 font-extrabold' : 'text-emerald-600'}`}>
            {totalPercent.toFixed(1)}% {Math.abs(totalPercent - 100) > 0.1 && '(ไม่ครบ 100%)'}
          </span>
        </div>

        <div className="flex justify-end gap-2 border-t border-slate-100 pt-4">
          <button
            type="button"
            onClick={onClose}
            className="rounded-xl border border-slate-300 bg-white px-4 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50"
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
