import { useState, useEffect } from 'react'
import type {
  BucketPlanDraftDTO,
  UpdateBucketPlanRequestDTO,
  PurposeBucketDTO,
} from '../../../api/types'
import { VIBRANT_BUCKET_PALETTE } from '../../../lib/bucketColors'

interface EssenceBucketBuilderProps {
  draft: BucketPlanDraftDTO
  portfolioId: string
  onUpdateDraft: (updates: UpdateBucketPlanRequestDTO) => Promise<void>
  onProceedToReview: () => void
  loading: boolean
  statusText?: string | null
}

export default function EssenceBucketBuilder({
  draft,
  portfolioId,
  onUpdateDraft,
  onProceedToReview,
  loading,
  statusText,
}: EssenceBucketBuilderProps) {
  const [buckets, setBuckets] = useState<PurposeBucketDTO[]>(draft.purpose_buckets || [])

  useEffect(() => {
    setBuckets(draft.purpose_buckets || [])
  }, [draft])

  const totalPercent = buckets.reduce((sum, b) => sum + (parseFloat(b.target_percent) || 0), 0)
  const is100Percent = Math.abs(totalPercent - 100) < 0.01

  const handleUpdateBucket = (
    index: number,
    field: keyof PurposeBucketDTO,
    value: string,
  ) => {
    const updated = [...buckets]
    updated[index] = { ...updated[index], [field]: value } as PurposeBucketDTO
    setBuckets(updated)
  }

  const handleAddBucket = () => {
    const nextIdx = buckets.length + 1
    const color = VIBRANT_BUCKET_PALETTE[buckets.length % VIBRANT_BUCKET_PALETTE.length] || '#3b82f6'
    const newB: PurposeBucketDTO = {
      bucket_id: `b_purpose_${Date.now()}`,
      name: `พอร์ตย่อย ${nextIdx}`,
      role: 'เพื่อสร้างผลตอบแทนหรือสภาพคล่อง',
      color,
      target_percent: '0',
      source_value_ids: [],
      source_axis_allocation_ids: [],
    }
    setBuckets([...buckets, newB])
  }

  const handleRemoveBucket = (index: number) => {
    if (buckets.length <= 1) return
    setBuckets(buckets.filter((_, i) => i !== index))
  }

  const handleSaveAndReview = async () => {
    await onUpdateDraft({
      purpose_buckets: buckets,
    })
    onProceedToReview()
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header Banner */}
      <div className="rounded-2xl border border-sky-100 bg-gradient-to-br from-sky-50/70 via-white to-blue-50/30 p-6 sm:p-8 shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-sky-100 pb-4">
          <div className="flex items-center gap-2.5">
            <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-tr from-sky-500 to-blue-600 text-white font-bold text-lg shadow-sm">
              🗂️
            </span>
            <div>
              <h1 className="text-xl sm:text-2xl font-extrabold text-zinc-900 tracking-tight">
                ออกแบบ Purpose Buckets จากแกนหลัก
              </h1>
              <p className="text-xs text-zinc-500 mt-0.5">
                พอร์ต: <span className="font-semibold text-zinc-700">{portfolioId}</span> | จัดกลุ่มพอร์ตตามบทบาทของเงิน (เช่น รายได้, เติบโต, สภาพคล่อง) โดยสอดคล้องกับแผนจัดสรรในแกนหลัก
              </p>
            </div>
          </div>

          {/* Live Total Percentage Badge */}
          <div className="flex items-center gap-2">
            <span
              className={`inline-flex items-center gap-1.5 rounded-full px-3.5 py-1 text-xs font-extrabold border ${
                is100Percent
                  ? 'bg-emerald-50 text-emerald-700 border-emerald-300'
                  : 'bg-amber-50 text-amber-700 border-amber-300'
              }`}
            >
              <span>รวมสัดส่วน:</span>
              <span className="text-sm font-black">{totalPercent.toFixed(1)}%</span>
              <span>/ 100%</span>
            </span>
          </div>
        </div>

        {/* Info card */}
        <div className="mt-4 rounded-xl bg-white p-4 border border-sky-200/80 text-xs text-zinc-600 space-y-1">
          <p>
            • <strong>Bucket คือบทบาทของเงิน:</strong> เช่น "เงินพร้อมใช้", "รายได้สม่ำเสมอ", หรือ "เติบโตระยะยาว"
          </p>
          <p>
            • สัดส่วนเริ่มต้นดึงมาจากแผนจัดสรรในแกนหลักที่คุณยืนยันไว้ คุณสามารถปรับแต่งชื่อ สี และสัดส่วนได้
          </p>
        </div>
      </div>

      {/* Purpose Buckets List */}
      <div className="rounded-2xl border border-sky-100 bg-white p-6 sm:p-8 shadow-sm space-y-6">
        <div className="flex items-center justify-between border-b border-sky-50 pb-3">
          <h2 className="text-base sm:text-lg font-bold text-zinc-900">
            รายการ Purpose Buckets ({buckets.length})
          </h2>
          <button
            type="button"
            onClick={handleAddBucket}
            disabled={loading}
            className="rounded-xl border border-sky-200 bg-sky-50 px-3.5 py-1.5 text-xs font-bold text-sky-700 hover:bg-sky-100 transition-all flex items-center gap-1"
          >
            <span>+ เพิ่ม Bucket</span>
          </button>
        </div>

        <div className="space-y-4">
          {buckets.map((b, idx) => (
            <div
              key={b.bucket_id}
              className="rounded-xl border border-sky-100 bg-white p-4 sm:p-5 shadow-xs hover:border-sky-300 transition-all space-y-3"
            >
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                {/* Color + Name */}
                <div className="flex items-center gap-3 flex-1 min-w-0">
                  <div className="relative group">
                    <span
                      className="block w-6 h-6 rounded-full border border-black/10 shadow-xs cursor-pointer"
                      style={{ backgroundColor: b.color }}
                      title="คลิกเพื่อสลับสี"
                    />
                  </div>

                  <div className="flex-1 min-w-0">
                    <label htmlFor={`b-name-${b.bucket_id}`} className="sr-only">ชื่อ Bucket</label>
                    <input
                      id={`b-name-${b.bucket_id}`}
                      type="text"
                      value={b.name}
                      onChange={(e) => handleUpdateBucket(idx, 'name', e.target.value)}
                      placeholder="ชื่อ Bucket เช่น รายได้สม่ำเสมอ"
                      className="w-full rounded-lg border border-zinc-200 px-3 py-1.5 text-sm font-bold text-zinc-800 focus:outline-none focus:ring-2 focus:ring-sky-400"
                    />
                  </div>
                </div>

                {/* Target Percent */}
                <div className="flex items-center gap-3 shrink-0">
                  <label htmlFor={`b-target-${b.bucket_id}`} className="text-xs font-semibold text-zinc-500">
                    สัดส่วน:
                  </label>
                  <div className="flex items-center gap-1">
                    <input
                      id={`b-target-${b.bucket_id}`}
                      type="number"
                      min="0"
                      max="100"
                      step="1"
                      value={b.target_percent}
                      onChange={(e) => handleUpdateBucket(idx, 'target_percent', e.target.value)}
                      className="w-20 rounded-lg border border-zinc-200 px-2.5 py-1.5 text-sm font-extrabold text-sky-700 text-right focus:outline-none focus:ring-2 focus:ring-sky-400"
                    />
                    <span className="text-xs font-bold text-zinc-500">%</span>
                  </div>

                  {buckets.length > 1 && (
                    <button
                      type="button"
                      onClick={() => handleRemoveBucket(idx)}
                      className="rounded-lg p-1.5 text-zinc-400 hover:text-rose-600 hover:bg-rose-50"
                      title="ลบ Bucket นี้"
                    >
                      🗑️
                    </button>
                  )}
                </div>
              </div>

              {/* Role */}
              <div>
                <label htmlFor={`b-role-${b.bucket_id}`} className="block text-xs font-semibold text-zinc-600 mb-1">
                  บทบาทและเหตุผลของเงินก้อนนี้:
                </label>
                <input
                  id={`b-role-${b.bucket_id}`}
                  type="text"
                  value={b.role}
                  onChange={(e) => handleUpdateBucket(idx, 'role', e.target.value)}
                  placeholder="เช่น สร้างกระแสเงินสดปันผลเพื่อค่าใช้จ่าย หรือ พอร์ตทดลองเรียนรู้"
                  className="w-full rounded-lg border border-zinc-200 bg-sky-50/20 px-3 py-1.5 text-xs text-zinc-700 focus:outline-none focus:ring-2 focus:ring-sky-400"
                />
              </div>
            </div>
          ))}
        </div>

        {/* Action Bar */}
        <div className="pt-6 border-t border-sky-100 flex flex-col sm:flex-row items-center justify-between gap-4 mt-6">
          <div className="text-xs text-zinc-500">
            {statusText ? (
              <span className="flex items-center gap-2 text-sky-700 font-semibold animate-pulse">
                <span className="h-2 w-2 rounded-full bg-sky-500" />
                {statusText}
              </span>
            ) : !is100Percent ? (
              <span className="text-amber-700 font-semibold">
                ⚠️ ผลรวมสัดส่วนต้องเท่ากับ 100% (ปัจจุบัน {totalPercent.toFixed(1)}%)
              </span>
            ) : (
              <span className="text-emerald-700 font-semibold">
                ✓ สัดส่วนรวมครบ 100% พร้อมสำหรับการตรวจสอบความปลอดภัย
              </span>
            )}
          </div>

          <div className="flex items-center gap-3 w-full sm:w-auto">
            <button
              type="button"
              onClick={handleSaveAndReview}
              disabled={!is100Percent || loading}
              className={`w-full sm:w-auto px-7 py-3 rounded-xl font-bold text-sm text-white shadow-md transition-all flex items-center justify-center gap-2 ${
                is100Percent && !loading
                  ? 'bg-gradient-to-r from-sky-600 to-blue-600 hover:from-sky-700 hover:to-blue-700 shadow-sky-500/25 active:scale-98 cursor-pointer'
                  : 'bg-zinc-300 text-zinc-500 cursor-not-allowed shadow-none'
              }`}
            >
              <span>ตรวจสอบก่อนใช้กับพอร์ต →</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
