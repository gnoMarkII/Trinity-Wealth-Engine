import { useState } from 'react'
import type {
  AllocationPreviewDTO,
  AllocationApplyReceiptDTO,
  BucketPlanDraftDTO,
} from '../../../api/types'

interface BucketApplyReviewProps {
  draft: BucketPlanDraftDTO
  preview: AllocationPreviewDTO | null
  applyReceipt: AllocationApplyReceiptDTO | null
  onApply: () => Promise<void>
  onBack: () => void
  onGoToOverview: () => void
  loading: boolean
  statusText?: string | null
}

export default function BucketApplyReview({
  draft,
  preview,
  applyReceipt,
  onApply,
  onBack,
  onGoToOverview,
  loading,
  statusText,
}: BucketApplyReviewProps) {
  const [confirmedSafe, setConfirmedSafe] = useState(false)

  if (applyReceipt) {
    return (
      <div className="max-w-3xl mx-auto space-y-6">
        <div className="rounded-2xl border border-emerald-200 bg-gradient-to-br from-emerald-50/80 via-white to-teal-50/40 p-8 text-center shadow-sm space-y-5 animate-scale-up">
          <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl bg-emerald-100 text-3xl text-emerald-600 shadow-xs">
            🎉
          </div>
          <div>
            <h2 className="text-xl sm:text-2xl font-extrabold text-zinc-900 tracking-tight">
              นำ Purpose Buckets ไปใช้กับพอร์ตสำเร็จ!
            </h2>
            <p className="text-xs sm:text-sm text-zinc-600 mt-1 max-w-lg mx-auto">
              ระบบได้ปรับเปลี่ยน Allocation Targets และจับคู่จัดหมวดหมู่สินทรัพย์ (Holdings) ในพอร์ตการลงทุนเรียบร้อยแล้ว
            </p>
          </div>

          <div className="rounded-xl bg-white p-4 border border-emerald-200/80 max-w-md mx-auto text-left text-xs space-y-2 text-zinc-700">
            <div className="flex justify-between border-b border-emerald-50 pb-1.5">
              <span className="font-semibold text-zinc-500">สถานะการบันทึก:</span>
              <span className="font-bold text-emerald-700">Committed (Transaction สมบูรณ์)</span>
            </div>
            <div className="flex justify-between border-b border-emerald-50 pb-1.5">
              <span className="font-semibold text-zinc-500">ลำดับ Checkpoint (Sequence):</span>
              <span className="font-mono font-bold text-zinc-800">#{applyReceipt.applied_sequence}</span>
            </div>
            <div className="flex justify-between">
              <span className="font-semibold text-zinc-500">Command ID:</span>
              <span className="font-mono text-[10px] text-zinc-500 truncate max-w-[200px]">
                {applyReceipt.command_id}
              </span>
            </div>
          </div>

          <div className="pt-4 flex items-center justify-center gap-3">
            <button
              type="button"
              onClick={onGoToOverview}
              className="rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 px-6 py-3 text-sm font-bold text-white shadow-md shadow-emerald-500/20 hover:from-emerald-700 hover:to-teal-700 active:scale-98 transition-all"
            >
              📊 ไปที่หน้า Strategy Buckets & Allocation
            </button>
          </div>
        </div>
      </div>
    )
  }

  const validatedTargets = preview?.validated_targets ?? []
  const affectedHoldings = preview?.affected_holdings ?? []
  const issues = preview?.issues ?? []

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div className="rounded-2xl border border-sky-100 bg-white p-6 sm:p-8 shadow-sm">
        <div className="flex items-center justify-between border-b border-sky-50 pb-4">
          <div className="flex items-center gap-3">
            <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-sky-100 text-sky-700 font-bold text-lg">
              🔍
            </span>
            <div>
              <h2 className="text-lg sm:text-xl font-bold text-zinc-900">
                ตรวจสอบก่อนใช้กับพอร์ต (Preview & Safety Check)
              </h2>
              <p className="text-xs text-zinc-500 mt-0.5">
                พอร์ต: <span className="font-semibold text-zinc-700">{draft.portfolio_id}</span> | ทบทวนการเปลี่ยนแปลงสัดส่วนเป้าหมายและผลกระทบต่อรายการสินทรัพย์เดิมก่อนยืนยัน
              </p>
            </div>
          </div>

          <button
            type="button"
            onClick={onBack}
            className="text-xs font-semibold text-zinc-500 hover:text-zinc-800"
          >
            ← กลับไปแก้ไข
          </button>
        </div>

        {/* Validation Issues Alert */}
        {issues.length > 0 && (
          <div className="mt-4 rounded-xl border border-amber-200 bg-amber-50 p-3.5 text-xs text-amber-900 space-y-1">
            <span className="font-bold">⚠️ ข้อควรระวังก่อนยืนยัน:</span>
            <ul className="list-disc list-inside">
              {issues.map((issue, idx) => (
                <li key={idx}>{issue}</li>
              ))}
            </ul>
          </div>
        )}

        {/* Targets comparison */}
        <div className="mt-6 space-y-3">
          <h3 className="text-sm font-bold text-zinc-900">
            🎯 สัดส่วนเป้าหมายชุดใหม่ (Validated Purpose Targets)
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
            {validatedTargets.map((t) => (
              <div
                key={t.bucket_id}
                className="p-3.5 rounded-xl border border-sky-100 bg-sky-50/30 flex items-center justify-between"
              >
                <div className="flex items-center gap-2 min-w-0">
                  <span
                    className="w-3.5 h-3.5 rounded-full shrink-0"
                    style={{ backgroundColor: t.color || '#3b82f6' }}
                  />
                  <span className="text-xs font-bold text-zinc-800 truncate">{t.name}</span>
                </div>
                <span className="text-sm font-extrabold text-sky-700 shrink-0">
                  {t.target_percent}%
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Affected Holdings */}
        <div className="mt-6 pt-4 border-t border-sky-50 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-zinc-900">
              📦 สินทรัพย์ที่ได้รับผลกระทบจากการจัดหมวดหมู่ ({affectedHoldings.length} รายการ)
            </h3>
          </div>

          {affectedHoldings.length > 0 ? (
            <div className="max-h-48 overflow-y-auto space-y-2 pr-1">
              {affectedHoldings.map((h, idx) => (
                <div
                  key={idx}
                  className="flex items-center justify-between p-2.5 rounded-lg bg-zinc-50 border border-zinc-200/80 text-xs"
                >
                  <span className="font-bold text-zinc-800">{h.symbol}</span>
                  <div className="flex items-center gap-2 text-zinc-600">
                    <span className="text-zinc-400">{h.old_bucket_id || 'ไม่มีหมวด'}</span>
                    <span>→</span>
                    <span className="font-bold text-emerald-700">{h.new_bucket_id}</span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-xs text-zinc-500 italic p-3 rounded-lg bg-zinc-50">
              ไม่มีสินทรัพย์เดิมที่ต้องสลับหมวด หรือพอร์ตยังไม่มีสินทรัพย์
            </p>
          )}
        </div>

        {/* Safety note & Confirmation checkbox */}
        <div className="mt-6 pt-4 border-t border-sky-50 space-y-3">
          <div className="rounded-xl border border-sky-200 bg-sky-50/60 p-4 text-xs text-sky-900 space-y-1">
            <span className="font-bold">🛡️ มาตรการความปลอดภัยของระบบ:</span>
            <p>
              • การใช้ Buckets จะบันทึก Targets และจัดหมวดสินทรัพย์ในธุรกรรมเดียว (Atomic Transaction)
            </p>
            <p>• จะไม่มีการส่งคำสั่งซื้อขาย ไม่มีผลต่อต้นทุน และมูลค่า NAV ของพอร์ตไม่เปลี่ยนแปลง</p>
          </div>

          <label className="flex items-center gap-2.5 cursor-pointer pt-2">
            <input
              type="checkbox"
              checked={confirmedSafe}
              onChange={(e) => setConfirmedSafe(e.target.checked)}
              className="h-4 w-4 rounded border-zinc-300 text-sky-600 focus:ring-sky-500 cursor-pointer"
            />
            <span className="text-xs font-semibold text-zinc-800">
              ฉันได้ตรวจสอบสัดส่วนเป้าหมายและรายการสินทรัพย์แล้ว และต้องการนำชุด Buckets นี้ไปใช้กับพอร์ต
            </span>
          </label>
        </div>

        {/* Action Buttons */}
        <div className="pt-6 border-t border-sky-100 flex flex-col sm:flex-row items-center justify-between gap-4 mt-6">
          <button
            type="button"
            onClick={onBack}
            disabled={loading}
            className="rounded-xl border border-zinc-200 px-4 py-2.5 text-xs font-semibold text-zinc-600 hover:bg-zinc-50"
          >
            ← กลับไปแก้ไข
          </button>

          <button
            type="button"
            onClick={onApply}
            disabled={!confirmedSafe || loading}
            className={`w-full sm:w-auto px-7 py-3 rounded-xl font-bold text-sm text-white shadow-md transition-all flex items-center justify-center gap-2 ${
              confirmedSafe && !loading
                ? 'bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-700 hover:to-teal-700 shadow-emerald-500/25 active:scale-98 cursor-pointer'
                : 'bg-zinc-300 text-zinc-500 cursor-not-allowed shadow-none'
            }`}
          >
            {loading ? (
              <span>{statusText || 'กำลังนำไปใช้กับพอร์ต...'}</span>
            ) : (
              <span>🚀 ยืนยันและนำไปใช้กับพอร์ตนี้</span>
            )}
          </button>
        </div>
      </div>
    </div>
  )
}
