import { useState, useEffect, useMemo, useCallback } from 'react'
import Modal from '../../ui/Modal'
import { api } from '../../../api/client'
import type {
  AllocationTargetDTO,
  ActualPortfolioStateDTO,
  HoldingsBucketRecommendationResponseDTO,
} from '../../../api/types'

interface Props {
  portfolioId: string
  targets: AllocationTargetDTO[]
  onClose: () => void
  onSuccess: (state: ActualPortfolioStateDTO) => void
}

export default function AiBucketAssignModal({ portfolioId, targets, onClose, onSuccess }: Props) {
  const [loading, setLoading] = useState(true)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [data, setData] = useState<HoldingsBucketRecommendationResponseDTO | null>(null)
  const [assignments, setAssignments] = useState<Record<string, string | null>>({})
  const [showOnlyChanged, setShowOnlyChanged] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')

  const targetMap = useMemo(() => {
    const map = new Map<string, AllocationTargetDTO>()
    targets.forEach((t) => map.set(t.bucket_id, t))
    return map
  }, [targets])

  const fetchRecommendations = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await api.recommendHoldingBuckets(portfolioId)
      setData(res)
      const initialMap: Record<string, string | null> = {}
      res.recommendations.forEach((item) => {
        initialMap[item.symbol] = item.recommended_bucket_id
      })
      setAssignments(initialMap)
      // If no changes recommended, default to show all items
      if (res.total_unassigned === 0 && res.total_mismatched === 0) {
        setShowOnlyChanged(false)
      }
    } catch (err: any) {
      setError(err?.message || 'ไม่สามารถโหลดคำแนะนำจาก AI ได้ กรุณาลองใหม่อีกครั้ง')
    } finally {
      setLoading(false)
    }
  }, [portfolioId])

  useEffect(() => {
    void fetchRecommendations()
  }, [fetchRecommendations])

  const handleBucketChange = (symbol: string, newBucketId: string) => {
    setAssignments((prev) => ({
      ...prev,
      [symbol]: newBucketId || null,
    }))
  }

  // Calculate changes compared to current buckets
  const changedItemsCount = useMemo(() => {
    if (!data) return 0
    return data.recommendations.filter((item) => {
      const current = item.current_bucket_id ?? null
      const assigned = assignments[item.symbol] ?? null
      return current !== assigned
    }).length
  }, [data, assignments])

  const filteredRecommendations = useMemo(() => {
    if (!data) return []
    return data.recommendations.filter((item) => {
      const assigned = assignments[item.symbol] ?? null
      const current = item.current_bucket_id ?? null
      const isChanged = current !== assigned

      if (showOnlyChanged && !isChanged) {
        return false
      }

      if (searchQuery.trim()) {
        const query = searchQuery.trim().toLowerCase()
        const matchSymbol = item.symbol.toLowerCase().includes(query)
        const matchAsset = item.asset_type.toLowerCase().includes(query)
        const matchReason = item.reason.toLowerCase().includes(query)
        if (!matchSymbol && !matchAsset && !matchReason) return false
      }

      return true
    })
  }, [data, assignments, showOnlyChanged, searchQuery])

  const handleSubmit = async () => {
    if (!data) return
    setSubmitting(true)
    setError(null)
    try {
      const payloadAssignments = data.recommendations.map((item) => ({
        symbol: item.symbol,
        bucket_id: assignments[item.symbol] || null,
      }))

      const updatedState = await api.batchReassignHoldingBuckets(
        { assignments: payloadAssignments },
        portfolioId
      )
      onSuccess(updatedState)
      onClose()
    } catch (err: any) {
      setError(err?.message || 'บันทึกการจัดกลุ่มไม่สำเร็จ กรุณาลองใหม่อีกครั้ง')
    } finally {
      setSubmitting(false)
    }
  }

  const getBucketBadge = (bucketId?: string | null) => {
    if (!bucketId || bucketId.toLowerCase() === 'unassigned') {
      return (
        <span className="inline-flex items-center rounded-md bg-amber-50 px-2 py-0.5 text-xs font-semibold text-amber-700 border border-amber-200">
          ⚠️ ยังไม่มีกลุ่ม (Unassigned)
        </span>
      )
    }
    const target = targetMap.get(bucketId)
    return (
      <span
        className="inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-xs font-bold text-zinc-800 border"
        style={{
          borderColor: target?.color ? `${target.color}40` : '#e2e8f0',
          backgroundColor: target?.color ? `${target.color}15` : '#f8fafc',
        }}
      >
        <span
          className="w-2 h-2 rounded-full inline-block"
          style={{ backgroundColor: target?.color || '#3b82f6' }}
        />
        {target ? target.name : bucketId}
      </span>
    )
  }

  return (
    <Modal
      titleId="ai-bucket-assign-modal-title"
      onClose={onClose}
      panelClassName="max-w-4xl max-h-[90vh] flex flex-col rounded-2xl border border-sky-100 bg-white p-6 shadow-2xl overflow-hidden"
    >
      {/* Header */}
      <div className="flex items-start justify-between border-b border-zinc-100 pb-3 flex-shrink-0">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xl">✨</span>
            <h3 id="ai-bucket-assign-modal-title" className="text-lg font-bold text-zinc-900">
              จัดกลุ่มสินทรัพย์อัตโนมัติด้วย AI
            </h3>
          </div>
          <p className="mt-1 text-xs text-zinc-500">
            ระบบ AI วิเคราะห์ความเหมาะสมของสินทรัพย์ตามเป้าหมายของแต่ละ Strategy Bucket
            และนำเสนอแผนจัดกลุ่มพร้อมเหตุผล
          </p>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="rounded-lg p-1 text-zinc-400 hover:bg-zinc-100 hover:text-zinc-600 transition-colors"
        >
          ✕
        </button>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 overflow-y-auto py-4 space-y-4">
        {error && (
          <div className="rounded-xl border border-rose-200 bg-rose-50 p-3.5 text-xs font-medium text-rose-800 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span>⚠️</span>
              <span>{error}</span>
            </div>
            <button
              type="button"
              onClick={fetchRecommendations}
              className="rounded-lg bg-white px-2.5 py-1 text-xs font-bold text-rose-700 border border-rose-300 hover:bg-rose-100 transition-colors"
            >
              🔄 ลองใหม่อีกครั้ง
            </button>
          </div>
        )}

        {loading ? (
          <div className="flex flex-col items-center justify-center py-16 space-y-4 text-center">
            <div className="relative">
              <div className="w-14 h-14 rounded-full border-4 border-sky-100 border-t-flow-blue animate-spin" />
              <div className="absolute inset-0 flex items-center justify-center text-xl">🧠</div>
            </div>
            <div className="space-y-1">
              <h4 className="text-sm font-bold text-zinc-800">
                AI กำลังวิเคราะห์สินทรัพย์และโครงสร้าง Strategy Buckets...
              </h4>
              <p className="text-xs text-zinc-500 max-w-md">
                เปรียบเทียบประเภทสินทรัพย์, บทบาทของกลุ่มเป้าหมาย และจับคู่สินทรัพย์ที่ยังไม่มีกลุ่มหรืออยู่ผิดกลุ่ม
              </p>
            </div>
          </div>
        ) : data ? (
          <>
            {/* KPI Cards & AI Summary */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div className="rounded-xl border border-sky-100 bg-sky-50/50 p-3 flex items-center gap-3">
                <div className="w-9 h-9 rounded-lg bg-sky-100 text-sky-700 flex items-center justify-center font-bold text-base">
                  📁
                </div>
                <div>
                  <div className="text-[11px] font-semibold text-zinc-500">สินทรัพย์ทั้งหมด</div>
                  <div className="text-lg font-black text-zinc-900 tabular-nums">
                    {data.recommendations.length} รายการ
                  </div>
                </div>
              </div>

              <div className="rounded-xl border border-amber-100 bg-amber-50/50 p-3 flex items-center gap-3">
                <div className="w-9 h-9 rounded-lg bg-amber-100 text-amber-700 flex items-center justify-center font-bold text-base">
                  ⚠️
                </div>
                <div>
                  <div className="text-[11px] font-semibold text-zinc-500">สินทรัพย์ไม่มีกลุ่ม</div>
                  <div className="text-lg font-black text-amber-700 tabular-nums">
                    {data.total_unassigned} รายการ
                  </div>
                </div>
              </div>

              <div className="rounded-xl border border-indigo-100 bg-indigo-50/50 p-3 flex items-center gap-3">
                <div className="w-9 h-9 rounded-lg bg-indigo-100 text-indigo-700 flex items-center justify-center font-bold text-base">
                  🔄
                </div>
                <div>
                  <div className="text-[11px] font-semibold text-zinc-500">ปรับเปลี่ยน / ย้ายกลุ่ม</div>
                  <div className="text-lg font-black text-indigo-700 tabular-nums">
                    {changedItemsCount} รายการ
                  </div>
                </div>
              </div>
            </div>

            {/* AI Summary Banner */}
            {data.summary && (
              <div className="rounded-xl border border-sky-200/80 bg-gradient-to-r from-sky-50 via-indigo-50/40 to-white p-3.5 text-xs text-zinc-700 flex items-start gap-2.5">
                <span className="text-base flex-shrink-0 mt-0.5">💡</span>
                <div className="leading-relaxed">
                  <span className="font-bold text-zinc-900">สรุปการวิเคราะห์: </span>
                  {data.summary}
                </div>
              </div>
            )}

            {/* Filter and Search Bar */}
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2.5 pt-1">
              <label className="inline-flex items-center gap-2 text-xs font-semibold text-zinc-700 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={showOnlyChanged}
                  onChange={(e) => setShowOnlyChanged(e.target.checked)}
                  className="rounded border-zinc-300 text-flow-blue focus:ring-flow-blue w-4 h-4"
                />
                <span>แสดงเฉพาะรายการที่มีการปรับเปลี่ยน ({changedItemsCount} รายการ)</span>
              </label>

              <div className="w-full sm:w-64">
                <input
                  type="text"
                  placeholder="ค้นหา Symbol หรือ Asset Type..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full rounded-xl border border-zinc-200 bg-zinc-50/60 px-3 py-1.5 text-xs text-zinc-800 placeholder-zinc-400 focus:bg-white focus:border-flow-blue focus:outline-none transition-colors"
                />
              </div>
            </div>

            {/* Table */}
            <div className="border border-zinc-200 rounded-xl overflow-hidden shadow-2xs">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="bg-zinc-50 border-b border-zinc-200 text-zinc-600 font-bold uppercase tracking-wider">
                    <th className="py-2.5 px-3">สินทรัพย์</th>
                    <th className="py-2.5 px-3">กลุ่มเดิม</th>
                    <th className="py-2.5 px-3">กลุ่มแนะนำ / ปรับแต่ง</th>
                    <th className="py-2.5 px-3">เหตุผลของ AI</th>
                    <th className="py-2.5 px-3 text-center">สถานะ</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-100">
                  {filteredRecommendations.length === 0 ? (
                    <tr>
                      <td colSpan={5} className="py-8 text-center text-zinc-400">
                        {showOnlyChanged
                          ? 'ไม่มีรายการที่มีการปรับเปลี่ยนกลุ่ม (ทุกสินทรัพย์จัดสรรถูกต้องแล้ว)'
                          : 'ไม่พบรายการที่ตรงกับคำค้นหา'}
                      </td>
                    </tr>
                  ) : (
                    filteredRecommendations.map((item) => {
                      const currentBucket = item.current_bucket_id ?? null
                      const selectedBucket = assignments[item.symbol] ?? null
                      const isUnassigned = !currentBucket || currentBucket.toLowerCase() === 'unassigned'
                      const isChanged = currentBucket !== selectedBucket

                      return (
                        <tr
                          key={item.symbol}
                          className={`hover:bg-sky-50/30 transition-colors ${
                            isChanged ? 'bg-amber-50/20' : ''
                          }`}
                        >
                          {/* Symbol & Type */}
                          <td className="py-2.5 px-3 whitespace-nowrap">
                            <div className="font-bold text-zinc-900 font-mono">{item.symbol}</div>
                            <span className="text-[10px] font-semibold text-zinc-500 uppercase">
                              {item.asset_type}
                            </span>
                          </td>

                          {/* Current Bucket */}
                          <td className="py-2.5 px-3 whitespace-nowrap">
                            {getBucketBadge(currentBucket)}
                          </td>

                          {/* Recommended Bucket Dropdown */}
                          <td className="py-2.5 px-3">
                            <select
                              value={selectedBucket || ''}
                              onChange={(e) => handleBucketChange(item.symbol, e.target.value)}
                              className={`rounded-lg border px-2.5 py-1 text-xs font-semibold text-zinc-800 transition-colors focus:outline-none focus:ring-1 focus:ring-flow-blue ${
                                isChanged
                                  ? 'border-indigo-300 bg-indigo-50/60 font-bold text-indigo-900'
                                  : 'border-zinc-300 bg-white'
                              }`}
                            >
                              <option value="">-- ไม่ระบุกลุ่ม (Unassigned) --</option>
                              {targets.map((t) => (
                                <option key={t.bucket_id} value={t.bucket_id}>
                                  {t.name} ({t.target_percent}%)
                                </option>
                              ))}
                            </select>
                          </td>

                          {/* AI Rationale */}
                          <td className="py-2.5 px-3 text-zinc-600 max-w-xs leading-relaxed">
                            {item.reason}
                          </td>

                          {/* Status Badge */}
                          <td className="py-2.5 px-3 text-center whitespace-nowrap">
                            {isUnassigned && isChanged ? (
                              <span className="inline-flex items-center rounded-md bg-emerald-50 px-2 py-0.5 text-[11px] font-bold text-emerald-700 border border-emerald-200">
                                ✨ จัดกลุ่มใหม่
                              </span>
                            ) : isChanged ? (
                              <span className="inline-flex items-center rounded-md bg-amber-50 px-2 py-0.5 text-[11px] font-bold text-amber-700 border border-amber-200">
                                🔄 ย้ายกลุ่ม
                              </span>
                            ) : (
                              <span className="inline-flex items-center rounded-md bg-zinc-50 px-2 py-0.5 text-[11px] font-semibold text-zinc-500 border border-zinc-200">
                                ✓ กลุ่มเดิม
                              </span>
                            )}
                          </td>
                        </tr>
                      )
                    })
                  )}
                </tbody>
              </table>
            </div>
          </>
        ) : null}
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between border-t border-zinc-100 pt-3 flex-shrink-0">
        <div className="text-xs text-zinc-500">
          {changedItemsCount > 0 ? (
            <span className="text-indigo-700 font-semibold">
              ✏️ มีการปรับเปลี่ยนกลุ่ม {changedItemsCount} รายการ
            </span>
          ) : (
            <span>ไม่มีรายการเปลี่ยนแปลง</span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={onClose}
            disabled={submitting}
            className="rounded-xl border border-zinc-300 bg-white px-4 py-2 text-xs font-semibold text-zinc-700 hover:bg-zinc-50 transition-colors disabled:opacity-50"
          >
            ยกเลิก
          </button>
          <button
            type="button"
            onClick={handleSubmit}
            disabled={loading || submitting || changedItemsCount === 0}
            className="rounded-xl bg-flow-blue px-5 py-2 text-xs font-bold text-white shadow-md hover:bg-sky-600 transition-all disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {submitting ? 'กำลังบันทึกการจัดกลุ่ม...' : `ยืนยันการจัดกลุ่ม (${changedItemsCount} รายการ)`}
          </button>
        </div>
      </div>
    </Modal>
  )
}
