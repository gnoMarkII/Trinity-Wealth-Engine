import React, { useState, useMemo } from 'react'
import type { PromiseItemDTO } from '../../../api/types'
import { VerbatimQuoteModal } from './VerbatimQuoteModal'

interface PromiseLedgerTableProps {
  promises: PromiseItemDTO[]
}

type FilterTab = 'all' | 'pending' | 'delivered' | 'broken'

export const PromiseLedgerTable: React.FC<PromiseLedgerTableProps> = ({ promises }) => {
  const [activeTab, setActiveTab] = useState<FilterTab>('all')
  const [searchQuery, setSearchQuery] = useState<string>('')
  const [selectedQuarter, setSelectedQuarter] = useState<string>('all')
  const [selectedPromise, setSelectedPromise] = useState<PromiseItemDTO | null>(null)

  const availableQuarters = useMemo(() => {
    const set = new Set<string>()
    for (const p of promises) {
      if (p.made_in) set.add(p.made_in)
    }
    return Array.from(set).sort()
  }, [promises])

  const filteredPromises = useMemo(() => {
    return promises.filter((p) => {
      // 1. Status Filter
      if (activeTab === 'pending' && p.status !== 'Pending') return false
      if (activeTab === 'delivered' && p.status !== 'Delivered') return false
      if (activeTab === 'broken') {
        const isBroken =
          p.status === 'Missed' ||
          p.status === 'Partial' ||
          p.status === 'Quietly dropped' ||
          (p.status === 'Restated' && p.restated_direction === 'lowered')
        if (!isBroken) return false
      }

      // 2. Quarter Filter
      if (selectedQuarter !== 'all' && p.made_in !== selectedQuarter) return false

      // 3. Search Query Filter
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase().trim()
        const match =
          (p.metric || '').toLowerCase().includes(q) ||
          (p.metric_th || '').toLowerCase().includes(q) ||
          (p.speaker || '').toLowerCase().includes(q) ||
          (p.quote || '').toLowerCase().includes(q) ||
          (p.id || '').toLowerCase().includes(q) ||
          (p.category || '').toLowerCase().includes(q) ||
          (p.due || '').toLowerCase().includes(q) ||
          (p.evidence_quote || '').toLowerCase().includes(q) ||
          (p.notes || '').toLowerCase().includes(q) ||
          (p.notes_th || '').toLowerCase().includes(q)
        if (!match) return false
      }

      return true
    })
  }, [promises, activeTab, selectedQuarter, searchQuery])

  const statusBadge = (status: string, restatedDir?: string | null) => {
    switch (status) {
      case 'Delivered':
        return (
          <span className="px-2 py-0.5 rounded text-[11px] font-semibold bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300">
            ✅ สำเร็จ
          </span>
        )
      case 'Partial':
        return (
          <span className="px-2 py-0.5 rounded text-[11px] font-semibold bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300">
            ⚠️ สำเร็จบางส่วน
          </span>
        )
      case 'Missed':
        return (
          <span className="px-2 py-0.5 rounded text-[11px] font-semibold bg-rose-100 text-rose-800 dark:bg-rose-950 dark:text-rose-300">
            ❌ พลาดเป้า
          </span>
        )
      case 'Restated':
        return (
          <span className="px-2 py-0.5 rounded text-[11px] font-semibold bg-purple-100 text-purple-800 dark:bg-purple-950 dark:text-purple-300">
            🔄 {restatedDir === 'lowered' ? 'ลดเป้าลง' : 'ปรับเป้า'}
          </span>
        )
      case 'Quietly dropped':
        return (
          <span className="px-2 py-0.5 rounded text-[11px] font-semibold bg-zinc-100 text-zinc-800 dark:bg-zinc-800 dark:text-zinc-300">
            👻 ละทิ้งเงียบๆ
          </span>
        )
      case 'Pending':
        return (
          <span className="px-2 py-0.5 rounded text-[11px] font-semibold bg-blue-100 text-blue-800 dark:bg-blue-950 dark:text-blue-300">
            ⏳ รอถึงกำหนด
          </span>
        )
      default:
        return (
          <span className="px-2 py-0.5 rounded text-[11px] font-semibold bg-zinc-100 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400">
            {status}
          </span>
        )
    }
  }

  return (
    <div className="rounded-2xl border border-edge bg-surface p-5 shadow-sm space-y-4">
      {/* Title & Filter Tabs */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-edge/60 pb-3">
        <div>
          <h4 className="font-bold text-zinc-900 dark:text-zinc-100 text-sm flex items-center gap-2">
            <span>📋</span>
            <span>ตารางตรวจสอบคำสัญญาของผู้บริหาร (Commitment Ledger)</span>
          </h4>
          <p className="text-xs text-zinc-500 mt-0.5">
            บันทึกคำมั่นสัญญาที่ตรวจสอบได้จริง พร้อมถอดความต้นฉบับและไตรมาสที่ส่งมอบ
          </p>
        </div>

        {/* Filter Pills */}
        <div className="flex items-center gap-1 bg-surface-muted/60 p-1 rounded-xl border border-edge/60 text-xs font-semibold">
          <button
            type="button"
            onClick={() => setActiveTab('all')}
            className={`px-3 py-1 rounded-lg transition-all ${
              activeTab === 'all'
                ? 'bg-surface text-zinc-900 dark:text-zinc-100 shadow-sm'
                : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-900'
            }`}
          >
            ทั้งหมด ({promises.length})
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('pending')}
            className={`px-3 py-1 rounded-lg transition-all ${
              activeTab === 'pending'
                ? 'bg-surface text-zinc-900 dark:text-zinc-100 shadow-sm'
                : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-900'
            }`}
          >
            รอผล ({promises.filter((p) => p.status === 'Pending').length})
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('delivered')}
            className={`px-3 py-1 rounded-lg transition-all ${
              activeTab === 'delivered'
                ? 'bg-surface text-zinc-900 dark:text-zinc-100 shadow-sm'
                : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-900'
            }`}
          >
            สำเร็จ ({promises.filter((p) => p.status === 'Delivered').length})
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('broken')}
            className={`px-3 py-1 rounded-lg transition-all ${
              activeTab === 'broken'
                ? 'bg-surface text-zinc-900 dark:text-zinc-100 shadow-sm'
                : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-900'
            }`}
          >
            พลาด / ทิ้งสัญญา (
            {
              promises.filter(
                (p) =>
                  p.status === 'Missed' ||
                  p.status === 'Partial' ||
                  p.status === 'Quietly dropped' ||
                  (p.status === 'Restated' && p.restated_direction === 'lowered')
              ).length
            }
            )
          </button>
        </div>
      </div>

      {/* Search & Quarter Filter Toolbar */}
      <div className="flex flex-col sm:flex-row gap-2.5 items-stretch sm:items-center justify-between pt-1">
        {/* Search Input */}
        <div className="relative flex-1">
          <span className="absolute left-3 top-1/2 -translate-y-1/2 text-zinc-400 text-xs pointer-events-none">
            🔍
          </span>
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="ค้นหาคำสัญญา, ชื่อผู้บริหาร, ตัวเลข หรือรหัสสัญญา..."
            className="w-full pl-8 pr-8 py-2 rounded-xl border border-edge/60 bg-surface-muted/30 text-xs text-zinc-900 dark:text-zinc-100 placeholder:text-zinc-400 focus:outline-none focus:border-sky-500 focus:ring-1 focus:ring-sky-500 transition-colors"
          />
          {searchQuery && (
            <button
              type="button"
              onClick={() => setSearchQuery('')}
              className="absolute right-2.5 top-1/2 -translate-y-1/2 text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200 text-xs p-1"
              title="ล้างคำค้นหา"
            >
              ✕
            </button>
          )}
        </div>

        {/* Quarter Dropdown Selector & Counter */}
        <div className="flex items-center gap-2">
          <select
            value={selectedQuarter}
            onChange={(e) => setSelectedQuarter(e.target.value)}
            aria-label="กรองตามไตรมาส"
            className="px-3 py-2 rounded-xl border border-edge/60 bg-surface-muted/30 text-xs text-zinc-800 dark:text-zinc-200 focus:outline-none focus:border-sky-500 transition-colors cursor-pointer"
          >
            <option value="all">ทุกไตรมาสที่สัญญา ({availableQuarters.length} ไตรมาส)</option>
            {availableQuarters.map((q) => (
              <option key={q} value={q}>
                {q}
              </option>
            ))}
          </select>

          <span className="text-[11px] text-zinc-500 whitespace-nowrap pl-1">
            แสดง <strong className="text-zinc-800 dark:text-zinc-200">{filteredPromises.length}</strong> จาก {promises.length}
          </span>
        </div>
      </div>

      {/* Table Content */}
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs border-collapse">
          <thead>
            <tr className="border-b border-edge bg-surface-muted/40 text-zinc-600 dark:text-zinc-400 font-semibold">
              <th className="py-2.5 px-3">ไตรมาส / รหัส</th>
              <th className="py-2.5 px-3">ผู้แถลง & หมวดหมู่</th>
              <th className="py-2.5 px-3">เป้าหมายที่สัญญา</th>
              <th className="py-2.5 px-3">กำหนดส่งมอบ</th>
              <th className="py-2.5 px-3 text-center">สถานะ</th>
              <th className="py-2.5 px-3 text-right">หลักฐานต้นฉบับ</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-edge/50">
            {filteredPromises.length === 0 ? (
              <tr>
                <td colSpan={6} className="py-8 text-center text-zinc-500 italic space-y-2">
                  <div>ไม่พบคำสัญญาที่ตรงกับเงื่อนไขการค้นหาหรือตัวกรอง</div>
                  {(searchQuery || selectedQuarter !== 'all' || activeTab !== 'all') && (
                    <button
                      type="button"
                      onClick={() => {
                        setSearchQuery('')
                        setSelectedQuarter('all')
                        setActiveTab('all')
                      }}
                      className="px-3 py-1 rounded-lg text-xs font-semibold bg-sky-100 text-sky-800 dark:bg-sky-950 dark:text-sky-300 hover:bg-sky-200 transition-colors"
                    >
                      ล้างตัวกรองทั้งหมด
                    </button>
                  )}
                </td>
              </tr>
            ) : (
              filteredPromises.map((p) => (
                <tr
                  key={p.id}
                  className="hover:bg-surface-muted/30 transition-colors group"
                >
                  <td className="py-3 px-3">
                    <span className="font-bold text-zinc-900 dark:text-zinc-100 block">
                      {p.made_in}
                    </span>
                    <span className="text-[10px] text-zinc-400 font-mono">
                      {p.id}
                    </span>
                  </td>
                  <td className="py-3 px-3">
                    <span className="font-medium text-zinc-800 dark:text-zinc-200 block truncate max-w-[160px]">
                      {p.speaker}
                    </span>
                    <span className="text-[10px] px-1.5 py-0.2 rounded bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400 uppercase">
                      {p.category}
                    </span>
                  </td>
                  <td className="py-3 px-3 max-w-xs">
                    <span className="font-semibold text-zinc-900 dark:text-zinc-100 block">
                      {p.metric_th || p.metric}
                    </span>
                    <span className="text-[11px] text-zinc-500 truncate block">
                      "{p.quote}"
                    </span>
                  </td>
                  <td className="py-3 px-3">
                    <span className="font-semibold text-zinc-800 dark:text-zinc-200 block">
                      {p.due}
                    </span>
                    <span className="text-[10px] text-zinc-400 block">
                      {p.due_precision === 'explicit' ? 'ระบุตรงชัดเจน' : 'อนุมาน'}
                    </span>
                  </td>
                  <td className="py-3 px-3 text-center">
                    {statusBadge(p.status, p.restated_direction)}
                    {p.resolved_in && (
                      <span className="block text-[10px] text-zinc-400 mt-0.5">
                        ใน {p.resolved_in}
                      </span>
                    )}
                  </td>
                  <td className="py-3 px-3 text-right">
                    <button
                      type="button"
                      onClick={() => setSelectedPromise(p)}
                      className="px-2.5 py-1 rounded-lg border border-edge/60 bg-surface text-sky-600 dark:text-sky-400 hover:bg-sky-50 dark:hover:bg-sky-950/40 text-[11px] font-semibold transition-colors"
                    >
                      ดูคำพูดต้นฉบับ 🔍
                    </button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Verbatim Quote Detail Modal */}
      <VerbatimQuoteModal
        promise={selectedPromise}
        onClose={() => setSelectedPromise(null)}
      />
    </div>
  )
}
