import { useState } from 'react'
import type { ActualWatchlistItemDTO, ActualWatchlistStateDTO } from '../../api/types'
import { api } from '../../api/client'
import WatchlistModal from './Modals/WatchlistModal'

interface Props {
  items: ActualWatchlistItemDTO[]
  lastUpdated: string | null
  onSuccess?: (state: ActualWatchlistStateDTO) => void
}

export default function PortfolioWatchlistTab({ items, lastUpdated, onSuccess }: Props) {
  const [modalOpen, setModalOpen] = useState(false)
  const [editingItem, setEditingItem] = useState<ActualWatchlistItemDTO | null>(null)

  const handleDelete = async (symbol: string) => {
    if (!onSuccess) return
    if (!window.confirm(`คุณต้องการลบ ${symbol} ออกจาก Watchlist หรือไม่?`)) return
    try {
      const state = await api.removeWatchlistItem(symbol)
      onSuccess(state)
    } catch (err: any) {
      alert(err?.message || 'ลบ Watchlist ไม่สำเร็จ')
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-center">
        <div>
          <h3 className="text-base font-bold text-zinc-900">Watchlist ({items.length} รายการ)</h3>
          <p className="text-xs text-zinc-500">
            รายการสินทรัพย์ที่เฝ้าติดตามราคาเป้าหมาย (Target Price) และบันทึกข้อสังเกตจาก Trading Journal
          </p>
        </div>
        <div className="flex items-center gap-3">
          {lastUpdated && (
            <span className="text-xs text-zinc-400">
              อัปเดตล่าสุด: {new Date(lastUpdated).toLocaleString('th-TH')}
            </span>
          )}
          {onSuccess && (
            <button
              type="button"
              onClick={() => {
                setEditingItem(null)
                setModalOpen(true)
              }}
              className="rounded-xl bg-flow-blue px-4 py-2 text-xs font-bold text-white shadow-md hover:bg-sky-600 transition-colors"
            >
              + Add to Watchlist
            </button>
          )}
        </div>
      </div>

      <div className="rounded-2xl border border-sky-100 bg-panel shadow-sm overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-sky-100 bg-zinc-50/80 text-xs font-semibold uppercase tracking-wider text-zinc-500">
                <th className="px-6 py-3.5">Symbol</th>
                <th className="px-6 py-3.5">Asset Type</th>
                <th className="px-6 py-3.5 text-right">Target Price</th>
                <th className="px-6 py-3.5">Added Date</th>
                <th className="px-6 py-3.5">Notes</th>
                <th className="px-6 py-3.5 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-sky-50">
              {items.map((item) => (
                <tr key={item.symbol} className="hover:bg-sky-50/50 transition-colors">
                  <td className="px-6 py-4 font-bold text-zinc-900">{item.symbol}</td>
                  <td className="px-6 py-4">
                    <span className="inline-block rounded-md bg-sky-50 px-2 py-0.5 text-xs font-semibold text-sky-800">
                      {item.asset_type}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-right font-mono tabular-nums font-bold text-flow-blue">
                    {item.target_price !== null && item.target_price !== undefined
                      ? item.target_price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
                      : 'N/A'}
                  </td>
                  <td className="px-6 py-4 text-xs font-mono tabular-nums text-zinc-500">{item.added_date}</td>
                  <td className="px-6 py-4 text-xs text-zinc-600 max-w-xs truncate">{item.notes || '-'}</td>
                  <td className="px-6 py-4 text-right space-x-2">
                    {onSuccess && (
                      <>
                        <button
                          type="button"
                          onClick={() => setEditingItem(item)}
                          className="text-xs font-bold text-flow-blue hover:underline"
                        >
                          ✏️ Edit
                        </button>
                        <button
                          type="button"
                          onClick={() => handleDelete(item.symbol)}
                          className="text-xs font-bold text-rose-600 hover:underline"
                        >
                          🗑️ Delete
                        </button>
                      </>
                    )}
                  </td>
                </tr>
              ))}
              {items.length === 0 && (
                <tr>
                  <td colSpan={6} className="py-16 text-center text-zinc-400">
                    ยังไม่มีรายการ Watchlist ในขณะนี้
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {(modalOpen || editingItem) && onSuccess && (
        <WatchlistModal
          initialItem={editingItem}
          onClose={() => {
            setModalOpen(false)
            setEditingItem(null)
          }}
          onSuccess={(state) => {
            onSuccess(state)
          }}
        />
      )}
    </div>
  )
}
