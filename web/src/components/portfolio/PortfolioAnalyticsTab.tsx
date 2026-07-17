import type { PerformanceSnapshotDTO } from '../../api/types'
import { formatTHB } from './PortfolioSummaryCards'

interface Props {
  performanceRows: PerformanceSnapshotDTO[]
  daysRange: number | undefined
  onChangeDaysRange: (days: number | undefined) => void
}

export default function PortfolioAnalyticsTab({
  performanceRows,
  daysRange,
  onChangeDaysRange,
}: Props) {
  return (
    <div className="space-y-4 animate-fade-in">
      <div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-center">
        <div>
          <h3 className="text-base font-bold text-zinc-900">
            Performance History ({performanceRows.length} จุดข้อมูล)
          </h3>
          <p className="text-xs text-zinc-500">
            ติดตามประวัติ NAV Snapshot การเปลี่ยนแปลงต้นทุน และกำไร/ขาดทุนที่ยังไม่เกิดขึ้นจริง (Unrealized PnL)
          </p>
        </div>

        {/* Days selector bar */}
        <div className="flex items-center gap-2 rounded-xl border border-sky-100 bg-panel p-2 shadow-2xs">
          <span className="text-xs font-semibold text-zinc-600 ml-1">ช่วงเวลา:</span>
          <div className="flex gap-1">
            {[
              { label: '30 วัน', val: 30 },
              { label: '90 วัน', val: 90 },
              { label: '1 ปี', val: 365 },
              { label: 'ทั้งหมด', val: undefined },
            ].map((opt) => (
              <button
                key={opt.label}
                type="button"
                onClick={() => onChangeDaysRange(opt.val)}
                className={`rounded-lg px-3 py-1 text-xs font-semibold transition-colors ${
                  daysRange === opt.val
                    ? 'bg-flow-blue text-white shadow-2xs font-bold'
                    : 'bg-sky-50 text-zinc-500 hover:bg-sky-100'
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Performance Table */}
      <div className="rounded-2xl border border-sky-100 bg-panel shadow-sm overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-sky-100 bg-zinc-50/80 text-xs font-semibold uppercase tracking-wider text-zinc-500">
                <th className="px-6 py-3.5">Date</th>
                <th className="px-6 py-3.5 text-right">Total NAV</th>
                <th className="px-6 py-3.5 text-right">Total Cost</th>
                <th className="px-6 py-3.5 text-right">Unrealized PnL</th>
                <th className="px-6 py-3.5 text-right">Cash Balance</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-sky-50 font-mono text-xs">
              {performanceRows.map((row) => {
                const isPos = row.Unrealized_PnL >= 0
                return (
                  <tr key={row.Date} className="hover:bg-sky-50/50 transition-colors">
                    <td className="px-6 py-4 font-bold text-zinc-900">{row.Date}</td>
                    <td className="px-6 py-4 text-right font-bold text-zinc-900">{formatTHB(row.Total_NAV)}</td>
                    <td className="px-6 py-4 text-right text-zinc-600">{formatTHB(row.Total_Cost)}</td>
                    <td className={`px-6 py-4 text-right font-bold ${isPos ? 'text-emerald-600' : 'text-rose-600'}`}>
                      {isPos ? '+' : ''}
                      {formatTHB(row.Unrealized_PnL)}
                    </td>
                    <td className="px-6 py-4 text-right text-zinc-700">{formatTHB(row.Cash_Balance)}</td>
                  </tr>
                )
              })}
              {performanceRows.length === 0 && (
                <tr>
                  <td colSpan={5} className="py-16 text-center font-sans text-zinc-400">
                    ยังไม่มีประวัติการบันทึก NAV Snapshot ในช่วงเวลาที่เลือก
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
