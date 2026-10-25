import type { FundConstituentsDTO } from '../../api/types'
import { useFocusTrap } from '../../hooks/useFocusTrap'
import type { WatchlistDraft } from './Modals/WatchlistModal'

export interface AvailableFundItem {
  symbol: string
  name?: string
  asset_type?: string
}

interface FundLookThroughDrawerProps {
  isOpen: boolean
  onClose: () => void
  selectedFundSymbol: string | null
  onSelectFund: (symbol: string) => void
  availableFunds: AvailableFundItem[]
  constituentsData?: FundConstituentsDTO | null
  loading?: boolean
  onQuickAddToWatchlist?: (draft: WatchlistDraft) => void
  directHoldingsSymbols?: string[]
  watchlistedSymbols?: string[]
}

export default function FundLookThroughDrawer({
  isOpen,
  onClose,
  selectedFundSymbol,
  onSelectFund,
  availableFunds,
  constituentsData,
  loading = false,
  onQuickAddToWatchlist,
  directHoldingsSymbols,
  watchlistedSymbols,
}: FundLookThroughDrawerProps) {
  const panelRef = useFocusTrap<HTMLDivElement>(isOpen, onClose)

  if (!isOpen) return null

  const topHoldings = constituentsData?.top_holdings || []
  const sectors = constituentsData?.sector_breakdown || []
  const alloc = constituentsData?.asset_allocation || {}

  return (
    <div className="fixed inset-0 z-50 overflow-hidden" role="region" aria-label="Fund Constituents Look-Through Drawer">
      {/* Backdrop */}
      <div
        aria-hidden="true"
        className="animate-fade-in fixed inset-0 bg-black/40 backdrop-blur-xs transition-opacity"
        onClick={onClose}
      />

      {/* Slide-over Container */}
      <div className="fixed inset-y-0 right-0 flex max-w-full pl-6 sm:pl-10">
        <div
          ref={panelRef}
          role="dialog"
          aria-modal="true"
          aria-labelledby="fund-look-through-drawer-title"
          className="animate-drawer-in flex w-screen max-w-xl flex-col bg-white shadow-2xl border-l border-zinc-200"
        >
          {/* 1. Header (Bloomberg Obsidian Style) */}
          <div className="flex items-start justify-between border-b border-zinc-800 bg-zinc-950 px-6 py-4 text-white shrink-0">
            <div className="space-y-1">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-purple-500/20 text-purple-300 text-sm border border-purple-500/30">
                  🔍
                </span>
                <h3 id="fund-look-through-drawer-title" className="font-mono font-extrabold text-sm sm:text-base tracking-tight text-white">
                  Glass-Box Look-Through: ไส้ในกองทุน [{selectedFundSymbol}]
                </h3>
              </div>
              <p className="text-xs text-zinc-400 font-medium">
                {constituentsData?.symbol === selectedFundSymbol && constituentsData?.as_of_date
                  ? `ข้อมูลส่องทะลุ ณ วันที่: ${constituentsData.as_of_date}`
                  : 'เจาะลึกสินทรัพย์ที่กองทุนถือครอง (Underlying Holdings & Sectors)'}
              </p>
            </div>
            <button
              type="button"
              onClick={onClose}
              className="rounded-lg bg-zinc-900 px-3 py-1.5 text-xs font-medium text-zinc-400 hover:text-white hover:bg-zinc-800 border border-zinc-800 transition-colors"
              title="ปิดลิ้นชัก (Esc)"
            >
              ✕ ปิด
            </button>
          </div>

          {/* 2. Quick Switcher Bar (Switch between funds without closing drawer) */}
          {availableFunds.length > 1 && (
            <div className="border-b border-zinc-200 bg-zinc-50 px-6 py-2.5 flex items-center gap-2 overflow-x-auto custom-scrollbar shrink-0">
              <span className="text-[11px] font-bold text-zinc-500 shrink-0">กองทุนอื่นในพอร์ต:</span>
              <div className="flex items-center gap-1.5">
                {availableFunds.map((fund) => {
                  const isActive = fund.symbol === selectedFundSymbol
                  return (
                    <button
                      key={fund.symbol}
                      type="button"
                      onClick={() => onSelectFund(fund.symbol)}
                      className={`px-2.5 py-1 rounded-lg text-xs font-mono font-semibold transition-all whitespace-nowrap ${
                        isActive
                          ? 'bg-purple-600 text-white shadow-xs'
                          : 'bg-white hover:bg-zinc-100 text-zinc-700 border border-zinc-200'
                      }`}
                    >
                      {fund.symbol}
                    </button>
                  )
                })}
              </div>
            </div>
          )}

          {/* 3. Drawer Body (Scrollable) */}
          <div className="flex-1 overflow-y-auto custom-scrollbar p-6 space-y-5 text-left bg-white">
            {loading ? (
              <div className="py-16 flex flex-col items-center justify-center gap-3 text-slate-500">
                <span className="animate-spin text-3xl">⏳</span>
                <span className="text-xs font-medium">
                  กำลังดึงข้อมูลไส้ในกองทุน {selectedFundSymbol} และประมวลผล 2-Tier Master Catalog...
                </span>
              </div>
            ) : constituentsData ? (
              <>
                {/* Master Fund Info Hero Card */}
                {constituentsData.master_fund_name && (
                  <div className="rounded-xl border border-indigo-200/90 bg-gradient-to-r from-indigo-50/80 via-white to-purple-50/60 p-4 shadow-2xs space-y-1.5">
                    <div className="flex items-center justify-between">
                      <span className="text-[11px] font-bold uppercase tracking-wider text-indigo-700 flex items-center gap-1">
                        <span>🌐</span> กองทุนหลักต่างประเทศ (Master Fund)
                      </span>
                      {constituentsData.master_fund_ticker && (
                        <span className="font-mono text-xs font-extrabold px-2 py-0.5 rounded bg-indigo-600 text-white shadow-2xs">
                          {constituentsData.master_fund_ticker}
                        </span>
                      )}
                    </div>
                    <div className="font-bold text-sm text-slate-900">
                      {constituentsData.master_fund_name}
                    </div>
                    <p className="text-[11px] text-slate-500">
                      ส่องทะลุผ่าน Feeder Fund โดยคำนวณสัดส่วนถือครองตามน้ำหนักการลงทุนจริงของ Master ETF
                    </p>
                  </div>
                )}

                {/* Top 10 Holdings Progress Bars */}
                <div className="space-y-3">
                  <div className="flex items-center justify-between pb-1 border-b border-zinc-200">
                    <span className="text-xs font-bold text-zinc-900 flex items-center gap-1.5">
                      <span>🏆</span>
                      <span>Top Holdings ({topHoldings.length} รายการแรก)</span>
                    </span>
                    <span className="text-[11px] text-zinc-400 font-mono">สัดส่วนในกองทุน</span>
                  </div>

                  {topHoldings.length === 0 ? (
                    <div className="p-4 rounded-xl bg-zinc-50 border border-zinc-200 text-xs text-zinc-500 text-center">
                      กองทุนนี้ไม่มีข้อมูลหุ้นรายตัวโดยตรง (แสดงสัดส่วนระดับ Sector และ Asset Allocation)
                    </div>
                  ) : (
                    <div className="space-y-2.5">
                      {topHoldings.slice(0, 10).map((item, idx) => {
                        const pctVal = item.percent ?? 0
                        return (
                          <div key={idx} className="space-y-1">
                            <div className="flex items-center justify-between text-xs">
                              <div className="flex items-center gap-2 max-w-[75%]">
                                <span className="font-mono text-[10px] text-zinc-400 w-4 text-right shrink-0">
                                  {idx + 1}.
                                </span>
                                {item.ticker && (
                                  <span className="font-mono font-extrabold text-flow-blue px-1.5 py-0.2 rounded bg-sky-50 border border-sky-200 text-[11px] shrink-0">
                                    {item.ticker}
                                  </span>
                                )}
                                <span className="font-medium text-zinc-800 truncate" title={item.name}>
                                  {item.name}
                                </span>
                              </div>
                              <div className="flex items-center gap-2 shrink-0">
                                {item.ticker && (
                                  <>
                                    {directHoldingsSymbols?.includes(item.ticker) ? (
                                      <span className="text-[10px] font-bold text-blue-600 bg-blue-50 px-1.5 py-0.5 rounded border border-blue-200">
                                        ถือตรง
                                      </span>
                                    ) : watchlistedSymbols?.includes(item.ticker) ? (
                                      <span className="text-[10px] font-bold text-amber-600 bg-amber-50 px-1.5 py-0.5 rounded border border-amber-200">
                                        ใน Watchlist
                                      </span>
                                    ) : onQuickAddToWatchlist ? (
                                      <button
                                        type="button"
                                        onClick={() =>
                                          onQuickAddToWatchlist({
                                            symbol: item.ticker!,
                                            asset_type: 'Stock',
                                            suggested_bucket_id: 'core_equities',
                                            notes: `From ${selectedFundSymbol || 'Fund'} look-through (${pctVal.toFixed(2)}%)`,
                                            source_refs: [selectedFundSymbol ? `fund:${selectedFundSymbol}` : 'fund:unknown'],
                                          })
                                        }
                                        className="text-[10px] font-bold text-flow-blue hover:text-sky-700 bg-sky-50 hover:bg-sky-100 px-1.5 py-0.5 rounded border border-sky-200 transition-colors"
                                        title={`เพิ่ม ${item.ticker} เข้า Watchlist`}
                                      >
                                        + Watchlist
                                      </button>
                                    ) : null}
                                  </>
                                )}
                                <span className="font-mono font-bold text-zinc-900">
                                  {pctVal.toFixed(2)}%
                                </span>
                              </div>
                            </div>
                            <div className="w-full bg-zinc-100 rounded-full h-1.5 overflow-hidden">
                              <div
                                className="bg-gradient-to-r from-sky-500 via-indigo-500 to-purple-600 h-1.5 rounded-full transition-all duration-300"
                                style={{ width: `${Math.min(pctVal * 3.5, 100)}%` }}
                              />
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  )}
                </div>

                {/* Sector Breakdown */}
                <div className="space-y-2.5 pt-2 border-t border-zinc-200">
                  <div className="text-xs font-bold text-zinc-900 flex items-center gap-1.5">
                    <span>🏭</span>
                    <span>สัดส่วนตามกลุ่มอุตสาหกรรม (Sector Breakdown)</span>
                  </div>
                  {sectors.length === 0 ? (
                    <div className="p-3 rounded-xl bg-zinc-50 border border-zinc-200 text-xs text-zinc-400">
                      ไม่มีข้อมูล Sector Breakdown
                    </div>
                  ) : (
                    <div className="flex flex-wrap gap-1.5">
                      {sectors.map((sec, sIdx) => (
                        <div
                          key={sIdx}
                          className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs bg-zinc-50 border border-zinc-200 text-zinc-800 shadow-2xs"
                        >
                          <span className="font-medium text-zinc-700">{sec.sector_name_en}</span>
                          <span className="font-mono text-purple-700 font-bold">
                            {(sec.percent ?? 0).toFixed(1)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Asset Allocation */}
                {Object.keys(alloc).length > 0 && (
                  <div className="space-y-2.5 pt-2 border-t border-zinc-200">
                    <div className="text-xs font-bold text-zinc-900 flex items-center gap-1.5">
                      <span>📊</span>
                      <span>ประเภทสินทรัพย์ (Asset Allocation)</span>
                    </div>
                    <div className="flex flex-wrap gap-2 text-xs">
                      {Object.entries(alloc).map(([assetKey, pctVal], aIdx) => (
                        <div
                          key={aIdx}
                          className="px-2.5 py-1 rounded-md bg-purple-50 text-purple-900 border border-purple-100 flex items-center gap-1.5 font-medium"
                        >
                          <span className="capitalize">{assetKey}:</span>
                          <span className="font-mono font-bold text-purple-700">
                            {Number(pctVal).toFixed(1)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div className="py-12 text-center text-xs text-zinc-400 space-y-2">
                <span className="text-2xl block">🔍</span>
                <p>ไม่พบข้อมูลไส้ในของกองทุน {selectedFundSymbol}</p>
                <p className="text-[11px] text-zinc-500">
                  อาจเป็นกองทุนใหม่ หรือยังไม่มีรายงานพอร์ตล่าสุดจาก Finnomena API
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
