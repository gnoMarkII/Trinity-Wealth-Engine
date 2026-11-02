import React from 'react'
import { ContextTooltip } from './explainer/ContextTooltip'

export interface TraderClassUI {
  class_name: string
  long_contracts: number
  short_contracts: number
  net_contracts: number
  change_long?: number
  change_short?: number
  pct_of_oi_long?: number
  pct_of_oi_short?: number
}

export interface MetalsCotCardProps {
  commodity: string
  commodityCode: string
  asOfDate: string // Tuesday
  publishedAt: string // Friday
  openInterest: number
  netManagedMoney: number
  percentile52w: number
  managedMoney: TraderClassUI
  swapDealers: TraderClassUI
  producerMerchant: TraderClassUI
  otherReportables?: TraderClassUI
  className?: string
}

export const MetalsCotCard: React.FC<MetalsCotCardProps> = ({
  commodity,
  asOfDate,
  publishedAt,
  openInterest,
  netManagedMoney,
  percentile52w = 50,
  managedMoney,
  swapDealers,
  producerMerchant,
  className = '',
}) => {
  const safeNetMm = typeof netManagedMoney === 'number' && !isNaN(netManagedMoney) ? netManagedMoney : 0
  const isMmBullish = safeNetMm > 0
  const safePercentile = typeof percentile52w === 'number' && !isNaN(percentile52w) ? percentile52w : 0
  const safeOi = typeof openInterest === 'number' && !isNaN(openInterest) ? openInterest : 0

  return (
    <div
      className={`rounded-2xl border border-sky-100 bg-white/80 p-5 shadow-[0_8px_25px_rgba(14,165,233,0.06)] backdrop-blur-md ${className}`}
    >
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-sky-100/60 pb-3">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="font-semibold text-zinc-900 tracking-tight">
              {commodity.toUpperCase()} Futures Positioning (CFTC COT)
            </h3>
            <ContextTooltip
              title="เรดาร์ส่องไพ่เฮดจ์ฟันด์โลก (CFTC COT Report)"
              whatIsIt="รายงานทางการสหรัฐฯ เปิดเผยสถานะสัญญา期貨 (Futures) ของกองทุนเก็งกำไรระดับโลกในทองคำ"
              howToRead="ดู Net Managed Money: ฝั่ง Long ชนะ แปลว่ากองทุนมองขึ้น ถ้า Percentile 52w เกิน 90% แปลว่าเปิด Long แน่นมาก อาจใกล้จุดกลับตัว"
              impactOnMoney="ช่วยให้เราไม่ไปไล่ซื้อทองคำตอนที่กองทุนทั่วโลกกำลังจ้องทำกำไร"
            />
            <span className="rounded-full bg-amber-100 px-2 py-0.5 font-mono text-[10px] font-bold text-amber-800">
              DISAGGREGATED
            </span>
          </div>
          <p className="text-xs text-zinc-500 mt-0.5">
            CME / NYMEX Futures & Options • Weekly Hedge Fund / Speculative Flows
          </p>
          <div className="mt-2 text-[11px] text-amber-950 bg-amber-50/80 rounded-lg p-2 border border-amber-100/70 font-medium leading-relaxed">
            👁️ <strong>เรดาร์ส่องไพ่กองทุนเฮดจ์ฟันด์โลก:</strong> ตรวจสอบว่าเงินฉลาด (Smart Money) กำลังวางเงินเก็งกำไรฝั่งซื้อหรือขายสุทธิในทองคำ
          </div>
        </div>

        {/* Date Separation (Tuesday as_of vs Friday published) */}
        <div className="text-right">
          <div className="flex items-center gap-2">
            <span className="rounded-md bg-slate-100 px-2 py-0.5 font-mono text-[11px] font-semibold text-zinc-700">
              As of: {asOfDate} (Tue)
            </span>
            <span className="rounded-md bg-sky-50 px-2 py-0.5 font-mono text-[11px] font-semibold text-sky-700">
              Pub: {publishedAt} (Fri)
            </span>
          </div>
        </div>
      </div>

      {/* Semantic Guidance Banner */}
      <div className="mt-3 flex items-start gap-2 rounded-lg bg-amber-50/60 p-2 text-[11px] text-amber-800 border border-amber-200/50">
        <span className="font-bold text-amber-600 shrink-0">ℹ Disaggregated Contract:</span>
        <span>
          Reports CME futures/options contracts, not physical vault reserves. Managed Money (CTAs & Hedge
          Funds) is tracked separately from Swap Dealers. There is no generic &ldquo;commercials&rdquo; bucket.
        </span>
      </div>

      {/* Key Metric Highlights */}
      <div className="mt-4 grid grid-cols-1 sm:grid-cols-3 gap-3">
        {/* Net Managed Money */}
        <div className="rounded-xl bg-slate-50/80 p-3 border border-slate-100">
          <span className="text-[10px] uppercase font-bold text-zinc-400">
            Managed Money Net Positioning
          </span>
          <p
            className={`font-mono text-xl font-extrabold mt-0.5 ${
              isMmBullish ? 'text-emerald-600' : 'text-rose-600'
            }`}
          >
            {isMmBullish ? `+${safeNetMm.toLocaleString()}` : safeNetMm.toLocaleString()}
          </p>
          <span className="text-[11px] text-zinc-500">
            Longs {(managedMoney?.long_contracts ?? 0).toLocaleString()} • Shorts{' '}
            {(managedMoney?.short_contracts ?? 0).toLocaleString()}
          </span>
        </div>

        {/* 52-Week Percentile */}
        <div className="rounded-xl bg-slate-50/80 p-3 border border-slate-100">
          <span className="text-[10px] uppercase font-bold text-zinc-400">
            52-Week Percentile Rank
          </span>
          <p className="font-mono text-xl font-extrabold text-sky-800 mt-0.5">
            {safePercentile.toFixed(1)}%
          </p>
          <div className="mt-1 h-2 w-full overflow-hidden rounded-full bg-slate-200">
            <div
              className="h-full rounded-full bg-sky-600 transition-all duration-500"
              style={{ width: `${Math.min(100, Math.max(0, safePercentile))}%` }}
            />
          </div>
        </div>

        {/* Total Open Interest */}
        <div className="rounded-xl bg-slate-50/80 p-3 border border-slate-100">
          <span className="text-[10px] uppercase font-bold text-zinc-400">Total Open Interest</span>
          <p className="font-mono text-xl font-extrabold text-zinc-900 mt-0.5">
            {safeOi.toLocaleString()}
          </p>
          <span className="text-[11px] text-zinc-500">Aggregate CME Market Contracts</span>
        </div>
      </div>

      {/* Disaggregated Breakdown Table */}
      <div className="mt-4">
        <h4 className="text-xs font-bold uppercase tracking-wider text-zinc-600 mb-2">
          Disaggregated Trader Classes
        </h4>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-100 text-[10px] uppercase font-bold text-zinc-400">
                <th className="pb-1.5 font-semibold">Trader Category</th>
                <th className="pb-1.5 font-semibold text-right">Longs</th>
                <th className="pb-1.5 font-semibold text-right">Shorts</th>
                <th className="pb-1.5 font-semibold text-right">Net Contracts</th>
                <th className="pb-1.5 font-semibold text-right">Chg Long</th>
                <th className="pb-1.5 font-semibold text-right">Chg Short</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-mono">
              {[
                managedMoney || { class_name: 'Managed Money', long_contracts: 0, short_contracts: 0, net_contracts: 0 },
                swapDealers || { class_name: 'Swap Dealers', long_contracts: 0, short_contracts: 0, net_contracts: 0 },
                producerMerchant || { class_name: 'Producer / Merchant', long_contracts: 0, short_contracts: 0, net_contracts: 0 },
              ].map((cls) => {
                const net = cls?.net_contracts ?? 0
                const isNetPos = net >= 0
                return (
                  <tr key={cls.class_name} className="hover:bg-slate-50/60 transition-colors">
                    <td className="py-2 text-zinc-800 font-medium font-sans">{cls.class_name}</td>
                    <td className="py-2 text-right text-zinc-700">
                      {(cls.long_contracts ?? 0).toLocaleString()}
                    </td>
                    <td className="py-2 text-right text-zinc-700">
                      {(cls.short_contracts ?? 0).toLocaleString()}
                    </td>
                    <td
                      className={`py-2 text-right font-bold ${
                        isNetPos ? 'text-emerald-600' : 'text-rose-600'
                      }`}
                    >
                      {isNetPos ? `+${net.toLocaleString()}` : net.toLocaleString()}
                    </td>
                    <td className="py-2 text-right text-zinc-500">
                      {cls.change_long !== undefined ? cls.change_long.toLocaleString() : '—'}
                    </td>
                    <td className="py-2 text-right text-zinc-500">
                      {cls.change_short !== undefined ? cls.change_short.toLocaleString() : '—'}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
