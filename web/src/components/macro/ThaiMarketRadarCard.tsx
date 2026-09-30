import { useEffect, useState, useCallback } from 'react'
import { api } from '../../api/client'
import type {
  ThaiFundFlowDTO,
  ThaiRetailGoldDTO,
  MarketValuationDTO,
  MarketBreadthDTO,
} from '../../api/types'

interface Props {
  className?: string
}

function formatThbMil(val: number | null | undefined): string {
  if (val === null || val === undefined || isNaN(val)) return '—'
  const inMil = val / 1_000_000
  const sign = inMil > 0 ? '+' : ''
  return `${sign}${inMil.toLocaleString('th-TH', { minimumFractionDigits: 1, maximumFractionDigits: 1 })} M`
}

function formatThb(val: number | null | undefined): string {
  if (val === null || val === undefined || isNaN(val)) return '—'
  return val.toLocaleString('th-TH', { minimumFractionDigits: 0, maximumFractionDigits: 0 })
}

export function ThaiMarketRadarCard({ className = '' }: Props) {
  const [flow, setFlow] = useState<ThaiFundFlowDTO | null>(null)
  const [gold, setGold] = useState<ThaiRetailGoldDTO | null>(null)
  const [valuation, setValuation] = useState<MarketValuationDTO | null>(null)
  const [breadth, setBreadth] = useState<MarketBreadthDTO | null>(null)
  const [loading, setLoading] = useState(false)
  const [lastRefreshed, setLastRefreshed] = useState<string | null>(null)
  const [refreshError, setRefreshError] = useState<string | null>(null)

  const fetchThaiMarketData = useCallback(async () => {
    setLoading(true)
    setRefreshError(null)
    try {
      const [f, g, v, b] = await Promise.allSettled([
        api.getThaiInvestorFlow('SET'),
        api.getThaiRetailGold(),
        api.getThaiMarketValuation('SET'),
        api.getThaiMarketBreadth('SET'),
      ])

      if (f.status === 'fulfilled') setFlow(f.value)
      if (g.status === 'fulfilled') setGold(g.value)
      if (v.status === 'fulfilled') setValuation(v.value)
      if (b.status === 'fulfilled') setBreadth(b.value)

      setLastRefreshed(new Date().toLocaleTimeString('th-TH', { hour: '2-digit', minute: '2-digit', second: '2-digit' }))
    } catch (e: any) {
      setRefreshError(e?.message || 'ไม่สามารถโหลดข้อมูลสภาวะตลาดไทยได้')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchThaiMarketData()
  }, [fetchThaiMarketData])

  // Computed metrics
  const foreignRow = flow?.investors?.find((i) => i.investor_type.toLowerCase().includes('foreign') || i.investor_type.includes('ต่างชาติ'))
  const foreignNet = foreignRow?.net_value ?? null
  const instRow = flow?.investors?.find((i) => i.investor_type.toLowerCase().includes('institution') || i.investor_type.includes('สถาบัน'))
  const instNet = instRow?.net_value ?? null
  const propRow = flow?.investors?.find((i) => i.investor_type.toLowerCase().includes('prop') || i.investor_type.includes('บัญชี บล.'))
  const propNet = propRow?.net_value ?? null
  const retailRow = flow?.investors?.find((i) => i.investor_type.toLowerCase().includes('retail') || i.investor_type.includes('รายย่อย'))
  const retailNet = retailRow?.net_value ?? null

  const gainers = breadth?.gainers ?? 0
  const losers = breadth?.losers ?? 0
  const unchanged = breadth?.unchanged ?? 0
  const totalBreadth = gainers + losers + unchanged
  const adRatio = losers > 0 ? (gainers / losers).toFixed(2) : '—'

  return (
    <div className={`space-y-4 rounded-2xl border border-sky-100 bg-gradient-to-br from-white/95 via-sky-50/40 to-emerald-50/30 p-6 shadow-[0_8px_30px_rgba(14,165,233,0.06)] backdrop-blur-sm ${className}`}>
      {/* Header Bar */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between border-b border-sky-100/80 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xl">🇹🇭</span>
            <h2 className="text-base font-bold tracking-tight text-zinc-900">
              Thailand Market Radar & Stance
            </h2>
            <span className="rounded-full bg-emerald-100 px-2.5 py-0.5 font-mono text-[10px] font-bold text-emerald-800">
              TERMINAL V2
            </span>
          </div>
          <p className="mt-0.5 text-xs text-zinc-500">
            SET & mai Market Microstructure, Foreign Investor Flow, Market Breadth & GTA Physical Gold
          </p>
        </div>

        <div className="flex items-center gap-2">
          {lastRefreshed && (
            <span className="text-[11px] text-zinc-400">
              อัปเดตเมื่อ {lastRefreshed}
            </span>
          )}
          <button
            onClick={fetchThaiMarketData}
            disabled={loading}
            className="flex items-center gap-1.5 rounded-lg border border-sky-200 bg-white px-3 py-1.5 text-xs font-semibold text-sky-700 shadow-sm transition-all hover:bg-sky-50 disabled:opacity-50"
            title="รีเฟรชข้อมูลตลาดไทยสดจาก Terminal V2"
          >
            <span>{loading ? '⏳' : '🔄'}</span>
            <span>{loading ? 'กำลังดึงข้อมูล...' : 'รีเฟรชสด'}</span>
          </button>
        </div>
      </div>

      {/* Dual-Track Fail-Closed Data Gap Notice */}
      <div className="rounded-xl border border-amber-200/80 bg-amber-50/70 p-3.5 text-xs">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <span className="rounded-md border border-amber-300 bg-amber-100 px-2 py-0.5 font-mono text-[10px] font-bold uppercase text-amber-900">
              Thailand Macro Regime: UNKNOWN
            </span>
            <span className="text-zinc-500 font-medium">Confidence: 0.0% (Fail-Closed Policy)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-[11px] font-medium text-amber-900">Data Gaps:</span>
            <span className="rounded bg-amber-200/70 px-1.5 py-0.5 font-mono text-[10px] text-amber-950">
              NESDC Real GDP YoY
            </span>
            <span className="rounded bg-amber-200/70 px-1.5 py-0.5 font-mono text-[10px] text-amber-950">
              MOC TPSO CPI YoY
            </span>
          </div>
        </div>
        <p className="mt-1.5 leading-relaxed text-amber-900/90 text-[11px]">
          ตามข้อกำหนด Dual-Track Revision 5 ข้อมูลฮาร์ดดาต้าเศรษฐกิจไทยอยู่ระหว่างการเชื่อมต่อ API ทางการ เพื่อไม่ให้โมเดลสร้างค่าจำลอง (Mock/Hallucination) สภาวะเศรษฐกิจไทยจึงถูกตั้งค่าเป็น UNKNOWN และการประเมินสภาวะตลาดในประเทศจะอ้างอิงจาก Microstructure จริงของ Terminal V2 ด้านล่างเท่านั้น
        </p>
      </div>

      {refreshError && (
        <div className="rounded-lg bg-red-50 p-2.5 text-xs text-red-700">
          ⚠️ {refreshError}
        </div>
      )}

      {/* 4 Dimension Metrics Grid */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {/* Panel 1: 4-Investor Flow */}
        <div className="rounded-xl border border-edge bg-white/80 p-4 shadow-sm">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-bold uppercase tracking-wider text-zinc-500">SET Flow (4 กลุ่ม)</span>
            <span className="text-[10px] font-medium text-zinc-400">Settrade ({flow?.as_of || 'T-0'})</span>
          </div>

          <div className="mt-3">
            <div className="text-[11px] text-zinc-500">Foreign Net Flow (ต่างชาติ)</div>
            <div className={`font-mono text-xl font-extrabold ${foreignNet !== null && foreignNet >= 0 ? 'text-emerald-600' : 'text-red-600'}`}>
              {formatThbMil(foreignNet)}
            </div>
          </div>

          <div className="mt-3 space-y-1.5 border-t border-zinc-100 pt-2 text-[11px]">
            <div className="flex justify-between">
              <span className="text-zinc-600">สถาบันในประเทศ:</span>
              <span className={`font-mono font-semibold ${instNet !== null && instNet >= 0 ? 'text-emerald-600' : 'text-red-600'}`}>
                {formatThbMil(instNet)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-zinc-600">บัญชี บล. (Prop):</span>
              <span className={`font-mono font-semibold ${propNet !== null && propNet >= 0 ? 'text-emerald-600' : 'text-red-600'}`}>
                {formatThbMil(propNet)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-zinc-600">รายย่อย (Retail):</span>
              <span className={`font-mono font-semibold ${retailNet !== null && retailNet >= 0 ? 'text-emerald-600' : 'text-red-600'}`}>
                {formatThbMil(retailNet)}
              </span>
            </div>
          </div>
        </div>

        {/* Panel 2: Market Breadth */}
        <div className="rounded-xl border border-edge bg-white/80 p-4 shadow-sm">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-bold uppercase tracking-wider text-zinc-500">Market Breadth</span>
            <span className="text-[10px] font-medium text-zinc-400">Settrade ({breadth?.as_of || 'T-0'})</span>
          </div>

          <div className="mt-3">
            <div className="text-[11px] text-zinc-500">A/D Ratio (Advance / Decline)</div>
            <div className="flex items-baseline gap-2">
              <span className="font-mono text-xl font-extrabold text-zinc-900">{adRatio}x</span>
              <span className={`rounded px-1.5 py-0.5 text-[10px] font-bold uppercase ${
                losers > 0 && gainers / losers >= 1.2
                  ? 'bg-emerald-100 text-emerald-800'
                  : losers > 0 && gainers / losers <= 0.8
                  ? 'bg-red-100 text-red-800'
                  : 'bg-zinc-100 text-zinc-700'
              }`}>
                {losers > 0 && gainers / losers >= 1.2 ? 'Bullish' : losers > 0 && gainers / losers <= 0.8 ? 'Bearish' : 'Neutral'}
              </span>
            </div>
          </div>

          {/* Visual Distribution Bar */}
          {totalBreadth > 0 && (
            <div className="mt-3">
              <div className="h-2 w-full overflow-hidden rounded-full bg-zinc-100 flex">
                <div style={{ width: `${(gainers / totalBreadth) * 100}%` }} className="bg-emerald-500" title={`บวก: ${gainers}`} />
                <div style={{ width: `${(unchanged / totalBreadth) * 100}%` }} className="bg-zinc-300" title={`เสมอ: ${unchanged}`} />
                <div style={{ width: `${(losers / totalBreadth) * 100}%` }} className="bg-red-500" title={`ลบ: ${losers}`} />
              </div>
            </div>
          )}

          <div className="mt-3 grid grid-cols-3 gap-1 border-t border-zinc-100 pt-2 text-center text-[11px]">
            <div>
              <div className="text-emerald-700 font-bold font-mono">{gainers}</div>
              <div className="text-[10px] text-zinc-400">บวก</div>
            </div>
            <div>
              <div className="text-zinc-600 font-bold font-mono">{unchanged}</div>
              <div className="text-[10px] text-zinc-400">เสมอ</div>
            </div>
            <div>
              <div className="text-red-700 font-bold font-mono">{losers}</div>
              <div className="text-[10px] text-zinc-400">ลบ</div>
            </div>
          </div>
        </div>

        {/* Panel 3: SET Valuation */}
        <div className="rounded-xl border border-edge bg-white/80 p-4 shadow-sm">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-bold uppercase tracking-wider text-zinc-500">SET Valuation</span>
            <span className="text-[10px] font-medium text-zinc-400">Settrade ({valuation?.as_of || 'T-0'})</span>
          </div>

          <div className="mt-3">
            <div className="text-[11px] text-zinc-500">SET P/E Ratio</div>
            <div className="flex items-baseline gap-2">
              <span className="font-mono text-xl font-extrabold text-zinc-900">
                {valuation?.pe_ratio !== null && valuation?.pe_ratio !== undefined ? `${valuation.pe_ratio.toFixed(2)}x` : '—'}
              </span>
              {valuation?.pe_ratio && (
                <span className={`rounded px-1.5 py-0.5 text-[10px] font-bold ${
                  valuation.pe_ratio < 15 ? 'bg-emerald-100 text-emerald-800' : valuation.pe_ratio > 18 ? 'bg-amber-100 text-amber-800' : 'bg-zinc-100 text-zinc-700'
                }`}>
                  {valuation.pe_ratio < 15 ? 'Undervalued' : valuation.pe_ratio > 18 ? 'Premium' : 'Fair'}
                </span>
              )}
            </div>
          </div>

          <div className="mt-3 space-y-1.5 border-t border-zinc-100 pt-2 text-[11px]">
            <div className="flex justify-between">
              <span className="text-zinc-600">P/BV Ratio:</span>
              <span className="font-mono font-semibold text-zinc-900">
                {valuation?.pbv_ratio !== null && valuation?.pbv_ratio !== undefined ? `${valuation.pbv_ratio.toFixed(2)}x` : '—'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-zinc-600">Dividend Yield:</span>
              <span className="font-mono font-semibold text-emerald-700">
                {valuation?.dividend_yield !== null && valuation?.dividend_yield !== undefined ? `${valuation.dividend_yield.toFixed(2)}%` : '—'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-zinc-600">Turnover Ratio:</span>
              <span className="font-mono font-semibold text-zinc-700">
                {valuation?.turnover_ratio !== null && valuation?.turnover_ratio !== undefined ? `${valuation.turnover_ratio.toFixed(2)}%` : '—'}
              </span>
            </div>
          </div>
        </div>

        {/* Panel 4: GTA Retail Physical Gold */}
        <div className="rounded-xl border border-edge bg-white/80 p-4 shadow-sm">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-bold uppercase tracking-wider text-amber-700">GTA Physical Gold</span>
            <span className="text-[10px] font-medium text-zinc-400">
              {gold?.revision ? `รอบที่ ${gold.revision}` : 'GTA Thailand'}
            </span>
          </div>

          <div className="mt-3">
            <div className="text-[11px] text-zinc-500">ทองคำแท่ง 96.5% (ขายออก)</div>
            <div className="font-mono text-xl font-extrabold text-amber-700">
              {formatThb(gold?.bar?.sell)} <span className="text-xs font-normal text-zinc-500">THB/บาท</span>
            </div>
          </div>

          <div className="mt-3 space-y-1.5 border-t border-zinc-100 pt-2 text-[11px]">
            <div className="flex justify-between">
              <span className="text-zinc-600">ทองคำแท่ง (รับซื้อ):</span>
              <span className="font-mono font-semibold text-zinc-900">
                {formatThb(gold?.bar?.buy)} THB
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-zinc-600">ทองรูปพรรณ (ขายออก):</span>
              <span className="font-mono font-semibold text-zinc-900">
                {formatThb(gold?.ornament?.sell)} THB
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-zinc-600">ประกาศเมื่อ:</span>
              <span className="font-mono text-[10px] text-zinc-500">
                {gold?.announced_at || 'T-0'}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
