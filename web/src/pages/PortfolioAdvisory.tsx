import { useState, useEffect, useCallback, useRef } from 'react'
import { api } from '../api/client'
import type {
  PortfolioAdvisoryResponseDTO,
  PortfolioMetaDTO,
  HoldingAdvisoryDTO,
  LotSizingDTO,
} from '../api/types'
import {
  PortfolioAdvisorySkeleton,
  AiPlaybookSynthesisLoader,
  TopLoadingRibbon,
} from '../components/ui/FlowSkeleton'

export default function PortfolioAdvisory() {
  const [selectedPortfolioId, setSelectedPortfolioId] = useState<string>(() => {
    if (typeof window !== 'undefined') {
      return new URLSearchParams(window.location.search).get('portfolio_id') || 'default'
    }
    return 'default'
  })
  const [portfolios, setPortfolios] = useState<PortfolioMetaDTO[]>([])
  const [data, setData] = useState<PortfolioAdvisoryResponseDTO | null>(null)
  const [loading, setLoading] = useState<boolean>(true)
  const [refreshing, setRefreshing] = useState<boolean>(false)
  const [synthesizing, setSynthesizing] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)
  const [filterAction, setFilterAction] = useState<string>('ALL')

  const fetchIdRef = useRef(0)
  const selectedPortfolioIdRef = useRef(selectedPortfolioId)
  useEffect(() => {
    selectedPortfolioIdRef.current = selectedPortfolioId
  }, [selectedPortfolioId])

  useEffect(() => {
    if (typeof api.listPortfolios === 'function') {
      api
        .listPortfolios()
        .then((list) => {
          if (Array.isArray(list)) setPortfolios(list)
        })
        .catch(() => {})
    }
  }, [])

  const fetchAdvisory = useCallback(
    async (refreshPrices = false) => {
      const myFetchId = ++fetchIdRef.current
      try {
        if (refreshPrices) setRefreshing(true)
        else setLoading(true)
        setError(null)
        const res =
          selectedPortfolioId === 'default'
            ? await api.getPortfolioAdvisory(refreshPrices)
            : await api.getPortfolioAdvisory(refreshPrices, selectedPortfolioId)
        if (fetchIdRef.current !== myFetchId) return
        setData(res)
      } catch (err: any) {
        if (fetchIdRef.current !== myFetchId) return
        setError(err?.message || 'ไม่สามารถโหลดข้อมูล Portfolio Advisory ได้')
      } finally {
        if (fetchIdRef.current === myFetchId) {
          setLoading(false)
          setRefreshing(false)
        }
      }
    },
    [selectedPortfolioId]
  )

  useEffect(() => {
    fetchAdvisory(false)
  }, [fetchAdvisory])

  const handleSynthesize = async () => {
    const currentPortfolioId = selectedPortfolioId
    try {
      setSynthesizing(true)
      setError(null)
      const narrative =
        currentPortfolioId === 'default'
          ? await api.synthesizeAdvisoryPlaybook(true)
          : await api.synthesizeAdvisoryPlaybook(true, currentPortfolioId)
      if (selectedPortfolioIdRef.current !== currentPortfolioId) return
      setData((prev) => (prev ? { ...prev, cached_narrative: narrative } : prev))
    } catch (err: any) {
      if (selectedPortfolioIdRef.current === currentPortfolioId) {
        setError(err?.message || 'การประมวลผล AI Playbook ขัดข้อง กรุณาลองใหม่อีกครั้ง')
      }
    } finally {
      if (selectedPortfolioIdRef.current === currentPortfolioId) {
        setSynthesizing(false)
      }
    }
  }

  if (loading) {
    return (
      <div className="p-4 sm:p-6 lg:p-8 max-w-7xl mx-auto">
        <PortfolioAdvisorySkeleton />
      </div>
    )
  }

  if (error && !data) {
    return (
      <div className="p-8 max-w-4xl mx-auto text-center space-y-4">
        <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-rose-100 text-rose-600">
          ⚠️
        </div>
        <h3 className="text-lg font-semibold text-slate-800">เกิดข้อผิดพลาดในการโหลดข้อมูล</h3>
        <p className="text-sm text-slate-600">{error}</p>
        <button
          onClick={() => fetchAdvisory(false)}
          className="px-4 py-2 rounded-xl bg-sky-600 text-white text-sm font-medium hover:bg-sky-700 transition"
        >
          ลองใหม่อีกครั้ง
        </button>
      </div>
    )
  }

  if (!data) return null

  const immediateHoldings = data.holdings_advisory.filter((h) => h.is_immediate_execution)
  const conditionalHoldings = data.holdings_advisory.filter(
    (h) => !h.is_immediate_execution && h.action_code === 'ACCUMULATE_ON_DIP'
  )

  const filteredHoldings =
    filterAction === 'ALL'
      ? data.holdings_advisory
      : filterAction === 'IMMEDIATE'
      ? immediateHoldings
      : filterAction === 'CONDITIONAL'
      ? conditionalHoldings
      : data.holdings_advisory.filter((h) => h.action_code === filterAction)

  return (
    <div className="p-4 sm:p-6 lg:p-8 max-w-7xl mx-auto space-y-6 animate-fade-in">
      {refreshing && <TopLoadingRibbon label="กำลังอัปเดตราคาตลาดล่าสุดและคำนวณ Guardrails ใหม่..." />}

      {/* 1. Header & Quick Action Bar */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-2 border-b border-edge">
        <div>
          <div className="flex items-center gap-3 flex-wrap">
            <h1 className="text-2xl font-bold tracking-tight text-slate-900 flex items-center gap-2">
              <span>⚡ Portfolio Advisory Desk</span>
            </h1>
            <span
              className={`px-2.5 py-0.5 rounded-full text-xs font-semibold uppercase tracking-wider ${
                data.data_mode === 'FULL_CROSS_SCALE'
                  ? 'bg-emerald-100 text-emerald-800 border border-emerald-200'
                  : data.data_mode === 'PORTFOLIO_PLUS_MICRO'
                  ? 'bg-sky-100 text-sky-800 border border-sky-200'
                  : 'bg-amber-100 text-amber-800 border border-amber-200'
              }`}
            >
              {data.data_mode === 'FULL_CROSS_SCALE'
                ? 'Full Cross-Scale Mode'
                : data.data_mode === 'PORTFOLIO_PLUS_MICRO'
                ? 'Portfolio + Micro Signals'
                : 'Bucket Only Baseline'}
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            คำนวณด้วย Python Deterministic Math Engine (&lt;50ms) • ประเมินเมื่อ:{' '}
            <span className="font-mono">{new Date(data.evaluated_at).toLocaleString('th-TH')}</span>
            {' • '}State Hash: <span className="font-mono text-slate-400">{data.portfolio_state_hash}</span>
          </p>
        </div>

        <div className="flex items-center gap-2.5 flex-wrap">
          {portfolios.length > 0 && (
            <div className="flex items-center gap-1.5 rounded-xl border border-edge bg-panel px-3 py-1.5 shadow-xs">
              <span className="text-xs font-semibold text-slate-600">📁 พอร์ต:</span>
              <select
                value={selectedPortfolioId}
                onChange={(e) => {
                  const nextId = e.target.value
                  setSelectedPortfolioId(nextId)
                  if (typeof window !== 'undefined') {
                    const url = new URL(window.location.href)
                    url.searchParams.set('portfolio_id', nextId)
                    window.history.replaceState({}, '', url.toString())
                  }
                }}
                className="text-xs font-bold text-slate-800 bg-transparent focus:outline-none cursor-pointer"
              >
                {portfolios.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name} {p.is_default ? '(หลัก)' : ''}
                  </option>
                ))}
              </select>
            </div>
          )}

          <button
            onClick={() => fetchAdvisory(true)}
            disabled={refreshing}
            className="px-3.5 py-2 rounded-xl text-xs font-medium border border-edge bg-panel hover:bg-slate-50 text-slate-700 transition flex items-center gap-1.5 shadow-xs disabled:opacity-50"
          >
            <span>{refreshing ? '⏳' : '🔄'}</span>
            <span>รีเฟรชราคาตลาด</span>
          </button>

          <button
            onClick={handleSynthesize}
            disabled={synthesizing}
            className="px-4 py-2 rounded-xl text-xs font-semibold bg-gradient-to-r from-sky-600 via-indigo-600 to-purple-600 hover:from-sky-500 hover:to-purple-500 text-white shadow-md shadow-indigo-500/20 transition flex items-center gap-2 disabled:opacity-50"
          >
            <span>{synthesizing ? '✨' : '🧠'}</span>
            <span>{data.cached_narrative ? 'วิเคราะห์กลยุทธ์ใหม่ (AI Playbook)' : 'ประมวลผล AI Playbook'}</span>
          </button>
        </div>
      </div>

      {/* 2. Anomaly Warnings Banner (Guardrails) */}
      {data.anomaly_warnings.length > 0 && (
        <div className="space-y-2">
          {data.anomaly_warnings.map((w, idx) => (
            <div
              key={idx}
              className={`p-4 rounded-xl border flex items-start gap-3 text-sm ${
                w.severity === 'critical'
                  ? 'bg-rose-50 border-rose-200 text-rose-900'
                  : 'bg-amber-50 border-amber-200 text-amber-900'
              }`}
            >
              <span className="text-lg shrink-0">{w.severity === 'critical' ? '🚨' : '⚠️'}</span>
              <div className="space-y-0.5 flex-1">
                <div className="flex items-center gap-2">
                  <span className="font-bold text-xs uppercase tracking-wide px-2 py-0.5 rounded bg-white/70 border border-current">
                    {w.guardrail_code}
                  </span>
                  {w.ticker && <span className="font-semibold">{w.ticker}</span>}
                </div>
                <p className="text-xs sm:text-sm">{w.message}</p>
                {w.remedy_hint && <p className="text-xs opacity-80 pt-0.5">💡 แนวทางแก้ไข: {w.remedy_hint}</p>}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* 3. Dual-Currency Cash Plan & Risk Radar */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* NAV & Total Cash Card */}
        <div className="flow-panel rounded-2xl border border-edge bg-panel p-5 shadow-xs space-y-3">
          <div className="flex items-center justify-between text-xs text-slate-500">
            <span className="font-medium">Total NAV &amp; FX Rate</span>
            <span className="font-mono">USD/THB: {data.cash_plan.usd_thb_fx_rate.toFixed(2)}</span>
          </div>
          <div>
            <div className="text-2xl font-bold tracking-tight text-slate-900">
              ฿{data.total_nav_thb.toLocaleString('th-TH', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </div>
            <div className="text-xs text-slate-500 mt-0.5">มูลค่าสินทรัพย์สุทธิรวมเงินสด</div>
          </div>
          <div className="pt-2 border-t border-edge/60 flex items-center justify-between text-xs">
            <span className="text-slate-600">เงินสดคงเหลือ:</span>
            <span className="font-semibold text-slate-800">
              ฿{data.cash_plan.cash_thb_available.toLocaleString('th-TH')} | ${data.cash_plan.cash_usd_available.toLocaleString('en-US')}
            </span>
          </div>
        </div>

        {/* Action Budget: Immediate Execution */}
        <div className="flow-panel rounded-2xl border border-sky-100 bg-sky-50/30 p-5 shadow-xs space-y-3">
          <div className="flex items-center justify-between text-xs text-sky-800">
            <span className="font-semibold flex items-center gap-1">
              <span>⚡</span> Immediate Deploy Budget
            </span>
            <span className="text-xs font-mono bg-sky-100 px-2 py-0.5 rounded text-sky-700">
              {immediateHoldings.length} คำสั่งวันนี้
            </span>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <div className="text-xs text-slate-500">งบเคาะซื้อ THB</div>
              <div className="text-lg font-bold text-sky-900">
                ฿{data.cash_plan.immediate_deploy_thb.toLocaleString('th-TH', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </div>
            </div>
            <div>
              <div className="text-xs text-slate-500">งบเคาะซื้อ USD</div>
              <div className="text-lg font-bold text-indigo-900">
                ${data.cash_plan.immediate_deploy_usd.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </div>
            </div>
          </div>
          {data.cash_plan.fx_conversion_needed_thb > 0 && (
            <div className="pt-2 border-t border-sky-200/60 text-xs text-amber-700 font-medium flex items-center gap-1">
              <span>⚠️</span>
              <span>ต้องแลกเงินเพิ่ม ฿{data.cash_plan.fx_conversion_needed_thb.toLocaleString('th-TH')} เข้า USD</span>
            </div>
          )}
        </div>

        {/* Risk Radar & Concentration */}
        <div className="flow-panel rounded-2xl border border-edge bg-panel p-5 shadow-xs space-y-3">
          <div className="flex items-center justify-between text-xs">
            <span className="font-bold text-slate-800 flex items-center gap-1">
              <span>🛡️</span> Risk &amp; Concentration Radar
            </span>
            {data.risk_radar.concentration_risk_level && (
              <span
                className={`text-[10px] font-bold px-2 py-0.5 rounded-full border ${
                  data.risk_radar.concentration_risk_level === 'HIGH'
                    ? 'bg-rose-100 text-rose-800 border-rose-300'
                    : data.risk_radar.concentration_risk_level === 'MODERATE'
                    ? 'bg-amber-100 text-amber-800 border-amber-300'
                    : 'bg-emerald-100 text-emerald-800 border-emerald-300'
                }`}
              >
                {data.risk_radar.concentration_risk_level === 'HIGH'
                  ? 'กระจุกตัวสูง (High Risk)'
                  : data.risk_radar.concentration_risk_level === 'MODERATE'
                  ? 'กระจุกตัวปานกลาง'
                  : 'กระจายตัวสมดุล (Balanced)'}
              </span>
            )}
          </div>

          {/* Effective Holdings & HHI */}
          <div className="grid grid-cols-2 gap-2 pt-1 pb-1 border-y border-edge/60">
            <div>
              <div className="text-[11px] text-slate-500">Effective Holdings (N<sub>eff</sub>)</div>
              <div className="text-base font-bold text-slate-900 font-mono">
                {data.risk_radar.effective_holdings_count !== null && data.risk_radar.effective_holdings_count !== undefined
                  ? `${data.risk_radar.effective_holdings_count.toFixed(1)}`
                  : 'N/A'}{' '}
                <span className="text-[10px] text-slate-400 font-normal">
                  / {data.holdings_advisory.filter((h) => !h.asset_type.toLowerCase().startsWith('cash')).length} ตัว
                </span>
              </div>
            </div>
            <div>
              <div className="text-[11px] text-slate-500">HHI / Top 3 Weight</div>
              <div className="text-base font-bold text-slate-900 font-mono">
                {data.risk_radar.top_3_weight_pct !== undefined ? `${data.risk_radar.top_3_weight_pct.toFixed(1)}%` : '-'}
                <span className="text-[10px] text-slate-400 font-normal font-sans ml-1">
                  (HHI {data.risk_radar.hhi ? data.risk_radar.hhi.toFixed(3) : '-'})
                </span>
              </div>
            </div>
          </div>

          <div className="flex items-center justify-between text-xs">
            <span className="text-slate-500">Beta พอร์ต:</span>
            <span className="font-semibold text-slate-700 font-mono">
              {data.risk_radar.weighted_beta ? data.risk_radar.weighted_beta.toFixed(2) : 'N/A'}
            </span>
          </div>

          {(data.risk_radar.bucket_active_share_pct !== undefined &&
            data.risk_radar.bucket_active_share_pct !== null) ||
          data.risk_radar.total_portfolio_volatility_pct !== undefined ? (
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-500">Active Share / Vol:</span>
              <span className="font-semibold text-slate-700 font-mono">
                {data.risk_radar.bucket_active_share_pct !== undefined &&
                data.risk_radar.bucket_active_share_pct !== null
                  ? `${data.risk_radar.bucket_active_share_pct.toFixed(1)}%`
                  : 'N/A'}{' '}
                /{' '}
                {data.risk_radar.total_portfolio_volatility_pct !== undefined &&
                data.risk_radar.total_portfolio_volatility_pct !== null
                  ? `${data.risk_radar.total_portfolio_volatility_pct.toFixed(1)}%`
                  : 'N/A'}
              </span>
            </div>
          ) : null}

          <div className="flex items-center gap-2 text-xs">
            <span className="text-slate-500">FX Exposure:</span>
            <div className="flex items-center gap-1.5 flex-1 justify-end font-mono">
              <span className="px-1.5 py-0.2 rounded bg-slate-100 text-slate-700 text-[11px]">
                THB {data.risk_radar.fx_exposure_pct['THB'] ? data.risk_radar.fx_exposure_pct['THB'].toFixed(1) : '0'}%
              </span>
              <span className="px-1.5 py-0.2 rounded bg-slate-100 text-slate-700 text-[11px]">
                USD {data.risk_radar.fx_exposure_pct['USD'] ? data.risk_radar.fx_exposure_pct['USD'].toFixed(1) : '0'}%
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* 3.5 Look-Through Radar & True Sector Exposure */}
      {data.look_through_summary && (
        <div className="space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <span className="text-xl">🔍</span>
              <div>
                <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
                  <span>Glass-Box Portfolio: Look-Through Radar &amp; True Exposure</span>
                  <span className="text-xs px-2 py-0.5 rounded-full bg-purple-100 text-purple-800 border border-purple-200 font-semibold">
                    100% Transparency
                  </span>
                </h2>
                <p className="text-xs text-slate-500">
                  ส่องทะลุไส้ในกองทุนรวมและ PVD รวมกับหุ้นตรง เพื่อขจัดจุดบอดการลงทุนและคุม Single-Stock Cap 15%
                </p>
              </div>
            </div>
            {data.look_through_summary.cap_breached_count > 0 ? (
              <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-amber-100 text-amber-900 border border-amber-300 animate-pulse">
                <span>⚠️</span> พบหุ้นเกิน 15% Cap ({data.look_through_summary.cap_breached_count} ตัว)
              </span>
            ) : (
              <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-emerald-100 text-emerald-800 border border-emerald-200">
                <span>✓</span> ไม่พบความเสี่ยงกระจุกตัวเกิน 15% Cap
              </span>
            )}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
            {/* Left 7 cols: Cross-Portfolio Stock Overlap Radar */}
            <div className="lg:col-span-7 flow-panel rounded-2xl border border-edge bg-panel p-5 shadow-xs space-y-3">
              <div className="flex items-center justify-between pb-2 border-b border-edge/60">
                <div className="flex items-center gap-2">
                  <span className="font-bold text-sm text-slate-900">📡 หุ้นที่มีการถือซ้ำข้ามพอร์ตและกองทุน (Overlap Radar)</span>
                </div>
                <span className="text-xs text-slate-500 font-mono">
                  {(data.look_through_summary.top_overlapped_stocks || []).length > 0
                    ? `${data.look_through_summary.top_overlapped_stocks.length} หุ้นที่ถือร่วม`
                    : `${(data.look_through_summary.holdings_exposure || []).length} หุ้นทั้งหมด`}
                </span>
              </div>

              {(data.look_through_summary.hidden_concentration_warnings || []).length > 0 && (
                <div className="p-3 rounded-xl bg-amber-50 border border-amber-200 text-xs text-amber-900 space-y-1">
                  {(data.look_through_summary.hidden_concentration_warnings || []).map((warn, wIdx) => (
                    <div key={wIdx} className="flex items-start gap-1.5">
                      <span className="font-bold shrink-0">🚨</span>
                      <span>{warn}</span>
                    </div>
                  ))}
                </div>
              )}

              <div className="overflow-x-auto custom-scrollbar">
                <table className="w-full text-left text-xs min-w-[520px]">
                  <thead>
                    <tr className="border-b border-edge/80 text-slate-500 font-semibold uppercase tracking-wider text-[11px]">
                      <th className="py-2 px-2">หุ้น / บริษัท</th>
                      <th className="py-2 px-2 text-right">ถือตรง (Direct)</th>
                      <th className="py-2 px-2 text-right">ในกองทุน (Indirect)</th>
                      <th className="py-2 px-2 text-right">รวมสุทธิ (Total)</th>
                      <th className="py-2 px-2">กองทุนที่ถือ</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-edge/40">
                    {(data.look_through_summary.top_overlapped_stocks.length > 0
                      ? data.look_through_summary.top_overlapped_stocks
                      : data.look_through_summary.holdings_exposure.slice(0, 8)
                    ).map((stk) => {
                      const isBreached = stk.is_cap_breached
                      return (
                        <tr
                          key={stk.ticker}
                          className={`hover:bg-slate-50/50 transition-colors ${
                            isBreached ? 'bg-amber-50/40' : ''
                          }`}
                        >
                          <td className="py-2.5 px-2">
                            <div className="flex items-center gap-1.5">
                              <span className="font-mono font-extrabold text-slate-900">{stk.ticker}</span>
                              <span className="text-[10px] px-1 py-0.2 rounded bg-slate-100 text-slate-600 font-semibold">
                                {stk.market}
                              </span>
                              {isBreached && (
                                <span className="text-[10px] px-1.5 py-0.2 rounded bg-amber-100 text-amber-800 font-bold border border-amber-300">
                                  &gt;15% Cap
                                </span>
                              )}
                            </div>
                            <div className="text-[11px] text-slate-500 line-clamp-1">{stk.company_name}</div>
                          </td>

                          <td className="py-2.5 px-2 text-right font-mono tabular-nums">
                            {stk.direct_value_thb > 0 ? (
                              <div>
                                <span className="font-semibold text-slate-800">
                                  ฿{stk.direct_value_thb.toLocaleString('th-TH', { maximumFractionDigits: 0 })}
                                </span>
                                <div className="text-[10px] text-slate-500">{stk.direct_weight_pct.toFixed(1)}%</div>
                              </div>
                            ) : (
                              <span className="text-slate-400">-</span>
                            )}
                          </td>

                          <td className="py-2.5 px-2 text-right font-mono tabular-nums">
                            {stk.indirect_value_thb > 0 ? (
                              <div>
                                <span className="font-semibold text-purple-700">
                                  ฿{stk.indirect_value_thb.toLocaleString('th-TH', { maximumFractionDigits: 0 })}
                                </span>
                                <div className="text-[10px] text-purple-600">{stk.indirect_weight_pct.toFixed(1)}%</div>
                              </div>
                            ) : (
                              <span className="text-slate-400">-</span>
                            )}
                          </td>

                          <td className="py-2.5 px-2 text-right font-mono tabular-nums">
                            <div className={`font-extrabold ${isBreached ? 'text-amber-700 font-bold' : 'text-slate-900'}`}>
                              ฿{stk.total_value_thb.toLocaleString('th-TH', { maximumFractionDigits: 0 })}
                            </div>
                            <div className={`text-[10px] font-bold ${isBreached ? 'text-amber-700' : 'text-slate-600'}`}>
                              {stk.total_weight_pct.toFixed(1)}%
                            </div>
                          </td>

                          <td className="py-2.5 px-2">
                            {stk.source_funds.length > 0 ? (
                              <div className="flex flex-wrap gap-1">
                                {stk.source_funds.map(([fundSym], fIdx) => (
                                  <span
                                    key={fIdx}
                                    className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-purple-50 text-purple-800 border border-purple-200"
                                    title={fundSym}
                                  >
                                    {fundSym}
                                  </span>
                                ))}
                              </div>
                            ) : (
                              <span className="text-[11px] text-slate-400">ถือตรงเท่านั้น</span>
                            )}
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Right 5 cols: True Sector Exposure vs Macro Direction */}
            <div className="lg:col-span-5 flow-panel rounded-2xl border border-edge bg-panel p-5 shadow-xs space-y-3">
              <div className="flex items-center justify-between pb-2 border-b border-edge/60">
                <span className="font-bold text-sm text-slate-900">🧭 True Sector Exposure vs Macro</span>
                <span className="text-xs text-slate-500">สัดส่วนรวมจริง</span>
              </div>

              <div className="space-y-3">
                {data.look_through_summary.sector_exposure.slice(0, 6).map((sec) => {
                  const macro = data.macro_regime_summary
                  const leading = macro?.sector_analysis?.leading_sectors || []
                  const lagging = macro?.sector_analysis?.lagging_sectors || []
                  const isLeading = leading.some((l: string) => l.toLowerCase() === sec.sector_name.toLowerCase())
                  const isLagging = lagging.some((l: string) => l.toLowerCase() === sec.sector_name.toLowerCase())

                  return (
                    <div key={sec.sector_name} className="space-y-1">
                      <div className="flex items-center justify-between text-xs">
                        <div className="flex items-center gap-1.5 flex-wrap">
                          <span className="font-bold text-slate-800">{sec.sector_name}</span>
                          {sec.is_crowded && (
                            <span className="text-[10px] px-1.5 py-0.2 rounded bg-rose-100 text-rose-800 font-bold border border-rose-300">
                              Crowded &gt;30%
                            </span>
                          )}
                          {isLeading && (
                            <span className="text-[10px] px-1.5 py-0.2 rounded bg-emerald-100 text-emerald-800 font-bold border border-emerald-300">
                              🟢 Leading
                            </span>
                          )}
                          {isLagging && (
                            <span className="text-[10px] px-1.5 py-0.2 rounded bg-rose-100 text-rose-800 font-bold border border-rose-300">
                              🔴 Lagging
                            </span>
                          )}
                        </div>
                        <span className="font-mono font-bold text-slate-900">{sec.weight_pct.toFixed(1)}%</span>
                      </div>

                      {/* Stacked bar: Direct (sky) vs Fund (purple) */}
                      <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden flex">
                        {sec.direct_stocks_pct > 0 && (
                          <div
                            className="bg-sky-500 h-2 transition-all"
                            style={{ width: `${Math.min(sec.direct_stocks_pct * 2.5, 100)}%` }}
                            title={`ถือตรง: ${sec.direct_stocks_pct.toFixed(1)}%`}
                          />
                        )}
                        {sec.funds_pct > 0 && (
                          <div
                            className="bg-purple-500 h-2 transition-all"
                            style={{ width: `${Math.min(sec.funds_pct * 2.5, 100)}%` }}
                            title={`ในกองทุน: ${sec.funds_pct.toFixed(1)}%`}
                          />
                        )}
                      </div>

                      <div className="flex items-center justify-between text-[10px] text-slate-400">
                        <span>ตรง: {sec.direct_stocks_pct.toFixed(1)}% · กองทุน: {sec.funds_pct.toFixed(1)}%</span>
                        <span>฿{sec.total_value_thb.toLocaleString('th-TH', { maximumFractionDigits: 0 })}</span>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 4. Tier 2 AI Strategic Playbook Hero Card */}
      {synthesizing ? (
        <AiPlaybookSynthesisLoader />
      ) : data.cached_narrative ? (
        <div className="flow-panel rounded-2xl border border-indigo-200/80 bg-gradient-to-br from-indigo-50/40 via-white to-purple-50/30 p-6 shadow-sm space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 border-b border-indigo-100">
            <div className="flex items-center gap-2.5">
              <span className="text-xl">🧠</span>
              <h2 className="text-base font-bold text-slate-900">
                AI Strategic Playbook
              </h2>
              {data.cached_narrative.consistency_verified && (
                <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-100 text-emerald-800 border border-emerald-200">
                  <span>✓</span> ตัวเลขสอดคล้อง 100%
                </span>
              )}
            </div>
            <span className="text-xs text-slate-500 font-mono">
              สร้างเมื่อ: {new Date(data.cached_narrative.synthesized_at).toLocaleTimeString('th-TH')}
            </span>
          </div>

          <div className="text-sm text-slate-800 leading-relaxed bg-white/80 p-4 rounded-xl border border-indigo-100/60 shadow-xs">
            <p className="font-medium text-indigo-950 mb-1">ภาพรวมกลยุทธ์มหภาค (Executive Macro Summary):</p>
            <p className="text-slate-700">{data.cached_narrative.macro_executive_summary}</p>
          </div>

          {/* Strategic RM Lens: What's Working vs What Concerns Me */}
          {((data.cached_narrative.whats_working && data.cached_narrative.whats_working.length > 0) ||
            (data.cached_narrative.what_concerns_me && data.cached_narrative.what_concerns_me.length > 0)) && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {data.cached_narrative.whats_working && data.cached_narrative.whats_working.length > 0 && (
                <div className="bg-emerald-50/50 p-4 rounded-xl border border-emerald-200/70 space-y-2">
                  <h4 className="text-xs font-bold uppercase tracking-wider text-emerald-900 flex items-center gap-1.5">
                    <span>📈</span> สิ่งที่กำลังได้เปรียบในภาวะตลาด (What's Working)
                  </h4>
                  <ul className="text-xs text-slate-700 space-y-1.5">
                    {data.cached_narrative.whats_working.map((item, idx) => (
                      <li key={idx} className="flex items-start gap-1.5">
                        <span className="text-emerald-600 shrink-0 font-bold">✓</span>
                        <span>{item}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {data.cached_narrative.what_concerns_me && data.cached_narrative.what_concerns_me.length > 0 && (
                <div className="bg-rose-50/50 p-4 rounded-xl border border-rose-200/70 space-y-2">
                  <h4 className="text-xs font-bold uppercase tracking-wider text-rose-900 flex items-center gap-1.5">
                    <span>⚠️</span> สิ่งที่น่ากังวล &amp; ความเสี่ยงแฝง (What Concerns Me)
                  </h4>
                  <ul className="text-xs text-slate-700 space-y-1.5">
                    {data.cached_narrative.what_concerns_me.map((item, idx) => (
                      <li key={idx} className="flex items-start gap-1.5">
                        <span className="text-rose-600 shrink-0 font-bold">•</span>
                        <span>{item}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 pt-1">
            {/* Playbook Column 1: Immediate */}
            <div className="bg-white/80 p-4 rounded-xl border border-sky-100 space-y-2">
              <h4 className="text-xs font-bold uppercase tracking-wider text-sky-800 flex items-center gap-1.5">
                <span>⚡</span> คำสั่งปฏิบัติการวันนี้ (Immediate)
              </h4>
              <ul className="text-xs text-slate-700 space-y-2">
                {data.cached_narrative.immediate_action_playbook.map((item, idx) => (
                  <li key={idx} className="flex items-start gap-1.5">
                    <span className="text-sky-500 shrink-0 font-bold">•</span>
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Playbook Column 2: Dip Plan */}
            <div className="bg-white/80 p-4 rounded-xl border border-emerald-100 space-y-2">
              <h4 className="text-xs font-bold uppercase tracking-wider text-emerald-800 flex items-center gap-1.5">
                <span>🎯</span> แผนตั้งรับ Buy on Dip
              </h4>
              <ul className="text-xs text-slate-700 space-y-2">
                {data.cached_narrative.conditional_dip_plan.map((item, idx) => (
                  <li key={idx} className="flex items-start gap-1.5">
                    <span className="text-emerald-500 shrink-0 font-bold">•</span>
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Playbook Column 3: Risk & Hedging */}
            <div className="bg-white/80 p-4 rounded-xl border border-purple-100 space-y-2">
              <h4 className="text-xs font-bold uppercase tracking-wider text-purple-800 flex items-center gap-1.5">
                <span>🛡️</span> การบริหารความเสี่ยง &amp; พักเงิน
              </h4>
              <ul className="text-xs text-slate-700 space-y-2">
                {data.cached_narrative.risk_mitigation_and_hedging.map((item, idx) => (
                  <li key={idx} className="flex items-start gap-1.5">
                    <span className="text-purple-500 shrink-0 font-bold">•</span>
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* Strategic Discussion Points */}
          {data.cached_narrative.strategic_discussion_points &&
            data.cached_narrative.strategic_discussion_points.length > 0 && (
              <div className="bg-slate-50/90 p-4 rounded-xl border border-slate-200/80 space-y-2">
                <h4 className="text-xs font-bold uppercase tracking-wider text-slate-800 flex items-center gap-1.5">
                  <span>💡</span> ประเด็นชวนคุยเชิงกลยุทธ์ (Strategic Discussion Points)
                </h4>
                <ol className="text-xs text-slate-700 space-y-1.5 pl-4 list-decimal">
                  {data.cached_narrative.strategic_discussion_points.map((item, idx) => (
                    <li key={idx} className="pl-1">
                      <span>{item}</span>
                    </li>
                  ))}
                </ol>
              </div>
            )}
        </div>
      ) : (
        <div className="p-6 rounded-2xl border border-dashed border-indigo-200 bg-indigo-50/20 text-center space-y-3">
          <div className="text-2xl">✨</div>
          <h3 className="text-sm font-semibold text-slate-800">ยังไม่มีบทวิเคราะห์ AI Strategic Playbook สำหรับพอร์ตโฟลิโอสถานะปัจจุบัน</h3>
          <p className="text-xs text-slate-500 max-w-md mx-auto">
            กดปุ่มประมวลผลเพื่อสังเคราะห์แผนกลยุทธ์เชิงลึก เชื่อมโยง Macro Regime และปัจจัยทางเทคนิค โดยตรวจสอบความสอดคล้องกับตัวเลขจริง
          </p>
          <button
            onClick={handleSynthesize}
            className="px-4 py-2 rounded-xl text-xs font-semibold bg-indigo-600 text-white hover:bg-indigo-700 transition"
          >
            ประมวลผล AI Playbook เดี๋ยวนี้
          </button>
        </div>
      )}

      {/* 5. Holdings Advisory Table with Action Badges & Lot Sizing */}
      <div className="flow-panel rounded-2xl border border-edge bg-panel shadow-xs overflow-hidden">
        {/* Table Filter Toolbar */}
        <div className="p-4 border-b border-edge flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-bold text-slate-900">รายการคำแนะนำสินทรัพย์รายตัว (Action Stance)</h3>
            <span className="text-xs text-slate-500 font-mono">({filteredHoldings.length})</span>
          </div>

          <div className="flex items-center gap-1 overflow-x-auto custom-scrollbar pb-1 sm:pb-0">
            {[
              { id: 'ALL', label: 'ทั้งหมด' },
              { id: 'IMMEDIATE', label: `เคาะซื้อ/ตัดขายทันที (${immediateHoldings.length})` },
              { id: 'CONDITIONAL', label: `ตั้งรับตามกรอบ (${conditionalHoldings.length})` },
              { id: 'URGENT_REDUCE', label: 'Urgent Reduce' },
              { id: 'ACCUMULATE_NOW', label: 'Accumulate' },
              { id: 'LET_WINNER_RUN', label: 'Winner Run' },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setFilterAction(tab.id)}
                className={`px-3 py-1.5 rounded-lg text-xs sm:text-[13px] font-semibold whitespace-nowrap transition ${
                  filterAction === tab.id
                    ? 'bg-sky-600 text-white shadow-xs'
                    : 'text-slate-600 hover:bg-slate-100'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        {/* Table Body */}
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50/75 border-b border-edge text-slate-500 uppercase tracking-wider font-semibold">
              <tr>
                <th className="py-3 px-4">สินทรัพย์ (Asset)</th>
                <th className="py-3 px-3">สัดส่วนปัจจุบัน vs เป้าหมาย</th>
                <th className="py-3 px-3">คำแนะนำ (Action)</th>
                <th className="py-3 px-3">ขนาดคำสั่งซื้อขายจริง (Lot Sizing)</th>
                <th className="py-3 px-3">Conviction &amp; Stage</th>
                <th className="py-3 px-4">เหตุผลเชิงปริมาณ (Rationale)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-edge/60">
              {filteredHoldings.map((h) => (
                <HoldingRow key={h.symbol} holding={h} />
              ))}
              {filteredHoldings.length === 0 && (
                <tr>
                  <td colSpan={6} className="text-center py-8 text-slate-400">
                    ไม่พบรายการสินทรัพย์ในหมวดหมู่นี้
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* 6. Quarantined Holdings Safe Drawer (if any) */}
      {data.quarantined_holdings.length > 0 && (
        <div className="flow-panel rounded-2xl border border-rose-200 bg-rose-50/30 p-5 space-y-3">
          <div className="flex items-center gap-2 text-rose-900 font-bold text-sm">
            <span>🛡️</span>
            <span>สินทรัพย์ที่ถูกกักกันชั่วคราว (Quarantined Data Anomalies)</span>
            <span className="text-xs font-normal text-rose-700">
              — ระงับการออกคำสั่ง rebalance เพื่อป้องกันความเสียหายจากข้อมูลผิดปกติ
            </span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
            {data.quarantined_holdings.map((q) => (
              <div key={q.symbol} className="p-3.5 rounded-xl border border-rose-200 bg-white space-y-1 text-xs">
                <div className="flex items-center justify-between font-bold text-slate-800">
                  <span>{q.symbol}</span>
                  <span className="text-rose-600 font-mono text-[11px]">{q.guardrail_code}</span>
                </div>
                <p className="text-slate-600">{q.reason}</p>
                {q.details && <p className="text-slate-400 font-mono text-[10px]">{q.details}</p>}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function HoldingRow({ holding: h }: { holding: HoldingAdvisoryDTO }) {
  const getActionBadgeClass = (code: string) => {
    switch (code) {
      case 'URGENT_REDUCE':
        return 'bg-rose-100 text-rose-800 border-rose-300 animate-pulse'
      case 'TRIM_OVERWEIGHT':
        return 'bg-amber-100 text-amber-800 border-amber-300'
      case 'LET_WINNER_RUN':
        return 'bg-indigo-100 text-indigo-800 border-indigo-300'
      case 'ACCUMULATE_NOW':
        return 'bg-emerald-100 text-emerald-800 border-emerald-300 font-bold'
      case 'BREAKOUT_BUY':
        return 'bg-cyan-100 text-cyan-800 border-cyan-300 font-bold'
      case 'ACCUMULATE_ON_DIP':
        return 'bg-teal-50 text-teal-800 border-teal-200'
      case 'HOLD_LOCKED_VEHICLE':
        return 'bg-slate-100 text-slate-700 border-slate-200'
      case 'QUARANTINED_DATA_ANOMALY':
        return 'bg-red-50 text-red-700 border-red-200'
      default:
        return 'bg-slate-50 text-slate-600 border-slate-200'
    }
  }

  const formatLotSizing = (lot?: LotSizingDTO | null) => {
    if (!lot || lot.amount_native === 0) return '-'
    if (lot.market === 'TH') {
      const boardStr = lot.board_lot_shares ? `${lot.board_lot_shares.toLocaleString()} หุ้น (Board Lot)` : ''
      const oddStr = lot.odd_lot_shares ? ` + ${lot.odd_lot_shares} หุ้น (Odd Lot)` : ''
      return (
        <div className="space-y-0.5">
          <div className="font-semibold text-slate-800">
            ฿{lot.amount_thb.toLocaleString('th-TH', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </div>
          <div className="text-[11px] text-slate-500 font-mono">
            {boardStr}
            {oddStr}
          </div>
        </div>
      )
    } else if (lot.market === 'US') {
      return (
        <div className="space-y-0.5">
          <div className="font-semibold text-slate-800">
            ${lot.amount_native.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            <span className="text-[11px] text-slate-500 font-normal"> (฿{lot.amount_thb.toLocaleString('th-TH')})</span>
          </div>
          <div className="text-[11px] text-slate-500 font-mono">
            {lot.fractional_shares?.toFixed(4)} หุ้น ({lot.whole_shares} เต็ม + เศษ)
          </div>
        </div>
      )
    } else {
      return (
        <div className="font-semibold text-slate-800">
          ฿{lot.amount_thb.toLocaleString('th-TH', { minimumFractionDigits: 2 })}
        </div>
      )
    }
  }

  return (
    <tr className="hover:bg-slate-50/60 transition">
      {/* Symbol & Market */}
      <td className="py-3 px-4">
        <div className="flex items-center gap-2">
          <span className="text-base">{h.market === 'TH' ? '🇹🇭' : '🇺🇸'}</span>
          <div>
            <div className="font-bold text-slate-900 flex items-center gap-1.5">
              <span>{h.symbol}</span>
              {h.is_immediate_execution && (
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-ping" />
              )}
            </div>
            <div className="text-[11px] text-slate-500 truncate max-w-[150px]">{h.asset_name}</div>
          </div>
        </div>
      </td>

      {/* Weights */}
      <td className="py-3 px-3">
        <div className="space-y-0.5">
          <div className="font-semibold text-slate-800">
            {h.current_weight_pct.toFixed(1)}%{' '}
            <span className="text-slate-400 font-normal">/ เป้า {h.target_weight_pct.toFixed(1)}%</span>
          </div>
          <div
            className={`text-[11px] font-mono ${
              h.weight_drift_pct > 2.5
                ? 'text-amber-600 font-semibold'
                : h.weight_drift_pct < -2.5
                ? 'text-sky-600 font-semibold'
                : 'text-slate-400'
            }`}
          >
            Drift: {h.weight_drift_pct > 0 ? `+${h.weight_drift_pct.toFixed(1)}` : h.weight_drift_pct.toFixed(1)}%
          </div>
          {h.risk_contribution_pct !== undefined && h.risk_contribution_pct !== null && (
            <div className="text-[10px] text-slate-500 font-mono">
              Risk: {h.risk_contribution_pct.toFixed(1)}%
              {h.is_volatility_estimated && (
                <span className="ml-1 text-amber-600" title="Estimated 20% default volatility">
                  (est. vol)
                </span>
              )}
            </div>
          )}
        </div>
      </td>

      {/* Action Code Badge */}
      <td className="py-3 px-3">
        <span
          className={`inline-block px-2.5 py-1 rounded-lg text-xs border tracking-tight ${getActionBadgeClass(
            h.action_code
          )}`}
        >
          {h.action_code}
        </span>
      </td>

      {/* Lot Sizing */}
      <td className="py-3 px-3">
        {formatLotSizing(h.lot_sizing)}
        {h.estimated_friction_bps !== undefined && h.estimated_friction_bps !== null && (
          <div className="text-[10px] text-slate-400 font-mono mt-0.5">
            Est. friction: {h.estimated_friction_bps.toFixed(1)} bps
          </div>
        )}
      </td>

      {/* Conviction & Stage */}
      <td className="py-3 px-3">
        <div className="space-y-0.5 text-[11px]">
          {h.core_conviction_score !== null && h.core_conviction_score !== undefined ? (
            <div className="font-medium text-slate-700">
              Conviction:{' '}
              <span className="font-bold text-indigo-600">{h.core_conviction_score.toFixed(1)}/10</span>
            </div>
          ) : (
            <div className="text-slate-400">-</div>
          )}
          {h.price_stage && (
            <div className="font-mono text-slate-500 text-[10px] truncate max-w-[130px]">{h.price_stage}</div>
          )}
          {h.macro_alignment !== 'unavailable' && (
            <div
              className={`font-semibold text-[10px] ${
                h.macro_alignment === 'tailwind'
                  ? 'text-emerald-600'
                  : h.macro_alignment === 'headwind'
                  ? 'text-rose-600'
                  : 'text-slate-500'
              }`}
            >
              Macro: {h.macro_alignment}
            </div>
          )}
        </div>
      </td>

      {/* Rationale */}
      <td className="py-3 px-4">
        <div className="text-xs text-slate-700 max-w-[260px] leading-snug">
          {h.action_rationale}
          {h.conditional_buy_zone_max && (
            <div className="text-[11px] text-teal-700 font-mono mt-0.5">
              โซนสะสม: {h.conditional_buy_zone_min?.toFixed(2)} - {h.conditional_buy_zone_max.toFixed(2)}{' '}
              {h.native_currency}
            </div>
          )}
        </div>
      </td>
    </tr>
  )
}
