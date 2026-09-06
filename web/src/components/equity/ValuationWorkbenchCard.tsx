import type { DCFResultDTO, MarginMetricItemDTO, ReverseDCFResultDTO, DCFScenarioDTO } from '../../api/types'

export interface ReverseDcfProps extends Partial<ReverseDCFResultDTO> {}

export interface PiotroskiProps {
  f_score?: number
  status?: string
  profitability_points?: number
  leverage_liquidity_points?: number
  operating_efficiency_points?: number
  exclusion_reason?: string
}

export interface CapitalQualityProps {
  roic_pct?: number | null
  fcf_margin_pct?: number | null
  fcf_yield_pct?: number | null
  ocf_to_net_income?: number | null
  solvency_score?: number | null
}

interface Props {
  reverseDcf?: ReverseDcfProps | null
  dcf?: DCFResultDTO | null
  piotroski?: PiotroskiProps | null
  qualityMetrics?: CapitalQualityProps | null
  gaapOperatingMargin?: MarginMetricItemDTO | null
  nonGaapOperatingMargin?: MarginMetricItemDTO | null
  providerEbitMargin?: MarginMetricItemDTO | null
  valuationMarginSourceUsed?: string | null
  dcfDiscrepancyWarning?: string | null
  currency?: string
}

function formatLargeNum(val?: number | null, prefix = '$'): string {
  if (val == null || isNaN(val)) return 'N/A'
  const abs = Math.abs(val)
  if (abs >= 1e12) return `${prefix}${(val / 1e12).toFixed(2)}T`
  if (abs >= 1e9) return `${prefix}${(val / 1e9).toFixed(2)}B`
  if (abs >= 1e6) return `${prefix}${(val / 1e6).toFixed(2)}M`
  return `${prefix}${val.toLocaleString()}`
}

export default function ValuationWorkbenchCard({
  reverseDcf,
  dcf,
  piotroski,
  qualityMetrics,
  gaapOperatingMargin,
  nonGaapOperatingMargin,
  providerEbitMargin,
  valuationMarginSourceUsed,
  dcfDiscrepancyWarning,
  currency = '$',
}: Props) {
  // If no valuation data at all
  if (!reverseDcf && !dcf && !piotroski) return null

  const targetPrice = reverseDcf?.target_price_12m
  const upside = reverseDcf?.upside_12m_pct
  const verdict = (reverseDcf?.valuation_verdict && reverseDcf.valuation_verdict !== 'unavailable')
    ? reverseDcf.valuation_verdict
    : (reverseDcf?.status === 'unavailable' || !reverseDcf ? 'unavailable' : (upside != null ? (upside > 15 ? 'undervalued' : upside < -15 ? 'overvalued' : 'fairly_valued') : 'unavailable'))

  const scenarios = dcf?.scenarios
    ? [
        { key: 'bear', label: 'Bear Case', color: 'bg-rose-500', textCol: 'text-rose-700', bgCol: 'bg-rose-50 border-rose-200', data: dcf.scenarios.bear },
        { key: 'base', label: 'Base Case', color: 'bg-sky-500', textCol: 'text-sky-700', bgCol: 'bg-sky-50 border-sky-200', data: dcf.scenarios.base },
        { key: 'bull', label: 'Bull Case', color: 'bg-emerald-500', textCol: 'text-emerald-700', bgCol: 'bg-emerald-50 border-emerald-200', data: dcf.scenarios.bull },
      ].filter((s): s is { key: string; label: string; color: string; textCol: string; bgCol: string; data: DCFScenarioDTO & { target_price: number; upside_pct: number } } => s.data != null && s.data.target_price != null && s.data.upside_pct != null)
    : []

  const maxScenarioPrice = scenarios.length > 0 ? Math.max(...scenarios.map(s => s.data.target_price), 1) : 100

  const ebitLabel = reverseDcf?.ebit_margin_source_tier === 'filing_authoritative' ? 'GAAP Operating Margin' : 'Reported EBIT Margin'
  const ebitMeta = reverseDcf?.ebit_margin_fiscal_period
    ? `${reverseDcf.ebit_margin_fiscal_period}${reverseDcf.ebit_margin_period_type ? `, ${reverseDcf.ebit_margin_period_type}` : ''}`
    : ''

  const invalidationReasons = dcf?.invalidation_reasons || (reverseDcf?.actionability_reason ? [reverseDcf.actionability_reason] : [])

  return (
    <div className="flow-panel rounded-2xl border border-edge/80 p-6 shadow-sm transition-all animate-card-in space-y-6">
      {/* Card Header: Title & Valuation Verdict Badge */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-edge/60 pb-4">
        <div>
          <div className="text-[11px] font-bold uppercase tracking-wider text-sky-700/80">
            Valuation & Quality Workbench
          </div>
          <h3 className="text-lg font-bold text-zinc-900 flex items-center gap-2 mt-0.5">
            <span>🎯 5Y Reverse DCF & Expectations</span>
          </h3>
        </div>

        <div className="flex items-center gap-3">
          {/* Wall Street Consensus Target */}
          {reverseDcf?.consensus_target_price != null && (
            <div className="text-right border-r border-edge/60 pr-3 hidden sm:block">
              <div className="text-xs text-zinc-400">Wall Street Consensus</div>
              <div className="flex items-center gap-1 font-bold text-sky-700 text-sm justify-end">
                <span>{currency}{reverseDcf.consensus_target_price.toFixed(2)}</span>
                {reverseDcf.analyst_count != null && (
                  <span className="text-[10px] text-zinc-400 font-normal">({reverseDcf.analyst_count} analysts)</span>
                )}
              </div>
            </div>
          )}

          {/* DCF Fair Value (Base Case or Exit Multiple) */}
          <div className="text-right">
            <div className="text-xs text-zinc-400">
              {dcf?.scenarios?.base?.target_price != null ? 'DCF Fair Value (Base)' : 'DCF 12M Outlook'}
            </div>
            {dcf?.scenarios?.base?.target_price != null && dcf?.scenarios?.base?.upside_pct != null ? (
              <div className="flex items-center gap-1.5 font-bold text-zinc-900 text-sm justify-end">
                <span>{currency}{dcf.scenarios.base.target_price.toFixed(2)}</span>
                <span
                  className={`text-xs px-2 py-0.5 rounded-full font-bold border ${
                    dcf.scenarios.base.upside_pct >= 0
                      ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                      : 'bg-rose-50 text-rose-700 border-rose-200'
                  }`}
                >
                  {dcf.scenarios.base.upside_pct >= 0 ? '+' : ''}{dcf.scenarios.base.upside_pct.toFixed(1)}%
                </span>
              </div>
            ) : targetPrice != null ? (
              <div className="flex items-center gap-1.5 font-bold text-zinc-900 text-sm justify-end">
                <span>{currency}{targetPrice.toFixed(2)}</span>
                {upside != null && (
                  <span
                    className={`text-xs px-2 py-0.5 rounded-full font-bold border ${
                      upside >= 0
                        ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                        : 'bg-rose-50 text-rose-700 border-rose-200'
                    }`}
                  >
                    {upside >= 0 ? '+' : ''}{upside.toFixed(1)}%
                  </span>
                )}
              </div>
            ) : (
              <div className="font-bold text-zinc-400 text-sm justify-end">
                <span>Unavailable</span>
              </div>
            )}
          </div>

          <div className="flex flex-col items-end">
            <span className="text-[10px] font-bold text-zinc-400 uppercase tracking-wider mb-0.5">Reverse DCF Verdict (12M)</span>
            <span
              title="Reverse DCF 12M Valuation Verdict"
              className={`px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider border ${
                verdict === 'undervalued'
                  ? 'bg-emerald-50 text-emerald-700 border-emerald-200 shadow-xs'
                  : verdict === 'overvalued'
                  ? 'bg-rose-50 text-rose-700 border-rose-200 shadow-xs'
                  : verdict === 'fairly_valued'
                  ? 'bg-sky-50 text-sky-700 border-sky-200 shadow-xs'
                  : 'bg-zinc-100 text-zinc-600 border-zinc-200 shadow-xs'
              }`}
            >
              {verdict.replace('_', ' ')}
            </span>
          </div>
        </div>
      </div>

      {/* Discrepancy Warning Banner */}
      {dcfDiscrepancyWarning && (
        <div className="rounded-xl border border-amber-300 bg-amber-50/90 p-3.5 text-xs text-amber-900 flex items-start gap-2.5 shadow-xs">
          <span className="text-base">⚠️</span>
          <div>
            <div className="font-bold">Valuation Model Discrepancy Alert</div>
            <p className="mt-0.5 leading-relaxed">{dcfDiscrepancyWarning}</p>
          </div>
        </div>
      )}

      {/* Non-Actionable Valuation Model Banner with Dynamic Reason List */}
      {(reverseDcf?.is_actionable === false || dcf?.is_actionable === false) && (
        <div className="rounded-xl border border-amber-300 bg-amber-50/90 p-3.5 text-xs text-amber-900 flex items-start gap-2.5 shadow-xs">
          <span className="text-base">ℹ️</span>
          <div className="space-y-1">
            <div className="font-bold">Informational-Only Valuation Model (Non-Actionable)</div>
            <p className="leading-relaxed">
              แบบจำลองถูกระงับการออกเป้าหมายราคาเนื่องจากความผิดปกติของสภาวะตลาดหรือสมมติฐานทางเศรษฐศาสตร์มหภาค:
            </p>
            {invalidationReasons.length > 0 && (
              <ul className="list-disc list-inside space-y-0.5 text-amber-950 font-medium">
                {invalidationReasons.map((r, i) => (
                  <li key={i}>{r}</li>
                ))}
              </ul>
            )}
          </div>
        </div>
      )}

      {/* 3-Tier Margin Provenance Strip */}
      {(gaapOperatingMargin || nonGaapOperatingMargin || providerEbitMargin) && (
        <div className="rounded-xl border border-edge/60 bg-surface/50 p-3.5 space-y-2">
          <div className="flex items-center justify-between text-xs text-zinc-500 font-semibold">
            <span>🏷️ 3-Tier Operating Margin Provenance</span>
            {valuationMarginSourceUsed && (
              <span className="text-sky-700 text-[11px] font-bold">Used in DCF: {valuationMarginSourceUsed}</span>
            )}
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5 text-xs">
            {gaapOperatingMargin && (
              <div className="rounded-lg border border-edge/50 bg-white/70 p-2 space-y-0.5">
                <span className="text-[10px] text-zinc-400 font-bold uppercase block">1. GAAP Operating Margin</span>
                <span className="text-sm font-extrabold text-zinc-900">{gaapOperatingMargin.value_pct != null ? `${gaapOperatingMargin.value_pct.toFixed(1)}%` : 'N/A'}</span>
                <p className="text-[10px] text-zinc-500">{gaapOperatingMargin.period_end} ({gaapOperatingMargin.source_provenance})</p>
              </div>
            )}
            {nonGaapOperatingMargin && (
              <div className="rounded-lg border border-edge/50 bg-white/70 p-2 space-y-0.5">
                <span className="text-[10px] text-zinc-400 font-bold uppercase block">2. Non-GAAP Margin (Guidance)</span>
                <span className="text-sm font-extrabold text-indigo-700">{nonGaapOperatingMargin.value_pct != null ? `${nonGaapOperatingMargin.value_pct.toFixed(1)}%` : 'N/A'}</span>
                <p className="text-[10px] text-zinc-500">{nonGaapOperatingMargin.period_end} ({nonGaapOperatingMargin.source_provenance})</p>
              </div>
            )}
            {providerEbitMargin && (
              <div className="rounded-lg border border-edge/50 bg-white/70 p-2 space-y-0.5">
                <span className="text-[10px] text-zinc-400 font-bold uppercase block">3. Provider EBIT Proxy</span>
                <span className="text-sm font-extrabold text-zinc-700">{providerEbitMargin.value_pct != null ? `${providerEbitMargin.value_pct.toFixed(1)}%` : 'N/A'}</span>
                <p className="text-[10px] text-zinc-500">{providerEbitMargin.source_provenance}</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Main 2-Pane Workbench */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Pane: Expectation Gap & Reverse DCF Forensics (7 Cols) */}
        <div className="lg:col-span-7 space-y-4">
          <div className="text-xs font-semibold text-zinc-500 uppercase tracking-wider flex items-center gap-1.5">
            <span>📉 Market Expectation Gap Analysis</span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
            {/* Market Implied Growth */}
            <div className="rounded-xl border border-edge/60 bg-surface/70 p-4 transition-all hover:bg-surface hover:border-sky-300/80">
              <div className="flex items-center justify-between">
                <span className="text-xs text-zinc-500 font-medium block">Market Implied 5Y Growth</span>
                {reverseDcf?.base_revenue != null && (
                  <span
                    className="text-[10px] px-1.5 py-0.5 rounded bg-sky-50 font-semibold text-sky-700 border border-sky-200"
                    title={`Base Revenue: ${formatLargeNum(reverseDcf.base_revenue, currency)} (${reverseDcf.base_revenue_period_type ? reverseDcf.base_revenue_period_type.toUpperCase() : 'TTM'})`}
                  >
                    Base: {formatLargeNum(reverseDcf.base_revenue, currency)} ({reverseDcf.base_revenue_period_type ? reverseDcf.base_revenue_period_type.toUpperCase() : 'TTM'})
                  </span>
                )}
              </div>
              <div className="flex items-baseline gap-1 mt-1">
                <span className="text-2xl font-bold text-zinc-900">
                  {reverseDcf?.market_implied_growth_pct != null
                    ? `${reverseDcf.market_implied_growth_pct.toFixed(1)}%`
                    : 'N/A'}
                </span>
                <span className="text-xs text-zinc-400">CAGR</span>
              </div>
              <p className="text-[11px] text-zinc-500 mt-1.5 leading-tight">
                อัตราเติบโตกระแสเงินสดแฝงที่ราคาตลาดปัจจุบัน Price-in ไว้
              </p>
            </div>

            {/* Market Implied Operating Margin with Reported Baseline */}
            <div className="rounded-xl border border-edge/60 bg-surface/70 p-4 transition-all hover:bg-surface hover:border-sky-300/80">
              <div className="flex items-center justify-between">
                <span className="text-xs text-zinc-500 font-medium block">Market Implied EBIT Margin</span>
                {reverseDcf?.reported_ebit_margin_pct != null && (
                  <span
                    className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-100 font-semibold text-zinc-600 border border-zinc-200"
                    title={`${ebitLabel}: ${reverseDcf.reported_ebit_margin_pct}% (${ebitMeta})`}
                  >
                    Base: {reverseDcf.reported_ebit_margin_pct.toFixed(1)}%
                  </span>
                )}
              </div>
              <div className="flex items-baseline gap-1 mt-1">
                <span className="text-2xl font-bold text-zinc-900">
                  {reverseDcf?.market_implied_margin_pct != null
                    ? `${reverseDcf.market_implied_margin_pct.toFixed(1)}%`
                    : (reverseDcf?.reported_ebit_margin_pct != null ? `${reverseDcf.reported_ebit_margin_pct.toFixed(1)}%` : 'N/A')}
                </span>
                <span className="text-xs text-zinc-400">
                  {reverseDcf?.market_implied_margin_pct != null ? 'Margin' : 'Base Margin'}
                </span>
              </div>
              <p className="text-[11px] text-zinc-500 mt-1.5 leading-tight">
                {ebitMeta ? `${ebitLabel} ${reverseDcf?.reported_ebit_margin_pct?.toFixed(1)}% (${ebitMeta})` : 'อัตรากำไรจากการดำเนินงานที่ตลาดคาดหวัง'}
              </p>
            </div>
          </div>

          {/* Intrinsic & Enterprise Value Breakdown */}
          {reverseDcf && (
            <div className="rounded-xl border border-edge/50 bg-surface-strong/40 p-3.5 space-y-2 text-xs">
              <div className="flex justify-between text-zinc-600 border-b border-edge/30 pb-1.5">
                <span>มูลค่าพื้นฐาน ณ วันนี้ (Gordon Growth):</span>
                <span className="font-semibold text-zinc-900">
                  {reverseDcf.intrinsic_value_today != null ? `${currency}${reverseDcf.intrinsic_value_today.toFixed(2)}` : 'N/A'}
                  {reverseDcf.target_price_12m != null && (
                    <span className="text-zinc-500 font-normal ml-1">
                      (12M: {currency}{reverseDcf.target_price_12m.toFixed(2)})
                    </span>
                  )}
                </span>
              </div>
              {reverseDcf.intrinsic_value_exit_multiple != null && (
                <div className="flex justify-between text-zinc-600 border-b border-edge/30 pb-1.5">
                  <span>มูลค่าพื้นฐาน ณ วันนี้ ({reverseDcf.exit_multiple_used || 25}x Exit Multiple):</span>
                  <span className="font-semibold text-sky-800">
                    {currency}{reverseDcf.intrinsic_value_exit_multiple.toFixed(2)}
                    {reverseDcf.target_price_exit_multiple_12m != null && (
                      <span className="text-zinc-500 font-normal ml-1">
                        (12M: {currency}{reverseDcf.target_price_exit_multiple_12m.toFixed(2)})
                      </span>
                    )}
                  </span>
                </div>
              )}
              <div className="flex justify-between text-zinc-600 border-b border-edge/30 pb-1.5">
                <span>Enterprise Value / Equity Value:</span>
                <span className="font-semibold text-zinc-900">
                  {formatLargeNum(reverseDcf.enterprise_value, currency)} / {formatLargeNum(reverseDcf.equity_value, currency)}
                </span>
              </div>
              <div className="flex justify-between text-zinc-600 border-b border-edge/30 pb-1.5">
                <span>Net Cash (เงินสดหักหนี้สินสุทธิ):</span>
                <span className={`font-semibold ${reverseDcf.net_cash_debt != null && reverseDcf.net_cash_debt >= 0 ? 'text-emerald-700' : 'text-zinc-900'}`}>
                  {reverseDcf.net_cash_debt != null && reverseDcf.net_cash_debt >= 0 ? '+' : ''}
                  {formatLargeNum(reverseDcf.net_cash_debt, currency)}
                </span>
              </div>
              <div className="flex justify-between text-zinc-600">
                <span>PV of 5Y FCF / Terminal Value PV:</span>
                <span className="font-semibold text-zinc-900">
                  {formatLargeNum(reverseDcf.sum_pv_5y_fcf, currency)} / {formatLargeNum(reverseDcf.terminal_value_pv, currency)}
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Right Pane: DCF Scenarios Visualizer (5 Cols) */}
        <div className="lg:col-span-5 space-y-4">
          <div className="flex items-center justify-between text-xs font-semibold text-zinc-500 uppercase tracking-wider">
            <span title="Constant 5Y FCF Growth Baseline Model">📊 Generic FCF DCF Scenarios (Constant Growth)</span>
            {dcf?.wacc_pct != null && (
              <span className="text-sky-700 font-bold lowercase">wacc {dcf.wacc_pct}%</span>
            )}
          </div>

          {scenarios.length > 0 ? (
            <div className="space-y-3 rounded-xl border border-edge/60 bg-surface/70 p-4">
              {scenarios.map(sc => {
                const pct = Math.min(100, Math.max(12, (sc.data.target_price / maxScenarioPrice) * 100))
                return (
                  <div key={sc.key} className="space-y-1">
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-medium text-zinc-700">{sc.label}</span>
                      <div className="flex items-center gap-1.5">
                        <span className="font-bold text-zinc-900">${sc.data.target_price}</span>
                        <span className={`px-1.5 py-0.2 rounded text-[10px] font-bold border ${sc.bgCol} ${sc.textCol}`}>
                          {sc.data.upside_pct >= 0 ? '+' : ''}{sc.data.upside_pct}%
                        </span>
                      </div>
                    </div>
                    <div className="w-full bg-zinc-200/80 rounded-full h-2 overflow-hidden">
                      <div
                        className={`h-2 rounded-full ${sc.color} transition-all duration-500`}
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                  </div>
                )
              })}
            </div>
          ) : (
            <div className="rounded-xl border border-dashed border-edge p-5 text-center text-xs text-zinc-400">
              ไม่มีข้อมูล Scenario Model
            </div>
          )}
        </div>
      </div>

      {/* Forensic Footer: Piotroski F-Score & Capital Quality Strip */}
      {(piotroski || qualityMetrics) && (
        <div className="rounded-xl border border-sky-200/80 bg-gradient-to-r from-sky-50/70 via-surface to-panel p-4 space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-sky-200/50 pb-2">
            <div className="flex items-center gap-2">
              <span className="text-base">🩺</span>
              <span className="text-xs font-bold uppercase tracking-wider text-sky-900">
                Financial Health & Forensics (Piotroski & Capital Quality)
              </span>
            </div>

            {piotroski?.f_score != null && (
              <div className="flex items-center gap-1.5">
                <span className="text-xs text-zinc-600 font-medium">Piotroski F-Score:</span>
                <span className="px-2.5 py-0.5 rounded-full bg-sky-100 text-sky-800 text-xs font-bold border border-sky-200">
                  {piotroski.f_score} / 9
                </span>
                <span className="text-[11px] text-zinc-500 font-semibold uppercase">
                  ({piotroski.status || 'AVAILABLE'})
                </span>
              </div>
            )}
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
            {/* ROIC */}
            <div className="bg-surface/80 rounded-lg p-2.5 border border-edge/40">
              <span className="text-zinc-400 block text-[10px] uppercase">ROIC (ผลตอบแทนลงทุน)</span>
              <span className="font-bold text-zinc-900 text-sm">
                {qualityMetrics?.roic_pct != null ? `${qualityMetrics.roic_pct}%` : 'N/A'}
              </span>
            </div>

            {/* FCF Margin */}
            <div className="bg-surface/80 rounded-lg p-2.5 border border-edge/40">
              <span className="text-zinc-400 block text-[10px] uppercase">FCF Margin (กระแสเงินสด)</span>
              <span className="font-bold text-zinc-900 text-sm">
                {qualityMetrics?.fcf_margin_pct != null ? `${qualityMetrics.fcf_margin_pct}%` : 'N/A'}
              </span>
            </div>

            {/* FCF Yield */}
            <div className="bg-surface/80 rounded-lg p-2.5 border border-edge/40">
              <span className="text-zinc-400 block text-[10px] uppercase">FCF Yield / Mcap</span>
              <span className="font-bold text-zinc-900 text-sm">
                {qualityMetrics?.fcf_yield_pct != null ? `${qualityMetrics.fcf_yield_pct}%` : 'N/A'}
              </span>
            </div>

            {/* OCF to Net Income */}
            <div className="bg-surface/80 rounded-lg p-2.5 border border-edge/40">
              <span className="text-zinc-400 block text-[10px] uppercase">OCF / Net Income Quality</span>
              <span className="font-bold text-zinc-900 text-sm">
                {qualityMetrics?.ocf_to_net_income != null ? `${qualityMetrics.ocf_to_net_income}x` : 'N/A'}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
