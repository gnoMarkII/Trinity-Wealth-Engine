import { useState } from 'react'
import CapitalAllocationYieldMeter from './CapitalAllocationYieldMeter'
import ScenarioTargetSpreadBar from './ScenarioTargetSpreadBar'
import { IllustrativeValueScale } from './explainer/IllustrativeValueScale'
import { ContextTooltip } from '../macro/explainer/ContextTooltip'
import type {
  DCFResultDTO,
  MarginMetricItemDTO,
  ReverseDCFResultDTO,
  DCFScenarioDTO,
  SensitivityCellDTO,
  OperatingSensitivityCellDTO,
  MacroSensitivityCellDTO,
  OperatingSensitivityMatrixDTO,
  MacroSensitivityMatrixDTO,
  SensitivityMatrixDTO,
} from '../../api/types'

export interface ReverseDcfProps extends Partial<ReverseDCFResultDTO> {
  shares_outstanding?: number | null
  current_stock_price?: number | null
  terminal_growth_pct?: number | null
  beta?: number | null
}

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

export interface SectorKpisProps {
  sector?: string
  sector_name?: string
  sector_health_score?: number | null
  status?: string
  kpis?: Array<{
    name: string
    value: number | null
    unit?: string
    benchmark?: string
    benchmark_status?: string
    definition?: string
    period_basis?: string
  }>
}

interface Props {
  reverseDcf?: (ReverseDcfProps & {
    alternative_valuation?: any
    alternative_valuation_model?: string | null
    alternative_intrinsic_value?: number | null
    alternative_target_price_12m?: number | null
    alternative_upside_12m_pct?: number | null
    market_implied_roe_pct?: number | null
  }) | null
  dcf?: DCFResultDTO | null
  piotroski?: PiotroskiProps | null
  qualityMetrics?: CapitalQualityProps | null
  gaapOperatingMargin?: MarginMetricItemDTO | null
  nonGaapOperatingMargin?: MarginMetricItemDTO | null
  providerEbitMargin?: MarginMetricItemDTO | null
  valuationMarginSourceUsed?: string | null
  dcfDiscrepancyWarning?: string | null
  currency?: string
  netShareholderYieldPct?: number | null
  netBuybackYieldPct?: number | null
  dividendYieldPct?: number | null
  sectorKpis?: SectorKpisProps | null
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
  netShareholderYieldPct,
  netBuybackYieldPct,
  dividendYieldPct,
  sectorKpis,
}: Props) {
  // If no valuation data at all
  if (!reverseDcf && !dcf && !piotroski) return null

  // Interactive Scenario Selection State (Bear, Base, Bull)
  const [selectedScenarioKey, setSelectedScenarioKey] = useState<'bear' | 'base' | 'bull'>('base')
  const [isWaccModalOpen, setIsWaccModalOpen] = useState(false)
  const [isScheduleOpen, setIsScheduleOpen] = useState(false)
  const [isSensitivityOpen, setIsSensitivityOpen] = useState(false)
  const [sensitivityTab, setSensitivityTab] = useState<'valuation' | 'operating' | 'macro'>('valuation')
  const [selectedCell, setSelectedCell] = useState<{
    type: 'valuation' | 'operating' | 'macro'
    data: SensitivityCellDTO | OperatingSensitivityCellDTO | MacroSensitivityCellDTO
  } | null>(null)

  const dcfScenariosList = dcf?.scenarios
    ? [
        { key: 'bear' as const, label: 'Bear Case', color: 'bg-rose-500', textCol: 'text-rose-700', bgCol: 'bg-rose-50 border-rose-200', data: dcf.scenarios.bear },
        { key: 'base' as const, label: 'Base Case', color: 'bg-sky-500', textCol: 'text-sky-700', bgCol: 'bg-sky-50 border-sky-200', data: dcf.scenarios.base },
        { key: 'bull' as const, label: 'Bull Case', color: 'bg-emerald-500', textCol: 'text-emerald-700', bgCol: 'bg-emerald-50 border-emerald-200', data: dcf.scenarios.bull },
      ].filter((s): s is { key: 'bear' | 'base' | 'bull'; label: string; color: string; textCol: string; bgCol: string; data: DCFScenarioDTO & { target_price: number; upside_pct: number } } => s.data != null && s.data.target_price != null && s.data.upside_pct != null)
    : []

  const scenarios = dcfScenariosList.length > 0
    ? dcfScenariosList
    : (reverseDcf?.target_price_12m != null
        ? [
            {
              key: 'bear' as const,
              label: 'Bear Case',
              color: 'bg-rose-500',
              textCol: 'text-rose-700',
              bgCol: 'bg-rose-50 border-rose-200',
              data: {
                target_price: reverseDcf.consensus_target_low ?? (reverseDcf.target_price_12m * 0.8),
                upside_pct: reverseDcf.current_stock_price
                  ? (((reverseDcf.consensus_target_low ?? (reverseDcf.target_price_12m * 0.8)) - reverseDcf.current_stock_price) / reverseDcf.current_stock_price) * 100
                  : -15.0,
                margin_of_safety_pct: -15.0,
                implied_exit_ebitda_multiple: reverseDcf.implied_exit_ebitda_multiple ? reverseDcf.implied_exit_ebitda_multiple * 0.85 : 10.0,
              },
            },
            {
              key: 'base' as const,
              label: 'Base Case (Perpetuity DCF)',
              color: 'bg-sky-500',
              textCol: 'text-sky-700',
              bgCol: 'bg-sky-50 border-sky-200',
              data: {
                target_price: reverseDcf.target_price_12m,
                upside_pct: reverseDcf.upside_12m_pct ?? (reverseDcf.current_stock_price ? ((reverseDcf.target_price_12m - reverseDcf.current_stock_price) / reverseDcf.current_stock_price) * 100 : 0),
                margin_of_safety_pct: reverseDcf.current_margin_of_safety_pct ?? reverseDcf.margin_of_safety_pct ?? 15.0,
                implied_exit_ebitda_multiple: reverseDcf.implied_exit_ebitda_multiple ?? 12.5,
              },
            },
            {
              key: 'bull' as const,
              label: 'Bull Case (Exit Multiple)',
              color: 'bg-emerald-500',
              textCol: 'text-emerald-700',
              bgCol: 'bg-emerald-50 border-emerald-200',
              data: {
                target_price: reverseDcf.target_price_exit_multiple_12m ?? reverseDcf.consensus_target_high ?? (reverseDcf.target_price_12m * 1.3),
                upside_pct: reverseDcf.current_stock_price
                  ? (((reverseDcf.target_price_exit_multiple_12m ?? reverseDcf.consensus_target_high ?? (reverseDcf.target_price_12m * 1.3)) - reverseDcf.current_stock_price) / reverseDcf.current_stock_price) * 100
                  : 45.0,
                margin_of_safety_pct: 35.0,
                implied_exit_ebitda_multiple: reverseDcf.exit_multiple_used ?? 20.0,
              },
            },
          ]
        : [])

  const activeScenarioItem = scenarios.find(s => s.key === selectedScenarioKey)
  const activeScenarioData = activeScenarioItem?.data

  const targetPrice = selectedScenarioKey === 'base' || !activeScenarioData
    ? reverseDcf?.target_price_12m
    : activeScenarioData.target_price

  const upside = selectedScenarioKey === 'base' || !activeScenarioData
    ? reverseDcf?.upside_12m_pct
    : activeScenarioData.upside_pct

  const verdict = (reverseDcf?.valuation_verdict && reverseDcf.valuation_verdict !== 'unavailable')
    ? reverseDcf.valuation_verdict
    : (reverseDcf?.status === 'unavailable' || !reverseDcf ? 'unavailable' : (upside != null ? (upside > 15 ? 'undervalued' : upside < -15 ? 'overvalued' : 'fairly_valued') : 'unavailable'))

  const maxScenarioPrice = scenarios.length > 0 ? Math.max(...scenarios.map(s => s.data.target_price), 1) : 100

  const ebitLabel = reverseDcf?.ebit_margin_source_tier === 'filing_authoritative' ? 'GAAP Operating Margin' : 'Reported EBIT Margin'
  const ebitMeta = reverseDcf?.ebit_margin_fiscal_period
    ? `${reverseDcf.ebit_margin_fiscal_period}${reverseDcf.ebit_margin_period_type ? `, ${reverseDcf.ebit_margin_period_type}` : ''}`
    : ''

  const invalidationReasons = dcf?.invalidation_reasons || (reverseDcf?.actionability_reason ? [reverseDcf.actionability_reason] : [])

  const sanityAudit = reverseDcf?.valuation_sanity_audit || dcf?.valuation_sanity_audit
  const isMidYear = reverseDcf?.discounting_convention === 'mid_year' || dcf?.discounting_convention === 'mid_year'

  // Sensitivity Suite Integration (Valuation, Operating, Macro Shock)
  const rawSensitivity = reverseDcf?.sensitivity_matrix || dcf?.sensitivity_matrix
  const matrixSet = reverseDcf?.sensitivity_matrices
  const valuationMatrix: SensitivityMatrixDTO | null =
    matrixSet?.valuation ||
    ((rawSensitivity && !Array.isArray(rawSensitivity) && Array.isArray((rawSensitivity as any).grid))
      ? (rawSensitivity as SensitivityMatrixDTO)
      : null)
  const operatingMatrix: OperatingSensitivityMatrixDTO | null = matrixSet?.operating || null
  const macroMatrix: MacroSensitivityMatrixDTO | null = matrixSet?.macro || null
  const sensitivitySummary = reverseDcf?.sensitivity_summary || dcf?.sensitivity_summary
  const hasAnySensitivity = Boolean(valuationMatrix || operatingMatrix || macroMatrix)

  // 3Y Historical Actuals & 5Y Forecast Projections (Dynamic Multi-Scenario Flexing)
  const historical3Y = reverseDcf?.historical_financials_3y || []
  const forecast5Y = (
    reverseDcf?.explicit_forecast_5y_scenarios?.[selectedScenarioKey] ||
    reverseDcf?.explicit_forecast_5y ||
    []
  )
  const hasScheduleData = historical3Y.length > 0 || forecast5Y.length > 0
  const hasOpexBreakdown = forecast5Y.some(f => f.projected_rd != null || f.projected_sga != null || f.projected_gross_profit != null) || historical3Y.some(h => h.research_and_development != null || h.gross_profit != null)

  const latestCapexRatio =
    forecast5Y[forecast5Y.length - 1]?.capex_to_depreciation_ratio ??
    historical3Y[historical3Y.length - 1]?.capex_to_depreciation_ratio
  const latestLifecyclePhase =
    forecast5Y[forecast5Y.length - 1]?.reinvestment_lifecycle_phase ??
    historical3Y[historical3Y.length - 1]?.reinvestment_lifecycle_phase

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
              <div className="text-[10px] font-bold uppercase tracking-wider text-zinc-400">Wall Street Consensus</div>
              <div className="flex items-center gap-1 font-bold text-sky-700 text-base sm:text-lg font-mono justify-end">
                <span>{currency}{reverseDcf.consensus_target_price.toFixed(2)}</span>
                {reverseDcf.analyst_count != null && (
                  <span className="text-[10px] text-zinc-400 font-normal font-sans">({reverseDcf.analyst_count} analysts)</span>
                )}
              </div>
            </div>
          )}

          {/* Margin of Safety (MoS) Buy Target */}
          {reverseDcf?.mos_entry_price_today != null && (
            <div className="text-right border-r border-edge/60 pr-3 hidden sm:block">
              <div className="text-[10px] font-bold uppercase tracking-wider text-zinc-400">Buy Below (15% MoS)</div>
              <div className="flex items-center gap-1.5 justify-end">
                <span className="font-mono text-zinc-900 text-base sm:text-lg font-bold">{currency}{reverseDcf.mos_entry_price_today.toFixed(2)}</span>
                <span
                  className={`text-[10px] px-1.5 py-0.5 rounded font-bold border ${
                    reverseDcf.is_in_buy_zone
                      ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                      : 'bg-amber-50 text-amber-700 border-amber-200'
                  }`}
                >
                  {reverseDcf.is_in_buy_zone ? '🟢 Buy Zone' : `-${reverseDcf.required_pullback_pct?.toFixed(0)}%`}
                </span>
              </div>
            </div>
          )}

          {/* DCF Fair Value (Base Case, Selected Scenario, or Exit Multiple) */}
          <div className="text-right">
            <div className="text-[10px] font-bold uppercase tracking-wider text-zinc-400">
              {selectedScenarioKey === 'base'
                ? (dcf?.scenarios?.base?.target_price != null ? 'DCF Fair Value (Base)' : 'DCF 12M Outlook')
                : `DCF Fair Value (${selectedScenarioKey.toUpperCase()})`}
            </div>
            {targetPrice != null ? (
              <div className="flex items-center gap-2 justify-end">
                <span className="font-mono font-black text-xl sm:text-2xl text-zinc-900 tracking-tight">
                  {currency}{targetPrice.toFixed(2)}
                </span>
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
              <div className="font-bold text-zinc-400 text-base justify-end">
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

            {/* Reliability Tier & Mid-Year Badges */}
            {(isMidYear || (sanityAudit?.model_reliability_tier && sanityAudit.model_reliability_tier !== 'unavailable') || latestCapexRatio != null || reverseDcf?.company_archetype != null) && (
              <div className="flex flex-wrap items-center gap-1.5 mt-1 justify-end">
                {reverseDcf?.company_archetype && (
                  <span
                    title={
                      reverseDcf.archetype_rationale ||
                      `Company Archetype: ${reverseDcf.company_archetype}${
                        reverseDcf.is_mid_cycle_normalized
                          ? ` (Mid-Cycle Margin: ${reverseDcf.normalized_mid_cycle_ebit_margin_pct?.toFixed(1)}%)`
                          : ''
                      }`
                    }
                    className={`px-2 py-0.5 rounded text-[10px] font-bold border flex items-center gap-1 ${
                      reverseDcf.company_archetype === 'high_growth_tech'
                        ? 'bg-indigo-50 text-indigo-700 border-indigo-200'
                        : reverseDcf.company_archetype === 'cyclical'
                        ? reverseDcf.is_mid_cycle_normalized
                          ? 'bg-amber-100 text-amber-900 border-amber-300 font-extrabold'
                          : 'bg-amber-50 text-amber-800 border-amber-200'
                        : reverseDcf.company_archetype === 'mature_cash_cow'
                        ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
                        : reverseDcf.company_archetype === 'turnaround'
                        ? 'bg-rose-50 text-rose-800 border-rose-200'
                        : 'bg-zinc-100 text-zinc-700 border-zinc-200'
                    }`}
                  >
                    <span>
                      {reverseDcf.company_archetype === 'high_growth_tech' && '🚀 High-Growth Tech'}
                      {reverseDcf.company_archetype === 'cyclical' && (
                        reverseDcf.is_mid_cycle_normalized
                          ? `🔄 Cyclical (Mid-Cycle: ${reverseDcf.normalized_mid_cycle_ebit_margin_pct?.toFixed(1)}%)`
                          : '🔄 Cyclical'
                      )}
                      {reverseDcf.company_archetype === 'mature_cash_cow' && '🐄 Mature Cash Cow'}
                      {reverseDcf.company_archetype === 'turnaround' && '🛠️ Turnaround'}
                      {reverseDcf.company_archetype === 'general' && '🏢 Balanced Corporate'}
                    </span>
                  </span>
                )}
                {isMidYear && (
                  <span
                    title="Mid-Year Discounting Convention (t - 0.5): Reflects cash inflows occurring evenly throughout the year, removing ~3-5% downward bias."
                    className="px-2 py-0.5 rounded text-[10px] font-bold bg-sky-50 text-sky-700 border border-sky-200"
                  >
                    ⏱️ Mid-Year Convention
                  </span>
                )}
                {sanityAudit?.model_reliability_tier && sanityAudit.model_reliability_tier !== 'unavailable' && (
                  <span
                    title={`Reliability Score: ${sanityAudit.reliability_score ?? 'N/A'}/100. Terminal Value accounts for ${sanityAudit.terminal_value_pct_of_ev ?? 'N/A'}% of EV.`}
                    className={`px-2 py-0.5 rounded text-[10px] font-bold border flex items-center gap-1 ${
                      sanityAudit.model_reliability_tier === 'high'
                        ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                        : sanityAudit.model_reliability_tier === 'moderate'
                        ? 'bg-amber-50 text-amber-700 border-amber-200'
                        : 'bg-rose-50 text-rose-700 border-rose-200'
                    }`}
                  >
                    <span>
                      {sanityAudit.model_reliability_tier === 'high'
                        ? '🟢 High Model Reliability'
                        : sanityAudit.model_reliability_tier === 'moderate'
                        ? '🟡 Moderate Quality'
                        : '🔴 Terminal Dependent (>85%)'}
                    </span>
                    {sanityAudit.terminal_value_pct_of_ev != null && (
                      <span className="font-normal opacity-80">(TV {sanityAudit.terminal_value_pct_of_ev}%)</span>
                    )}
                  </span>
                )}
                {sanityAudit?.implied_exit_ebitda_multiple != null && (
                  <span
                    title={`Implied Exit EV/EBITDA Multiple: ${sanityAudit.implied_exit_ebitda_multiple}x Year 5 EBITDA. Standard range: 8x-18x.`}
                    className={`px-2 py-0.5 rounded text-[10px] font-bold border flex items-center gap-1 ${
                      sanityAudit.implied_exit_ebitda_multiple > 25
                        ? 'bg-rose-50 text-rose-700 border-rose-200'
                        : sanityAudit.implied_exit_ebitda_multiple > 18
                        ? 'bg-amber-50 text-amber-700 border-amber-200'
                        : 'bg-emerald-50 text-emerald-700 border-emerald-200'
                    }`}
                  >
                    <span>📊 Exit: {sanityAudit.implied_exit_ebitda_multiple.toFixed(1)}x EBITDA</span>
                  </span>
                )}
                {latestCapexRatio != null && (
                  <span
                    title={`CapEx / D&A Ratio: ${latestCapexRatio.toFixed(2)}x. Reinvestment Phase: ${latestLifecyclePhase || 'N/A'}`}
                    className={`px-2 py-0.5 rounded text-[10px] font-bold border flex items-center gap-1 ${
                      latestCapexRatio >= 1.5
                        ? 'bg-sky-50 text-sky-700 border-sky-200'
                        : latestCapexRatio >= 0.85
                        ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                        : 'bg-rose-50 text-rose-700 border-rose-200'
                    }`}
                  >
                    <span>
                      🏗️ CapEx/D&A: {latestCapexRatio.toFixed(2)}x (
                      {latestCapexRatio >= 1.5 ? 'Expansion' : latestCapexRatio >= 0.85 ? 'Sustainable' : 'Underinvestment'}
                      )
                    </span>
                  </span>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Beginner Illustrative Value Scale (100-Baht Model) */}
      {reverseDcf?.current_stock_price != null && (reverseDcf.intrinsic_value_today != null || targetPrice != null) && (
        <IllustrativeValueScale
          currentPrice={reverseDcf.current_stock_price}
          intrinsicValue={reverseDcf.intrinsic_value_today ?? targetPrice!}
          currencySymbol={currency}
        />
      )}

      {/* Scenario Target Spread Bar (Bear vs Base vs Bull) */}
      {scenarios.length > 0 && (
        <ScenarioTargetSpreadBar
          scenarios={scenarios}
          selectedScenarioKey={selectedScenarioKey}
          onSelectScenario={setSelectedScenarioKey}
          currentPrice={reverseDcf?.current_stock_price}
          currency={currency}
        />
      )}

      {/* 1. ⚖️ Dual Valuation Method Comparison Bar */}
      {(targetPrice != null || reverseDcf?.target_price_exit_multiple_12m != null || reverseDcf?.intrinsic_value_exit_multiple != null) && (
        <div className="rounded-xl border border-sky-200/80 bg-gradient-to-br from-sky-50/50 via-white to-sky-50/20 p-4 space-y-2.5 shadow-2xs">
          <div className="flex items-center justify-between text-xs font-bold uppercase tracking-wider text-sky-900 border-b border-sky-100 pb-2">
            <span className="flex items-center gap-1.5">
              <span>⚖️</span>
              <span>Dual Valuation Method Comparison</span>
            </span>
            <span className="text-[10px] text-zinc-400 font-normal normal-case hidden sm:inline">
              เปรียบเทียบราคาเป้าหมาย 12M ระหว่างวิธี Perpetuity Growth กับ Exit Multiple
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {/* Gordon Perpetuity Growth Model */}
            <div className="rounded-lg border border-sky-100 bg-white/95 p-3 space-y-1">
              <div className="flex items-center justify-between text-xs">
                <span className="font-semibold text-zinc-800">1. Perpetuity Growth DCF (Gordon Model)</span>
                <div className="flex items-center gap-1.5 font-mono">
                  <span className="text-[10px] text-zinc-500">
                    g = {reverseDcf?.terminal_growth_pct != null ? `${reverseDcf.terminal_growth_pct.toFixed(1)}%` : '2.5%'}
                  </span>
                  {reverseDcf?.implied_exit_ebitda_multiple != null && (
                    <span
                      title={`Gordon Growth Terminal Value implies an exit multiple of ${reverseDcf.implied_exit_ebitda_multiple}x Year 5 EBITDA`}
                      className={`px-1.5 py-0.5 rounded text-[10px] font-bold border ${
                        reverseDcf.implied_exit_ebitda_multiple > 25
                          ? 'bg-rose-50 text-rose-700 border-rose-200'
                          : reverseDcf.implied_exit_ebitda_multiple > 18
                          ? 'bg-amber-50 text-amber-700 border-amber-200'
                          : 'bg-emerald-50 text-emerald-700 border-emerald-200'
                      }`}
                    >
                      Exit: {reverseDcf.implied_exit_ebitda_multiple.toFixed(1)}x EBITDA
                    </span>
                  )}
                </div>
              </div>
              <div className="flex items-baseline justify-between mt-1">
                <div className="flex items-baseline gap-1.5">
                  <span className="text-xl font-extrabold text-zinc-900 font-mono">
                    {reverseDcf?.target_price_12m != null ? `${currency}${reverseDcf.target_price_12m.toFixed(2)}` : (targetPrice != null ? `${currency}${targetPrice.toFixed(2)}` : 'N/A')}
                  </span>
                  {reverseDcf?.intrinsic_value_today != null && (
                    <span className="text-[11px] text-zinc-400 font-normal">
                      (Today: {currency}{reverseDcf.intrinsic_value_today.toFixed(2)})
                    </span>
                  )}
                </div>
                {reverseDcf?.upside_12m_pct != null && (
                  <span
                    className={`px-2 py-0.5 rounded-full text-xs font-bold border ${
                      reverseDcf.upside_12m_pct >= 15
                        ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                        : reverseDcf.upside_12m_pct <= -15
                        ? 'bg-rose-50 text-rose-700 border-rose-200'
                        : 'bg-sky-50 text-sky-700 border-sky-200'
                    }`}
                  >
                    {reverseDcf.upside_12m_pct >= 0 ? '+' : ''}{reverseDcf.upside_12m_pct.toFixed(1)}%
                  </span>
                )}
              </div>

              {/* MoS Entry Price Target */}
              {reverseDcf?.mos_entry_price_today != null && (
                <div className="flex items-center justify-between pt-1.5 border-t border-sky-100/60 text-[11px]">
                  <span className="text-zinc-500 font-medium">🎯 Buy Below (15% MoS):</span>
                  <span className="font-mono font-bold text-zinc-800 flex items-center gap-1.5">
                    {currency}{reverseDcf.mos_entry_price_today.toFixed(2)}
                    {reverseDcf.is_in_buy_zone ? (
                      <span className="px-1.5 py-0.5 rounded text-[9px] bg-emerald-100 text-emerald-800 font-bold border border-emerald-200">
                        In Buy Zone
                      </span>
                    ) : (
                      <span className="px-1.5 py-0.5 rounded text-[9px] bg-amber-100 text-amber-800 font-bold border border-amber-200">
                        Pullback -{reverseDcf.required_pullback_pct?.toFixed(0)}%
                      </span>
                    )}
                  </span>
                </div>
              )}
            </div>

            {/* Exit Multiple Method */}
            <div className="rounded-lg border border-sky-100 bg-white/95 p-3 space-y-1">
              <div className="flex items-center justify-between text-xs">
                <span className="font-semibold text-zinc-800">
                  2. Exit Multiple Method ({reverseDcf?.exit_multiple_used ?? 16.0}x EV/EBITDA)
                </span>
                <div className="flex items-center gap-1.5 font-mono">
                  {reverseDcf?.implied_perpetual_growth_from_exit_pct != null && (
                    <span
                      title={`Exit Multiple of ${reverseDcf.exit_multiple_used}x EBITDA implies an equivalent perpetual growth rate of ${reverseDcf.implied_perpetual_growth_from_exit_pct}%`}
                      className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-sky-50 text-sky-700 border border-sky-200"
                    >
                      Implied g: {reverseDcf.implied_perpetual_growth_from_exit_pct.toFixed(1)}%
                    </span>
                  )}
                  <span className="text-[10px] text-zinc-500">
                    {reverseDcf?.exit_multiple_used ?? 16.0}x EBITDA
                  </span>
                </div>
              </div>
              <div className="flex items-baseline justify-between mt-1">
                <div className="flex items-baseline gap-1.5">
                  <span className="text-xl font-extrabold text-sky-950 font-mono">
                    {reverseDcf?.target_price_exit_multiple_12m != null
                      ? `${currency}${reverseDcf.target_price_exit_multiple_12m.toFixed(2)}`
                      : (reverseDcf?.intrinsic_value_exit_multiple != null
                        ? `${currency}${(reverseDcf.intrinsic_value_exit_multiple * 1.08).toFixed(2)}`
                        : 'N/A')}
                  </span>
                  {reverseDcf?.intrinsic_value_exit_multiple != null && (
                    <span className="text-[11px] text-zinc-400 font-normal">
                      (Today: {currency}{reverseDcf.intrinsic_value_exit_multiple.toFixed(2)})
                    </span>
                  )}
                </div>
                {(() => {
                  const exitTarget = reverseDcf?.target_price_exit_multiple_12m
                  const currentPrice = reverseDcf?.current_stock_price
                  if (exitTarget != null && currentPrice != null && currentPrice > 0) {
                    const exitUpside = ((exitTarget - currentPrice) / currentPrice) * 100
                    return (
                      <span
                        className={`px-2 py-0.5 rounded-full text-xs font-bold border ${
                          exitUpside >= 15
                            ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                            : exitUpside <= -15
                            ? 'bg-rose-50 text-rose-700 border-rose-200'
                            : 'bg-sky-50 text-sky-700 border-sky-200'
                        }`}
                      >
                        {exitUpside >= 0 ? '+' : ''}{exitUpside.toFixed(1)}%
                      </span>
                    )
                  }
                  return null
                })()}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 2. 📊 Enterprise to Equity Value Bridge (Visual Waterfall) */}
      {reverseDcf?.enterprise_value != null && (
        <div className="rounded-xl border border-edge/70 bg-surface/70 p-4 space-y-3 shadow-2xs">
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-edge/40 pb-2">
            <div className="flex items-center gap-2">
              <span className="text-base">📊</span>
              <div>
                <span className="text-xs font-bold uppercase tracking-wider text-zinc-800 block">
                  Enterprise to Equity Value Bridge (Visual Waterfall)
                </span>
                <span className="text-[10px] text-zinc-500">
                  ขั้นตอนการเชื่อมโยงมูลค่าจาก PV of Explicit Cash Flows สู่ Enterprise Value และ Equity Value ต่อหุ้น
                </span>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2.5 items-stretch text-xs">
            {/* Step 1: PV 5Y FCF */}
            <div className="rounded-lg border border-sky-100 bg-sky-50/40 p-2.5 flex flex-col justify-between">
              <span className="text-[10px] uppercase font-bold text-sky-800">1. PV 5Y FCF</span>
              <div className="my-1">
                <div className="text-sm font-extrabold text-zinc-900 font-mono">
                  {formatLargeNum(reverseDcf.sum_pv_5y_fcf, currency)}
                </div>
                {reverseDcf.enterprise_value > 0 && reverseDcf.sum_pv_5y_fcf != null && (
                  <div className="text-[10px] text-sky-700 font-semibold font-mono">
                    ({((reverseDcf.sum_pv_5y_fcf / reverseDcf.enterprise_value) * 100).toFixed(1)}% of EV)
                  </div>
                )}
              </div>
              <span className="text-[9px] text-zinc-400">Explicit Forecast</span>
            </div>

            {/* Step 2: PV Terminal Value */}
            <div className="rounded-lg border border-indigo-100 bg-indigo-50/40 p-2.5 flex flex-col justify-between">
              <span className="text-[10px] uppercase font-bold text-indigo-800">(+) 2. PV Terminal Value</span>
              <div className="my-1">
                <div className="text-sm font-extrabold text-zinc-900 font-mono">
                  {formatLargeNum(reverseDcf.terminal_value_pv, currency)}
                </div>
                {reverseDcf.enterprise_value > 0 && reverseDcf.terminal_value_pv != null && (
                  <div className="text-[10px] text-indigo-700 font-semibold font-mono">
                    ({((reverseDcf.terminal_value_pv / reverseDcf.enterprise_value) * 100).toFixed(1)}% of EV
                    {reverseDcf.implied_exit_ebitda_multiple != null ? ` | ${reverseDcf.implied_exit_ebitda_multiple.toFixed(1)}x EBITDA` : ''})
                  </div>
                )}
              </div>
              <span className="text-[9px] text-zinc-400">Perpetuity Horizon</span>
            </div>

            {/* Step 3: Enterprise Value */}
            <div className="rounded-lg border border-edge/70 bg-zinc-100/70 p-2.5 flex flex-col justify-between">
              <span className="text-[10px] uppercase font-bold text-zinc-700">(=) 3. Enterprise Value</span>
              <div className="my-1">
                <div className="text-sm font-extrabold text-zinc-900 font-mono">
                  {formatLargeNum(reverseDcf.enterprise_value, currency)}
                </div>
                <div className="text-[10px] text-zinc-500 font-semibold font-mono">
                  (100.0% Firm EV)
                </div>
              </div>
              <span className="text-[9px] text-zinc-400">Operating Assets</span>
            </div>

            {/* Step 4: Net Cash / Debt */}
            <div className="rounded-lg border border-edge/70 bg-surface p-2.5 flex flex-col justify-between">
              <span className="text-[10px] uppercase font-bold text-zinc-700">
                {(reverseDcf.net_cash_debt ?? 0) >= 0 ? '(+) 4. Net Cash' : '(-) 4. Net Debt'}
              </span>
              <div className="my-1">
                <div className={`text-sm font-extrabold font-mono ${(reverseDcf.net_cash_debt ?? 0) >= 0 ? 'text-emerald-700' : 'text-rose-700'}`}>
                  {(reverseDcf.net_cash_debt ?? 0) >= 0 ? '+' : ''}
                  {formatLargeNum(reverseDcf.net_cash_debt, currency)}
                </div>
                <div className="text-[10px] text-zinc-500 font-semibold">
                  {(reverseDcf.net_cash_debt ?? 0) >= 0 ? 'Net Cash Position' : 'Net Debt Burden'}
                </div>
              </div>
              <span className="text-[9px] text-zinc-400">Balance Sheet Net</span>
            </div>

            {/* Step 5: Equity Value & Implied Target */}
            <div className="rounded-lg border border-emerald-200 bg-emerald-50/50 p-2.5 flex flex-col justify-between col-span-2 sm:col-span-1">
              <span className="text-[10px] uppercase font-bold text-emerald-800">(=) 5. Equity Value</span>
              <div className="my-1">
                <div className="text-sm font-extrabold text-emerald-900 font-mono">
                  {formatLargeNum(reverseDcf.equity_value, currency)}
                </div>
                {reverseDcf.shares_outstanding != null && reverseDcf.shares_outstanding > 0 && (
                  <div className="text-[10px] text-emerald-700 font-semibold font-mono">
                    ÷ {formatLargeNum(reverseDcf.shares_outstanding, '').trim()} Shares
                  </div>
                )}
              </div>
              <span className="text-[9px] font-bold text-emerald-800">
                = {currency}{reverseDcf.target_price_12m != null ? reverseDcf.target_price_12m.toFixed(2) : (reverseDcf.intrinsic_value_today != null ? reverseDcf.intrinsic_value_today.toFixed(2) : '-')} / sh
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Valuation Sanity Audit & Exit Multiple Strip */}
      {sanityAudit && (sanityAudit.implied_exit_fcf_multiple != null || (sanityAudit.sanity_warnings && sanityAudit.sanity_warnings.length > 0)) && (
        <div className="rounded-xl border border-edge/60 bg-surface/50 p-3.5 space-y-2 text-xs">
          <div className="flex flex-wrap items-center justify-between gap-2 text-zinc-600 font-semibold">
            <span className="flex items-center gap-1.5">
              <span>🛡️ Valuation Health & Multiple Cross-Check</span>
              {sanityAudit.reliability_score != null && (
                <span className="text-[10px] px-1.5 py-0.2 rounded bg-zinc-200 text-zinc-700 font-bold">
                  Score: {sanityAudit.reliability_score}/100
                </span>
              )}
            </span>
            {sanityAudit.implied_exit_fcf_multiple != null && (
              <span className="text-zinc-500 font-normal text-[11px]">
                Implied Exit Multiple: <strong className="text-zinc-900">{sanityAudit.implied_exit_fcf_multiple}x FCF</strong>
                {sanityAudit.benchmark_sector_multiple != null && (
                  <span> (Sector Median: <strong>{sanityAudit.benchmark_sector_multiple}x</strong>)</span>
                )}
              </span>
            )}
          </div>
          {sanityAudit.sanity_warnings && sanityAudit.sanity_warnings.length > 0 && (
            <ul className="list-disc list-inside space-y-0.5 text-zinc-600 text-[11px]">
              {sanityAudit.sanity_warnings.map((w, idx) => (
                <li key={idx} className={w.includes('Terminal Value') && sanityAudit.is_terminal_dependent ? 'text-amber-800 font-medium' : ''}>
                  {w}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

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

      {/* Main 2-Pane Workbench: Expectation Gap & Scenarios Visualizer */}
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
              {(() => {
                const altModel = reverseDcf.alternative_valuation_model || reverseDcf.alternative_valuation?.model_name
                const altVal = reverseDcf.alternative_intrinsic_value ?? reverseDcf.alternative_valuation?.intrinsic_value
                const altUpside = reverseDcf.alternative_upside_12m_pct ?? reverseDcf.alternative_valuation?.upside_pct
                if (!altModel && altVal == null) return null

                const modelLabel = altModel === 'justified_pbv'
                  ? 'Justified P/BV (Financials)'
                  : altModel === 'ddm_yield_spread'
                  ? 'DDM Yield Spread (REIT)'
                  : (altModel || 'Alternative Sector Valuation')

                return (
                  <div className="flex justify-between text-zinc-600 border-b border-edge/30 pb-1.5">
                    <span>{modelLabel}:</span>
                    <span className="font-semibold text-emerald-800">
                      {currency}{altVal != null ? altVal.toFixed(2) : 'N/A'}
                      {altUpside != null && (
                        <span className="text-emerald-700 font-bold ml-1">
                          ({altUpside >= 0 ? '+' : ''}{altUpside.toFixed(1)}%)
                        </span>
                      )}
                    </span>
                  </div>
                )
              })()}
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

        {/* Right Pane: DCF Scenarios Visualizer (5 Cols) & Interactive Toggle */}
        <div className="lg:col-span-5 space-y-4">
          <div className="flex items-center justify-between text-xs font-semibold text-zinc-500 uppercase tracking-wider">
            <span title="Generic 5Y FCF Growth Baseline Model">📊 DCF Scenarios Visualizer</span>
            <div className="flex items-center gap-1.5">
              {dcf?.wacc_pct != null && (
                <button
                  type="button"
                  onClick={() => setIsWaccModalOpen(true)}
                  className="text-sky-700 hover:text-sky-900 font-bold lowercase hover:underline cursor-pointer flex items-center gap-1"
                  title="คลิกเพื่อดูการแจกแจงส่วนประกอบ WACC (CAPM Decomposition)"
                >
                  <span>wacc {dcf.wacc_pct}%</span>
                  <span className="text-[10px]">ℹ️</span>
                </button>
              )}
            </div>
          </div>

          {/* Interactive Case Selector (Bear / Base / Bull Toggle) */}
          {scenarios.length > 0 && (
            <div className="flex items-center rounded-lg bg-zinc-100 p-1 border border-edge/60 text-xs">
              <button
                type="button"
                onClick={() => setSelectedScenarioKey('bear')}
                className={`flex-1 py-1 px-2 rounded-md font-semibold transition-all ${
                  selectedScenarioKey === 'bear'
                    ? 'bg-white text-rose-700 shadow-xs border border-rose-200'
                    : 'text-zinc-600 hover:text-zinc-900'
                }`}
              >
                Bear
              </button>
              <button
                type="button"
                onClick={() => setSelectedScenarioKey('base')}
                className={`flex-1 py-1 px-2 rounded-md font-semibold transition-all ${
                  selectedScenarioKey === 'base'
                    ? 'bg-white text-sky-700 shadow-xs border border-sky-200'
                    : 'text-zinc-600 hover:text-zinc-900'
                }`}
              >
                Base
              </button>
              <button
                type="button"
                onClick={() => setSelectedScenarioKey('bull')}
                className={`flex-1 py-1 px-2 rounded-md font-semibold transition-all ${
                  selectedScenarioKey === 'bull'
                    ? 'bg-white text-emerald-700 shadow-xs border border-emerald-200'
                    : 'text-zinc-600 hover:text-zinc-900'
                }`}
              >
                Bull
              </button>
            </div>
          )}

          {scenarios.length > 0 ? (
            <div className="space-y-3 rounded-xl border border-edge/60 bg-surface/70 p-4">
              {scenarios.map(sc => {
                const pct = Math.min(100, Math.max(12, (sc.data.target_price / maxScenarioPrice) * 100))
                const isSelected = selectedScenarioKey === sc.key
                return (
                  <div
                    key={sc.key}
                    onClick={() => setSelectedScenarioKey(sc.key as 'bear' | 'base' | 'bull')}
                    className={`space-y-1 p-2 rounded-lg cursor-pointer transition-all ${
                      isSelected ? 'bg-sky-50/60 ring-1 ring-sky-300' : 'hover:bg-zinc-50/60'
                    }`}
                  >
                    <div className="flex items-center justify-between text-xs">
                      <span className={`font-medium ${isSelected ? 'text-zinc-900 font-bold' : 'text-zinc-700'}`}>
                        {sc.label}
                        {isSelected && <span className="ml-1 text-[10px] text-sky-700">(Active)</span>}
                      </span>
                      <div className="flex items-center gap-1.5">
                        <span className="font-bold text-zinc-900">{currency}{typeof sc.data.target_price === 'number' ? sc.data.target_price.toFixed(2) : sc.data.target_price}</span>
                        <span className={`px-1.5 py-0.2 rounded text-[10px] font-bold border ${sc.bgCol} ${sc.textCol}`}>
                          {sc.data.upside_pct >= 0 ? '+' : ''}{typeof sc.data.upside_pct === 'number' ? sc.data.upside_pct.toFixed(1) : sc.data.upside_pct}%
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

      {/* 3. 📅 3Y History Actuals | 5Y Staged Forecast FCF Schedule Table (8 Columns) */}
      {hasScheduleData && (
        <div className="rounded-xl border border-edge/70 bg-surface/60 p-4 space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <span className="text-base">📅</span>
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold uppercase tracking-wider text-zinc-800 block">
                    3Y History Actuals | 5Y Staged Forecast FCF Schedule Table
                  </span>
                  <span
                    className={`px-2 py-0.5 rounded text-[10px] font-bold border ${
                      selectedScenarioKey === 'bull'
                        ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                        : selectedScenarioKey === 'bear'
                        ? 'bg-rose-50 text-rose-700 border-rose-200'
                        : 'bg-sky-50 text-sky-700 border-sky-200'
                    }`}
                  >
                    Scenario: {selectedScenarioKey.toUpperCase()}
                  </span>
                </div>
                <span className="text-[10px] text-zinc-500">
                  ตาราง 8 คอลัมน์แจกแจงงบจริง 3 ปี ย้อนหลัง และประมาณการกระแสเงินสด 5 ปี พร้อม Net CapEx และ ΔNWC
                </span>
              </div>
            </div>
            <button
              type="button"
              onClick={() => setIsScheduleOpen(!isScheduleOpen)}
              className="text-xs font-semibold px-2.5 py-1 rounded-lg border border-edge/80 bg-white hover:bg-zinc-50 text-sky-700 transition-all flex items-center gap-1 shadow-2xs"
            >
              <span>{isScheduleOpen ? 'ย่อตารางงบ' : 'แสดงตารางงบกระแสเงินสด (8 คอลัมน์)'}</span>
              <span className="text-[10px]">{isScheduleOpen ? '▲' : '▼'}</span>
            </button>
          </div>

          {isScheduleOpen && (
            <div className="space-y-3 pt-1 animate-fade-in overflow-x-auto">
              <table className="w-full text-right border-collapse text-xs">
                <thead>
                  <tr className="border-b border-edge/60 bg-zinc-50/70">
                    <th className="p-2 text-left text-[11px] font-bold text-zinc-700 font-sans">
                      Financial Line Item
                    </th>
                    {/* 3 Historical Columns */}
                    {historical3Y.map((item, idx) => (
                      <th key={`hist-${idx}`} className="p-2 text-[11px] font-bold text-zinc-600 bg-zinc-100/70 font-mono">
                        {item.fiscal_year || `Year -${3 - idx}`}A
                        <span className="block text-[9px] font-normal text-zinc-400">Actual</span>
                      </th>
                    ))}
                    {/* 5 Forecast Columns */}
                    {forecast5Y.map((f, idx) => (
                      <th key={`fore-${idx}`} className="p-2 text-[11px] font-bold text-sky-800 bg-sky-50/40 font-mono">
                        Year {f.year_index}E
                        <span className="block text-[9px] font-normal text-sky-600">Forecast</span>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-edge/30">
                  {/* Revenue */}
                  <tr className="hover:bg-zinc-50/40">
                    <td className="p-2 text-left font-semibold text-zinc-800">Revenue ({currency}M)</td>
                    {historical3Y.map((h, i) => (
                      <td key={i} className="p-2 font-mono text-zinc-700 bg-zinc-50/30">
                        {formatLargeNum(h.revenue, currency)}
                      </td>
                    ))}
                    {forecast5Y.map((f, i) => (
                      <td key={i} className="p-2 font-mono text-zinc-900 font-medium">
                        {formatLargeNum(f.projected_revenue, currency)}
                      </td>
                    ))}
                  </tr>

                  {/* Revenue Growth % */}
                  <tr className="hover:bg-zinc-50/40 text-[11px]">
                    <td className="p-2 text-left text-zinc-500 pl-4">└ YoY Revenue Growth (%)</td>
                    {historical3Y.map((h, i) => (
                      <td key={i} className="p-2 font-mono text-zinc-500 bg-zinc-50/30">
                        {h.revenue_growth_pct != null ? `${h.revenue_growth_pct.toFixed(1)}%` : '-'}
                      </td>
                    ))}
                    {forecast5Y.map((f, i) => (
                      <td key={i} className="p-2 font-mono text-sky-700 font-semibold">
                        {f.projected_revenue_growth_pct != null ? `${f.projected_revenue_growth_pct.toFixed(1)}%` : '-'}
                      </td>
                    ))}
                  </tr>

                  {/* Functional OpEx Breakdown: Gross Profit, R&D, SG&A */}
                  {hasOpexBreakdown && (
                    <>
                      {/* Gross Profit */}
                      <tr className="hover:bg-zinc-50/40 bg-zinc-50/20 text-zinc-800">
                        <td className="p-2 text-left font-semibold">Gross Profit</td>
                        {historical3Y.map((h, i) => (
                          <td key={i} className="p-2 font-mono text-zinc-700 bg-zinc-50/30">
                            {h.gross_profit != null ? formatLargeNum(h.gross_profit, currency) : '-'}
                          </td>
                        ))}
                        {forecast5Y.map((f, i) => (
                          <td key={i} className="p-2 font-mono text-zinc-900 font-medium">
                            {f.projected_gross_profit != null ? formatLargeNum(f.projected_gross_profit, currency) : '-'}
                          </td>
                        ))}
                      </tr>

                      {/* Gross Margin % */}
                      <tr className="hover:bg-zinc-50/40 text-[11px] text-zinc-500">
                        <td className="p-2 text-left pl-4">└ Gross Margin (%)</td>
                        {historical3Y.map((h, i) => (
                          <td key={i} className="p-2 font-mono text-zinc-500 bg-zinc-50/30">
                            {h.gross_margin_pct != null ? `${h.gross_margin_pct.toFixed(1)}%` : '-'}
                          </td>
                        ))}
                        {forecast5Y.map((f, i) => (
                          <td key={i} className="p-2 font-mono text-zinc-600">
                            {f.projected_gross_margin_pct != null ? `${f.projected_gross_margin_pct.toFixed(1)}%` : '-'}
                          </td>
                        ))}
                      </tr>

                      {/* R&D Expense */}
                      <tr className="hover:bg-zinc-50/40 text-zinc-700">
                        <td className="p-2 text-left pl-4">(-) Research & Development (R&D)</td>
                        {historical3Y.map((h, i) => (
                          <td key={i} className="p-2 font-mono text-zinc-600 bg-zinc-50/30">
                            {h.research_and_development != null ? formatLargeNum(h.research_and_development, currency) : '-'}
                          </td>
                        ))}
                        {forecast5Y.map((f, i) => (
                          <td key={i} className="p-2 font-mono text-zinc-700">
                            {f.projected_rd != null ? formatLargeNum(f.projected_rd, currency) : '-'}
                          </td>
                        ))}
                      </tr>

                      {/* R&D % of Revenue */}
                      <tr className="hover:bg-zinc-50/40 text-[11px] text-zinc-400">
                        <td className="p-2 text-left pl-6">└ R&D % of Revenue</td>
                        {historical3Y.map((h, i) => (
                          <td key={i} className="p-2 font-mono bg-zinc-50/30">
                            {h.rd_pct_of_rev != null ? `${h.rd_pct_of_rev.toFixed(1)}%` : '-'}
                          </td>
                        ))}
                        {forecast5Y.map((f, i) => (
                          <td key={i} className="p-2 font-mono text-sky-600">
                            {f.projected_rd_pct_of_rev != null ? `${f.projected_rd_pct_of_rev.toFixed(1)}%` : '-'}
                          </td>
                        ))}
                      </tr>

                      {/* SG&A Expense */}
                      <tr className="hover:bg-zinc-50/40 text-zinc-700">
                        <td className="p-2 text-left pl-4">(-) SG&A Expenses</td>
                        {historical3Y.map((h, i) => (
                          <td key={i} className="p-2 font-mono text-zinc-600 bg-zinc-50/30">
                            {h.selling_general_admin != null ? formatLargeNum(h.selling_general_admin, currency) : '-'}
                          </td>
                        ))}
                        {forecast5Y.map((f, i) => (
                          <td key={i} className="p-2 font-mono text-zinc-700">
                            {f.projected_sga != null ? formatLargeNum(f.projected_sga, currency) : '-'}
                          </td>
                        ))}
                      </tr>

                      {/* SG&A % of Revenue */}
                      <tr className="hover:bg-zinc-50/40 text-[11px] text-zinc-400">
                        <td className="p-2 text-left pl-6">└ SG&A % of Revenue</td>
                        {historical3Y.map((h, i) => (
                          <td key={i} className="p-2 font-mono bg-zinc-50/30">
                            {h.sga_pct_of_rev != null ? `${h.sga_pct_of_rev.toFixed(1)}%` : '-'}
                          </td>
                        ))}
                        {forecast5Y.map((f, i) => (
                          <td key={i} className="p-2 font-mono text-sky-600">
                            {f.projected_sga_pct_of_rev != null ? `${f.projected_sga_pct_of_rev.toFixed(1)}%` : '-'}
                          </td>
                        ))}
                      </tr>
                    </>
                  )}

                  {/* EBIT */}
                  <tr className="hover:bg-zinc-50/40">
                    <td className="p-2 text-left font-semibold text-zinc-800">EBIT (Operating Profit)</td>
                    {historical3Y.map((h, i) => (
                      <td key={i} className="p-2 font-mono text-zinc-700 bg-zinc-50/30">
                        {formatLargeNum(h.ebit, currency)}
                      </td>
                    ))}
                    {forecast5Y.map((f, i) => (
                      <td key={i} className="p-2 font-mono text-zinc-900 font-medium">
                        {formatLargeNum(f.projected_ebit, currency)}
                      </td>
                    ))}
                  </tr>

                  {/* EBIT Margin % */}
                  <tr className="hover:bg-zinc-50/40 text-[11px]">
                    <td className="p-2 text-left text-zinc-500 pl-4">
                      └ EBIT Margin (%)
                      {reverseDcf?.is_mid_cycle_normalized && reverseDcf.normalized_mid_cycle_ebit_margin_pct != null && (
                        <span
                          className="ml-1 text-[9px] px-1 py-0.5 rounded bg-amber-100 text-amber-900 font-semibold border border-amber-200"
                          title="โมเดลปรับสมมติฐาน Operating Margin ทยอยลู่เข้าสู่ค่ากึ่งกลางวัฏจักร Mid-Cycle เพื่อป้องกันการประเมินมูลค่าสูงหรือต่ำเกินจริง"
                        >
                          Mid-Cycle ({reverseDcf.normalized_mid_cycle_ebit_margin_pct.toFixed(1)}%)
                        </span>
                      )}
                    </td>
                    {historical3Y.map((h, i) => (
                      <td key={i} className="p-2 font-mono text-zinc-500 bg-zinc-50/30">
                        {h.ebit_margin_pct != null ? `${h.ebit_margin_pct.toFixed(1)}%` : '-'}
                      </td>
                    ))}
                    {forecast5Y.map((f, i) => (
                      <td key={i} className="p-2 font-mono text-sky-700 font-semibold">
                        {f.projected_ebit_margin_pct != null ? `${f.projected_ebit_margin_pct.toFixed(1)}%` : '-'}
                      </td>
                    ))}
                  </tr>

                  {/* EBITDA Row */}
                  <tr className="hover:bg-zinc-50/40 bg-zinc-50/20 font-medium">
                    <td className="p-2 text-left text-zinc-800">
                      EBITDA (EBIT + D&A)
                    </td>
                    {historical3Y.map((h, i) => {
                      const histEbitda = h.ebit + (h.depreciation || 0)
                      return (
                        <td key={i} className="p-2 font-mono text-zinc-800 bg-zinc-50/30">
                          {formatLargeNum(histEbitda, currency)}
                        </td>
                      )
                    })}
                    {forecast5Y.map((f, i) => {
                      const projEbitda = f.projected_ebit + f.projected_depreciation
                      return (
                        <td key={i} className="p-2 font-mono text-zinc-900 font-semibold">
                          {formatLargeNum(projEbitda, currency)}
                        </td>
                      )
                    })}
                  </tr>

                  {/* EBITDA Margin % */}
                  <tr className="hover:bg-zinc-50/40 text-[11px] text-zinc-500">
                    <td className="p-2 text-left pl-4">└ EBITDA Margin (%)</td>
                    {historical3Y.map((h, i) => {
                      const histEbitda = h.ebit + (h.depreciation || 0)
                      const margin = h.revenue > 0 ? (histEbitda / h.revenue) * 100 : null
                      return (
                        <td key={i} className="p-2 font-mono text-zinc-500 bg-zinc-50/30">
                          {margin != null ? `${margin.toFixed(1)}%` : '-'}
                        </td>
                      )
                    })}
                    {forecast5Y.map((f, i) => {
                      const projEbitda = f.projected_ebit + f.projected_depreciation
                      const margin = f.projected_revenue > 0 ? (projEbitda / f.projected_revenue) * 100 : null
                      return (
                        <td key={i} className="p-2 font-mono text-sky-700 font-semibold">
                          {margin != null ? `${margin.toFixed(1)}%` : '-'}
                        </td>
                      )
                    })}
                  </tr>

                  {/* Taxes */}
                  <tr className="hover:bg-zinc-50/40 text-zinc-600">
                    <td className="p-2 text-left pl-4">(-) Taxes</td>
                    {historical3Y.map((_, i) => (
                      <td key={i} className="p-2 font-mono text-zinc-400 bg-zinc-50/30">-</td>
                    ))}
                    {forecast5Y.map((f, i) => (
                      <td key={i} className="p-2 font-mono text-zinc-600">
                        {formatLargeNum(f.projected_ebit - f.projected_nopat, currency)}
                      </td>
                    ))}
                  </tr>

                  {/* NOPAT */}
                  <tr className="hover:bg-zinc-50/40 font-semibold text-zinc-800 bg-zinc-50/20">
                    <td className="p-2 text-left">(=) NOPAT (EBIT × (1 - t))</td>
                    {historical3Y.map((_, i) => (
                      <td key={i} className="p-2 font-mono text-zinc-400 bg-zinc-50/30">-</td>
                    ))}
                    {forecast5Y.map((f, i) => (
                      <td key={i} className="p-2 font-mono text-zinc-900">
                        {formatLargeNum(f.projected_nopat, currency)}
                      </td>
                    ))}
                  </tr>

                  {/* Net CapEx */}
                  <tr className="hover:bg-zinc-50/40 text-zinc-600">
                    <td className="p-2 text-left pl-4">(-) Net CapEx (CapEx - D&A)</td>
                    {historical3Y.map((h, i) => (
                      <td key={i} className="p-2 font-mono text-zinc-500 bg-zinc-50/30">
                        {h.capex != null && h.depreciation != null
                          ? formatLargeNum(Math.max(0, h.capex - h.depreciation), currency)
                          : (h.capex != null ? formatLargeNum(h.capex, currency) : '-')}
                      </td>
                    ))}
                    {forecast5Y.map((f, i) => (
                      <td key={i} className="p-2 font-mono text-zinc-600">
                        {formatLargeNum(f.projected_net_capex, currency)}
                      </td>
                    ))}
                  </tr>

                  {/* CapEx / D&A Coverage Ratio & Lifecycle Phase */}
                  <tr className="hover:bg-zinc-50/40 text-[11px] text-zinc-500">
                    <td className="p-2 text-left pl-6" title="อัตราส่วนการลงทุนเทียบกับค่าเสื่อม: >1.5x ขยายกำลังผลิต, 0.85-1.3x รักษากำลังผลิต, <0.85x ชะลอลงทุน">
                      └ CapEx / D&A Coverage
                    </td>
                    {historical3Y.map((h, i) => {
                      const ratio = h.capex_to_depreciation_ratio ?? (h.capex != null && h.depreciation != null && h.depreciation > 0 ? Number((Math.abs(h.capex) / Math.abs(h.depreciation)).toFixed(2)) : null)
                      return (
                        <td key={i} className="p-2 font-mono text-zinc-500 bg-zinc-50/30">
                          {ratio != null ? (
                            <span className="flex items-center justify-end gap-1">
                              <span>{ratio.toFixed(2)}x</span>
                              <span
                                className={`text-[9px] px-1 rounded ${
                                  ratio >= 1.5
                                    ? 'bg-sky-50 text-sky-700'
                                    : ratio >= 0.85
                                    ? 'bg-emerald-50 text-emerald-700'
                                    : 'bg-rose-50 text-rose-700 font-bold'
                                }`}
                              >
                                {ratio >= 1.5 ? 'Exp' : ratio >= 0.85 ? 'Maint' : 'Under'}
                              </span>
                            </span>
                          ) : '-'}
                        </td>
                      )
                    })}
                    {forecast5Y.map((f, i) => {
                      const ratio = f.capex_to_depreciation_ratio ?? (f.projected_capex > 0 && f.projected_depreciation > 0 ? Number((f.projected_capex / f.projected_depreciation).toFixed(2)) : null)
                      return (
                        <td key={i} className="p-2 font-mono text-zinc-600">
                          {ratio != null ? (
                            <span className="flex items-center justify-end gap-1">
                              <span>{ratio.toFixed(2)}x</span>
                              <span
                                className={`text-[9px] px-1 rounded ${
                                  ratio >= 1.5
                                    ? 'bg-sky-50 text-sky-700'
                                    : ratio >= 0.85
                                    ? 'bg-emerald-50 text-emerald-700'
                                    : 'bg-rose-50 text-rose-700 font-bold'
                                }`}
                              >
                                {ratio >= 1.5 ? 'Exp' : ratio >= 0.85 ? 'Maint' : 'Under'}
                              </span>
                            </span>
                          ) : '-'}
                        </td>
                      )
                    })}
                  </tr>

                  {/* Delta NWC */}
                  <tr className="hover:bg-zinc-50/40 text-zinc-600">
                    <td className="p-2 text-left pl-4">(-) ΔNWC (Change in Working Capital)</td>
                    {historical3Y.map((_, i) => (
                      <td key={i} className="p-2 font-mono text-zinc-400 bg-zinc-50/30">-</td>
                    ))}
                    {forecast5Y.map((f, i) => (
                      <td key={i} className="p-2 font-mono text-zinc-600">
                        {formatLargeNum(f.projected_delta_nwc, currency)}
                      </td>
                    ))}
                  </tr>

                  {/* Reinvestment */}
                  <tr className="hover:bg-zinc-50/40 text-zinc-700">
                    <td className="p-2 text-left pl-4">(=) Total Reinvestment (Net CapEx + ΔNWC)</td>
                    {historical3Y.map((_, i) => (
                      <td key={i} className="p-2 font-mono text-zinc-400 bg-zinc-50/30">-</td>
                    ))}
                    {forecast5Y.map((f, i) => (
                      <td key={i} className="p-2 font-mono text-zinc-700 font-semibold">
                        {formatLargeNum(f.projected_reinvestment_currency, currency)}
                      </td>
                    ))}
                  </tr>

                  {/* UFCF */}
                  <tr className="hover:bg-zinc-50/40 font-bold text-emerald-800 bg-emerald-50/20">
                    <td className="p-2 text-left">(=) Unlevered Free Cash Flow (UFCF)</td>
                    {historical3Y.map((h, i) => (
                      <td key={i} className="p-2 font-mono text-zinc-700 bg-zinc-50/30">
                        {h.fcf != null ? formatLargeNum(h.fcf, currency) : '-'}
                      </td>
                    ))}
                    {forecast5Y.map((f, i) => (
                      <td key={i} className="p-2 font-mono text-emerald-900 font-extrabold">
                        {formatLargeNum(f.projected_fcf, currency)}
                      </td>
                    ))}
                  </tr>

                  {/* Mid-Year Discount Factor */}
                  <tr className="hover:bg-zinc-50/40 text-[11px] text-zinc-500">
                    <td className="p-2 text-left pl-4">(×) Mid-Year Discount Factor (t - 0.5)</td>
                    {historical3Y.map((_, i) => (
                      <td key={i} className="p-2 font-mono text-zinc-400 bg-zinc-50/30">-</td>
                    ))}
                    {forecast5Y.map((f, i) => (
                      <td key={i} className="p-2 font-mono text-sky-800 font-semibold">
                        {f.discount_factor != null ? f.discount_factor.toFixed(4) : '-'}
                      </td>
                    ))}
                  </tr>

                  {/* PV of FCF */}
                  <tr className="hover:bg-zinc-50/40 font-bold text-sky-900 bg-sky-50/30 border-t border-edge/60">
                    <td className="p-2 text-left">(=) PV of Cash Flow ({currency}M)</td>
                    {historical3Y.map((_, i) => (
                      <td key={i} className="p-2 font-mono text-zinc-400 bg-zinc-50/30">-</td>
                    ))}
                    {forecast5Y.map((f, i) => (
                      <td key={i} className="p-2 font-mono text-sky-950 font-extrabold">
                        {formatLargeNum(f.pv_fcf, currency)}
                      </td>
                    ))}
                  </tr>
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* 4. 📐 3-Dimensional Sensitivity Matrix Switcher (75 Cells Total) */}
      {hasAnySensitivity && (
        <div className="rounded-xl border border-edge/70 bg-surface/60 p-4 space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <span className="text-base">📐</span>
              <div>
                <span className="text-xs font-bold uppercase tracking-wider text-zinc-800 block">
                  5×5 Valuation Sensitivity Matrix (3D Suite)
                </span>
                <span className="text-[10px] text-zinc-500">
                  ทดสอบความทนทานต่อ Valuation (WACC vs g), Operating (Growth vs Margin), และ Macro Shock (Beta vs 10Y Yield) — 💡 ตัวช่วยดูว่าถ้าเศรษฐกิจเปลี่ยน มูลค่าหุ้นจะไปอยู่ตรงไหน
                </span>
              </div>
            </div>
            <button
              type="button"
              onClick={() => setIsSensitivityOpen(!isSensitivityOpen)}
              className="text-xs font-semibold px-2.5 py-1 rounded-lg border border-edge/80 bg-white hover:bg-zinc-50 text-sky-700 transition-all flex items-center gap-1 shadow-2xs"
            >
              <span>{isSensitivityOpen ? 'ย่อตาราง' : 'แสดงตาราง Sensitivity (5×5)'}</span>
              <span className="text-[10px]">{isSensitivityOpen ? '▲' : '▼'}</span>
            </button>
          </div>

          {isSensitivityOpen && (
            <div className="space-y-3 pt-1 animate-fade-in">
              {/* 3 Sensitivity Dimension Tabs */}
              <div className="flex items-center gap-2 border-b border-edge/60 pb-2 text-xs">
                <button
                  type="button"
                  onClick={() => {
                    setSensitivityTab('valuation')
                    setSelectedCell(null)
                  }}
                  className={`px-3 py-1.5 rounded-lg font-bold transition-all ${
                    sensitivityTab === 'valuation'
                      ? 'bg-sky-600 text-white shadow-xs'
                      : 'bg-zinc-100 text-zinc-600 hover:bg-zinc-200'
                  }`}
                >
                  1. Valuation (WACC vs g)
                </button>
                {operatingMatrix && (
                  <button
                    type="button"
                    onClick={() => {
                      setSensitivityTab('operating')
                      setSelectedCell(null)
                    }}
                    className={`px-3 py-1.5 rounded-lg font-bold transition-all ${
                      sensitivityTab === 'operating'
                        ? 'bg-sky-600 text-white shadow-xs'
                        : 'bg-zinc-100 text-zinc-600 hover:bg-zinc-200'
                    }`}
                  >
                    2. Operating (Growth vs Margin)
                  </button>
                )}
                {macroMatrix && (
                  <button
                    type="button"
                    onClick={() => {
                      setSensitivityTab('macro')
                      setSelectedCell(null)
                    }}
                    className={`px-3 py-1.5 rounded-lg font-bold transition-all ${
                      sensitivityTab === 'macro'
                        ? 'bg-sky-600 text-white shadow-xs'
                        : 'bg-zinc-100 text-zinc-600 hover:bg-zinc-200'
                    }`}
                  >
                    3. Macro Shock (Beta vs 10Y Yield)
                  </button>
                )}
              </div>

              {/* Tab 1: Valuation Sensitivity (WACC vs Terminal g) */}
              {sensitivityTab === 'valuation' && valuationMatrix && valuationMatrix.grid && valuationMatrix.grid.length > 0 && (
                <div className="overflow-x-auto">
                  <table className="w-full text-center border-collapse text-xs">
                    <thead>
                      <tr>
                        <th className="p-1.5 text-[10px] text-zinc-400 font-sans border-b border-edge/60 bg-zinc-50/50">
                          WACC \ g
                        </th>
                        {valuationMatrix.terminal_growth_values.map((g, idx) => (
                          <th
                            key={idx}
                            className="p-1.5 text-[11px] font-bold text-zinc-700 border-b border-edge/60 bg-zinc-50/50 font-mono"
                          >
                            {g.toFixed(1)}%
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {valuationMatrix.grid.map((row, rIdx) => {
                        const rowWacc = valuationMatrix.wacc_values[rIdx]
                        return (
                          <tr key={rIdx} className="border-b border-edge/30">
                            <td className="p-1.5 text-[11px] font-bold text-zinc-700 bg-zinc-50/50 text-right pr-2 font-mono">
                              {rowWacc != null ? `${rowWacc.toFixed(1)}%` : `Row ${rIdx}`}
                            </td>
                            {row.map((cell, cIdx) => {
                              const isSelected = selectedCell?.type === 'valuation' && selectedCell.data === cell
                              const bgClass =
                                cell.verdict === 'undervalued'
                                  ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
                                  : cell.verdict === 'overvalued'
                                  ? 'bg-rose-50 text-rose-800 border-rose-200'
                                  : cell.verdict === 'unavailable'
                                  ? 'bg-zinc-100 text-zinc-400 border-zinc-200'
                                  : 'bg-sky-50 text-sky-800 border-sky-200'

                              return (
                                <td
                                  key={cIdx}
                                  onClick={() => setSelectedCell({ type: 'valuation', data: cell })}
                                  className={`p-1.5 cursor-pointer transition-all border ${bgClass} ${
                                    cell.is_base ? 'ring-2 ring-indigo-500 bg-[#BDD7EE]/40 font-extrabold z-10' : ''
                                  } ${isSelected ? 'ring-2 ring-sky-500 scale-105' : 'hover:opacity-80'}`}
                                  title={`WACC: ${cell.wacc_pct}%, g: ${cell.terminal_growth_pct}% -> Target: ${currency}${cell.target_price} (${cell.upside_pct >= 0 ? '+' : ''}${cell.upside_pct}%)${cell.is_base ? ' [Base Case Anchor]' : ''}`}
                                >
                                  <div className="text-[11px] font-bold leading-tight font-mono">
                                    {cell.target_price > 0 ? `${currency}${cell.target_price.toFixed(1)}` : 'N/A'}
                                  </div>
                                  <div className="text-[9px] opacity-75 leading-none mt-0.5 font-mono">
                                    {cell.target_price > 0 ? `${cell.upside_pct >= 0 ? '+' : ''}${cell.upside_pct.toFixed(0)}%` : '-'}
                                  </div>
                                  {cell.is_base && (
                                    <div className="text-[8px] uppercase tracking-wider text-indigo-700 font-sans font-bold mt-0.5">
                                      Base
                                    </div>
                                  )}
                                </td>
                              )
                            })}
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              )}

              {/* Tab 2: Operating Sensitivity (Growth vs Margin) */}
              {sensitivityTab === 'operating' && operatingMatrix && operatingMatrix.grid && operatingMatrix.grid.length > 0 && (
                <div className="overflow-x-auto">
                  <table className="w-full text-center border-collapse text-xs">
                    <thead>
                      <tr>
                        <th className="p-1.5 text-[10px] text-zinc-400 font-sans border-b border-edge/60 bg-zinc-50/50">
                          Growth \ Margin
                        </th>
                        {operatingMatrix.margin_values.map((m, idx) => (
                          <th
                            key={idx}
                            className="p-1.5 text-[11px] font-bold text-zinc-700 border-b border-edge/60 bg-zinc-50/50 font-mono"
                          >
                            {m.toFixed(1)}%
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {operatingMatrix.grid.map((row, rIdx) => {
                        const rowGrowth = operatingMatrix.growth_values[rIdx]
                        return (
                          <tr key={rIdx} className="border-b border-edge/30">
                            <td className="p-1.5 text-[11px] font-bold text-zinc-700 bg-zinc-50/50 text-right pr-2 font-mono">
                              {rowGrowth != null ? `${rowGrowth.toFixed(1)}%` : `Row ${rIdx}`}
                            </td>
                            {row.map((cell, cIdx) => {
                              const isSelected = selectedCell?.type === 'operating' && selectedCell.data === cell
                              const bgClass =
                                cell.verdict === 'undervalued'
                                  ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
                                  : cell.verdict === 'overvalued'
                                  ? 'bg-rose-50 text-rose-800 border-rose-200'
                                  : cell.verdict === 'unavailable'
                                  ? 'bg-zinc-100 text-zinc-400 border-zinc-200'
                                  : 'bg-sky-50 text-sky-800 border-sky-200'

                              return (
                                <td
                                  key={cIdx}
                                  onClick={() => setSelectedCell({ type: 'operating', data: cell })}
                                  className={`p-1.5 cursor-pointer transition-all border ${bgClass} ${
                                    cell.is_base ? 'ring-2 ring-indigo-500 bg-[#BDD7EE]/40 font-extrabold z-10' : ''
                                  } ${isSelected ? 'ring-2 ring-sky-500 scale-105' : 'hover:opacity-80'}`}
                                  title={`Growth: ${cell.revenue_growth_pct}%, Margin: ${cell.ebit_margin_pct}% -> Target: ${currency}${cell.target_price} (${cell.upside_pct >= 0 ? '+' : ''}${cell.upside_pct}%)${cell.is_base ? ' [Base Case Anchor]' : ''}`}
                                >
                                  <div className="text-[11px] font-bold leading-tight font-mono">
                                    {cell.target_price > 0 ? `${currency}${cell.target_price.toFixed(1)}` : 'N/A'}
                                  </div>
                                  <div className="text-[9px] opacity-75 leading-none mt-0.5 font-mono">
                                    {cell.target_price > 0 ? `${cell.upside_pct >= 0 ? '+' : ''}${cell.upside_pct.toFixed(0)}%` : '-'}
                                  </div>
                                  {cell.is_base && (
                                    <div className="text-[8px] uppercase tracking-wider text-indigo-700 font-sans font-bold mt-0.5">
                                      Base
                                    </div>
                                  )}
                                </td>
                              )
                            })}
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              )}

              {/* Tab 3: Macro Shock Sensitivity (Beta vs 10Y Yield) */}
              {sensitivityTab === 'macro' && macroMatrix && macroMatrix.grid && macroMatrix.grid.length > 0 && (
                <div className="overflow-x-auto">
                  <table className="w-full text-center border-collapse text-xs">
                    <thead>
                      <tr>
                        <th className="p-1.5 text-[10px] text-zinc-400 font-sans border-b border-edge/60 bg-zinc-50/50">
                          Beta \ 10Y Yield
                        </th>
                        {macroMatrix.yield_values.map((y, idx) => (
                          <th
                            key={idx}
                            className="p-1.5 text-[11px] font-bold text-zinc-700 border-b border-edge/60 bg-zinc-50/50 font-mono"
                          >
                            {y.toFixed(2)}%
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {macroMatrix.grid.map((row, rIdx) => {
                        const rowBeta = macroMatrix.beta_values[rIdx]
                        return (
                          <tr key={rIdx} className="border-b border-edge/30">
                            <td className="p-1.5 text-[11px] font-bold text-zinc-700 bg-zinc-50/50 text-right pr-2 font-mono">
                              {rowBeta != null ? `${rowBeta.toFixed(2)}β` : `Row ${rIdx}`}
                            </td>
                            {row.map((cell, cIdx) => {
                              const isSelected = selectedCell?.type === 'macro' && selectedCell.data === cell
                              const bgClass =
                                cell.verdict === 'undervalued'
                                  ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
                                  : cell.verdict === 'overvalued'
                                  ? 'bg-rose-50 text-rose-800 border-rose-200'
                                  : cell.verdict === 'unavailable'
                                  ? 'bg-zinc-100 text-zinc-400 border-zinc-200'
                                  : 'bg-sky-50 text-sky-800 border-sky-200'

                              return (
                                <td
                                  key={cIdx}
                                  onClick={() => setSelectedCell({ type: 'macro', data: cell })}
                                  className={`p-1.5 cursor-pointer transition-all border ${bgClass} ${
                                    cell.is_base ? 'ring-2 ring-indigo-500 bg-[#BDD7EE]/40 font-extrabold z-10' : ''
                                  } ${isSelected ? 'ring-2 ring-sky-500 scale-105' : 'hover:opacity-80'}`}
                                  title={`Beta: ${cell.beta}, 10Y Yield: ${cell.risk_free_rate_pct}% (Eff. WACC: ${cell.effective_wacc_pct}%) -> Target: ${currency}${cell.target_price} (${cell.upside_pct >= 0 ? '+' : ''}${cell.upside_pct}%)${cell.is_base ? ' [Base Case Anchor]' : ''}`}
                                >
                                  <div className="text-[11px] font-bold leading-tight font-mono">
                                    {cell.target_price > 0 ? `${currency}${cell.target_price.toFixed(1)}` : 'N/A'}
                                  </div>
                                  <div className="text-[9px] opacity-75 leading-none mt-0.5 font-mono">
                                    {cell.target_price > 0 ? `${cell.upside_pct >= 0 ? '+' : ''}${cell.upside_pct.toFixed(0)}%` : '-'}
                                  </div>
                                  {cell.is_base && (
                                    <div className="text-[8px] uppercase tracking-wider text-indigo-700 font-sans font-bold mt-0.5">
                                      Base
                                    </div>
                                  )}
                                </td>
                              )
                            })}
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              )}

              {/* Inspector for Clicked Cell */}
              {selectedCell && (
                <div className="p-2.5 rounded-lg border border-sky-200 bg-sky-50/70 text-xs flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <span className="font-bold text-sky-950">🔍 Selected Scenario:</span>
                    {selectedCell.type === 'valuation' && (
                      <span className="text-zinc-700">
                        WACC: <strong>{(selectedCell.data as SensitivityCellDTO).wacc_pct.toFixed(1)}%</strong> |
                        Terminal g: <strong>{(selectedCell.data as SensitivityCellDTO).terminal_growth_pct.toFixed(1)}%</strong>
                      </span>
                    )}
                    {selectedCell.type === 'operating' && (
                      <span className="text-zinc-700">
                        Revenue Growth: <strong>{(selectedCell.data as OperatingSensitivityCellDTO).revenue_growth_pct.toFixed(1)}%</strong> |
                        EBIT Margin: <strong>{(selectedCell.data as OperatingSensitivityCellDTO).ebit_margin_pct.toFixed(1)}%</strong>
                      </span>
                    )}
                    {selectedCell.type === 'macro' && (
                      <span className="text-zinc-700">
                        Beta: <strong>{(selectedCell.data as MacroSensitivityCellDTO).beta.toFixed(2)}</strong> |
                        10Y Yield: <strong>{(selectedCell.data as MacroSensitivityCellDTO).risk_free_rate_pct.toFixed(2)}%</strong>
                        (Effective WACC: <strong>{(selectedCell.data as MacroSensitivityCellDTO).effective_wacc_pct.toFixed(2)}%</strong>)
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-2 font-mono">
                    <span className="font-bold text-zinc-900">
                      Target Price: {currency}{selectedCell.data.target_price.toFixed(2)}
                    </span>
                    <span className={`px-2 py-0.5 rounded-full font-bold text-[11px] ${
                      selectedCell.data.upside_pct >= 0 ? 'bg-emerald-100 text-emerald-800' : 'bg-rose-100 text-rose-800'
                    }`}>
                      {selectedCell.data.upside_pct >= 0 ? '+' : ''}{selectedCell.data.upside_pct.toFixed(1)}%
                    </span>
                  </div>
                </div>
              )}

              {/* Sensitivity Summary Footer */}
              {sensitivitySummary && (
                <div className="flex flex-wrap items-center justify-between gap-2 p-2 rounded-lg bg-zinc-100/70 text-[11px] text-zinc-600 font-sans">
                  <div className="flex items-center gap-3">
                    <span>
                      ช่วงราคาประเมิน: <strong>{currency}{sensitivitySummary.min_target_price.toFixed(1)}</strong> ถึง <strong>{currency}{sensitivitySummary.max_target_price.toFixed(1)}</strong>
                    </span>
                    <span className="text-zinc-400">|</span>
                    <span>
                      Upside Spread: <strong>{sensitivitySummary.min_upside_pct.toFixed(0)}%</strong> ถึง <strong>{sensitivitySummary.max_upside_pct.toFixed(0)}%</strong>
                    </span>
                  </div>
                  {sensitivitySummary.wacc_spread_pct.length === 2 && (
                    <div className="text-zinc-500 text-[10px]">
                      WACC: {sensitivitySummary.wacc_spread_pct[0]}% - {sensitivitySummary.wacc_spread_pct[1]}% | g: {sensitivitySummary.growth_spread_pct[0]}% - {sensitivitySummary.growth_spread_pct[1]}%
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Forensic Footer: Piotroski F-Score & Capital Quality Strip */}
      {(piotroski || qualityMetrics || netShareholderYieldPct != null) && (
        <div className="rounded-xl border border-sky-200/80 bg-gradient-to-r from-sky-50/70 via-surface to-panel p-4 space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-sky-200/50 pb-2">
            <div className="flex items-center gap-2">
              <span className="text-base">🩺</span>
              <span className="text-xs font-bold uppercase tracking-wider text-sky-900">
                Financial Health & Forensics (Piotroski & Capital Quality)
              </span>
              <ContextTooltip
                title="Piotroski F-Score & Capital Quality"
                what="คะแนนตรวจสุขภาพการเงิน 9 ด้าน ครอบคลุมการทำกำไร (Profitability), หนี้สิน/สภาพคล่อง (Leverage), และประสิทธิภาพดำเนินงาน (Operating Efficiency)"
                normalBenchmark="คะแนนเต็ม 9; คะแนน 7-9 = สุขภาพการเงินแกร่งมาก; ต่ำกว่า 3 = เสี่ยงมีปัญหาสภาพคล่องหรือความเสี่ยงสูง"
                decisionImpact="ช่วยคัดกรองไม่ให้เราซื้อ 'หุ้นเน่าราคาถูก (Value Trap)' ที่ตัวเลข P/E ดูถูกแต่กิจการกำลังแย่ลง"
              />
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

          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3 text-xs">
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

          {/* Infographic 6: Capital Allocation & Shareholder Yield Meter */}
          {(netShareholderYieldPct != null || netBuybackYieldPct != null || dividendYieldPct != null) && (
            <CapitalAllocationYieldMeter
              dividendYieldPct={dividendYieldPct ?? ((netShareholderYieldPct != null && netBuybackYieldPct != null) ? Math.max(0, Number((netShareholderYieldPct - netBuybackYieldPct).toFixed(2))) : (netShareholderYieldPct ?? 0))}
              buybackYieldPct={netBuybackYieldPct}
              netShareholderYieldPct={netShareholderYieldPct}
              fcfYieldPct={qualityMetrics?.fcf_yield_pct}
            />
          )}
        </div>
      )}

      {/* Sector-Specific Operating KPIs */}
      {sectorKpis?.status === 'available' && sectorKpis.kpis && sectorKpis.kpis.length > 0 && (
        <div className="rounded-xl border border-sky-200/80 bg-gradient-to-r from-sky-50/70 via-surface to-panel p-4 space-y-3">
          <div className="flex items-center justify-between border-b border-sky-200/50 pb-2">
            <div className="flex items-center gap-2">
              <span className="text-base">🏭</span>
              <span className="text-xs font-bold uppercase tracking-wider text-sky-900">
                Sector Operating KPIs ({sectorKpis.sector_name || sectorKpis.sector || 'Industry Baseline'})
              </span>
            </div>
            {sectorKpis.sector_health_score != null && (
              <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-sky-100 text-sky-800 border border-sky-200">
                Health Score: {sectorKpis.sector_health_score}/100
              </span>
            )}
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-xs">
            {sectorKpis.kpis.map((k, idx) => (
              <div key={idx} className="bg-surface/80 rounded-lg p-2.5 border border-edge/40">
                <span className="text-zinc-400 block text-[10px] uppercase truncate" title={k.definition || k.name}>
                  {k.name}
                </span>
                <div className="flex items-baseline justify-between mt-1">
                  <span className="font-bold text-zinc-900 text-sm">
                    {k.value != null ? `${k.value}${k.unit ? (k.unit === 'percent' || k.unit === '%' ? '%' : ` ${k.unit}`) : ''}` : 'N/A'}
                  </span>
                  {k.benchmark && (
                    <span className="text-[10px] text-zinc-500 font-medium">
                      {k.benchmark}
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 5. 💼 WACC & Capital Structure Inspector Modal */}
      {isWaccModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-xs p-4 animate-fade-in">
          <div className="bg-white rounded-2xl border border-edge/80 shadow-2xl max-w-lg w-full p-6 space-y-4">
            <div className="flex items-center justify-between border-b border-edge/60 pb-3">
              <div className="flex items-center gap-2">
                <span className="text-lg">💼</span>
                <h4 className="font-bold text-zinc-900 text-sm uppercase tracking-wider">
                  WACC & Capital Structure Decomposition
                </h4>
              </div>
              <button
                type="button"
                onClick={() => setIsWaccModalOpen(false)}
                className="text-zinc-400 hover:text-zinc-700 font-bold p-1 rounded-md text-sm"
              >
                ✕
              </button>
            </div>

            <div className="space-y-3 text-xs text-zinc-600">
              <div className="rounded-lg bg-sky-50/70 p-3 border border-sky-100">
                <div className="text-[11px] font-bold text-sky-900 uppercase">WACC Formula & Components</div>
                <div className="font-mono text-zinc-800 text-xs mt-1">
                  WACC = (E / V × K_e) + (D / V × K_d × (1 - t))
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 rounded-lg border border-edge/50 bg-surface/50">
                  <span className="text-[10px] text-zinc-400 uppercase font-bold block">Cost of Equity (K_e)</span>
                  <div className="text-base font-extrabold text-zinc-900 font-mono mt-0.5">
                    {dcf?.cost_of_equity_pct != null ? `${dcf.cost_of_equity_pct.toFixed(2)}%` : 'N/A'}
                  </div>
                  <div className="text-[10px] text-zinc-500 mt-1">
                    CAPM: R_f ({dcf?.risk_free_rate_pct ?? 4.25}%) + β × ERP ({dcf?.erp_pct ?? 5.5}%)
                  </div>
                </div>

                <div className="p-3 rounded-lg border border-edge/50 bg-surface/50">
                  <span className="text-[10px] text-zinc-400 uppercase font-bold block">Cost of Debt (K_d)</span>
                  <div className="text-base font-extrabold text-zinc-900 font-mono mt-0.5">
                    {dcf?.cost_of_debt_pct != null ? `${dcf.cost_of_debt_pct.toFixed(2)}%` : '5.00%'}
                  </div>
                  <div className="text-[10px] text-zinc-500 mt-1">
                    Effective after-tax borrowing cost
                  </div>
                </div>
              </div>

              {/* Credit Rating & Damodaran Spread Benchmark Card */}
              {(reverseDcf?.synthetic_credit_rating || dcf?.synthetic_credit_rating) && (
                <div className="rounded-xl border border-emerald-200/80 bg-gradient-to-br from-emerald-50/60 via-white to-emerald-50/30 p-3.5 space-y-2 shadow-2xs">
                  <div className="flex items-center justify-between text-xs border-b border-emerald-100 pb-2">
                    <span className="font-bold text-emerald-950 flex items-center gap-1.5">
                      <span>💳</span>
                      <span>Damodaran Credit Rating Benchmark</span>
                    </span>
                    <span className="px-2 py-0.5 rounded font-mono font-bold text-xs bg-emerald-600 text-white shadow-2xs">
                      {reverseDcf?.synthetic_credit_rating || dcf?.synthetic_credit_rating}
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-xs pt-1">
                    <div>
                      <span className="text-[10px] text-zinc-400 uppercase font-bold block">Interest Coverage (ICR)</span>
                      <span className="font-mono font-bold text-zinc-800">
                        {(reverseDcf?.interest_coverage_ratio ?? dcf?.interest_coverage_ratio) != null
                          ? `${(reverseDcf?.interest_coverage_ratio ?? dcf?.interest_coverage_ratio)?.toFixed(1)}x`
                          : 'N/A (Net Cash)'}
                      </span>
                    </div>
                    <div>
                      <span className="text-[10px] text-zinc-400 uppercase font-bold block">Spread over 10Y Yield</span>
                      <span className="font-mono font-bold text-emerald-700">
                        +{(reverseDcf?.credit_spread_bps ?? dcf?.credit_spread_bps ?? 60).toFixed(0)} bps
                      </span>
                    </div>
                  </div>
                  <p className="text-[10px] text-zinc-500 leading-tight">
                    อันดับความน่าเชื่อถือจำลองสะท้อนความสามารถในการชำระดอกเบี้ยจาก EBIT ตามเกณฑ์ Damodaran Coverage Table
                  </p>
                </div>
              )}

              <div className="rounded-lg border border-edge/50 bg-surface/70 p-3 flex items-center justify-between">
                <span className="font-semibold text-zinc-700">Effective Blended WACC:</span>
                <span className="text-base font-extrabold text-sky-700 font-mono">
                  {dcf?.wacc_pct != null ? `${dcf.wacc_pct.toFixed(2)}%` : 'N/A'}
                </span>
              </div>
            </div>

            <div className="flex justify-end pt-2 border-t border-edge/40">
              <button
                type="button"
                onClick={() => setIsWaccModalOpen(false)}
                className="px-4 py-1.5 rounded-lg bg-sky-600 text-white font-semibold text-xs hover:bg-sky-700 transition-all"
              >
                ปิดหน้าต่าง
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
