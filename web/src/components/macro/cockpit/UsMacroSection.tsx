import React, { useState } from 'react'
import type {
  TreasuryYieldCurveDTO,
  CommodityVolSnapshotDTO,
  AuctionDemandSnapshotDTO,
  UsNationalDebtDTO,
  MacroDashboardDTO,
} from '../../../api/types'
import { YieldCurveChart } from './YieldCurveChart'
import { OfrStressBar } from './OfrStressBar'
import { MetalsCotCard } from '../MetalsCotCard'
import { CommodityVolCard } from '../CommodityVolCard'
import { TreasuryAuctionDemandCard } from '../TreasuryAuctionDemandCard'
import { StackedAreaChart } from '../../charts/StackedAreaChart'
import TradingViewMiniWidget from '../../TradingViewMiniWidget'
import { SourceProvenanceBadge } from './SourceProvenanceBadge'
import RegimeProbabilityChart from '../../RegimeProbabilityChart'
import PortfolioStanceBar from '../../PortfolioStanceBar'
import WarningPanel from '../../WarningPanel'
import { stanceCategory, type StanceCategory } from '../../../lib/stance'

interface UsMacroSectionProps {
  yieldCurve: TreasuryYieldCurveDTO | null
  financialStress: any | null
  metalsCot: any | null
  commodityVol: CommodityVolSnapshotDTO[] | null
  auctionDemandNote: AuctionDemandSnapshotDTO | null
  auctionDemandBill: AuctionDemandSnapshotDTO | null
  nationalDebt: UsNationalDebtDTO[] | null
  aiData: MacroDashboardDTO | null
  aiLoading?: boolean
  aiError?: string | null
  loadingYield?: boolean
  loadingStress?: boolean
  errorYield?: string | null
  errorStress?: string | null
}

const STANCE_CLASS: Record<StanceCategory, string> = {
  overweight: 'bg-emerald-50 text-emerald-700 border-emerald-200',
  underweight: 'bg-red-50 text-red-700 border-red-200',
  neutral: 'bg-surface-strong text-zinc-700 border-edge',
}

function confidenceBadgeClass(confidence: string): string {
  const c = confidence.toLowerCase()
  if (c === 'high') return 'border-emerald-200 bg-emerald-50 text-emerald-700'
  if (c === 'medium') return 'border-amber-200 bg-amber-50 text-amber-700'
  if (c === 'low') return 'border-red-200 bg-red-50 text-red-700'
  return 'border-edge bg-surface text-zinc-600'
}

const cardClass =
  'space-y-3 rounded-xl border border-sky-100 bg-panel p-4 shadow-[0_8px_26px_rgba(14,165,233,0.05)] backdrop-blur-sm transition-all duration-150 hover:border-sky-200 hover:shadow-md'

export const UsMacroSection: React.FC<UsMacroSectionProps> = ({
  yieldCurve,
  financialStress,
  metalsCot,
  commodityVol,
  auctionDemandNote,
  auctionDemandBill,
  nationalDebt,
  aiData,
  aiLoading = false,
  aiError = null,
  loadingYield = false,
  loadingStress = false,
  errorYield = null,
  errorStress = null,
}) => {
  const [showGlobalContext, setShowGlobalContext] = useState(false)
  const [stanceFilter, setStanceFilter] = useState<'all' | 'overweight' | 'underweight' | 'neutral'>('all')

  const assetAllocations = aiData?.asset_allocation ?? []
  const pairTrades = aiData?.pair_trades ?? []
  const riskScenarios = aiData?.risk_scenarios ?? []

  // Filter out Thailand-specific assets (like USD/THB or SET) from US tab
  const usAssetAllocations = assetAllocations.filter((a) => {
    if (a.region === 'Thailand') return false
    const nameLower = (a.asset_class || '').toLowerCase()
    if (nameLower.includes('usd/thb') || nameLower.includes('set ') || nameLower.includes('thailand') || nameLower.includes('thai ')) return false
    return true
  })

  const filteredAssets = usAssetAllocations.filter(
    (a) => stanceFilter === 'all' || stanceCategory(a.stance) === stanceFilter
  )

  return (
    <div className="space-y-6">
      {/* 1. US Primary Market & Yield Observables (Verified Data) */}
      <div className="space-y-3">
        <div className="flex items-center justify-between border-b border-sky-100/70 pb-2">
          <div>
            <h2 className="text-base font-bold text-zinc-900 tracking-tight flex items-center gap-2">
              <span>ข้อมูลตลาดและอัตราผลตอบแทนสหรัฐฯ (US Observables)</span>
            </h2>
            <p className="text-xs text-zinc-500">
              ข้อมูลปฐมภูมิทางการจาก US Treasury และ OFR ใช้ประเมินเส้นอัตราผลตอบแทนและความตึงตัวของระบบการเงิน
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 items-start">
          <YieldCurveChart data={yieldCurve} loading={loadingYield} error={errorYield} />
          <OfrStressBar
            fsiValue={financialStress?.fsi_value}
            asOfDate={financialStress?.as_of_date}
            regime={financialStress?.regime}
            categories={financialStress?.categories}
            loading={loadingStress}
            error={errorStress}
          />
        </div>
      </div>

      {/* 2. Global Context with US Impact (Collapsible Accordion) */}
      <div className="rounded-2xl border border-slate-200/90 bg-white/80 p-4 shadow-sm">
        <button
          type="button"
          onClick={() => setShowGlobalContext(!showGlobalContext)}
          className="flex items-center justify-between w-full text-left font-semibold text-sm text-zinc-800 hover:text-sky-700 transition-colors"
        >
          <div className="flex items-center gap-2">
            <span>🌐</span>
            <span>บริบทโลกที่มีผลต่อสหรัฐฯ (Commodity Vol, Gold COT, Debt & Auctions)</span>
            <span className="text-[11px] font-normal text-zinc-400">
              {showGlobalContext ? '(คลิกเพื่อย่อ)' : '(คลิกเพื่อขยายดูรายละเอียด 4 แผง)'}
            </span>
          </div>
          <span className="font-mono text-xs text-zinc-400">{showGlobalContext ? '▲' : '▼'}</span>
        </button>

        {showGlobalContext && (
          <div className="mt-4 space-y-5 border-t border-slate-100 pt-4">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 items-start">
              {/* CFTC Gold COT */}
              {metalsCot ? (
                <MetalsCotCard
                  commodity={metalsCot.commodity}
                  commodityCode={metalsCot.commodity_code}
                  asOfDate={metalsCot.as_of_date}
                  publishedAt={metalsCot.published_at}
                  openInterest={metalsCot.open_interest}
                  netManagedMoney={metalsCot.net_managed_money}
                  percentile52w={metalsCot.percentile_52w}
                  managedMoney={metalsCot.managed_money}
                  swapDealers={metalsCot.swap_dealers}
                  producerMerchant={metalsCot.producer_merchant}
                />
              ) : (
                <div className="rounded-xl border border-slate-100 bg-slate-50 p-4 text-xs text-zinc-500">
                  ไม่มีข้อมูล CFTC Gold COT
                </div>
              )}

              {/* Commodity Volatility */}
              {Array.isArray(commodityVol) && commodityVol.length > 0 && (
                <CommodityVolCard indices={commodityVol} />
              )}
            </div>

            {/* Treasury Auction Demand */}
            {(auctionDemandNote || auctionDemandBill) && (
              <TreasuryAuctionDemandCard
                demandNote={auctionDemandNote}
                demandBill={auctionDemandBill}
              />
            )}

            {/* US National Debt to the Penny */}
            {Array.isArray(nationalDebt) && nationalDebt.length > 0 && (
              <StackedAreaChart
                data={[...nationalDebt]
                  .filter((d) => d && d.record_date)
                  .sort((a, b) => (a.record_date || '').localeCompare(b.record_date || ''))
                  .map((d) => ({
                    date: d.record_date,
                    values_by_category: {
                      public: d.debt_held_by_public_usd ? d.debt_held_by_public_usd / 1e12 : 0,
                      intragov: d.intragovernmental_holdings_usd ? d.intragovernmental_holdings_usd / 1e12 : 0,
                    },
                  }))}
                categories={[
                  { key: 'public', label: 'Debt Held by Public', color: '#0284c7' },
                  { key: 'intragov', label: 'Intragovernmental Holdings', color: '#8b5cf6' },
                ]}
                title="US National Debt: Debt to the Penny (Held by Public vs Intragovernmental)"
                subtitle="ข้อมูลหนี้สาธารณะสหรัฐฯ รายวันจาก US Treasury Fiscal Data ไม่นับรวม Maturity Composition ตามนิยามข้อมูล"
                unit="$T"
                height={260}
                defaultMode="absolute"
              />
            )}
          </div>
        )}
      </div>

      {/* 3. AI Analysis for US & Global Portfolio */}
      {aiData ? (
        <div className="space-y-4">
          <div className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between border-b border-sky-100/70 pb-2">
            <div>
              <h2 className="text-base font-bold text-zinc-900 tracking-tight flex items-center gap-2">
                <span>AI วิเคราะห์ภาวะเศรษฐกิจสหรัฐฯ / ภาพรวม (AI Regime Analysis)</span>
              </h2>
              <p className="text-xs text-zinc-500">
                การประเมินภาพรวมโดย Agent AI จากหลักฐาน 5 มิติ สมมติฐานหลัก และคำแนะนำการจัดสรรสินทรัพย์
              </p>
            </div>
            <SourceProvenanceBadge
              origin="ai"
              evaluatedAt={aiData.evaluated_at}
              compact
            />
          </div>

          {/* AI Banner Summary */}
          <div className="rounded-xl border border-sky-100 bg-gradient-to-r from-sky-50/70 via-white to-amber-50/50 p-4">
            <div className="flex flex-wrap items-center gap-2">
              <span className="rounded-lg border border-zinc-900 bg-zinc-900 px-3 py-1 text-xs font-bold text-white">
                AI สภาวะเศรษฐกิจสมอหลัก (Global/US Anchor): {aiData.overall_regime}
              </span>
              {aiData.conviction_level && (
                <span className="rounded-lg border border-amber-300 bg-amber-100/90 px-2.5 py-1 text-xs font-semibold uppercase text-amber-900">
                  Conviction: {aiData.conviction_level}
                </span>
              )}
              {aiData.quant_narrative_alignment && (
                <span className="rounded-lg border border-edge bg-panel px-2.5 py-1 text-xs font-medium text-zinc-700">
                  Alignment: {aiData.quant_narrative_alignment}
                </span>
              )}
              <span className="rounded-lg border border-edge bg-panel px-2.5 py-1 text-xs font-medium text-zinc-600">
                Horizon: {aiData.time_horizon}
              </span>
            </div>
            {aiData.conviction_rationale && (
              <p className="mt-2 text-xs leading-relaxed text-zinc-700">{aiData.conviction_rationale}</p>
            )}
          </div>

          <WarningPanel warnings={aiData.warnings} />

          {/* 2-Column Responsive: Left Regime & Evidence, Right Allocation */}
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
            <div className="space-y-5 lg:col-span-5">
              {/* Regime Scenarios (Not Calibrated Probability) */}
              <div className="rounded-xl border border-edge bg-panel p-4 shadow-sm">
                <div className="mb-2 flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-zinc-900">ฉากทัศน์ที่ AI ประเมิน (Assessed Scenarios)</h3>
                </div>
                <p className="text-[10px] text-zinc-400 mb-3">
                  สัดส่วนคะแนนที่โมเดลประเมินจากหลักฐานเชิงปริมาณ (ไม่ใช่ Probability ทางสถิติที่ผ่านการ Calibrate)
                </p>
                <RegimeProbabilityChart probabilities={aiData.regime_probabilities} />
              </div>

              {/* 5-Dimension Evidence */}
              {(aiData.regime_evidence ?? []).length > 0 && (
                <div className="rounded-xl border border-edge bg-panel p-4 shadow-sm">
                  <h3 className="mb-3 text-sm font-semibold text-zinc-900">หลักฐาน 5 มิติ (5-Dimension Evidence)</h3>
                  <div className="space-y-2.5">
                    {(aiData.regime_evidence ?? []).map((re, idx) => (
                      <div key={idx} className="rounded-lg border border-edge bg-zinc-50/60 p-2.5 text-xs">
                        <div className="flex items-center justify-between gap-2">
                          <span className="font-semibold uppercase tracking-wide text-zinc-800 text-[11px]">
                            {re.dimension}
                          </span>
                          <span className={`rounded-full border px-2 py-0.2 text-[9px] font-semibold ${confidenceBadgeClass(re.confidence)}`}>
                            {re.confidence}
                          </span>
                        </div>
                        <div className="mt-1 font-medium text-zinc-900">{re.signal}</div>
                        <p className="mt-1 text-[11px] text-zinc-600 leading-relaxed">{re.evidence}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* External Widget */}
              <div className="rounded-xl border border-edge bg-panel p-4 shadow-sm">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-sm font-semibold text-zinc-900">US 10Y Benchmark Rate</h3>
                  <SourceProvenanceBadge origin="external" sourceName="TradingView" compact />
                </div>
                <p className="text-[10px] text-zinc-400 mb-3">
                  ⚠️ เวลาของกราฟอาจเหลื่อมจาก Snapshot ทางการของ US Treasury
                </p>
                <TradingViewMiniWidget symbol="TVC:US10Y" title="US 10-Year Treasury Yield" />
              </div>
            </div>

            {/* Right Column: Asset Allocation & Pair Trades */}
            <div className="space-y-5 lg:col-span-7">
              <div className="rounded-xl border border-edge bg-panel p-4 shadow-sm">
                <div className="mb-4 flex flex-col justify-between gap-2 sm:flex-row sm:items-center">
                  <div>
                    <h3 className="text-sm font-semibold text-zinc-900">คำแนะนำจัดสรรสัดส่วนสินทรัพย์ (Asset Stance)</h3>
                    <p className="text-[11px] text-zinc-500">ตามภาพรวมสภาวะเศรษฐกิจโลกและสหรัฐฯ</p>
                  </div>
                  <div className="flex w-fit gap-1 rounded-lg border border-edge bg-surface p-1">
                    {(['all', 'overweight', 'underweight', 'neutral'] as const).map((key) => (
                      <button
                        key={key}
                        type="button"
                        onClick={() => setStanceFilter(key)}
                        aria-pressed={stanceFilter === key}
                        className={`rounded px-2 py-0.5 text-xs font-medium transition-colors ${
                          stanceFilter === key
                            ? 'bg-zinc-900 text-white shadow-xs'
                            : 'text-zinc-600 hover:text-zinc-900'
                        }`}
                      >
                        {key === 'all' ? 'ทั้งหมด' : key.charAt(0).toUpperCase() + key.slice(1)}
                      </button>
                    ))}
                  </div>
                </div>

                {assetAllocations.length > 0 && (
                  <div className="mb-4 rounded-xl border border-edge bg-surface p-3">
                    <PortfolioStanceBar allocations={assetAllocations} />
                  </div>
                )}

                <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                  {filteredAssets.map((a, idx) => (
                    <div key={idx} className={cardClass}>
                      <div className="flex items-start justify-between gap-2">
                        <div>
                          <h4 className="font-semibold text-zinc-900 text-xs">{a.asset_class}</h4>
                          {a.asset_bucket && (
                            <span className="text-[10px] font-medium uppercase tracking-wider text-zinc-400">
                              {a.asset_bucket}
                            </span>
                          )}
                        </div>
                        <span className={`rounded-md border px-2 py-0.2 text-[10px] font-bold uppercase ${STANCE_CLASS[stanceCategory(a.stance)]}`}>
                          {a.stance}
                        </span>
                      </div>
                      <p className="text-xs text-zinc-600 leading-relaxed">{a.rationale}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Pair Trades */}
              {pairTrades.length > 0 && (
                <div className="rounded-xl border border-edge bg-panel p-4 shadow-sm">
                  <h3 className="text-sm font-semibold text-zinc-900 mb-1">กลยุทธ์จับคู่การเทรด (Tactical Pair Trades)</h3>
                  <p className="text-[11px] text-zinc-500 mb-3">กลยุทธ์ภาพรวมจาก AI (ขอบเขตข้ามสินทรัพย์)</p>
                  <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                    {pairTrades.map((pt, idx) => (
                      <div key={idx} className={cardClass}>
                        <div className="font-semibold text-xs text-zinc-900">
                          Long <span className="text-emerald-600">{pt.long_leg}</span> / Short{' '}
                          <span className="text-rose-600">{pt.short_leg}</span>
                        </div>
                        <p className="text-xs text-zinc-600">{pt.thesis}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Risk Scenarios */}
              {riskScenarios.length > 0 && (
                <div className="rounded-xl border border-edge bg-panel p-4 shadow-sm">
                  <h3 className="text-sm font-semibold text-zinc-900 mb-1">การบริหารความเสี่ยงหางแถว (Tail Risk Scenarios)</h3>
                  <p className="text-[11px] text-zinc-500 mb-3">แผนรองรับเหตุการณ์ไม่คาดฝันระดับโลก</p>
                  <div className="space-y-3">
                    {riskScenarios.map((rs, idx) => (
                      <div key={idx} className={cardClass}>
                        <div className="font-semibold text-xs text-zinc-900">{rs.tail_risk}</div>
                        {rs.mitigation_strategy && (
                          <p className="text-xs text-zinc-600">{rs.mitigation_strategy}</p>
                        )}
                        <div className="rounded bg-amber-50/70 p-2 text-[11px] text-amber-900">
                          <span className="font-semibold">Trigger:</span> {rs.trigger_to_activate}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      ) : aiLoading ? (
        <div className="rounded-2xl border border-sky-100 bg-white/80 p-8 text-center text-xs text-zinc-500 animate-pulse">
          กำลังโหลดบทวิเคราะห์ภาพรวมเศรษฐกิจจาก AI...
        </div>
      ) : aiError ? (
        <div className="rounded-2xl border border-amber-200 bg-amber-50/70 p-6 text-xs text-amber-900">
          <div className="font-bold flex items-center gap-1.5 mb-1">
            <span>ℹ️</span>
            <span>สถานะบทวิเคราะห์ AI:</span>
          </div>
          <p>{aiError}</p>
          <p className="mt-2 text-zinc-500">
            คุณสามารถกดปุ่ม &ldquo;อัปเดตบทวิเคราะห์&rdquo; ที่แถบด้านบนเพื่อสั่ง AI สร้างรายงานล่าสุด
          </p>
        </div>
      ) : (
        <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50/60 p-6 text-center text-xs text-zinc-500">
          ยังไม่มีบทวิเคราะห์ภาพรวมเศรษฐกิจจาก AI ในคลัง สามารถกดปุ่ม &ldquo;อัปเดตบทวิเคราะห์&rdquo; ที่แถบด้านบนเพื่อเริ่มงาน
        </div>
      )}
    </div>
  )
}
