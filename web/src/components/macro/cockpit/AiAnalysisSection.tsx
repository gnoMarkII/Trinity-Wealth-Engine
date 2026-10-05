import React, { useState } from 'react'
import type { MacroDashboardDTO } from '../../../api/types'
import RegimeProbabilityChart from '../../RegimeProbabilityChart'
import PortfolioStanceBar from '../../PortfolioStanceBar'
import WarningPanel from '../../WarningPanel'
import { stanceCategory, type StanceCategory } from '../../../lib/stance'
import { SourceProvenanceBadge } from './SourceProvenanceBadge'
import { SectorAiPanel } from './SectorAiPanel'

interface AiAnalysisSectionProps {
  aiData: MacroDashboardDTO | null
  aiLoading?: boolean
  aiError?: string | null
}

const STANCE_CLASS: Record<StanceCategory, string> = {
  overweight: 'bg-emerald-50 text-emerald-700 border-emerald-200',
  underweight: 'bg-rose-50 text-rose-700 border-rose-200',
  neutral: 'bg-surface-strong text-zinc-700 border-edge',
}

function confidenceBadgeClass(confidence?: string): string {
  const c = (confidence || '').toLowerCase()
  if (c === 'high') return 'border-emerald-200 bg-emerald-50 text-emerald-700'
  if (c === 'medium') return 'border-amber-200 bg-amber-50 text-amber-700'
  if (c === 'low') return 'border-rose-200 bg-rose-50 text-rose-700'
  return 'border-edge bg-surface text-zinc-600'
}

const cardClass =
  'space-y-2.5 rounded-xl border border-sky-100 bg-panel p-4 shadow-[0_4px_20px_rgba(14,165,233,0.03)] backdrop-blur-sm transition-all duration-150 hover:border-sky-200 hover:shadow-sm'

function synthesizeThaiNarrative(
  marketStance?: MacroDashboardDTO['thailand_market_stance'],
  thaiAssets?: MacroDashboardDTO['asset_allocation']
): string | null {
  if (marketStance?.rationale && marketStance.rationale.trim().length > 0) {
    return marketStance.rationale
  }

  const parts: string[] = []
  if (marketStance?.valuation?.pe_ratio) {
    const peStr = `SET Index ซื้อขายที่ระดับ P/E ${marketStance.valuation.pe_ratio.toFixed(2)} เท่า`
    const divStr = marketStance.valuation.dividend_yield
      ? ` (Dividend Yield ${marketStance.valuation.dividend_yield.toFixed(2)}%)`
      : ''
    parts.push(peStr + divStr)
  }
  if (marketStance?.market_breadth?.advance_decline_ratio) {
    const ad = marketStance.market_breadth.advance_decline_ratio
    const sent = marketStance.market_breadth.sentiment
    parts.push(`Market Breadth (A/D Ratio) อยู่ที่ ${ad.toFixed(2)} เท่า${sent ? ` บ่งชี้ทัศนะ ${sent}` : ''}`)
  }
  if (
    marketStance?.investor_flow?.foreign_net_mb !== undefined &&
    marketStance?.investor_flow?.foreign_net_mb !== null
  ) {
    const fFlow = marketStance.investor_flow.foreign_net_mb
    const fAct = fFlow > 0 ? 'ต่างชาติซื้อสุทธิ' : 'ต่างชาติขายสุทธิ'
    const fStr = `${fAct} ${fFlow > 0 ? '+' : ''}${fFlow.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })} ล้านบาท`
    const iFlow = marketStance.investor_flow.institution_net_mb
    const iStr =
      iFlow !== undefined && iFlow !== null
        ? ` โดยสถาบันในประเทศ ${iFlow > 0 ? 'ซื้อสุทธิ' : 'ขายสุทธิ'} ${iFlow > 0 ? '+' : ''}${iFlow.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })} ล้านบาท`
        : ''
    parts.push(fStr + iStr)
  }
  if (
    marketStance?.policy_spread_bps !== undefined &&
    marketStance?.policy_spread_bps !== null
  ) {
    parts.push(
      `ส่วนต่างอัตราดอกเบี้ยนโยบาย Fed-BOT อยู่ที่ ${marketStance.policy_spread_bps > 0 ? '+' : ''}${marketStance.policy_spread_bps} bps`
    )
  }
  if (thaiAssets && thaiAssets.length > 0) {
    const recs = thaiAssets.map((a) => `${a.asset_class} (${a.stance})`).join(', ')
    parts.push(`คำแนะนำจัดสรรสินทรัพย์: ${recs}`)
  }

  return parts.length > 0 ? parts.join(' • ') : null
}

export const AiAnalysisSection: React.FC<AiAnalysisSectionProps> = ({
  aiData,
  aiLoading = false,
  aiError = null,
}) => {
  const [stanceFilter, setStanceFilter] = useState<'all' | 'overweight' | 'underweight' | 'neutral'>('all')

  if (aiLoading) {
    return (
      <div className="rounded-2xl border border-sky-100 bg-white/80 p-8 text-center text-xs text-zinc-500 animate-pulse">
        กำลังโหลดบทวิเคราะห์ภาพรวมเศรษฐกิจจาก AI...
      </div>
    )
  }

  if (!aiData) {
    return (
      <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50/60 p-6 text-center text-xs text-zinc-500">
        {aiError || 'ยังไม่มีบทวิเคราะห์ภาพรวมเศรษฐกิจจาก AI ในคลัง สามารถกดปุ่ม "อัปเดตบทวิเคราะห์" ที่กล่องสรุปด้านบนเพื่อเริ่มงาน'}
      </div>
    )
  }

  const assetAllocations = aiData.asset_allocation ?? []
  const pairTrades = aiData.pair_trades ?? []
  const riskScenarios = aiData.risk_scenarios ?? []

  // Global & All Regions included in AI Analysis tab
  const filteredAssets = assetAllocations.filter(
    (a) => stanceFilter === 'all' || stanceCategory(a.stance) === stanceFilter
  )

  // Extract Thailand-specific asset allocations
  const thaiAssets = assetAllocations.filter((a) => {
    if (a.region === 'Thailand') return true
    const nameLower = (a.asset_class || '').toLowerCase()
    return (
      nameLower.includes('usd/thb') ||
      nameLower.includes('set ') ||
      nameLower.includes('thailand') ||
      nameLower.includes('thai ') ||
      nameLower.includes('baht') ||
      nameLower.includes('gta')
    )
  })

  // Extract Thailand-specific references from News Funnel
  const thaiReferences = (aiData.report_references ?? []).filter((ref) => {
    const text = `${ref.title} ${ref.summary} ${ref.publisher}`.toLowerCase()
    return (
      text.includes('ไทย') ||
      text.includes('บาท') ||
      text.includes('thailand') ||
      text.includes('thai') ||
      text.includes('set') ||
      text.includes('bot') ||
      text.includes('กนง') ||
      text.includes('ประชาชาติ') ||
      text.includes('bangkok post')
    )
  })

  const marketStance = aiData.thailand_market_stance
  const thaiNarrative = synthesizeThaiNarrative(marketStance, thaiAssets)

  return (
    <div className="space-y-6">
      {/* Section Title Bar */}
      <div className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between border-b border-sky-100/70 pb-2">
        <div>
          <h2 className="text-base font-bold text-zinc-900 tracking-tight flex items-center gap-2">
            <span>บทวิเคราะห์สภาวะเศรษฐกิจและการจัดสรรสินทรัพย์เชิงลึก (In-Depth AI Regime & Strategy)</span>
          </h2>
          <p className="text-xs text-zinc-500">
            ศูนย์รวมบทวิเคราะห์ AI ครบทุกมิติ: ฉากทัศน์สภาวะเศรษฐกิจ, เสาหลักฐาน 5 มิติ, การหมุนเวียนกลุ่มอุตสาหกรรม, และกลยุทธ์จัดสรรสินทรัพย์ทั่วโลกและไทย
          </p>
        </div>
        <SourceProvenanceBadge origin="ai" evaluatedAt={aiData.evaluated_at} compact />
      </div>

      <WarningPanel warnings={aiData.warnings} />

      {/* Block 1: Scenarios & 5D Evidence (2-Column Responsive) */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12 items-start">
        {/* Left Column: Scenarios & Key Assumptions (5 cols) */}
        <div className="space-y-5 lg:col-span-5">
          {/* Regime Scenarios */}
          <div className="rounded-xl border border-edge bg-panel p-4 shadow-sm">
            <div className="mb-2 flex items-center justify-between">
              <h3 className="text-sm font-semibold text-zinc-900">ฉากทัศน์ที่ AI ประเมิน (Assessed Scenarios)</h3>
            </div>
            <p className="text-xs text-zinc-500 mb-3">
              สัดส่วนคะแนนที่โมเดลประเมินจากหลักฐานเชิงปริมาณ (ไม่ใช่ Probability ทางสถิติที่ผ่านการ Calibrate)
            </p>
            <RegimeProbabilityChart probabilities={aiData.regime_probabilities} />
          </div>

          {/* Key Assumptions */}
          {(aiData.key_assumptions ?? []).length > 0 && (
            <div className="rounded-xl border border-edge bg-panel p-4 shadow-sm">
              <h3 className="mb-2 text-sm font-semibold text-zinc-900">สมมติฐานหลัก (Key Assumptions)</h3>
              <ul className="space-y-1.5 list-disc list-inside text-xs text-zinc-600">
                {aiData.key_assumptions.map((item, idx) => (
                  <li key={idx} className="leading-relaxed">
                    {item}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        {/* Right Column: 5-Dimension Evidence (7 cols) */}
        <div className="space-y-5 lg:col-span-7">
          {(aiData.regime_evidence ?? []).length > 0 && (
            <div className="rounded-xl border border-edge bg-panel p-4 shadow-sm">
              <div className="mb-3 flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-semibold text-zinc-900">หลักฐาน 5 มิติ (5-Dimension Evidence)</h3>
                  <p className="text-xs text-zinc-500">
                    สัญญาณชี้วัดเชิงปริมาณและระดับความเชื่อมั่นในแต่ละมิติเศรษฐกิจ
                  </p>
                </div>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {(aiData.regime_evidence ?? []).map((re, idx) => (
                  <div key={idx} className="rounded-lg border border-edge bg-zinc-50/70 p-3 text-xs space-y-1">
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-semibold uppercase tracking-wide text-zinc-800 text-xs">
                        {re.dimension}
                      </span>
                      <span
                        className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold ${confidenceBadgeClass(
                          re.confidence
                        )}`}
                      >
                        {re.confidence}
                      </span>
                    </div>
                    <div className="font-semibold text-zinc-900 text-xs sm:text-[13px]">{re.signal}</div>
                    <p className="text-xs text-zinc-600 leading-relaxed">{re.evidence}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Block 2: AI Sector Rotation Analysis */}
      <SectorAiPanel sectorAnalysis={aiData.sector_analysis} />

      {/* Block 3: Asset Allocation, Pair Trades & Tail Risk */}
      <div className="rounded-xl border border-edge bg-panel p-4 sm:p-5 shadow-sm space-y-5">
        <div className="flex flex-col justify-between gap-2 sm:flex-row sm:items-center border-b border-slate-100 pb-3">
          <div>
            <h3 className="text-base font-bold text-zinc-900">คำแนะนำจัดสรรสัดส่วนสินทรัพย์ (Asset Stance)</h3>
            <p className="text-xs text-zinc-500">ตามภาพรวมสภาวะเศรษฐกิจโลก สหรัฐฯ และไทย</p>
          </div>
          <div className="flex w-fit gap-1 rounded-lg border border-edge bg-surface p-1">
            {(['all', 'overweight', 'underweight', 'neutral'] as const).map((key) => (
              <button
                key={key}
                type="button"
                onClick={() => setStanceFilter(key)}
                aria-pressed={stanceFilter === key}
                className={`rounded px-2.5 py-1 text-xs font-medium transition-colors ${
                  stanceFilter === key
                    ? 'bg-sky-600 text-white shadow-xs font-semibold'
                    : 'text-zinc-600 hover:text-zinc-900'
                }`}
              >
                {key === 'all' ? 'ทั้งหมด' : key.charAt(0).toUpperCase() + key.slice(1)}
              </button>
            ))}
          </div>
        </div>

        {assetAllocations.length > 0 && (
          <div className="rounded-xl border border-edge bg-surface p-3">
            <PortfolioStanceBar allocations={assetAllocations} />
          </div>
        )}

        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {filteredAssets.map((a, idx) => (
            <div key={idx} className={cardClass}>
              <div className="flex items-start justify-between gap-2">
                <div>
                  <h4 className="font-semibold text-zinc-900 text-xs sm:text-[13px]">{a.asset_class}</h4>
                  <div className="flex items-center gap-1.5 mt-0.5">
                    {a.asset_bucket && (
                      <span className="text-[10px] font-medium uppercase tracking-wider text-zinc-400">
                        {a.asset_bucket}
                      </span>
                    )}
                    {a.region && (
                      <span className="text-[10px] text-zinc-400 font-medium">
                        • {a.region}
                      </span>
                    )}
                  </div>
                </div>
                <div className="flex items-center gap-1">
                  <span
                    className={`rounded-md border px-2 py-0.5 text-[10px] font-bold uppercase ${
                      STANCE_CLASS[stanceCategory(a.stance)]
                    }`}
                  >
                    {a.stance}
                  </span>
                  {a.confidence && (
                    <span
                      className={`rounded-md border px-1.5 py-0.5 text-[9px] font-semibold ${confidenceBadgeClass(
                        a.confidence
                      )}`}
                    >
                      {a.confidence}
                    </span>
                  )}
                </div>
              </div>
              <p className="text-xs text-zinc-600 leading-relaxed">{a.rationale}</p>
              {(a.supporting_data ?? []).length > 0 && (
                <div className="pt-1 text-[11px] text-zinc-500">
                  <span className="font-semibold text-zinc-600">ข้อมูลสนับสนุน: </span>
                  {a.supporting_data.join(', ')}
                </div>
              )}
              {a.allocation_delta && (
                <div className="text-[11px] text-zinc-500 flex items-center justify-between border-t border-slate-100 pt-1.5">
                  <span>การปรับสัดส่วน:</span>
                  <span className="font-semibold text-zinc-800">{a.allocation_delta}</span>
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Pair Trades & Tail Risks Row */}
        {(pairTrades.length > 0 || riskScenarios.length > 0) && (
          <div className="grid grid-cols-1 gap-5 lg:grid-cols-2 pt-2 border-t border-slate-100">
            {/* Pair Trades */}
            {pairTrades.length > 0 && (
              <div className="space-y-3">
                <div>
                  <h4 className="text-sm font-semibold text-zinc-900">กลยุทธ์จับคู่การเทรด (Tactical Pair Trades)</h4>
                  <p className="text-xs text-zinc-500">กลยุทธ์ความสัมพันธ์เชิงเปรียบเทียบข้ามสินทรัพย์</p>
                </div>
                <div className="space-y-2.5">
                  {pairTrades.map((pt, idx) => (
                    <div key={idx} className={cardClass}>
                      <div className="font-semibold text-xs sm:text-[13px] text-zinc-900">
                        Long <span className="text-emerald-600">{pt.long_leg}</span> / Short{' '}
                        <span className="text-rose-600">{pt.short_leg}</span>
                      </div>
                      <p className="text-xs text-zinc-600 leading-relaxed">{pt.thesis}</p>
                      {pt.catalyst && (
                        <p className="text-[11px] text-sky-800 bg-sky-50/70 rounded p-1.5 mt-1">
                          <span className="font-semibold">Catalyst:</span> {pt.catalyst}
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Tail Risk Scenarios */}
            {riskScenarios.length > 0 && (
              <div className="space-y-3">
                <div>
                  <h4 className="text-sm font-semibold text-zinc-900">การบริหารความเสี่ยงหางแถว (Tail Risk Scenarios)</h4>
                  <p className="text-xs text-zinc-500">แผนรองรับเหตุการณ์ไม่คาดฝันระดับโลก (High-Impact)</p>
                </div>
                <div className="space-y-2.5">
                  {riskScenarios.map((rs, idx) => (
                    <div key={idx} className={cardClass}>
                      <div className="font-semibold text-xs sm:text-[13px] text-zinc-900">{rs.tail_risk}</div>
                      {rs.mitigation_strategy && (
                        <p className="text-xs text-zinc-600 leading-relaxed">{rs.mitigation_strategy}</p>
                      )}
                      <div className="rounded bg-amber-50/80 p-2 text-xs text-amber-900">
                        <span className="font-semibold">Trigger:</span> {rs.trigger_to_activate}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Block 4: Regional AI Strategy - Thailand & Cross-Border Divergence */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2 items-start">
        {/* Thailand AI Market Stance & Strategy */}
        <div className="rounded-xl border border-sky-100 bg-panel p-4 sm:p-5 shadow-sm space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-1 border-b border-sky-100/70 pb-2.5">
            <div>
              <h3 className="text-sm font-bold text-zinc-900 flex items-center gap-2">
                <span>🇹🇭 AI วิเคราะห์สภาวะตลาดทุนและค่าเงินบาท (AI Thailand Market Stance)</span>
              </h3>
              <p className="text-xs text-zinc-500">
                บทวิเคราะห์เชิงคุณภาพและคำแนะนำจัดสรรสินทรัพย์ไทย/ค่าเงินบาท
              </p>
            </div>
            <span className="rounded bg-sky-50 text-sky-800 border border-sky-200 px-2 py-0.5 text-[10px] font-semibold w-fit">
              Strategic Allocator
            </span>
          </div>

          {/* Thai Stance Narrative */}
          {thaiNarrative ? (
            <div className="rounded-xl border border-sky-100 bg-sky-50/60 p-3.5 text-xs text-sky-950 leading-relaxed shadow-2xs">
              <div className="font-semibold text-sky-900 mb-1 flex items-center gap-1.5 text-xs">
                <span>💡 ทัศนะรวมสภาวะตลาดทุนไทย (Macro Stance Narrative):</span>
              </div>
              <p className="text-zinc-700 leading-relaxed">{thaiNarrative}</p>
            </div>
          ) : (
            <p className="text-xs text-zinc-500 italic bg-slate-50/70 p-3 rounded-lg border border-slate-100">
              ยังไม่มีบทสรุป AI สำหรับตลาดทุนไทยในรอบนี้ (ข้อมูลสถิติสดดูได้ที่แท็บประเทศไทย)
            </p>
          )}

          {/* Thai Assets Stance Cards */}
          <div className="space-y-2 pt-1">
            <h4 className="text-xs font-semibold text-zinc-800">
              คำแนะนำสินทรัพย์ไทยและค่าเงินบาท (Thai Asset & Currency Stance):
            </h4>
            {thaiAssets.length > 0 ? (
              <div className="space-y-2.5">
                {thaiAssets.map((asset, idx) => (
                  <div
                    key={idx}
                    className="rounded-xl border border-slate-200 bg-white p-3.5 shadow-2xs space-y-2"
                  >
                    <div className="flex items-center justify-between gap-2 border-b border-slate-100 pb-1.5">
                      <span className="font-bold text-xs sm:text-[13px] text-zinc-900">
                        {asset.asset_class}
                      </span>
                      <div className="flex items-center gap-1.5">
                        <span
                          className={`rounded border px-2 py-0.5 text-[10px] font-bold uppercase ${
                            STANCE_CLASS[stanceCategory(asset.stance)]
                          }`}
                        >
                          {asset.stance}
                        </span>
                        {asset.confidence && (
                          <span
                            className={`rounded border px-1.5 py-0.5 text-[9px] font-semibold ${confidenceBadgeClass(
                              asset.confidence
                            )}`}
                          >
                            {asset.confidence}
                          </span>
                        )}
                      </div>
                    </div>
                    <p className="text-xs text-zinc-600 leading-relaxed">{asset.rationale}</p>
                    {asset.supporting_data && asset.supporting_data.length > 0 && (
                      <div className="text-[11px] text-zinc-500 font-mono space-y-0.5 bg-slate-50 p-2 rounded border border-slate-100">
                        <span className="text-[10px] font-semibold text-zinc-400 block uppercase">
                          หลักฐานเชิงปริมาณรองรับ:
                        </span>
                        {asset.supporting_data.map((sd, sIdx) => (
                          <div key={sIdx} className="flex items-start gap-1">
                            <span className="text-sky-600">•</span>
                            <span>{sd}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-xs text-zinc-500">ยังไม่มีคำแนะนำเจาะจงรายสินทรัพย์ไทยในรอบนี้</p>
            )}
          </div>

          {/* Thai Domestic News References */}
          {thaiReferences.length > 0 && (
            <div className="space-y-2 pt-2 border-t border-sky-100/70">
              <h4 className="text-xs font-semibold text-zinc-700">
                📰 ปัจจัยและข่าวสารเศรษฐกิจในประเทศที่ AI ใช้อ้างอิง:
              </h4>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                {thaiReferences.slice(0, 4).map((ref, idx) => (
                  <a
                    key={idx}
                    href={ref.url}
                    target="_blank"
                    rel="noreferrer"
                    className="rounded-lg border border-slate-100 bg-slate-50/70 p-2.5 transition-all hover:bg-white hover:shadow-xs group block"
                  >
                    <div className="flex items-center justify-between text-[10px] text-zinc-400 font-mono mb-0.5">
                      <span className="font-semibold text-sky-700">{ref.publisher || 'ข่าวเศรษฐกิจ'}</span>
                      {ref.age_hours !== null && <span>{ref.age_hours} ชม. ที่แล้ว</span>}
                    </div>
                    <div className="text-xs font-semibold text-zinc-900 group-hover:text-sky-700 line-clamp-1">
                      {ref.title}
                    </div>
                    {ref.summary && (
                      <p className="text-[11px] text-zinc-500 line-clamp-2 mt-1">
                        {ref.summary}
                      </p>
                    )}
                  </a>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Cross-Border Transmission & Divergence Note */}
        {/* Cross-Border Transmission & Policy Divergence Card */}
        <div className="rounded-xl border border-sky-100 bg-panel p-4 sm:p-5 shadow-sm space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-1 border-b border-sky-100/70 pb-2.5">
            <div>
              <h3 className="text-sm font-bold text-zinc-900 flex items-center gap-2">
                <span>🌐 ความเชื่อมโยงข้ามพรมแดน (Policy Divergence & Transmission)</span>
              </h3>
              <p className="text-xs text-zinc-500">
                การสังเคราะห์ส่วนต่างนโยบายการเงินและการไหลของเงินทุนระหว่างประเทศ
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-1.5 shrink-0">
              {aiData.quant_narrative_alignment && (
                <span className={`rounded border px-2 py-0.5 text-[10px] font-semibold ${
                  aiData.quant_narrative_alignment.toLowerCase() === 'aligned'
                    ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
                    : 'bg-amber-50 text-amber-800 border-amber-200'
                }`}>
                  Alignment: {aiData.quant_narrative_alignment}
                </span>
              )}
              {typeof aiData.thailand_market_stance?.policy_spread_bps === 'number' && (
                <span className="rounded bg-indigo-50 text-indigo-800 border border-indigo-200 px-2 py-0.5 text-[10px] font-mono font-semibold">
                  Spread: {aiData.thailand_market_stance.policy_spread_bps > 0 ? '+' : ''}{aiData.thailand_market_stance.policy_spread_bps} bps
                </span>
              )}
            </div>
          </div>

          {/* Divergence Note or Quant-Narrative Alignment Synthesis */}
          {aiData.divergence_note ? (
            <div className="rounded-xl border border-indigo-100 bg-indigo-50/50 p-3.5 text-xs text-indigo-950 leading-relaxed shadow-2xs space-y-1">
              <div className="font-semibold text-indigo-900 flex items-center gap-1.5 text-xs">
                <span>📡 สรุปความแตกต่างเชิงนโยบาย (Divergence Note):</span>
              </div>
              <p className="text-zinc-700">{aiData.divergence_note}</p>
            </div>
          ) : (
            <div className="rounded-xl border border-slate-200 bg-slate-50/80 p-3.5 text-xs text-zinc-700 leading-relaxed shadow-2xs space-y-1.5">
              <div className="font-semibold text-zinc-800 flex items-center gap-1.5 text-xs">
                <span>⚖️ ความสอดคล้องของข้อมูลและนโยบาย (Policy & Quant Alignment):</span>
              </div>
              <p className="text-zinc-600">
                {aiData.quant_narrative_alignment?.toLowerCase() === 'aligned'
                  ? 'ข้อมูลเชิงปริมาณและปัจจัยเชิงคุณภาพสอดคล้องไปในทิศทางเดียวกัน (Aligned) ไม่พบสัญญาณขัดแย้งที่มีนัยสำคัญ โดยทิศทางเงินทุนและค่าเงินสะท้อนผ่านส่วนต่างอัตราดอกเบี้ยนโยบาย'
                  : 'บทวิเคราะห์ในรอบนี้ประเมินทิศทางเงินทุนและผลกระทบข้ามพรมแดนผ่านการจัดสรรสินทรัพย์ค่าเงินและส่วนต่างดอกเบี้ย'}
              </p>
              {typeof aiData.thailand_market_stance?.policy_spread_bps === 'number' && (
                <div className="pt-1 text-[11px] text-zinc-500 font-mono flex items-center gap-2 border-t border-slate-200/60 mt-1">
                  <span className="font-semibold text-zinc-700">ส่วนต่างอัตราดอกเบี้ยนโยบาย Fed - BoT:</span>
                  <span className="font-bold text-indigo-700">
                    {aiData.thailand_market_stance.policy_spread_bps > 0 ? '+' : ''}
                    {aiData.thailand_market_stance.policy_spread_bps} bps
                  </span>
                </div>
              )}
            </div>
          )}

          {/* Cross-Border / FX Assets */}
          {(() => {
            const crossAssets = assetAllocations.filter((a) => {
              const name = (a.asset_class || '').toLowerCase()
              return (
                name.includes('usd/thb') ||
                name.includes('usd vs thb') ||
                name.includes('dollar') ||
                name.includes('thb') ||
                a.asset_bucket === 'fx'
              )
            })

            if (crossAssets.length === 0) return null

            return (
              <div className="space-y-2 pt-1">
                <h4 className="text-xs font-semibold text-zinc-800">
                  คำแนะนำสินทรัพย์ที่ได้รับผลกระทบจากปัจจัยอัตราแลกเปลี่ยนและข้ามพรมแดน:
                </h4>
                <div className="space-y-2.5">
                  {crossAssets.map((asset, idx) => (
                    <div key={idx} className="rounded-xl border border-sky-100 bg-white p-3.5 space-y-1.5 shadow-2xs">
                      <div className="flex items-center justify-between">
                        <span className="font-semibold text-xs sm:text-[13px] text-zinc-900">{asset.asset_class}</span>
                        <div className="flex items-center gap-1">
                          <span
                            className={`rounded border px-2 py-0.5 text-[10px] font-bold uppercase ${
                              STANCE_CLASS[stanceCategory(asset.stance)]
                            }`}
                          >
                            {asset.stance}
                          </span>
                          {asset.confidence && (
                            <span
                              className={`rounded border px-1.5 py-0.5 text-[9px] font-semibold ${confidenceBadgeClass(
                                asset.confidence
                              )}`}
                            >
                              {asset.confidence}
                            </span>
                          )}
                        </div>
                      </div>
                      <p className="text-xs text-zinc-600 leading-relaxed">{asset.rationale}</p>
                      {asset.supporting_data && asset.supporting_data.length > 0 && (
                        <div className="text-[11px] text-zinc-500 font-mono">
                          {asset.supporting_data.join(', ')}
                        </div>
                      )}
                      {asset.allocation_delta && (
                        <div className="text-[11px] text-zinc-500 flex items-center justify-between border-t border-slate-100 pt-1">
                          <span>การปรับสัดส่วน:</span>
                          <span className="font-semibold text-zinc-800">{asset.allocation_delta}</span>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )
          })()}
        </div>
      </div>
    </div>
  )
}
