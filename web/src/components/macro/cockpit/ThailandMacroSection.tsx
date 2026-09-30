import React from 'react'
import type {
  ThaiFundFlowDTO,
  ThaiRetailGoldDTO,
  MarketValuationDTO,
  MarketBreadthDTO,
  MacroDashboardDTO,
} from '../../../api/types'
import { DivergingFlowBar } from './DivergingFlowBar'
import { BreadthStackedBar } from './BreadthStackedBar'
import { SourceProvenanceBadge } from './SourceProvenanceBadge'
import { stanceCategory, type StanceCategory } from '../../../lib/stance'

interface ThailandMacroSectionProps {
  flow: ThaiFundFlowDTO | null
  gold: ThaiRetailGoldDTO | null
  valuation: MarketValuationDTO | null
  breadth: MarketBreadthDTO | null
  aiData?: MacroDashboardDTO | null
  aiLoading?: boolean
  aiError?: string | null
  loading?: boolean
  error?: string | null
}

const STANCE_BADGE: Record<StanceCategory, string> = {
  overweight: 'bg-emerald-50 text-emerald-700 border-emerald-200',
  underweight: 'bg-rose-50 text-rose-700 border-rose-200',
  neutral: 'bg-zinc-100 text-zinc-700 border-zinc-200',
}

function confidenceBadgeClass(confidence: string): string {
  const c = confidence.toLowerCase()
  if (c === 'high') return 'border-emerald-200 bg-emerald-50 text-emerald-700'
  if (c === 'medium') return 'border-amber-200 bg-amber-50 text-amber-700'
  if (c === 'low') return 'border-rose-200 bg-rose-50 text-rose-700'
  return 'border-zinc-200 bg-zinc-50 text-zinc-600'
}

function formatThb(val: number | null | undefined): string {
  if (val === null || val === undefined || isNaN(val)) return '—'
  return val.toLocaleString('th-TH', { minimumFractionDigits: 0, maximumFractionDigits: 0 })
}

function formatPolicySpreadBps(val: number | null | undefined): { text: string; hasValue: boolean } {
  if (val === null || val === undefined || isNaN(val)) {
    return { text: 'ไม่มีข้อมูลส่วนต่างดอกเบี้ย', hasValue: false }
  }
  const prefix = val > 0 ? '+' : val < 0 ? '−' : ''
  const absVal = Math.abs(val)
  const formattedVal = Number.isInteger(absVal) ? absVal.toString() : absVal.toFixed(1)
  return { text: `${val < 0 ? '−' : prefix}${formattedVal} bps`, hasValue: true }
}

export const ThailandMacroSection: React.FC<ThailandMacroSectionProps> = ({
  flow,
  gold,
  valuation,
  breadth,
  aiData = null,
  aiLoading = false,
  aiError = null,
  loading = false,
  error = null,
}) => {
  // Extract Thailand-specific asset allocations (USD/THB, SET Equities, Thai Gold)
  const thaiAssets = (aiData?.asset_allocation ?? []).filter((a) => {
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
  const thaiReferences = (aiData?.report_references ?? []).filter((ref) => {
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

  const marketStance = aiData?.thailand_market_stance
  const foreignFlowMb = marketStance?.investor_flow?.foreign_net_mb
  const adRatio = marketStance?.market_breadth?.advance_decline_ratio
  const breadthSentiment = marketStance?.market_breadth?.sentiment
  const peRatio = marketStance?.valuation?.pe_ratio
  const goldBarSell = marketStance?.physical_gold?.bar_sell_thb
  const policySpreadBps = marketStance?.policy_spread_bps

  const thAssessment = aiData?.regional_assessments?.Thailand
  const thState = thAssessment?.economic_state ?? 'Unknown'
  const thConfidence =
    typeof thAssessment?.confidence === 'number'
      ? `${(thAssessment.confidence * 100).toFixed(1)}%`
      : '0.0%'
  const thGaps =
    thAssessment?.data_gaps && thAssessment.data_gaps.length > 0
      ? thAssessment.data_gaps
      : ['Real GDP YoY', 'Headline CPI YoY', 'Core CPI YoY']

  return (
    <div className="space-y-6">
      {/* 1. Real Economic State & Data Gaps Notice (Fail-Closed) */}
      {thState === 'Unknown' ? (
        <div className="rounded-2xl border border-amber-200/90 bg-amber-50/70 p-5 text-xs text-amber-900 shadow-sm">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between border-b border-amber-200/70 pb-3">
            <div className="flex items-center gap-2">
              <span className="rounded-md border border-amber-300 bg-amber-100 px-2 py-0.5 font-mono text-[10px] font-bold uppercase text-amber-900">
                สถานะเศรษฐกิจไทย: ยังประเมินไม่ได้ (Data Gaps)
              </span>
              <span className="font-semibold text-amber-800 text-xs">
                ความเชื่อมั่น: {thConfidence} (Fail-Closed Policy)
              </span>
            </div>

            <SourceProvenanceBadge
              origin="deterministic"
              sourceName="Dual-Track Policy"
              compact
            />
          </div>

          <div className="mt-3">
            <div className="font-semibold text-amber-950 mb-1">
              ช่องว่างข้อมูลฮาร์ดดาต้าที่อยู่ระหว่างเชื่อมต่อ API ทางการ:
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 mt-2">
              {thGaps.map((gap, gIdx) => (
                <div key={gIdx} className="rounded-lg bg-amber-100/60 p-2 border border-amber-200/80">
                  <span className="font-mono text-[10px] text-amber-800 block">
                    {gap.toLowerCase().includes('gdp') ? 'สศช. (NESDC)' : 'สนค. พาณิชย์ (MOC)'}
                  </span>
                  <span className="font-bold text-amber-950 text-xs">{gap}</span>
                  <span className="text-[10px] text-amber-700 block mt-0.5">รอเชื่อมต่อ API ทางการ</span>
                </div>
              ))}
            </div>
            <p className="mt-2 text-[11px] leading-relaxed text-amber-900/90">
              ตามข้อกำหนด Dual-Track Revision 5 เพื่อป้องกันไม่ให้โมเดลสร้างค่าจำลอง (Hallucination) สภาวะเศรษฐกิจไทยจึงถูกระบุเป็น &ldquo;ยังประเมินไม่ได้&rdquo; อย่างตรงไปตรงมา และการประเมินสภาวะตลาดในประเทศจะอ้างอิงจาก Microstructure จริงของตลาดทุนไทยด้านล่างเท่านั้น
            </p>
          </div>
        </div>
      ) : (
        <div className="rounded-2xl border border-emerald-200/90 bg-emerald-50/70 p-5 text-xs text-emerald-900 shadow-sm">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between border-b border-emerald-200/70 pb-3">
            <div className="flex items-center gap-2">
              <span className="rounded-md border border-emerald-300 bg-emerald-100 px-2 py-0.5 font-mono text-[10px] font-bold uppercase text-emerald-900">
                สถานะเศรษฐกิจไทย: {thState}
              </span>
              <span className="font-semibold text-emerald-800 text-xs">
                ความเชื่อมั่น: {thConfidence}
              </span>
            </div>

            <SourceProvenanceBadge
              origin="deterministic"
              sourceName="NESDC / MOC Feeds"
              compact
            />
          </div>
        </div>
      )}

      {error && (
        <div className="rounded-xl border border-rose-200 bg-rose-50 p-4 text-xs text-rose-700">
          ⚠️ {error}
        </div>
      )}

      {/* 2. Thai Market Microstructure Panel (Provider / Deterministic Data) */}
      <div className="rounded-2xl border border-slate-200 bg-white/95 p-5 shadow-sm space-y-4">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between border-b border-slate-100 pb-3">
          <div>
            <h2 className="text-base font-bold text-zinc-900 tracking-tight flex items-center gap-2">
              <span>🇹🇭 สภาวะตลาดทุนและสภาพคล่องไทย (Thai Market Microstructure)</span>
            </h2>
            <p className="text-xs text-zinc-500">
              ข้อมูลโครงสร้างตลาดสด: กระแสเงินทุนต่างชาติ ความกว้างตลาด อัตราส่วนมูลค่า และราคาทองคำแท่ง
            </p>
          </div>
          <SourceProvenanceBadge
            origin="provider"
            sourceName="SET / GTA / BIS"
            observedAt={flow?.as_of || breadth?.as_of}
            compact
          />
        </div>

        {/* Microstructure Cards */}
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 text-xs">
          <div className="rounded-xl bg-slate-50 p-3 border border-slate-100">
            <span className="text-[10px] text-zinc-500 block">Foreign Net Flow</span>
            <span className={`font-mono text-base font-extrabold mt-0.5 block ${
              foreignFlowMb !== undefined && foreignFlowMb > 0
                ? 'text-emerald-700'
                : foreignFlowMb !== undefined && foreignFlowMb < 0
                ? 'text-rose-700'
                : 'text-zinc-700'
            }`}>
              {foreignFlowMb !== undefined ? `${foreignFlowMb.toLocaleString('th-TH')} ลบ.` : 'รอประเมิน'}
            </span>
            <span className="text-[10px] text-zinc-400 mt-0.5 flex items-center justify-between">
              <span>{foreignFlowMb !== undefined && foreignFlowMb < 0 ? 'ต่างชาติขายสุทธิ' : foreignFlowMb !== undefined && foreignFlowMb > 0 ? 'ต่างชาติซื้อสุทธิ' : 'Settrade Data'}</span>
              <span className="text-[9px] bg-slate-200/70 px-1 py-0.2 rounded font-mono text-zinc-600">SET</span>
            </span>
          </div>

          <div className="rounded-xl bg-slate-50 p-3 border border-slate-100">
            <span className="text-[10px] text-zinc-500 block">Market Breadth (A/D)</span>
            <span className="font-mono text-base font-extrabold text-zinc-900 mt-0.5 block">
              {adRatio !== undefined ? `${adRatio.toFixed(2)}x` : 'รอประเมิน'}
            </span>
            <span className="text-[10px] text-zinc-400 mt-0.5 flex items-center justify-between">
              <span className="capitalize">{breadthSentiment ? `Sentiment: ${breadthSentiment}` : 'สัดส่วนหุ้นขึ้น/ตก'}</span>
              <span className="text-[9px] bg-slate-200/70 px-1 py-0.2 rounded font-mono text-zinc-600">SET</span>
            </span>
          </div>

          <div className="rounded-xl bg-slate-50 p-3 border border-slate-100">
            <span className="text-[10px] text-zinc-500 block">SET Valuation P/E</span>
            <span className="font-mono text-base font-extrabold text-zinc-900 mt-0.5 block">
              {peRatio !== undefined ? `${peRatio.toFixed(2)}x` : valuation?.pe_ratio ? `${valuation.pe_ratio.toFixed(2)}x` : 'รอประเมิน'}
            </span>
            <span className="text-[10px] text-zinc-400 mt-0.5 flex items-center justify-between">
              <span>ราคาเทียบกำไรตลาด</span>
              <span className="text-[9px] bg-slate-200/70 px-1 py-0.2 rounded font-mono text-zinc-600">SET</span>
            </span>
          </div>

          <div className="rounded-xl bg-slate-50 p-3 border border-slate-100">
            <span className="text-[10px] text-zinc-500 block">GTA Gold (96.5%)</span>
            <span className="font-mono text-base font-extrabold text-amber-800 mt-0.5 block">
              {goldBarSell !== undefined ? `${goldBarSell.toLocaleString('th-TH')} ฿` : gold?.bar?.sell ? `${gold.bar.sell.toLocaleString('th-TH')} ฿` : 'รอประกาศ'}
            </span>
            <span className="text-[10px] text-zinc-400 mt-0.5 flex items-center justify-between">
              <span>ทองคำแท่งขายออก</span>
              <span className="text-[9px] bg-amber-200/70 px-1 py-0.2 rounded font-mono text-amber-800">GTA</span>
            </span>
          </div>

          <div className="rounded-xl bg-slate-50 p-3 border border-slate-100 col-span-2 sm:col-span-1">
            <span className="text-[10px] text-zinc-500 block">US-TH Policy Spread</span>
            {(() => {
              const { text, hasValue } = formatPolicySpreadBps(policySpreadBps)
              return (
                <span
                  className={
                    hasValue
                      ? 'font-mono text-base font-extrabold text-amber-700 mt-0.5 block'
                      : 'text-xs text-zinc-400 font-medium mt-1 block'
                  }
                >
                  {text}
                </span>
              )
            })()}
            <span className="text-[10px] text-zinc-400 mt-0.5 flex items-center justify-between">
              <span>ส่วนต่าง Fed - BOT</span>
              <span className="text-[9px] bg-indigo-100 px-1 py-0.2 rounded font-mono text-indigo-700">BIS</span>
            </span>
          </div>
        </div>
      </div>

      {/* 3. AI Thailand Market Stance & Strategy Panel */}
      <div className="rounded-2xl border border-sky-100 bg-white/95 p-5 shadow-sm space-y-4">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between border-b border-sky-100/70 pb-3">
          <div>
            <h2 className="text-base font-bold text-zinc-900 tracking-tight flex items-center gap-2">
              <span>🇹🇭 AI วิเคราะห์สภาวะตลาดทุนและค่าเงินบาท (AI Thailand Market Stance & Strategy)</span>
            </h2>
            <p className="text-xs text-zinc-500">
              บทวิเคราะห์เชิงคุณภาพและคำแนะนำจัดสรรสินทรัพย์ไทย/ค่าเงินบาทโดย Strategic Allocator
            </p>
          </div>
          <SourceProvenanceBadge
            origin="ai"
            evaluatedAt={aiData?.evaluated_at}
            compact
          />
        </div>

        {aiLoading && (
          <div className="p-4 text-center text-xs text-zinc-500 animate-pulse">
            กำลังโหลดข้อมูลวิเคราะห์ AI...
          </div>
        )}

        {aiError && (
          <div className="rounded-xl border border-amber-200 bg-amber-50 p-3 text-xs text-amber-800">
            ⚠️ ไม่สามารถโหลดบทวิเคราะห์ AI ได้: {aiError}
          </div>
        )}

        {/* Microstructure AI Rationale / Summary */}
        {marketStance?.rationale ? (
          <div className="rounded-xl border border-sky-100 bg-sky-50/60 p-3.5 text-xs text-sky-950 leading-relaxed shadow-2xs">
            <div className="font-semibold text-sky-900 mb-1 flex items-center gap-1.5 text-xs">
              <span>💡 ทัศนะรวมสภาวะตลาดทุนไทย (Macro Stance Narrative):</span>
            </div>
            <p className="text-zinc-700">{marketStance.rationale}</p>
          </div>
        ) : (
          <div className="rounded-xl border border-zinc-200 bg-zinc-50 p-3 text-xs text-zinc-500 italic text-center">
            ยังไม่มีบทสรุป AI สำหรับตลาดทุนไทยในรอบนี้ (แสดงเฉพาะข้อมูลตลาดและผลคำนวณที่ยืนยันได้)
          </div>
        )}


        {/* Thai Asset Allocations & Currency Strategy */}
        <div className="space-y-3 pt-2">
          <h3 className="text-sm font-semibold text-zinc-900">
            กลยุทธ์จัดสรรสินทรัพย์และค่าเงินบาท (Thai Asset & Currency Stance)
          </h3>

          {thaiAssets.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {thaiAssets.map((asset, idx) => {
                const sCat = stanceCategory(asset.stance)
                return (
                  <div
                    key={idx}
                    className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm space-y-2.5 transition-all hover:border-sky-300"
                  >
                    <div className="flex items-center justify-between gap-2 border-b border-slate-100 pb-2">
                      <div>
                        <span className="font-bold text-sm text-zinc-900 block">
                          {asset.asset_class}
                        </span>
                        <span className="text-[10px] text-zinc-400 font-mono">
                          {asset.asset_bucket ? `หมวด: ${asset.asset_bucket}` : 'สินทรัพย์ตลาดไทย/ข้ามพรมแดน'}
                        </span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <span
                          className={`rounded-full border px-2.5 py-0.5 text-xs font-bold uppercase ${STANCE_BADGE[sCat]}`}
                        >
                          {asset.stance}
                        </span>
                        <span
                          className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold ${confidenceBadgeClass(
                            asset.confidence
                          )}`}
                        >
                          {asset.confidence}
                        </span>
                      </div>
                    </div>

                    <p className="text-xs leading-relaxed text-zinc-700">
                      {asset.rationale}
                    </p>

                    {asset.supporting_data && asset.supporting_data.length > 0 && (
                      <div className="rounded-lg bg-slate-50 p-2.5 border border-slate-100 text-[11px] text-zinc-600 space-y-1 font-mono">
                        <span className="text-[10px] font-semibold text-zinc-500 uppercase block">
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

                    {asset.allocation_delta && (
                      <div className="text-[11px] text-zinc-500 flex items-center justify-between border-t border-slate-100 pt-1.5">
                        <span>การปรับสัดส่วน:</span>
                        <span className="font-semibold text-zinc-800">{asset.allocation_delta}</span>
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          ) : (
            <div className="rounded-xl border border-dashed border-slate-200 bg-slate-50/60 p-4 text-xs text-zinc-500">
              💡 ระบบกำลังประมวลผลกลยุทธ์เฉพาะเจาะจงสำหรับสินทรัพย์ไทย — กลยุทธ์ค่าเงินบาท USD/THB จะแสดงผลที่นี่โดยอัตโนมัติเมื่อ AI สร้างแผนจัดสรรสินทรัพย์
            </div>
          )}
        </div>

        {/* Domestic News & Policy Developments */}
        {thaiReferences.length > 0 && (
          <div className="space-y-2 pt-2 border-t border-sky-100/70">
            <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-600 flex items-center gap-1.5">
              <span>📰 ปัจจัยและข่าวสารเศรษฐกิจในประเทศที่ AI ใช้อ้างอิง</span>
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
              {thaiReferences.slice(0, 4).map((ref, idx) => (
                <a
                  key={idx}
                  href={ref.url}
                  target="_blank"
                  rel="noreferrer"
                  className="rounded-lg border border-slate-100 bg-slate-50/70 p-2.5 transition-all hover:bg-white hover:shadow-xs group block"
                >
                  <div className="flex items-center justify-between text-[10px] text-zinc-400 font-mono mb-1">
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

      {/* 3. Primary Market Observables: Flow & Breadth */}
      <div className="space-y-3">
        <div className="flex items-center justify-between border-b border-sky-100/70 pb-2">
          <div>
            <h2 className="text-base font-bold text-zinc-900 tracking-tight flex items-center gap-2">
              <span>ข้อมูลจุลภาคตลาดทุนและทองคำไทย (Thai Market Observables)</span>
            </h2>
            <p className="text-xs text-zinc-500">
              ข้อมูลตรวจสอบได้รายวันจาก Settrade และสมาคมค้าทองคำ (GTA)
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 items-start">
          <DivergingFlowBar flow={flow} loading={loading} />
          <BreadthStackedBar breadth={breadth} loading={loading} />
        </div>
      </div>

      {/* 4. Secondary Market Observables: SET Valuation & Retail Gold */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 items-start">
        {/* SET Valuation Table Card */}
        <div className="rounded-2xl border border-sky-100 bg-white/90 p-5 shadow-sm">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between border-b border-sky-100/70 pb-3">
            <div>
              <h3 className="font-bold text-zinc-900 text-sm tracking-tight flex items-center gap-2">
                <span>SET Market Valuation</span>
                <span className="font-mono text-[10px] bg-slate-100 text-zinc-600 font-semibold px-2 py-0.5 rounded">
                  {valuation?.market || 'SET'}
                </span>
              </h3>
              <p className="text-xs text-zinc-500 mt-0.5">
                ระดับการประเมินมูลค่าตลาดหลักทรัพย์แห่งประเทศไทย (ไม่ใช้ Badge เกณฑ์สมมติ)
              </p>
            </div>
            <SourceProvenanceBadge
              origin="provider"
              sourceName="Settrade"
              observedAt={valuation?.as_of}
              compact
            />
          </div>

          <div className="mt-4 grid grid-cols-2 gap-3 text-xs">
            <div className="rounded-xl bg-slate-50 p-3 border border-slate-100">
              <span className="text-[10px] text-zinc-500 block">SET P/E Ratio</span>
              <span className="font-mono text-xl font-extrabold text-zinc-900 mt-0.5 block">
                {valuation?.pe_ratio !== null && valuation?.pe_ratio !== undefined
                  ? `${valuation.pe_ratio.toFixed(2)}x`
                  : '—'}
              </span>
              <span className="text-[10px] text-zinc-400 mt-1 block">ราคาต่อกำไรสุทธิ</span>
            </div>

            <div className="rounded-xl bg-slate-50 p-3 border border-slate-100">
              <span className="text-[10px] text-zinc-500 block">SET P/BV Ratio</span>
              <span className="font-mono text-xl font-extrabold text-zinc-900 mt-0.5 block">
                {valuation?.pbv_ratio !== null && valuation?.pbv_ratio !== undefined
                  ? `${valuation.pbv_ratio.toFixed(2)}x`
                  : '—'}
              </span>
              <span className="text-[10px] text-zinc-400 mt-1 block">ราคาต่อมูลค่าทางบัญชี</span>
            </div>

            <div className="rounded-xl bg-slate-50 p-3 border border-slate-100">
              <span className="text-[10px] text-zinc-500 block">Dividend Yield</span>
              <span className="font-mono text-xl font-extrabold text-emerald-700 mt-0.5 block">
                {valuation?.dividend_yield !== null && valuation?.dividend_yield !== undefined
                  ? `${valuation.dividend_yield.toFixed(2)}%`
                  : '—'}
              </span>
              <span className="text-[10px] text-zinc-400 mt-1 block">อัตราเงินปันผลตอบแทน</span>
            </div>

            <div className="rounded-xl bg-slate-50 p-3 border border-slate-100">
              <span className="text-[10px] text-zinc-500 block">Turnover Ratio</span>
              <span className="font-mono text-xl font-extrabold text-zinc-700 mt-0.5 block">
                {valuation?.turnover_ratio !== null && valuation?.turnover_ratio !== undefined
                  ? `${valuation.turnover_ratio.toFixed(2)}%`
                  : '—'}
              </span>
              <span className="text-[10px] text-zinc-400 mt-1 block">อัตราการหมุนเวียนซื้อขาย</span>
            </div>
          </div>

          <div className="mt-3 flex justify-between text-[11px] text-zinc-400 border-t border-slate-100 pt-2 font-mono">
            <span>Market Cap รวม:</span>
            <span className="font-semibold text-zinc-700">
              {valuation?.market_cap ? `${(valuation.market_cap / 1e12).toFixed(2)} ล้านล้านบาท` : '—'}
            </span>
          </div>
        </div>

        {/* GTA Retail Physical Gold Card */}
        <div className="rounded-2xl border border-sky-100 bg-white/90 p-5 shadow-sm">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between border-b border-sky-100/70 pb-3">
            <div>
              <h3 className="font-bold text-zinc-900 text-sm tracking-tight flex items-center gap-2">
                <span>ราคาทองคำแท่งและรูปพรรณ (GTA Physical Gold)</span>
                {gold?.revision && (
                  <span className="font-mono text-[10px] bg-amber-100 text-amber-800 font-semibold px-2 py-0.5 rounded">
                    รอบที่ {gold.revision}
                  </span>
                )}
              </h3>
              <p className="text-xs text-zinc-500 mt-0.5">
                ราคาซื้อขายทองคำแท้ 96.5% ตามประกาศสมาคมค้าทองคำแห่งประเทศไทย
              </p>
            </div>
            <SourceProvenanceBadge
              origin="provider"
              sourceName="GTA Thailand"
              observedAt={gold?.announced_at}
              compact
            />
          </div>

          <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
            {/* Gold Bar (96.5%) */}
            <div className="rounded-xl bg-amber-50/70 p-3.5 border border-amber-200/80">
              <span className="font-bold text-amber-900 text-xs block mb-1">ทองคำแท่ง (96.5%)</span>
              <div className="flex items-baseline justify-between">
                <span className="text-zinc-600">ขายออก:</span>
                <span className="font-mono text-lg font-extrabold text-amber-800">
                  {formatThb(gold?.bar?.sell)} <span className="text-xs font-normal text-zinc-500">THB</span>
                </span>
              </div>
              <div className="flex items-baseline justify-between mt-1 text-zinc-600">
                <span>รับซื้อ:</span>
                <span className="font-mono font-bold text-zinc-800">
                  {formatThb(gold?.bar?.buy)} THB
                </span>
              </div>
              <div className="mt-2 text-[10px] text-amber-800/80 font-mono border-t border-amber-200/60 pt-1">
                ส่วนต่างซื้อขาย: {gold?.bar?.sell && gold?.bar?.buy ? `${gold.bar.sell - gold.bar.buy} THB/บาททอง` : '—'}
              </div>
            </div>

            {/* Gold Ornament (96.5%) */}
            <div className="rounded-xl bg-slate-50 p-3.5 border border-slate-100">
              <span className="font-bold text-zinc-800 text-xs block mb-1">ทองรูปพรรณ (96.5%)</span>
              <div className="flex items-baseline justify-between">
                <span className="text-zinc-600">ขายออก:</span>
                <span className="font-mono text-lg font-extrabold text-zinc-900">
                  {formatThb(gold?.ornament?.sell)} <span className="text-xs font-normal text-zinc-500">THB</span>
                </span>
              </div>
              <div className="flex items-baseline justify-between mt-1 text-zinc-600">
                <span>รับซื้อ (ฐานภาษี):</span>
                <span className="font-mono font-bold text-zinc-700">
                  {formatThb(gold?.ornament?.buy)} THB
                </span>
              </div>
              <div className="mt-2 text-[10px] text-zinc-400 font-mono border-t border-slate-200 pt-1">
                หน่วย: บาททองคำ (15.244 กรัม)
              </div>
            </div>
          </div>

          <p className="mt-3 text-[11px] text-zinc-400">
            ℹ️ ราคาทองคำแท่งในประเทศแยกขาดจาก CME Gold Futures (USD/oz) และสะท้อนทั้งราคาทองคำโลกและอัตราแลกเปลี่ยน USD/THB
          </p>
        </div>
      </div>
    </div>
  )
}
