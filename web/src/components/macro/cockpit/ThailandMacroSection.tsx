import React from 'react'
import type {
  ThaiFundFlowDTO,
  ThaiRetailGoldDTO,
  MarketValuationDTO,
  MarketBreadthDTO,
  ThaiYieldCurveDTO,
  ThaiPublicDebtDTO,
  MacroDashboardDTO,
} from '../../../api/types'
import { DivergingFlowBar } from './DivergingFlowBar'
import { BreadthStackedBar } from './BreadthStackedBar'
import { SourceProvenanceBadge } from './SourceProvenanceBadge'
import { ContextTooltip } from '../explainer/ContextTooltip'
import { TrafficLightSubtitle } from '../explainer/TrafficLightSubtitle'

interface ThailandMacroSectionProps {
  flow: ThaiFundFlowDTO | null
  gold: ThaiRetailGoldDTO | null
  valuation: MarketValuationDTO | null
  breadth: MarketBreadthDTO | null
  thaiYieldCurve?: ThaiYieldCurveDTO | null
  thaiPublicDebt?: ThaiPublicDebtDTO | null
  aiData?: MacroDashboardDTO | null
  globalPolicyRates?: { as_of_date?: string; spreads_vs_bot_repo?: Record<string, number | null> } | null
  aiLoading?: boolean
  aiError?: string | null
  loading?: boolean
  error?: string | null
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

function parseGapDetails(gap: string): { authority: string; label: string; reason: string } {
  const lower = gap.toLowerCase()
  if (lower.includes('gdp') || lower.includes('สศช') || lower.includes('nesdc')) {
    return {
      authority: 'สศช. (NESDC)',
      label: 'Real GDP YoY',
      reason: 'รอรอบรายงานไตรมาส / เชื่อมต่อ API ทางการ',
    }
  }
  if (lower.includes('mpi') || lower.includes('สศอ') || lower.includes('manufacturing')) {
    return {
      authority: 'สศอ. (OIE)',
      label: 'Manufacturing Production Index (MPI)',
      reason: 'ตัวชี้วัดเสริมภาคการผลิตภาคอุตสาหกรรม',
    }
  }
  if (
    lower.includes('monetary') ||
    lower.includes('ดอกเบี้ย') ||
    lower.includes('policy rate') ||
    lower.includes('ธปท') ||
    lower.includes('bot')
  ) {
    if (gap.includes('ขาด Headline CPI') || gap.includes('Headline CPI เพื่อคำนวณ')) {
      return {
        authority: 'ธนาคารแห่งประเทศไทย (BOT)',
        label: 'Policy Rate & Real Rate Proxy',
        reason: 'ดอกเบี้ยนโยบายพร้อม แต่ขาด Headline CPI เพื่อคำนวณ Real Rate',
      }
    }
    return {
      authority: 'ธนาคารแห่งประเทศไทย (BOT)',
      label: 'Policy Rate & Yield Curve',
      reason: 'รอข้อมูลอัตราดอกเบี้ยนโยบาย / เส้นผลตอบแทน',
    }
  }
  if (lower.includes('core cpi')) {
    return {
      authority: 'สนค. พาณิชย์ (TPSO/MOC)',
      label: 'Core CPI YoY',
      reason: 'ข้อมูลเสริมเงินเฟ้อพื้นฐาน',
    }
  }
  if (lower.includes('cpi') || lower.includes('สนค') || lower.includes('moc')) {
    return {
      authority: 'สนค. พาณิชย์ (TPSO/MOC)',
      label: 'Headline CPI YoY',
      reason: 'รอรอบรายงานประจำเดือน / เชื่อมต่อ API ทางการ',
    }
  }
  if (lower.includes('debt') || lower.includes('หนี้') || lower.includes('mof')) {
    return {
      authority: 'กระทรวงการคลัง (MOF)',
      label: 'Public Debt & Debt-to-GDP',
      reason: 'รอข้อมูลหนี้สาธารณะคงค้าง',
    }
  }
  return {
    authority: 'ข้อมูลทางการ (Official Source)',
    label: gap,
    reason: 'รอข้อมูลที่ผ่านเกณฑ์ Dual-Track',
  }
}

export const ThailandMacroSection: React.FC<ThailandMacroSectionProps> = ({
  flow,
  gold,
  valuation,
  breadth,
  thaiYieldCurve: liveThaiYieldCurve = null,
  thaiPublicDebt: liveThaiPublicDebt = null,
  aiData = null,
  globalPolicyRates = null,
  loading = false,
  error = null,
}) => {
  const marketStance = aiData?.thailand_market_stance
  const foreignRow = flow?.investors.find((row) =>
    row.investor_type.toLowerCase().includes('foreign') || row.investor_type.includes('ต่างชาติ')
  )
  const foreignFlowMb = flow
    ? foreignRow?.net_value != null ? foreignRow.net_value / 1e6 : null
    : marketStance?.investor_flow?.foreign_net_mb ?? null
  const adRatio = breadth
    ? breadth.losers > 0 ? breadth.gainers / breadth.losers : null
    : marketStance?.market_breadth?.advance_decline_ratio ?? null
  const breadthSentiment = breadth ? null : marketStance?.market_breadth?.sentiment
  const peRatio = valuation ? valuation.pe_ratio ?? null : marketStance?.valuation?.pe_ratio ?? null
  const goldBarSell = gold ? gold.bar?.sell ?? null : marketStance?.physical_gold?.bar_sell_thb ?? null
  const policySpreadBps = globalPolicyRates
    ? globalPolicyRates.spreads_vs_bot_repo?.US ?? null
    : marketStance?.policy_spread_bps ?? null

  const thAssessment = aiData?.regional_assessments?.Thailand
  const thState = thAssessment?.economic_state ?? thAssessment?.state ?? 'Unknown'
  const thConfidence =
    typeof thAssessment?.confidence === 'number'
      ? `${(thAssessment.confidence * 100).toFixed(1)}%`
      : '0.0%'
  const thGaps =
    thAssessment
      ? thAssessment.data_gaps ?? []
      : ['Real GDP YoY', 'Headline CPI YoY', 'Core CPI YoY']

  const fiscalHealth = thAssessment?.fiscal_health
  const registry = aiData?.observable_registry
  const debtObs = registry?.['obs_th_debt_to_gdp_mof']?.is_valid === false ? null : registry?.['obs_th_debt_to_gdp_mof']
  const debtTotalObs = registry?.['obs_th_public_debt_mof']?.is_valid === false ? null : registry?.['obs_th_public_debt_mof']
  const resolvedDebtToGdp =
    liveThaiPublicDebt?.debt_to_gdp_pct !== undefined && liveThaiPublicDebt?.debt_to_gdp_pct !== null
      ? liveThaiPublicDebt.debt_to_gdp_pct
      : fiscalHealth?.debt_to_gdp_pct !== undefined && fiscalHealth?.debt_to_gdp_pct !== null
      ? fiscalHealth.debt_to_gdp_pct
      : debtObs?.value
      ? parseFloat(debtObs.value)
      : null
  const resolvedPublicDebtMb =
    liveThaiPublicDebt?.total_debt_thb !== undefined && liveThaiPublicDebt?.total_debt_thb !== null
      ? liveThaiPublicDebt.total_debt_thb / 1e6
      : fiscalHealth?.public_debt_million_thb !== undefined && fiscalHealth?.public_debt_million_thb !== null
      ? fiscalHealth.public_debt_million_thb
      : debtTotalObs?.value
      ? parseFloat(debtTotalObs.value.replace(/,/g, ''))
      : null
  const statLimit = fiscalHealth?.statutory_limit_pct ?? 70.0
  const debtObsDate = liveThaiPublicDebt?.reporting_month || debtObs?.observed_at || debtTotalObs?.observed_at

  const aiYieldCurve = (thAssessment as any)?.thai_yield_curve
  const y2Obs = registry?.['obs_th_gov_yield_2y']?.is_valid === false ? null : registry?.['obs_th_gov_yield_2y']
  const y10Obs = registry?.['obs_th_gov_yield_10y']?.is_valid === false ? null : registry?.['obs_th_gov_yield_10y']
  const spreadObs = registry?.['obs_th_gov_10y_2y_spread']?.is_valid === false ? null : registry?.['obs_th_gov_10y_2y_spread']

  const liveY2 = liveThaiYieldCurve?.yields?.find((y) => y.tenor === '2Y')?.yield_percent ?? null
  const liveY10 = liveThaiYieldCurve?.yields?.find((y) => y.tenor === '10Y')?.yield_percent ?? null
  const liveSpread = liveThaiYieldCurve?.spread_10y_2y_bps ?? null

  const resolvedY2 =
    liveY2 !== null && liveY2 !== undefined
      ? liveY2
      : aiYieldCurve?.yield_2y !== undefined && aiYieldCurve?.yield_2y !== null
      ? aiYieldCurve.yield_2y
      : y2Obs?.value
      ? parseFloat(y2Obs.value)
      : null

  const resolvedY10 =
    liveY10 !== null && liveY10 !== undefined
      ? liveY10
      : aiYieldCurve?.yield_10y !== undefined && aiYieldCurve?.yield_10y !== null
      ? aiYieldCurve.yield_10y
      : y10Obs?.value
      ? parseFloat(y10Obs.value)
      : null

  const resolvedSpread =
    liveSpread !== null && liveSpread !== undefined
      ? liveSpread
      : aiYieldCurve?.spread_10y_2y_bps !== undefined && aiYieldCurve?.spread_10y_2y_bps !== null
      ? aiYieldCurve.spread_10y_2y_bps
      : spreadObs?.value
      ? parseFloat(spreadObs.value)
      : null

  const yieldCurveObsDate = liveThaiYieldCurve?.observation_date || spreadObs?.observed_at || y10Obs?.observed_at

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
              {thGaps.map((gap, gIdx) => {
                const parsed = parseGapDetails(gap)
                return (
                  <div key={gIdx} className="rounded-lg bg-amber-100/60 p-2.5 border border-amber-200/80">
                    <span className="font-mono text-[10px] text-amber-800 block font-semibold">
                      {parsed.authority}
                    </span>
                    <span className="font-bold text-amber-950 text-xs block mt-0.5">{parsed.label}</span>
                    <span className="text-[10px] text-amber-700 block mt-1 leading-snug">{parsed.reason}</span>
                  </div>
                )
              })}
            </div>
            <p className="mt-2.5 text-[11px] leading-relaxed text-amber-900/90">
              ตามข้อกำหนด Dual-Track เพื่อป้องกันไม่ให้โมเดลสร้างค่าจำลอง (Hallucination) สภาวะเศรษฐกิจไทยจึงถูกระบุเป็น &ldquo;ยังประเมินไม่ได้&rdquo; อย่างตรงไปตรงมา และการประเมินสภาวะตลาดในประเทศจะอ้างอิงจาก Microstructure จริงของตลาดทุนไทยและข้อมูลความยั่งยืนทางการคลังด้านล่าง
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

      {/* 2. Thailand Sovereign Fiscal Health & Public Debt (MOF Data Services) */}
      <div className="rounded-2xl border border-slate-200 bg-white/95 p-5 shadow-sm space-y-4">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between border-b border-slate-100 pb-3">
          <div>
            <h2 className="text-base font-bold text-zinc-900 tracking-tight flex items-center gap-2">
              <span>🏛️ ความยั่งยืนทางการคลังและหนี้สาธารณะ (Thai Sovereign Fiscal Health)</span>
            </h2>
            <p className="text-xs text-zinc-500">
              สถิติหนี้สาธารณะคงค้างและสัดส่วนต่อ GDP จากสำนักงานบริหารหนี้สาธารณะ กระทรวงการคลัง (MOF Data Services)
            </p>
            {liveThaiPublicDebt?.is_stale && (
              <p className="text-[11px] text-amber-700 mt-1">
                เซิร์ฟเวอร์ MOF อยู่ระหว่างบำรุงรักษา แสดงสถิติล่าสุดจากสำนักงานบริหารหนี้สาธารณะ (สบน.)
              </p>
            )}
          </div>
          <SourceProvenanceBadge
            origin="provider"
            sourceName={liveThaiPublicDebt?.source || "กระทรวงการคลัง (MOF)"}
            observedAt={debtObsDate}
            compact
          />
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
          <div className="rounded-xl bg-slate-50 p-3.5 border border-slate-100">
            <span className="text-[10px] text-zinc-500 block">สัดส่วนหนี้สาธารณะต่อ GDP</span>
            <span className={`font-mono text-xl font-extrabold mt-0.5 block ${
              resolvedDebtToGdp !== null && resolvedDebtToGdp <= statLimit
                ? 'text-emerald-700'
                : resolvedDebtToGdp !== null && resolvedDebtToGdp > statLimit
                ? 'text-rose-700'
                : 'text-zinc-700'
            }`}>
              {resolvedDebtToGdp !== null ? `${resolvedDebtToGdp.toFixed(2)}%` : 'รอข้อมูล MOF'}
            </span>
            <div className="flex items-center justify-between text-[10px] text-zinc-400 mt-1">
              <span>เพดาน พ.ร.บ. วินัยการคลัง</span>
              <span className="font-mono font-semibold text-zinc-600">≤ {statLimit.toFixed(0)}%</span>
            </div>
          </div>

          <div className="rounded-xl bg-slate-50 p-3.5 border border-slate-100">
            <span className="text-[10px] text-zinc-500 block">ยอดหนี้สาธารณะคงค้างรวม</span>
            <span className="font-mono text-xl font-extrabold text-zinc-900 mt-0.5 block">
              {resolvedPublicDebtMb !== null
                ? `${(resolvedPublicDebtMb / 1e6).toFixed(2)} ล้านล้านบาท`
                : 'รอข้อมูล MOF'}
            </span>
            <div className="flex items-center justify-between text-[10px] text-zinc-400 mt-1">
              <span>หน่วย: ล้านล้านบาท</span>
              <span className="font-mono text-zinc-500">
                {resolvedPublicDebtMb !== null ? `${resolvedPublicDebtMb.toLocaleString('th-TH', { maximumFractionDigits: 2 })} ลบ.` : '—'}
              </span>
            </div>
          </div>

          <div className="rounded-xl bg-slate-50 p-3.5 border border-slate-100 flex flex-col justify-between">
            <div>
              <span className="text-[10px] text-zinc-500 block">สถานะวินัยการเงินการคลัง</span>
              <div className="mt-1">
                {resolvedDebtToGdp !== null ? (
                  resolvedDebtToGdp <= statLimit ? (
                    <span className="inline-flex items-center gap-1 rounded-md bg-emerald-100 px-2 py-1 text-xs font-semibold text-emerald-800 border border-emerald-200">
                      ✓ ปกติ (ต่ำกว่าเพดาน {statLimit}%)
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1 rounded-md bg-rose-100 px-2 py-1 text-xs font-semibold text-rose-800 border border-rose-200">
                      ⚠️ เกินเพดานกฎหมาย ({resolvedDebtToGdp.toFixed(1)}% &gt; {statLimit}%)
                    </span>
                  )
                ) : (
                  <span className="inline-flex items-center rounded-md bg-slate-200 px-2 py-1 text-xs font-semibold text-zinc-700">
                    รอข้อมูล
                  </span>
                )}
              </div>
            </div>
            <span className="text-[10px] text-zinc-400 mt-2 block">
              พ.ร.บ. วินัยการเงินการคลังของรัฐ พ.ศ. 2561
            </span>
          </div>
        </div>
      </div>

      {/* 3. Thai Government Bond Yield Curve (ThaiBMA) */}
      <div className="rounded-2xl border border-slate-200 bg-white/95 p-5 shadow-sm space-y-4">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between border-b border-slate-100 pb-3">
          <div>
            <h2 className="text-base font-bold text-zinc-900 tracking-tight flex items-center gap-2">
              <span>📈 เส้นผลตอบแทนพันธบัตรรัฐบาลไทย (Thai Government Bond Yield Curve)</span>
            </h2>
            <p className="text-xs text-zinc-500">
              เส้นอัตราผลตอบแทนพันธบัตรรัฐบาลไทยแบบจำลอง (Model Yield Curve) จากสมาคมตลาดตราสารหนี้ไทย (ThaiBMA)
            </p>
          </div>
          <SourceProvenanceBadge
            origin="provider"
            sourceName="สมาคมตลาดตราสารหนี้ไทย (ThaiBMA)"
            observedAt={yieldCurveObsDate}
            compact
          />
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
          <div className="rounded-xl bg-slate-50 p-3.5 border border-slate-100">
            <span className="text-[10px] text-zinc-500 block">อัตราผลตอบแทนพันธบัตร 2 ปี (2Y)</span>
            <span className="font-mono text-xl font-extrabold text-zinc-900 mt-0.5 block">
              {resolvedY2 !== null ? `${resolvedY2.toFixed(2)}%` : 'รอข้อมูล ThaiBMA'}
            </span>
            <span className="text-[10px] text-zinc-400 mt-1 block">ตัวแทนคาดการณ์ดอกเบี้ยระยะสั้น-กลาง</span>
          </div>

          <div className="rounded-xl bg-slate-50 p-3.5 border border-slate-100">
            <span className="text-[10px] text-zinc-500 block">อัตราผลตอบแทนพันธบัตร 10 ปี (10Y)</span>
            <span className="font-mono text-xl font-extrabold text-zinc-900 mt-0.5 block">
              {resolvedY10 !== null ? `${resolvedY10.toFixed(2)}%` : 'รอข้อมูล ThaiBMA'}
            </span>
            <span className="text-[10px] text-zinc-400 mt-1 block">Benchmark ผลตอบแทนพันธบัตรระยะยาว</span>
          </div>

          <div className="rounded-xl bg-slate-50 p-3.5 border border-slate-100 flex flex-col justify-between">
            <div>
              <span className="text-[10px] text-zinc-500 block">ส่วนต่างอัตราผลตอบแทน (10Y-2Y Spread)</span>
              <div className="flex items-baseline gap-2 mt-0.5">
                <span className={`font-mono text-xl font-extrabold ${
                  resolvedSpread !== null && resolvedSpread > 0
                    ? 'text-emerald-700'
                    : resolvedSpread !== null && resolvedSpread < 0
                    ? 'text-rose-700'
                    : 'text-zinc-700'
                }`}>
                  {resolvedSpread !== null ? `${resolvedSpread > 0 ? '+' : ''}${resolvedSpread.toFixed(1)} bps` : 'รอข้อมูล'}
                </span>
                {resolvedSpread !== null && (
                  <span className={`text-[10px] font-semibold px-1.5 py-0.5 rounded ${
                    resolvedSpread > 0 ? 'bg-emerald-100 text-emerald-800' : 'bg-rose-100 text-rose-800'
                  }`}>
                    {resolvedSpread > 0 ? 'Normal (ชันขึ้น)' : 'Inverted (ผกผัน)'}
                  </span>
                )}
              </div>
            </div>
            <span className="text-[10px] text-zinc-400 mt-2 block">
              {resolvedSpread !== null && resolvedSpread > 0
                ? 'เส้นผลตอบแทนลาดชันปกติ สะท้อนคาดการณ์เศรษฐกิจขยายตัว'
                : 'ส่วนต่างผลตอบแทน 10 ปี ลบ 2 ปี'}
            </span>
          </div>
        </div>
      </div>

      {/* 4. Thai Market Microstructure Panel (Provider / Deterministic Data) */}
      <div className="rounded-2xl border border-slate-200 bg-white/95 p-5 shadow-sm space-y-4">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between border-b border-slate-100 pb-3">
          <div>
            <h2 className="text-base font-bold text-zinc-900 tracking-tight flex items-center gap-2">
              <span>🇹🇭 สภาวะตลาดทุนและสภาพคล่องไทย (Thai Market Microstructure)</span>
            </h2>
            <p className="text-xs text-zinc-500">
              ข้อมูลโครงสร้างตลาดสด: กระแสเงินทุนต่างชาติ ความกว้างตลาด อัตราส่วนมูลค่า และราคาทองคำแท่ง
            </p>
            {aiData && (!flow || !breadth || !valuation || !gold) && (
              <p className="text-[11px] text-amber-700 mt-1">
                รายการที่ยังดึงข้อมูลตลาดไม่ได้อ้างอิงค่า ณ รอบรายงาน {aiData.evaluated_at}
              </p>
            )}
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
              foreignFlowMb !== null && foreignFlowMb > 0
                ? 'text-emerald-700'
                : foreignFlowMb !== null && foreignFlowMb < 0
                ? 'text-rose-700'
                : 'text-zinc-700'
            }`}>
              {foreignFlowMb !== null && (foreignFlowMb !== 0 || (flow?.total_value ?? 0) > 0)
                ? `${foreignFlowMb.toLocaleString('th-TH')} ลบ.`
                : 'ไม่มีข้อมูล'}
            </span>
            <span className="text-[10px] text-zinc-400 mt-0.5 flex items-center justify-between">
              <span>{foreignFlowMb !== null && foreignFlowMb < 0 ? 'ต่างชาติขายสุทธิ' : foreignFlowMb !== null && foreignFlowMb > 0 ? 'ต่างชาติซื้อสุทธิ' : 'Settrade Data'}</span>
              <span className="text-[9px] bg-slate-200/70 px-1 py-0.2 rounded font-mono text-zinc-600">SET</span>
            </span>
            <span className="text-[9px] text-zinc-400 block mt-1">{flow?.as_of || (foreignFlowMb !== null && aiData ? `ณ รอบรายงาน ${aiData.evaluated_at}` : '')}</span>
          </div>

          <div className="rounded-xl bg-slate-50 p-3 border border-slate-100">
            <span className="text-[10px] text-zinc-500 block">Market Breadth (A/D)</span>
            <span className="font-mono text-base font-extrabold text-zinc-900 mt-0.5 block">
              {adRatio !== null && adRatio > 0 ? `${adRatio.toFixed(2)}x` : 'ไม่มีข้อมูล'}
            </span>
            <span className="text-[10px] text-zinc-400 mt-0.5 flex items-center justify-between">
              <span className="capitalize">{breadthSentiment ? `Sentiment: ${breadthSentiment}` : 'สัดส่วนหุ้นขึ้น/ตก'}</span>
              <span className="text-[9px] bg-slate-200/70 px-1 py-0.2 rounded font-mono text-zinc-600">SET</span>
            </span>
          </div>

          <div className="rounded-xl bg-slate-50 p-3 border border-slate-100">
            <span className="text-[10px] text-zinc-500 block">SET Valuation P/E</span>
            <span className="font-mono text-base font-extrabold text-zinc-900 mt-0.5 block">
              {peRatio !== null && peRatio > 0 ? `${peRatio.toFixed(2)}x` : 'ไม่มีข้อมูล'}
            </span>
            <span className="text-[10px] text-zinc-400 mt-0.5 flex items-center justify-between">
              <span>ราคาเทียบกำไรตลาด</span>
              <span className="text-[9px] bg-slate-200/70 px-1 py-0.2 rounded font-mono text-zinc-600">SET</span>
            </span>
          </div>

          <div className="rounded-xl bg-slate-50 p-3 border border-slate-100">
            <span className="text-[10px] text-zinc-500 block">GTA Gold (96.5%)</span>
            <span className="font-mono text-base font-extrabold text-amber-800 mt-0.5 block">
              {goldBarSell !== null ? `${goldBarSell.toLocaleString('th-TH')} ฿` : 'รอประกาศ'}
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

      {/* 3. Primary Market Observables: Flow & Breadth */}
      <div className="space-y-3">
        <div className="flex items-center justify-between border-b border-sky-100/70 pb-2">
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-base font-bold text-zinc-900 tracking-tight flex items-center gap-2">
                <span>ข้อมูลจุลภาคตลาดทุนและทองคำไทย (Thai Market Observables)</span>
              </h2>
              <ContextTooltip
                title="กระแสเงินทุนและความกว้างตลาดหุ้นไทย (Fund Flow & Market Breadth)"
                whatIsIt="ยอดซื้อขายสุทธิของ 4 กลุ่มผู้เล่น (ต่างชาติ, สถาบัน, บัญชีบล., รายย่อย) และอัตราส่วนหุ้นขึ้นต่อหุ้นลง"
                howToRead="หากต่างชาติขายสุทธิต่อเนื่อง ดัชนี SET มักขึ้นได้ช้าและค่าเงินบาทจะเผชิญแรงกดดันอ่อนค่า"
                impactOnMoney="บอกว่าตลาดขับเคลื่อนด้วยแรงซื้อจริงของสถาบันหรือเป็นแค่การเก็งกำไรระยะสั้น"
              />
            </div>
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
              <div className="flex items-center gap-2">
                <h3 className="font-bold text-zinc-900 text-sm tracking-tight flex items-center gap-2">
                  <span>SET Market Valuation</span>
                  <span className="font-mono text-[10px] bg-slate-100 text-zinc-600 font-semibold px-2 py-0.5 rounded">
                    {valuation?.market || 'SET'}
                  </span>
                </h3>
                <ContextTooltip
                  title="ระดับการประเมินมูลค่าตลาดหุ้นไทย (SET Valuation)"
                  whatIsIt="อัตราส่วนความถูกแพงของดัชนี SET ได้แก่ P/E (ราคาต่อกำไร) และ Dividend Yield (เงินปันผล)"
                  howToRead="P/E ต่ำกว่า 15 เท่า และ Dividend Yield สูงกว่า 3.5% สะท้อนว่าตลาดอยู่ในโซนไม่แพงและมีส่วนเผื่อความปลอดภัย"
                  impactOnMoney="เป็นจังหวะทยอยสะสมหุ้นไทยพื้นฐานดีที่ปันผลสม่ำเสมอสำหรับเงินลงทุนระยะยาว"
                />
              </div>
              <p className="text-xs text-zinc-500 mt-0.5">
                ระดับการประเมินมูลค่าตลาดหลักทรัพย์แห่งประเทศไทย (ไม่ใช้ Badge เกณฑ์สมมติ)
              </p>
              {valuation?.pe_ratio != null && valuation.pe_ratio > 0 && (
                <div className="mt-2">
                  <TrafficLightSubtitle
                    status={valuation.pe_ratio <= 16 ? 'positive' : 'neutral'}
                    text={
                      valuation.pe_ratio <= 16
                        ? `🟢 SET Valuation น่าสนใจ (P/E ${valuation.pe_ratio.toFixed(2)}x): ราคาหุ้นเทียบกับกำไรอยู่ในระดับสมเหตุสมผล มีปันผลรองรับ`
                        : `⚪ SET P/E อยู่ที่ ${valuation.pe_ratio.toFixed(2)}x`
                    }
                  />
                </div>
              )}
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
                {valuation?.pe_ratio !== null && valuation?.pe_ratio !== undefined && valuation.pe_ratio > 0
                  ? `${valuation.pe_ratio.toFixed(2)}x`
                  : 'รอข้อมูล'}
              </span>
              <span className="text-[10px] text-zinc-400 mt-1 block">ราคาต่อกำไรสุทธิ</span>
            </div>

            <div className="rounded-xl bg-slate-50 p-3 border border-slate-100">
              <span className="text-[10px] text-zinc-500 block">SET P/BV Ratio</span>
              <span className="font-mono text-xl font-extrabold text-zinc-900 mt-0.5 block">
                {valuation?.pbv_ratio !== null && valuation?.pbv_ratio !== undefined && valuation.pbv_ratio > 0
                  ? `${valuation.pbv_ratio.toFixed(2)}x`
                  : 'รอข้อมูล'}
              </span>
              <span className="text-[10px] text-zinc-400 mt-1 block">ราคาต่อมูลค่าทางบัญชี</span>
            </div>

            <div className="rounded-xl bg-slate-50 p-3 border border-slate-100">
              <span className="text-[10px] text-zinc-500 block">Dividend Yield</span>
              <span className="font-mono text-xl font-extrabold text-emerald-700 mt-0.5 block">
                {valuation?.dividend_yield !== null && valuation?.dividend_yield !== undefined && valuation.dividend_yield > 0
                  ? `${valuation.dividend_yield.toFixed(2)}%`
                  : 'รอข้อมูล'}
              </span>
              <span className="text-[10px] text-zinc-400 mt-1 block">อัตราเงินปันผลตอบแทน</span>
            </div>

            <div className="rounded-xl bg-slate-50 p-3 border border-slate-100">
              <span className="text-[10px] text-zinc-500 block">Turnover Ratio</span>
              <span className="font-mono text-xl font-extrabold text-zinc-700 mt-0.5 block">
                {valuation?.turnover_ratio !== null && valuation?.turnover_ratio !== undefined && valuation.turnover_ratio > 0
                  ? `${valuation.turnover_ratio.toFixed(2)}%`
                  : 'รอข้อมูล'}
              </span>
              <span className="text-[10px] text-zinc-400 mt-1 block">อัตราการหมุนเวียนซื้อขาย</span>
            </div>
          </div>

          <div className="mt-3 flex justify-between text-[11px] text-zinc-400 border-t border-slate-100 pt-2 font-mono">
            <span>Market Cap รวม:</span>
            <span className="font-semibold text-zinc-700">
              {valuation?.market_cap && valuation.market_cap > 0 ? `${(valuation.market_cap / 1e12).toFixed(2)} ล้านล้านบาท` : 'รอข้อมูล'}
            </span>
          </div>
        </div>

        {/* GTA Retail Physical Gold Card */}
        <div className="rounded-2xl border border-sky-100 bg-white/90 p-5 shadow-sm">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between border-b border-sky-100/70 pb-3">
            <div>
              <div className="flex items-center gap-2">
                <h3 className="font-bold text-zinc-900 text-sm tracking-tight flex items-center gap-2">
                  <span>ราคาทองคำแท่งและรูปพรรณ (GTA Physical Gold)</span>
                  {gold?.revision && (
                    <span className="font-mono text-[10px] bg-amber-100 text-amber-800 font-semibold px-2 py-0.5 rounded">
                      รอบที่ {gold.revision}
                    </span>
                  )}
                </h3>
                <ContextTooltip
                  title="ราคาทองคำแท่งสมาคมค้าทองคำ (GTA Physical Gold)"
                  whatIsIt="ราคาประกาศทางการของทองคำแท่งแท้ 96.5% โดยสมาคมค้าทองคำแห่งประเทศไทย"
                  howToRead="ราคาในไทยคำนวณจาก Gold Spot โลกคูณอัตราแลกเปลี่ยน USD/THB หากบาทอ่อนค่าจะช่วยพยุงราคาทองในประเทศ"
                  impactOnMoney="คนไทยควรดูทั้งราคาทองโลกและค่าเงินบาทประกอบกัน หากบาทแข็งมาก ราคาทองในไทยอาจขึ้นช้ากว่าตลาดโลก"
                />
              </div>
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
