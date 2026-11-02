import React, { useState } from 'react'
import type { MacroDashboardDTO } from '../../../api/types'
import { getReportAgeInfo } from '../../../lib/reportAge'
import { ContextTooltip } from '../explainer/ContextTooltip'
import { SafetyShieldBanner } from '../explainer/SafetyShieldBanner'

interface AiBriefingCardProps {
  aiData: MacroDashboardDTO | null
  aiLoading?: boolean
  aiError?: string | null
  onUpdateMacro?: () => void
  updating?: boolean
  onNavigateToAiTab?: () => void
}

interface RegimeWeatherInfo {
  weatherIcon: string
  weatherName: string
  ruleOfThumb: string
  bgClass: string
  borderClass: string
  textClass: string
}

function getRegimeWeather(regimeStr: string): RegimeWeatherInfo {
  const lower = (regimeStr || '').toLowerCase()
  if (lower.includes('goldilocks') || lower.includes('expansion')) {
    return {
      weatherIcon: '☀️',
      weatherName: 'แดดออกแจ่มใส (Goldilocks)',
      ruleOfThumb: 'ยุคทองของหุ้น: เศรษฐกิจโต เงินเฟ้อต่ำ เป็นมิตรต่อตลาดหุ้น ทยอยสะสมหุ้นเติบโตและสินทรัพย์เสี่ยงได้เต็มที่',
      bgClass: 'bg-emerald-50/70',
      borderClass: 'border-emerald-200/80',
      textClass: 'text-emerald-950',
    }
  }
  if (lower.includes('reflation')) {
    return {
      weatherIcon: '🌤️',
      weatherName: 'แดดออกมีลมหนาว (Reflation)',
      ruleOfThumb: 'เศรษฐกิจฟื้นแต่ของเริ่มแพง: หุ้นยังไปได้แต่ห้ามไล่ราคาของแพง ทยอยเก็บหุ้นถูก (เช่น หุ้นไทย) ถือเงินสดสำรอง 20% รับมือความผันผวน',
      bgClass: 'bg-amber-50/70',
      borderClass: 'border-amber-200/80',
      textClass: 'text-amber-950',
    }
  }
  if (lower.includes('stagflation')) {
    return {
      weatherIcon: '⛈️',
      weatherName: 'พายุเศรษฐกิจ (Stagflation)',
      ruleOfThumb: 'ของแพงแต่เศรษฐกิจแย่: ห้ามทุ่มหุ้นเด็ดขาด โยกเงินไปพักที่ทองคำ พันธบัตรระยะสั้น และถือเงินสดเป็นเกราะกำบัง',
      bgClass: 'bg-rose-50/70',
      borderClass: 'border-rose-200/80',
      textClass: 'text-rose-950',
    }
  }
  if (lower.includes('recession') || lower.includes('slowdown')) {
    return {
      weatherIcon: '🌧️',
      weatherName: 'ฝนตกหนัก (Recession)',
      ruleOfThumb: 'เศรษฐกิจหดตัว: ดอกเบี้ยกำลังจะลง ตลาดหุ้นเสี่ยงสูง พันธบัตรรัฐบาลระยะยาวและเงินสดคือที่หลบภัยที่ดีที่สุด',
      bgClass: 'bg-blue-50/70',
      borderClass: 'border-blue-200/80',
      textClass: 'text-blue-950',
    }
  }
  return {
    weatherIcon: '⛅',
    weatherName: 'สภาวะผสมผสาน (Transition)',
    ruleOfThumb: 'จัดสรรตามรอบ กระจายความเสี่ยงในสินทรัพย์คุณภาพ และถือเงินสดสำรองเพื่อความยืดหยุ่น',
    bgClass: 'bg-sky-50/70',
    borderClass: 'border-sky-200/80',
    textClass: 'text-sky-950',
  }
}

function getConvictionExplanation(level: string): { label: string; note: string; detail: string } {
  const lower = (level || '').toLowerCase()
  if (lower.includes('high')) {
    return {
      label: 'สูง (ข้อมูลสอดคล้องกันมั่นใจ 🟢)',
      note: 'ข้อมูลเชิงปริมาณและมุมมองสอดคล้องกันสูงมาก ทยอยเข้าลงทุนตามแผนได้',
      detail: 'ตัวเลขเศรษฐกิจ ข้อมูลเชิงปริมาณ และทิศทางนโยบายชี้ไปในทางเดียวกันอย่างชัดเจน',
    }
  }
  if (lower.includes('low')) {
    return {
      label: 'ระมัดระวัง (ข้อมูลขัดแย้ง 🔴)',
      note: 'ข้อมูลมีความขัดแย้งสูง ควรถือเงินสดรอความชัดเจน',
      detail: 'ตัวชี้วัดเศรษฐกิจมีความผันผวนสูงและทิศทางยังไม่นิ่ง ไม่ควรเร่งรีบเข้าซื้อ',
    }
  }
  return {
    label: 'ปานกลาง (กระจายความเสี่ยง 🟡)',
    note: 'ตลาดก้ำกึ่ง ห้ามทุ่มเงินก้อนเดียว ให้กระจายความเสี่ยงและทยอยสะสม',
    detail: 'มีทั้งปัจจัยบวกและปัจจัยกดดันคละกัน ควรแบ่งเงินเป็นไม้ๆ ทยอยลงทุนตามรอบ',
  }
}

export const AiBriefingCard: React.FC<AiBriefingCardProps> = ({
  aiData,
  aiLoading = false,
  aiError = null,
  onUpdateMacro,
  updating = false,
  onNavigateToAiTab,
}) => {
  const [isExpanded, setIsExpanded] = useState(false)

  // 1. Loading State
  if (aiLoading) {
    return (
      <div className="rounded-2xl border border-sky-200/80 bg-linear-to-r from-sky-50/60 via-white to-slate-50/60 p-5 shadow-xs animate-pulse">
        <div className="flex items-center justify-between pb-3 border-b border-sky-100">
          <div className="h-6 w-48 bg-sky-200/60 rounded-md" />
          <div className="h-5 w-28 bg-slate-200/60 rounded-full" />
        </div>
        <div className="mt-4 space-y-2.5">
          <div className="h-4 w-3/4 bg-slate-200/60 rounded" />
          <div className="h-4 w-1/2 bg-slate-200/60 rounded" />
        </div>
      </div>
    )
  }

  // 2. Empty / No Data / Error State
  if (!aiData) {
    return (
      <div className="rounded-2xl border border-dashed border-amber-300/80 bg-amber-50/40 p-5 sm:p-6 shadow-xs">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="text-xl">🧠</span>
              <h2 className="text-base font-bold text-zinc-900 tracking-tight">
                AI Executive Briefing (สรุปสภาวะเศรษฐกิจมหภาค)
              </h2>
            </div>
            <p className="text-xs text-zinc-600 max-w-xl">
              {aiError ||
                'ยังไม่มีรายงานบทวิเคราะห์สภาวะเศรษฐกิจและการจัดสรรสินทรัพย์ในคลัง คุณสามารถสั่งการให้ Agent วิเคราะห์ข้อมูลล่าสุดได้ทันที'}
            </p>
          </div>

          {onUpdateMacro && (
            <button
              type="button"
              onClick={onUpdateMacro}
              disabled={updating}
              className="inline-flex items-center justify-center gap-2 rounded-xl bg-sky-600 px-4 py-2.5 text-xs font-semibold text-white shadow-sm hover:bg-sky-500 disabled:opacity-50 transition-all shrink-0"
            >
              <span>{updating ? '⏳' : '🚀'}</span>
              <span>{updating ? 'กำลังสั่งงาน...' : 'เริ่มวิเคราะห์ภาวะเศรษฐกิจ (AI Run)'}</span>
            </button>
          )}
        </div>
      </div>
    )
  }

  // 3. Normal State with AI Data (Pure Executive Summary)
  const ageInfo = getReportAgeInfo(aiData.evaluated_at)
  const themes = aiData.focus_themes ?? []
  const weather = getRegimeWeather(aiData.overall_regime)
  const conviction = getConvictionExplanation(aiData.conviction_level || 'MEDIUM')

  return (
    <div className="rounded-2xl border border-sky-200/90 bg-linear-to-br from-white via-sky-50/30 to-amber-50/20 p-5 sm:p-6 shadow-sm space-y-4">
      {/* Top Header Row: Title, Scope, Age & Actions */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 pb-3.5 border-b border-sky-100/80">
        {/* Left: Card Title & Subtitle */}
        <div className="flex items-center gap-2.5">
          <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-sky-100/80 text-xl shadow-2xs shrink-0">
            🧠
          </span>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-sm font-bold text-zinc-900 tracking-tight">
                AI Executive Briefing
              </h2>
              <span className="hidden sm:inline-block text-[11px] font-medium text-zinc-400">
                (สรุปสภาวะเศรษฐกิจมหภาค)
              </span>
            </div>
            <p className="text-[11px] text-zinc-500">
              วิเคราะห์ทิศทางนโยบาย วัฏจักรการเงิน และกรอบการจัดสรรสินทรัพย์
            </p>
          </div>
        </div>

        {/* Right: Freshness Status & Update Button */}
        <div className="flex items-center gap-2 self-start sm:self-center shrink-0">
          <div
            className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-semibold shadow-2xs ${ageInfo.badgeClass}`}
            title={ageInfo.detail}
          >
            <span className={`h-2 w-2 rounded-full ${ageInfo.dotClass}`} />
            <span>{ageInfo.label}</span>
          </div>

          {onUpdateMacro && (
            <button
              type="button"
              onClick={onUpdateMacro}
              disabled={updating}
              className="inline-flex items-center gap-1.5 rounded-xl border border-sky-200 bg-white px-3 py-1.5 text-xs font-semibold text-sky-800 shadow-2xs hover:bg-sky-50 hover:border-sky-300 transition-all disabled:opacity-50"
              title="สั่ง AI เริ่มวิเคราะห์ภาพรวมเศรษฐกิจรอบใหม่"
            >
              <span>{updating ? '⏳' : '🔄'}</span>
              <span>{updating ? 'กำลังสั่ง...' : 'อัปเดตบทวิเคราะห์'}</span>
            </button>
          )}
        </div>
      </div>

      {/* Key Diagnostic Badges: Regime, Conviction & Horizon */}
      <div className="flex flex-wrap items-center gap-2.5">
        {/* Regime Badge with ContextTooltip */}
        <div className="inline-flex items-center gap-2 rounded-xl border border-sky-200 bg-sky-50/80 px-3 py-1.5 text-xs font-semibold text-sky-950 shadow-2xs">
          <span>สภาวะเศรษฐกิจหลัก: {aiData.overall_regime}</span>
          <ContextTooltip
            title="สภาวะเศรษฐกิจมหภาค (Macro Regime)"
            whatIsIt="สภาวะการเติบโตของเศรษฐกิจและเงินเฟ้อในปัจจุบัน แบ่งเป็น 4 แบบหลัก: Goldilocks, Reflation, Stagflation, Recession"
            howToRead="ดูว่าเศรษฐกิจกำลังเติบโต (ลูกศรขึ้น) หรือเงินเฟ้อกำลังเร่งตัว เพื่อเลือกประเภทสินทรัพย์ที่ได้เปรียบในวัฏจักรนี้"
            impactOnMoney="บอกว่าเราควรเน้นถือหุ้นเติบโต, หุ้นถูก, ทองคำ, ตราสารหนี้ หรือควรเก็บเงินสดรอจังหวะ"
          />
        </div>

        {/* Conviction Level with ContextTooltip */}
        {aiData.conviction_level && (
          <div className="inline-flex items-center gap-2 rounded-xl border border-amber-200 bg-amber-50/80 px-3 py-1.5 text-xs font-semibold text-amber-950 shadow-2xs">
            <span>Conviction: {aiData.conviction_level}</span>
            <ContextTooltip
              title="ระดับความมั่นใจของบทวิเคราะห์ (Conviction Level)"
              whatIsIt="ระดับความสอดคล้องกันระหว่างตัวเลขสถิติเชิงปริมาณ (Quantitative) และมุมมองนโยบายเศรษฐกิจ (Narrative)"
              howToRead="HIGH = มั่นใจสูง ทิศทางตรงกัน, MEDIUM = ก้ำกึ่ง ทยอยกระจายความเสี่ยง, LOW = ข้อมูลขัดแย้ง ควรถือเงินสด"
              impactOnMoney="MEDIUM หมายถึง 'ห้ามทุ่มเงินก้อนเดียว ให้แบ่งไม้ทยอยสะสมและกระจายความเสี่ยง'"
            />
          </div>
        )}

        {/* Horizon */}
        {aiData.time_horizon && (
          <div className="inline-flex items-center gap-1.5 rounded-xl border border-zinc-200 bg-white/90 px-3 py-1.5 text-xs font-medium text-zinc-700 shadow-2xs">
            <span className="text-zinc-400">⏱️</span>
            <span>กรอบเวลา: {aiData.time_horizon}</span>
          </div>
        )}
      </div>

      {/* Beginner Navigator 2: Regime Weather Metaphor + Actionable Rule of Thumb */}
      <div
        className={`rounded-xl border p-3.5 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 ${weather.bgClass} ${weather.borderClass} ${weather.textClass}`}
      >
        <div className="flex items-center gap-2.5">
          <span className="text-2xl sm:text-3xl shrink-0">{weather.weatherIcon}</span>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-xs uppercase tracking-wider">
                สภาพอากาศเศรษฐกิจ: {weather.weatherName}
              </span>
            </div>
            <p className="text-xs font-medium mt-0.5 opacity-95">
              💡 <span className="font-bold">สูตรจำสำหรับเงินของคุณ:</span> "{weather.ruleOfThumb}"
            </p>
          </div>
        </div>
        <div className="shrink-0 text-[11px] font-semibold bg-white/80 border border-zinc-200/80 rounded-lg px-2.5 py-1 text-zinc-700">
          ความมั่นใจ: <span className="text-amber-800 font-bold">{conviction.label}</span>
        </div>
      </div>

      {/* Pure Executive Summary: Core Thesis (Full Width, No Stance/Warning Duplication) */}
      <div className="rounded-xl border border-slate-200/80 bg-white/85 p-4 shadow-2xs space-y-3">
        {aiData.conviction_rationale && (
          <div>
            <div className="text-xs font-bold text-zinc-900 mb-1.5 flex items-center justify-between">
              <span className="flex items-center gap-1.5">
                <span>📌</span>
                <span>สรุปข้อวินิจฉัยและเหตุผลหลัก (Core Thesis):</span>
              </span>
              {onNavigateToAiTab && (
                <button
                  type="button"
                  onClick={onNavigateToAiTab}
                  className="text-[11px] font-semibold text-sky-700 hover:text-sky-900 hover:underline flex items-center gap-1 transition-colors"
                >
                  <span>ดูสัดส่วนพอร์ตและหลักฐาน 5 มิติในแท็บด้านล่าง</span>
                  <span>↓</span>
                </button>
              )}
            </div>
            <p
              className={`text-xs sm:text-[13px] leading-relaxed text-zinc-700 ${
                !isExpanded ? 'line-clamp-3' : ''
              }`}
            >
              {aiData.conviction_rationale}
            </p>
            {aiData.conviction_rationale.length > 200 && (
              <button
                type="button"
                onClick={() => setIsExpanded(!isExpanded)}
                className="mt-1 text-xs font-semibold text-sky-700 hover:text-sky-900 transition-colors"
              >
                {isExpanded ? 'ย่อข้อความ ▲' : 'อ่านต่อ ▼'}
              </button>
            )}
          </div>
        )}

        <div className="flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-slate-100">
          {/* Focus Themes */}
          {themes.length > 0 && (
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="text-xs font-semibold text-zinc-500 mr-1">ธีมหลัก (Themes):</span>
              {themes.map((theme) => (
                <span
                  key={theme}
                  className="rounded-md border border-sky-200/80 bg-white px-2.5 py-0.5 text-xs font-medium text-sky-900 shadow-2xs"
                >
                  #{theme}
                </span>
              ))}
            </div>
          )}

          {/* Alignment */}
          {aiData.quant_narrative_alignment && (
            <p className="text-[11px] text-zinc-500">
              ความสอดคล้องเชิงปริมาณและมุมมอง:{' '}
              <span className="font-semibold text-zinc-700">{aiData.quant_narrative_alignment}</span>
            </p>
          )}
        </div>
      </div>

      {/* Safety Shield 3: Invalidation Trigger Bar (Circuit Breaker) */}
      <SafetyShieldBanner
        type="invalidation"
        customMessage="หาก US 10Y Yield พุ่งทะลุ 5.25% หรือ VIX Index พุ่งเกิน 25.0 จุด แปลว่าสภาวะตลาดหลุดกรอบและเข้าสู่ภาวะตื่นตระหนก ให้ระงับการซื้อสินทรัพย์เสี่ยงทันทีและเพิ่มเงินสดสำรองเป็น 40%"
      />
    </div>
  )
}
