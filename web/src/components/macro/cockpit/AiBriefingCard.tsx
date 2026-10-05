import React, { useState } from 'react'
import type { MacroDashboardDTO } from '../../../api/types'
import { getReportAgeInfo } from '../../../lib/reportAge'
import { stanceCategory } from '../../../lib/stance'

interface AiBriefingCardProps {
  aiData: MacroDashboardDTO | null
  aiLoading?: boolean
  aiError?: string | null
  onUpdateMacro?: () => void
  updating?: boolean
  onNavigateToAiTab?: () => void
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
      <div className="rounded-2xl border border-sky-200/80 bg-gradient-to-r from-sky-50/60 via-white to-slate-50/60 p-5 shadow-xs animate-pulse">
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
              {aiError || 'ยังไม่มีรายงานบทวิเคราะห์สภาวะเศรษฐกิจและการจัดสรรสินทรัพย์ในคลัง คุณสามารถสั่งการให้ Agent วิเคราะห์ข้อมูลล่าสุดได้ทันที'}
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

  // 3. Normal State with AI Data
  const ageInfo = getReportAgeInfo(aiData.evaluated_at)
  const allocations = aiData.asset_allocation ?? []
  const overweightAssets = allocations.filter((a) => stanceCategory(a.stance) === 'overweight')
  const underweightAssets = allocations.filter((a) => stanceCategory(a.stance) === 'underweight')
  const warnings = aiData.warnings ?? []
  const themes = aiData.focus_themes ?? []

  return (
    <div className="rounded-2xl border border-sky-200/90 bg-gradient-to-br from-white via-sky-50/30 to-amber-50/20 p-5 sm:p-6 shadow-sm">
      {/* Top Banner: Regime Badge + Conviction + Report Age + Update Button */}
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-3.5 pb-4 border-b border-sky-100/80">
        <div className="flex flex-wrap items-center gap-2">
          <div className="flex items-center gap-2 mr-1">
            <span className="text-xl">🧠</span>
            <span className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">
              AI Executive Briefing
            </span>
          </div>

          {/* Regime Badge */}
          <span className="rounded-lg border border-sky-300 bg-sky-600 px-3 py-1 text-xs font-bold text-white shadow-2xs">
            สภาวะเศรษฐกิจหลัก: {aiData.overall_regime}
          </span>

          {/* Conviction Level */}
          {aiData.conviction_level && (
            <span className="rounded-lg border border-amber-300 bg-amber-100/90 px-2.5 py-1 text-xs font-bold uppercase text-amber-900">
              Conviction: {aiData.conviction_level}
            </span>
          )}

          {/* Horizon */}
          {aiData.time_horizon && (
            <span className="rounded-lg border border-edge bg-white px-2.5 py-1 text-xs font-medium text-zinc-700 shadow-2xs">
              กรอบเวลา: {aiData.time_horizon}
            </span>
          )}
        </div>

        {/* Right side: Report Age Badge & Action */}
        <div className="flex flex-wrap items-center gap-2 shrink-0">
          <div
            className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-semibold ${ageInfo.badgeClass}`}
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
              className="inline-flex items-center gap-1.5 rounded-xl border border-sky-200 bg-sky-50/80 px-3 py-1.5 text-xs font-semibold text-sky-800 shadow-2xs hover:bg-sky-100 transition-all disabled:opacity-50"
              title="สั่ง AI เริ่มวิเคราะห์ภาพรวมเศรษฐกิจรอบใหม่"
            >
              <span>{updating ? '⏳' : '🔄'}</span>
              <span>{updating ? 'กำลังสั่ง...' : 'อัปเดตบทวิเคราะห์'}</span>
            </button>
          )}
        </div>
      </div>

      {/* Middle Section: 30-Second Summary */}
      <div className="mt-4 grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
        {/* Left Column: Conviction Rationale & Focus Themes (7 cols) */}
        <div className="space-y-3 lg:col-span-7">
          {aiData.conviction_rationale && (
            <div>
              <div className="text-xs font-semibold text-zinc-900 mb-1 flex items-center gap-1.5">
                <span>📌</span>
                <span>สรุปข้อวินิจฉัยและเหตุผลหลัก (Core Thesis):</span>
              </div>
              <p
                className={`text-xs sm:text-[13px] leading-relaxed text-zinc-700 ${
                  !isExpanded ? 'line-clamp-2' : ''
                }`}
              >
                {aiData.conviction_rationale}
              </p>
              {aiData.conviction_rationale.length > 150 && (
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

          {/* Focus Themes */}
          {themes.length > 0 && (
            <div className="flex flex-wrap items-center gap-1.5 pt-1">
              <span className="text-xs font-semibold text-zinc-500 mr-1">ธีมหลัก (Themes):</span>
              {themes.map((theme, idx) => (
                <span
                  key={idx}
                  className="rounded-md border border-sky-200/80 bg-white/90 px-2.5 py-0.5 text-xs font-medium text-sky-900 shadow-2xs"
                >
                  #{theme}
                </span>
              ))}
            </div>
          )}

          {/* Alignment */}
          {aiData.quant_narrative_alignment && (
            <p className="text-[11px] text-zinc-500">
              ความสอดคล้องเชิงปริมาณและมุมมอง: <span className="font-semibold text-zinc-700">{aiData.quant_narrative_alignment}</span>
            </p>
          )}
        </div>

        {/* Right Column: Asset Stance Highlights & Warnings (5 cols) */}
        <div className="space-y-3 lg:col-span-5 rounded-xl border border-slate-200/80 bg-white/80 p-3.5 shadow-2xs">
          <div className="text-xs font-bold text-zinc-900 tracking-tight flex items-center justify-between">
            <span>🎯 ภาพรวมท่าทีสินทรัพย์ (Stance Summary)</span>
            {onNavigateToAiTab && (
              <button
                type="button"
                onClick={onNavigateToAiTab}
                className="text-[11px] font-semibold text-sky-700 hover:underline"
              >
                ดูพอร์ตเต็ม →
              </button>
            )}
          </div>

          {/* Overweight Chips */}
          <div className="space-y-1">
            <span className="text-[10px] font-bold uppercase tracking-wider text-emerald-800 block">
              Overweight (เพิ่มน้ำหนัก):
            </span>
            <div className="flex flex-wrap gap-1">
              {overweightAssets.length > 0 ? (
                overweightAssets.map((a, idx) => (
                  <span
                    key={idx}
                    className="rounded border border-emerald-200 bg-emerald-50 px-2 py-0.5 text-[11px] font-semibold text-emerald-800"
                    title={a.rationale}
                  >
                    {a.asset_class}
                  </span>
                ))
              ) : (
                <span className="text-[11px] text-zinc-400 italic">ไม่มีคำแนะนำเพิ่มน้ำหนัก</span>
              )}
            </div>
          </div>

          {/* Underweight Chips */}
          <div className="space-y-1 pt-1">
            <span className="text-[10px] font-bold uppercase tracking-wider text-rose-800 block">
              Underweight (ลดน้ำหนัก):
            </span>
            <div className="flex flex-wrap gap-1">
              {underweightAssets.length > 0 ? (
                underweightAssets.map((a, idx) => (
                  <span
                    key={idx}
                    className="rounded border border-rose-200 bg-rose-50 px-2 py-0.5 text-[11px] font-semibold text-rose-800"
                    title={a.rationale}
                  >
                    {a.asset_class}
                  </span>
                ))
              ) : (
                <span className="text-[11px] text-zinc-400 italic">ไม่มีคำแนะนำลดน้ำหนัก</span>
              )}
            </div>
          </div>

          {/* Top Warning highlight if any */}
          {warnings.length > 0 && (
            <div className="mt-2 rounded-lg border border-amber-200 bg-amber-50/70 p-2 text-xs text-amber-900">
              <div className="flex items-center gap-1 font-bold text-[11px]">
                <span>⚠️</span>
                <span>คำเตือนความเสี่ยงสำคัญ ({warnings.length} รายการ):</span>
              </div>
              <p className="mt-0.5 text-[11px] leading-tight line-clamp-2">
                {warnings[0]?.message || 'มีปัจจัยเสี่ยงที่ต้องติดตามใกล้ชิด'}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
