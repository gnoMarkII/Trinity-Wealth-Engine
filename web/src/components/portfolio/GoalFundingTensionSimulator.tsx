import { useState } from 'react'
import type { GoalAdjustmentLeversDTO } from '../../api/types'
import { formatTHB } from '../../utils/formatters'

export interface GoalFundingTensionSimulatorProps {
  goalName: string
  targetAmountThb: number
  currentAmountThb: number
  monthlySavingsThb: number
  deadlineDaysLeft?: number | null
  requiredReturnPct?: number | null
  tensionLevel?: 'CRITICAL_TENSION' | 'MODERATE_TENSION' | 'BALANCED' | 'INSUFFICIENT_DATA' | string | null
  adjustmentLevers: GoalAdjustmentLeversDTO
  expectedAnnualReturn?: number
  defaultExpanded?: boolean
}

function formatMonthsToYears(months: number): string {
  if (months < 12) return `${months} เดือน`
  const years = Math.floor(months / 12)
  const rem = months % 12
  return rem === 0 ? `${years} ปี` : `${years} ปี ${rem} เดือน`
}

export default function GoalFundingTensionSimulator({
  goalName,
  targetAmountThb,
  currentAmountThb: _currentAmountThb,
  monthlySavingsThb,
  deadlineDaysLeft,
  requiredReturnPct,
  tensionLevel,
  adjustmentLevers,
  expectedAnnualReturn = 7.0,
  defaultExpanded = false,
}: GoalFundingTensionSimulatorProps) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded)
  const [selectedLever, setSelectedLever] = useState<
    'save_more' | 'extend_timeline' | 'downsize_target' | 'accept_risk'
  >('save_more')

  const isCritical = tensionLevel === 'CRITICAL_TENSION'
  const reqPct = requiredReturnPct ?? expectedAnnualReturn
  const gapPct = Math.max(0, reqPct - expectedAnnualReturn)

  // Current months remaining
  const currentMonths = deadlineDaysLeft != null && deadlineDaysLeft > 0
    ? Math.max(1, Math.ceil(deadlineDaysLeft / 30.4375))
    : (adjustmentLevers.total_months_needed - adjustmentLevers.extend_timeline_months) || 12

  // Gauge bar math (min 0%, max 35%)
  const gaugeMax = Math.max(30, Math.ceil(reqPct * 1.2))
  const basePos = Math.min(100, Math.max(5, (expectedAnnualReturn / gaugeMax) * 100))
  const reqPos = Math.min(95, Math.max(basePos + 8, (reqPct / gaugeMax) * 100))

  const targetReduction = Math.max(0, targetAmountThb - adjustmentLevers.downsized_target_thb)
  const targetReductionPct = targetAmountThb > 0 ? (targetReduction / targetAmountThb) * 100 : 0

  return (
    <div
      data-testid="goal-funding-tension-simulator"
      className={`rounded-2xl border transition-all shadow-2xs overflow-hidden ${
        isCritical
          ? 'border-rose-200/90 bg-gradient-to-b from-rose-50/70 via-white to-rose-50/30 text-rose-950'
          : 'border-amber-200/90 bg-gradient-to-b from-amber-50/70 via-white to-amber-50/30 text-amber-950'
      }`}
    >
      {/* 1. Collapsible Header Banner */}
      <button
        type="button"
        onClick={() => setIsExpanded(!isExpanded)}
        aria-expanded={isExpanded}
        className={`w-full text-left p-3 sm:p-3.5 transition-colors flex items-center justify-between gap-2.5 cursor-pointer ${
          isExpanded ? 'border-b border-edge/60 bg-black/[0.015]' : 'hover:bg-black/[0.02]'
        }`}
      >
        <div className="flex items-center gap-2.5 min-w-0">
          <span className="text-base sm:text-lg shrink-0">{isCritical ? '⚠️' : '⚡'}</span>
          <div className="min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <h4 className="font-bold text-xs sm:text-sm text-zinc-900 tracking-tight">
                {isCritical
                  ? 'เป้าหมายมีความตึงเครียดสูง (Required Return สูงกว่าพอร์ต)'
                  : 'เป้าหมายต้องพึ่งพาผลตอบแทนเชิงรุก'}
              </h4>
              <span
                className={`px-2 py-0.5 rounded-full text-[10px] font-bold border uppercase tracking-wider shrink-0 ${
                  isCritical
                    ? 'bg-rose-100 text-rose-800 border-rose-300'
                    : 'bg-amber-100 text-amber-800 border-amber-300'
                }`}
              >
                Risk Gap: +{gapPct.toFixed(1)}% ต่อปี
              </span>
            </div>
            <p className="text-[11px] text-zinc-500 mt-0.5 truncate">
              {isExpanded
                ? `ผลตอบแทนที่ต้องการ (${reqPct.toFixed(1)}%) สูงกว่าพอร์ต (${expectedAnnualReturn.toFixed(1)}%) • ปรับแผนได้ด้วย 4 คานงัด:`
                : `ผลตอบแทนที่ต้องการ (${reqPct.toFixed(1)}%) สูงกว่าพอร์ต (${expectedAnnualReturn.toFixed(1)}%) • คลิกเพื่อเปิดจำลอง 4 คานงัด`}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-1.5 shrink-0">
          <span className="hidden sm:inline-block text-[11px] font-bold text-zinc-600">
            {isExpanded ? 'หุบแบบจำลอง' : 'จำลอง 4 คานงัด'}
          </span>
          <span
            className={`w-6 h-6 rounded-full flex items-center justify-center border text-xs font-bold transition-transform duration-200 shadow-2xs ${
              isExpanded
                ? 'rotate-180 bg-zinc-800 text-white border-zinc-800'
                : 'bg-white text-zinc-700 border-zinc-300'
            }`}
          >
            ▾
          </span>
        </div>
      </button>

      {/* 2, 3, 4: Collapsible Details */}
      {isExpanded && (
        <div className="p-4 sm:p-5 pt-3.5 space-y-4">
          {/* 2. Tension Gap Meter (Visual Gauge) */}
          <div className="space-y-1.5 rounded-xl border border-edge/60 bg-white/90 p-3 shadow-2xs">
            <div className="flex items-center justify-between text-[11px] font-semibold text-zinc-600">
              <span className="flex items-center gap-1 text-emerald-700">
                <span>🛡️</span>
                <span>พอร์ตคาดหวังทั่วไป: <strong>{expectedAnnualReturn.toFixed(1)}% ต่อปี</strong></span>
              </span>
              <span className="flex items-center gap-1 text-rose-700 font-bold">
                <span>🎯</span>
                <span>เป้าหมายเรียกร้อง: <strong>{reqPct.toFixed(1)}% ต่อปี</strong></span>
              </span>
            </div>

            {/* Gauge Track */}
            <div className="relative w-full h-3.5 rounded-full bg-zinc-100 overflow-hidden border border-edge/50">
              {/* Gradient background track */}
              <div className="absolute inset-0 bg-gradient-to-r from-emerald-300 via-amber-300 to-rose-500 opacity-60" />

              {/* Gap highlight area */}
              <div
                className="absolute top-0 bottom-0 bg-rose-600/30 border-y border-rose-500"
                style={{
                  left: `${basePos}%`,
                  width: `${Math.max(5, reqPos - basePos)}%`,
                }}
              />

              {/* Baseline pin */}
              <div
                className="absolute top-0 bottom-0 w-1 bg-emerald-700 shadow-sm"
                style={{ left: `${basePos}%` }}
                title={`Baseline Return: ${expectedAnnualReturn.toFixed(1)}%`}
              />

              {/* Required return pin */}
              <div
                className="absolute top-0 bottom-0 w-1.5 bg-rose-700 shadow-sm animate-pulse"
                style={{ left: `${reqPos}%` }}
                title={`Required Return: ${reqPct.toFixed(1)}%`}
              />
            </div>

            <div className="flex items-center justify-between text-[10px] text-zinc-400 font-mono pt-0.5">
              <span>0%</span>
              <span className="text-rose-700 font-bold">
                ⚡ ช่องว่างผลตอบแทนที่เกินกำลังพอร์ต: +{gapPct.toFixed(1)}% ต่อปี
              </span>
              <span>{gaugeMax}%</span>
            </div>
          </div>

      {/* 3. The 4 Interactive Lever Cards */}
      <div className="space-y-2">
        <div className="flex flex-wrap items-center justify-between gap-1 text-[11px] font-semibold text-zinc-500">
          <span className="flex items-center gap-1">
            <span>🎮</span>
            <span>จำลอง 4 คานงัด (Interactive Simulator):</span>
          </span>
          <span className="text-zinc-400 text-[10px] shrink-0">คลิกการ์ดเพื่อสลับดูผล</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
          {/* Lever 1: Save More */}
          <button
            type="button"
            onClick={() => setSelectedLever('save_more')}
            className={`p-3 rounded-xl border text-left transition-all cursor-pointer relative ${
              selectedLever === 'save_more'
                ? 'bg-emerald-50/90 border-emerald-400 ring-2 ring-emerald-500 shadow-sm'
                : 'bg-white/90 border-edge/80 hover:border-emerald-300 hover:bg-emerald-50/30'
            }`}
          >
            <div className="flex items-center justify-between gap-1.5 mb-1.5">
              <span className="font-bold text-zinc-900 text-xs flex items-center gap-1.5 shrink-0">
                <span className={`w-3.5 h-3.5 rounded-full border-2 flex items-center justify-center shrink-0 ${
                  selectedLever === 'save_more'
                    ? 'border-emerald-600 bg-emerald-600'
                    : 'border-zinc-300 bg-white'
                }`}>
                  {selectedLever === 'save_more' && <span className="w-1.5 h-1.5 rounded-full bg-white block" />}
                </span>
                <span>💰 1. ออมเพิ่ม</span>
              </span>
              <span className="text-[9px] px-1.5 py-0.5 rounded font-bold uppercase tracking-wider bg-emerald-100 text-emerald-800 border border-emerald-200 shrink-0">
                ควบคุมได้
              </span>
            </div>
            <div className="font-mono font-black text-sm sm:text-base text-emerald-950 mt-1 whitespace-nowrap">
              +{formatTHB(adjustmentLevers.save_more_monthly_thb)}/ด.
            </div>
            <div className="text-[10px] text-zinc-500 mt-1 font-mono truncate">
              รวมใหม่: {formatTHB(adjustmentLevers.total_monthly_savings_needed_thb)}/ด.
            </div>
          </button>

          {/* Lever 2: Extend Timeline */}
          <button
            type="button"
            onClick={() => setSelectedLever('extend_timeline')}
            className={`p-3 rounded-xl border text-left transition-all cursor-pointer relative ${
              selectedLever === 'extend_timeline'
                ? 'bg-sky-50/90 border-sky-400 ring-2 ring-sky-500 shadow-sm'
                : 'bg-white/90 border-edge/80 hover:border-sky-300 hover:bg-sky-50/30'
            }`}
          >
            <div className="flex items-center justify-between gap-1.5 mb-1.5">
              <span className="font-bold text-zinc-900 text-xs flex items-center gap-1.5 shrink-0">
                <span className={`w-3.5 h-3.5 rounded-full border-2 flex items-center justify-center shrink-0 ${
                  selectedLever === 'extend_timeline'
                    ? 'border-sky-600 bg-sky-600'
                    : 'border-zinc-300 bg-white'
                }`}>
                  {selectedLever === 'extend_timeline' && <span className="w-1.5 h-1.5 rounded-full bg-white block" />}
                </span>
                <span>⏳ 2. ยืดเวลา</span>
              </span>
              <span className="text-[9px] px-1.5 py-0.5 rounded font-bold uppercase tracking-wider bg-sky-100 text-sky-800 border border-sky-200 shrink-0">
                เวลาทบต้น
              </span>
            </div>
            <div className="font-mono font-black text-sm sm:text-base text-sky-950 mt-1 whitespace-nowrap">
              +{adjustmentLevers.extend_timeline_months} เดือน
            </div>
            <div className="text-[10px] text-zinc-500 mt-1 font-mono truncate">
              ({formatMonthsToYears(adjustmentLevers.extend_timeline_months)}) • รวม {adjustmentLevers.total_months_needed} ด.
            </div>
          </button>

          {/* Lever 3: Downsize Target */}
          <button
            type="button"
            onClick={() => setSelectedLever('downsize_target')}
            className={`p-3 rounded-xl border text-left transition-all cursor-pointer relative ${
              selectedLever === 'downsize_target'
                ? 'bg-amber-50/90 border-amber-400 ring-2 ring-amber-500 shadow-sm'
                : 'bg-white/90 border-edge/80 hover:border-amber-300 hover:bg-amber-50/30'
            }`}
          >
            <div className="flex items-center justify-between gap-1.5 mb-1.5">
              <span className="font-bold text-zinc-900 text-xs flex items-center gap-1.5 shrink-0">
                <span className={`w-3.5 h-3.5 rounded-full border-2 flex items-center justify-center shrink-0 ${
                  selectedLever === 'downsize_target'
                    ? 'border-amber-600 bg-amber-600'
                    : 'border-zinc-300 bg-white'
                }`}>
                  {selectedLever === 'downsize_target' && <span className="w-1.5 h-1.5 rounded-full bg-white block" />}
                </span>
                <span>📉 3. ปรับเป้าหมาย</span>
              </span>
              <span className="text-[9px] px-1.5 py-0.5 rounded font-bold uppercase tracking-wider bg-amber-100 text-amber-800 border border-amber-200 shrink-0">
                เป้าสอดคล้อง
              </span>
            </div>
            <div className="font-mono font-black text-sm sm:text-base text-amber-950 mt-1 whitespace-nowrap">
              เหลือ {formatTHB(adjustmentLevers.downsized_target_thb)}
            </div>
            <div className="text-[10px] text-zinc-500 mt-1 font-mono truncate">
              ลดลง -{formatTHB(targetReduction)} (-{targetReductionPct.toFixed(1)}%)
            </div>
          </button>

          {/* Lever 4: Accept Risk */}
          <button
            type="button"
            onClick={() => setSelectedLever('accept_risk')}
            className={`p-3 rounded-xl border text-left transition-all cursor-pointer relative ${
              selectedLever === 'accept_risk'
                ? 'bg-rose-50/90 border-rose-400 ring-2 ring-rose-500 shadow-sm'
                : 'bg-white/90 border-edge/80 hover:border-rose-300 hover:bg-rose-50/30'
            }`}
          >
            <div className="flex items-center justify-between gap-1.5 mb-1.5">
              <span className="font-bold text-zinc-900 text-xs flex items-center gap-1.5 shrink-0">
                <span className={`w-3.5 h-3.5 rounded-full border-2 flex items-center justify-center shrink-0 ${
                  selectedLever === 'accept_risk'
                    ? 'border-rose-600 bg-rose-600'
                    : 'border-zinc-300 bg-white'
                }`}>
                  {selectedLever === 'accept_risk' && <span className="w-1.5 h-1.5 rounded-full bg-white block" />}
                </span>
                <span>🚀 4. รับเสี่ยงเพิ่ม</span>
              </span>
              <span className="text-[9px] px-1.5 py-0.5 rounded font-bold uppercase tracking-wider bg-rose-100 text-rose-800 border border-rose-200 shrink-0">
                เสี่ยงพิเศษ
              </span>
            </div>
            <div className="font-mono font-black text-sm sm:text-base text-rose-950 mt-1 whitespace-nowrap">
              พอร์ต {reqPct.toFixed(1)}% CAGR
            </div>
            <div className="text-[10px] text-zinc-500 mt-1 font-mono truncate">
              เกินเกณฑ์ปกติ +{gapPct.toFixed(1)}% ต่อปี
            </div>
          </button>
        </div>
      </div>

      {/* 4. Simulator Preview Drawer (Before vs After Plan Comparison) */}
      <div className="rounded-xl border border-edge/80 bg-white p-3.5 space-y-3 shadow-2xs">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-edge/50 pb-2">
          <div className="flex items-center gap-2">
            <div className="font-bold text-zinc-800 text-xs flex items-center gap-1.5">
              <span>📋</span>
              <span>
                ผลลัพธ์จำลองเมื่อเลือก:{' '}
                <strong className="text-zinc-950 underline decoration-indigo-400 underline-offset-2">
                  {selectedLever === 'save_more' && 'คานงัดที่ 1: ออมเพิ่มรายเดือน'}
                  {selectedLever === 'extend_timeline' && 'คานงัดที่ 2: ขยายระยะเวลา'}
                  {selectedLever === 'downsize_target' && 'คานงัดที่ 3: ปรับขนาดเป้าหมาย'}
                  {selectedLever === 'accept_risk' && 'คานงัดที่ 4: รับความเสี่ยงเพิ่มขึ้น'}
                </strong>
              </span>
            </div>
          </div>
          <span className="text-[10px] px-2 py-0.5 rounded bg-zinc-100 text-zinc-600 font-medium">
            เป้าหมาย: {goalName}
          </span>
        </div>

        {/* 4-Metric Before vs After Comparison Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 text-xs">
          {/* Monthly Savings */}
          <div className="rounded-lg bg-surface p-2.5 border border-edge/60 space-y-1">
            <span className="text-[10px] text-zinc-400 font-bold uppercase block">เงินออมต่อเดือน</span>
            <div className="font-mono">
              {selectedLever === 'save_more' ? (
                <div>
                  <div className="text-xs line-through text-zinc-400">{formatTHB(monthlySavingsThb)}/ด.</div>
                  <div className="text-sm font-bold text-emerald-700">
                    {formatTHB(adjustmentLevers.total_monthly_savings_needed_thb)}/ด.
                  </div>
                </div>
              ) : (
                <div className="text-xs font-semibold text-zinc-800">
                  {formatTHB(monthlySavingsThb)}/ด. <span className="text-[10px] text-zinc-400">(คงเดิม)</span>
                </div>
              )}
            </div>
          </div>

          {/* Timeline */}
          <div className="rounded-lg bg-surface p-2.5 border border-edge/60 space-y-1">
            <span className="text-[10px] text-zinc-400 font-bold uppercase block">ระยะเวลาที่เหลือ</span>
            <div className="font-mono">
              {selectedLever === 'extend_timeline' ? (
                <div>
                  <div className="text-xs line-through text-zinc-400">{currentMonths} เดือน</div>
                  <div className="text-sm font-bold text-sky-700">
                    {adjustmentLevers.total_months_needed} เดือน ({formatMonthsToYears(adjustmentLevers.total_months_needed)})
                  </div>
                </div>
              ) : (
                <div className="text-xs font-semibold text-zinc-800">
                  {currentMonths} เดือน <span className="text-[10px] text-zinc-400">(คงเดิม)</span>
                </div>
              )}
            </div>
          </div>

          {/* Target Amount */}
          <div className="rounded-lg bg-surface p-2.5 border border-edge/60 space-y-1">
            <span className="text-[10px] text-zinc-400 font-bold uppercase block">เงินเป้าหมาย</span>
            <div className="font-mono">
              {selectedLever === 'downsize_target' ? (
                <div>
                  <div className="text-xs line-through text-zinc-400">{formatTHB(targetAmountThb)}</div>
                  <div className="text-sm font-bold text-amber-700">
                    {formatTHB(adjustmentLevers.downsized_target_thb)}
                  </div>
                </div>
              ) : (
                <div className="text-xs font-semibold text-zinc-800">
                  {formatTHB(targetAmountThb)} <span className="text-[10px] text-zinc-400">(คงเดิม)</span>
                </div>
              )}
            </div>
          </div>

          {/* Required Return / Risk */}
          <div className="rounded-lg bg-surface p-2.5 border border-edge/60 space-y-1">
            <span className="text-[10px] text-zinc-400 font-bold uppercase block">ผลตอบแทนที่ต้องการ</span>
            <div className="font-mono">
              {selectedLever === 'accept_risk' ? (
                <div>
                  <div className="text-sm font-bold text-rose-700">{reqPct.toFixed(1)}% ต่อปี</div>
                  <span className="text-[9px] text-rose-600 font-sans block">⚠️ เสี่ยงสูงพิเศษ</span>
                </div>
              ) : (
                <div>
                  <div className="text-xs line-through text-zinc-400">{reqPct.toFixed(1)}% ต่อปี</div>
                  <div className="text-sm font-bold text-emerald-700">
                    {expectedAnnualReturn.toFixed(1)}% ต่อปี
                  </div>
                  <span className="text-[9px] text-emerald-600 font-sans block">✅ เกณฑ์ปลอดภัย</span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Narrative Strategic Guidance */}
        <div className="text-[11px] leading-relaxed p-2.5 rounded-lg bg-zinc-50 border border-edge/50 text-zinc-700">
          {selectedLever === 'save_more' && (
            <div className="space-y-0.5">
              <span className="font-bold text-emerald-900 block">💡 ข้อแนะนำเชิงกลยุทธ์ (วินัยการเงิน):</span>
              <span>
                เมื่อคุณเพิ่มเงินออมเป็น <strong>{formatTHB(adjustmentLevers.total_monthly_savings_needed_thb)}/ด.</strong> (+{formatTHB(adjustmentLevers.save_more_monthly_thb)}/ด.) พอร์ตจะสามารถบรรลุเป้าหมายเดิมได้ในกำหนดเวลาเดิม โดยพึ่งพาผลตอบแทนมาตรฐานเพียง <strong>{expectedAnnualReturn.toFixed(1)}% ต่อปี</strong> ซึ่งปลอดภัยและลดความเสี่ยงขาดทุนลงได้มากที่สุด
              </span>
            </div>
          )}

          {selectedLever === 'extend_timeline' && (
            <div className="space-y-0.5">
              <span className="font-bold text-sky-900 block">💡 ข้อแนะนำเชิงกลยุทธ์ (พลังเวลาทบต้น):</span>
              <span>
                หากขยายเวลาออกไปอีก <strong>+{adjustmentLevers.extend_timeline_months} เดือน</strong> ({formatMonthsToYears(adjustmentLevers.extend_timeline_months)}) คุณจะสามารถออมเงินด้วยยอดเดิม ({formatTHB(monthlySavingsThb)}/ด.) ต่อไปได้สบายๆ โดยปล่อยให้พลังดอกเบี้ยทบต้นทำงานที่ <strong>{expectedAnnualReturn.toFixed(1)}% ต่อปี</strong> จนบรรลุเป้าหมาย
              </span>
            </div>
          )}

          {selectedLever === 'downsize_target' && (
            <div className="space-y-0.5">
              <span className="font-bold text-amber-900 block">💡 ข้อแนะนำเชิงกลยุทธ์ (ปรับสเกลเป้าหมาย):</span>
              <span>
                หากคุณไม่ต้องการเพิ่มเงินออมและไม่ต้องการยืดเวลา การปรับขนาดเป้าหมายลงเหลือ <strong>{formatTHB(adjustmentLevers.downsized_target_thb)}</strong> จะทำให้แผนการเงินเป็นไปได้จริงทันทีโดยไม่ก่อให้เกิดความตึงเครียดต่อพอร์ตการลงทุน
              </span>
            </div>
          )}

          {selectedLever === 'accept_risk' && (
            <div className="space-y-0.5">
              <span className="font-bold text-rose-900 block">⚠️ ข้อแนะนำเชิงกลยุทธ์ (ความเสี่ยงสูงมาก):</span>
              <span>
                การคงแผนเดิมโดยไม่ปรับตัวแปรใดๆ จะบังคับให้พอร์ตต้องทำผลตอบแทนสูงถึง <strong>{reqPct.toFixed(1)}% ต่อปี</strong> ซึ่งเกินกว่าผลตอบแทนเฉลี่ยของตลาดหุ้นระยะยาวอย่างมาก และอาจทำให้พอร์ตเผชิญกับ Drawdown หนักหรือเป้าหมายล้มเหลวหากตลาดปรับฐาน
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Bottom quick-collapse toggle */}
      <div className="pt-0.5 flex justify-center">
        <button
          type="button"
          onClick={() => setIsExpanded(false)}
          className="inline-flex items-center gap-1.5 text-[11px] font-semibold text-zinc-500 hover:text-zinc-800 hover:bg-zinc-100 px-3.5 py-1 rounded-full border border-zinc-200 transition-colors cursor-pointer bg-white/80 shadow-2xs"
        >
          <span>▴</span>
          <span>หุบแบบจำลอง 4 คานงัด</span>
        </button>
      </div>
    </div>
  )}
</div>
  )
}
