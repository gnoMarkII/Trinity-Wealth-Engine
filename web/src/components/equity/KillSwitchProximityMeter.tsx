import React from 'react'
import { ContextTooltip } from './explainer'

export interface FalsifierItem {
  falsifier_id: string
  metric_name: string
  condition: string
  threshold_value: number | string | null
  source_ref?: string | null
  narrative_explanation: string
}

export interface KillSwitchProximityMeterProps {
  falsifiers: FalsifierItem[]
  currentOperatingMargin?: number | null
  currentPrice?: number | null
  currency?: string
}

function humanizeSourceRef(sourceRef?: string | null): { label: string; tooltip: string } {
  if (!sourceRef) return { label: 'Internal Audit', tooltip: 'Calculated from internal models' }
  if (sourceRef.includes('base_ebit_margin_pct')) {
    return { label: '📄 GAAP 10-Q Filing', tooltip: sourceRef }
  }
  if (sourceRef.includes('invalidation_stop_loss') || sourceRef.includes('tactical_setup')) {
    return { label: '📈 Weekly Pivot S1 (Technical)', tooltip: sourceRef }
  }
  return { label: 'Authoritative Metric', tooltip: sourceRef }
}

function parseThresholdNumeric(val?: number | string | null): number | null {
  if (val == null) return null
  if (typeof val === 'number') return isNaN(val) ? null : val
  const cleaned = parseFloat(String(val).replace(/[^0-9.-]/g, ''))
  return Number.isNaN(cleaned) ? null : cleaned
}

export const KillSwitchProximityMeter: React.FC<KillSwitchProximityMeterProps> = ({
  falsifiers,
  currentOperatingMargin,
  currentPrice,
  currency = '$',
}) => {
  if (!falsifiers || falsifiers.length === 0) return null

  // Extract threshold values if present
  const marginFalsifier = falsifiers.find(f =>
    f.falsifier_id.toLowerCase().includes('margin') ||
    f.metric_name.toLowerCase().includes('margin') ||
    f.metric_name.toLowerCase().includes('ebit')
  )
  const stopLossFalsifier = falsifiers.find(f =>
    f.falsifier_id.toLowerCase().includes('stop') ||
    f.falsifier_id.toLowerCase().includes('support') ||
    f.metric_name.toLowerCase().includes('support') ||
    f.metric_name.toLowerCase().includes('stop')
  )

  const marginThreshold = parseThresholdNumeric(marginFalsifier?.threshold_value)
  const stopLossThreshold = parseThresholdNumeric(stopLossFalsifier?.threshold_value)

  const marginBuffer = (currentOperatingMargin != null && marginThreshold != null)
    ? currentOperatingMargin - marginThreshold
    : null

  const stopBuffer = (currentPrice != null && stopLossThreshold != null)
    ? currentPrice - stopLossThreshold
    : null
  const stopBufferPct = (stopBuffer != null && stopLossThreshold != null && stopLossThreshold > 0)
    ? (stopBuffer / stopLossThreshold) * 100
    : null

  return (
    <section className="flow-panel rounded-2xl border border-rose-200/90 bg-rose-50/20 p-6 shadow-xs space-y-6 animate-card-in">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-rose-200/60 pb-4">
        <div className="flex items-center gap-2.5">
          <span className="text-xl">🛑</span>
          <div>
            <h3 className="text-sm font-bold uppercase tracking-wider text-rose-950 flex items-center gap-2">
              <span>Thesis Falsifiers & Invalidation Criteria</span>
              <ContextTooltip
                title="เกณฑ์ลบล้างสมมติฐาน (Thesis Falsifiers & Kill Switches)"
                whatIsIt="ชุดเงื่อนไขที่กำหนดไว้ล่วงหน้าว่า 'ถ้าเกิดเหตุการณ์นี้ขึ้น แปลว่าเราคิดผิดตั้งแต่แรก' เช่น อัตรากำไรทรุดหนัก หรือราคาหลุดแนวรับโครงสร้าง"
                howToRead="กองทุนมืออาชีพจะกำหนดจุดยอมแพ้ก่อนเข้าซื้อเสมอ หากแตะเกณฑ์ Kill Switch ต้องตัดขาดทุนทันที ห้ามหวังลมๆ แล้งๆ หรือเปลี่ยนใจถือยาว"
                impactOnMoney="กฎเหล็กที่ปกป้องพอร์ตไม่ให้เสียหายหนัก และป้องกันการติดดอยยาวข้ามปี"
              />
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-rose-100 text-rose-800 font-extrabold">
                {falsifiers.length} Kill-Switches Active
              </span>
            </h3>
            <p className="text-xs text-rose-800/80 mt-0.5">
              เกณฑ์ชี้วัดการลบล้างสมมติฐานการลงทุน — หากเงื่อนไขใดถูกละเมิด วิทยานิพนธ์การลงทุนจะถือว่าล้มเหลวทันที
            </p>
          </div>
        </div>

        <span className="text-xs px-2.5 py-1 rounded-xl bg-white/80 border border-rose-200 text-rose-700 font-semibold shadow-2xs">
          🛡️ Institutional Risk Governance
        </span>
      </div>

      {/* Practical Safety Shield 2: Stop Loss & Kill Switch Discipline Banner */}
      <div className="flex items-start gap-3 p-3.5 rounded-xl bg-rose-100/70 border border-rose-300/80 text-xs text-rose-950">
        <span className="text-base shrink-0 mt-0.5">🚨</span>
        <div className="space-y-0.5">
          <span className="font-bold block text-rose-950">กฎเหล็กการเอาตัวรอด (Kill Switch Discipline):</span>
          <p className="text-rose-900 leading-relaxed text-[11px]">
            หากราคาหลุดแนวรับสำคัญ หรือ Margin หดตัวต่ำกว่าเกณฑ์สมมติฐาน แปลว่า <em>'เหตุผลที่เราซื้อตั้งแต่แรกผิดไปแล้ว'</em> ต้องกล้ายอมรับความจริงและตัดขาดทุนตามแผน อย่าถือหุ้นด้วยความหวัง
          </p>
        </div>
      </div>

      {/* Infographic 4: Safety Buffer Meters */}
      {(marginBuffer != null || stopBuffer != null) && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Buffer 1: EBIT Margin Safety Buffer */}
          {marginBuffer != null && marginThreshold != null && currentOperatingMargin != null && (
            <div className="p-4 rounded-xl bg-white/70 border border-rose-200/80 shadow-2xs space-y-3">
              <div className="flex items-center justify-between text-xs">
                <span className="font-semibold text-zinc-700">1. EBIT Margin Floor Safety Buffer</span>
                <span
                  className={`px-2 py-0.5 rounded-full font-bold text-xs ${
                    marginBuffer > 1.5
                      ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                      : marginBuffer > 0
                      ? 'bg-amber-50 text-amber-700 border border-amber-200'
                      : 'bg-rose-100 text-rose-800 border border-rose-300'
                  }`}
                >
                  {marginBuffer >= 0 ? '+' : ''}{marginBuffer.toFixed(2)}% Buffer
                </span>
              </div>

              {/* Big Number Tier 2 Display */}
              <div className="flex items-baseline justify-between pt-1">
                <div>
                  <span className="text-[10px] font-bold uppercase text-zinc-400 block">Current EBIT Margin</span>
                  <span className="font-mono text-2xl font-bold text-zinc-900">{currentOperatingMargin.toFixed(2)}%</span>
                </div>
                <div className="text-right">
                  <span className="text-[10px] font-bold uppercase text-rose-600 block">Kill-Switch Threshold</span>
                  <span className="font-mono text-xl font-bold text-rose-600">&lt; {marginThreshold.toFixed(1)}%</span>
                </div>
              </div>

              {/* Visual Buffer Bar */}
              <div className="space-y-1">
                <div className="h-2.5 w-full rounded-full bg-rose-200 overflow-hidden flex">
                  <div
                    className="h-full bg-emerald-500 rounded-full"
                    style={{ width: `${Math.min(100, Math.max(15, (marginBuffer / 5) * 100))}%` }}
                    title={`Current safety buffer: +${marginBuffer.toFixed(2)}% above threshold`}
                  />
                </div>
                <div className="flex justify-between text-[10px] text-zinc-400">
                  <span>Threshold Floor ({marginThreshold.toFixed(1)}%)</span>
                  <span>Safe Operating Zone</span>
                </div>
              </div>
            </div>
          )}

          {/* Buffer 2: Structural Support Distance */}
          {stopBuffer != null && stopLossThreshold != null && currentPrice != null && (
            <div className="p-4 rounded-xl bg-white/70 border border-rose-200/80 shadow-2xs space-y-3">
              <div className="flex items-center justify-between text-xs">
                <span className="font-semibold text-zinc-700">2. Structural Stop Distance</span>
                <span
                  className={`px-2 py-0.5 rounded-full font-bold text-xs ${
                    stopBuffer > 0
                      ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                      : 'bg-rose-100 text-rose-800 border border-rose-300'
                  }`}
                >
                  +{currency}{stopBuffer.toFixed(2)} ({stopBufferPct?.toFixed(1)}% buffer)
                </span>
              </div>

              {/* Big Number Tier 2 Display */}
              <div className="flex items-baseline justify-between pt-1">
                <div>
                  <span className="text-[10px] font-bold uppercase text-zinc-400 block">Current Close</span>
                  <span className="font-mono text-2xl font-bold text-zinc-900">{currency}{currentPrice.toFixed(2)}</span>
                </div>
                <div className="text-right">
                  <span className="text-[10px] font-bold uppercase text-rose-600 block">Invalidation Stop</span>
                  <span className="font-mono text-xl font-bold text-rose-600">&lt; {currency}{stopLossThreshold.toFixed(2)}</span>
                </div>
              </div>

              {/* Visual Buffer Bar */}
              <div className="space-y-1">
                <div className="h-2.5 w-full rounded-full bg-rose-200 overflow-hidden flex">
                  <div
                    className="h-full bg-sky-500 rounded-full"
                    style={{ width: `${Math.min(100, Math.max(15, ((stopBufferPct || 0) / 10) * 100))}%` }}
                    title={`Buffer to stop: +${currency}${stopBuffer.toFixed(2)}`}
                  />
                </div>
                <div className="flex justify-between text-[10px] text-zinc-400">
                  <span>Weekly Close Breakdown ({currency}{stopLossThreshold.toFixed(2)})</span>
                  <span>Current Price Level</span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Clean Humanized Table */}
      <div className="overflow-x-auto rounded-xl bg-white/90 border border-rose-200/70">
        <table className="w-full text-left text-xs border-collapse">
          <thead>
            <tr className="border-b border-rose-200/80 bg-rose-50/40 text-rose-900 font-semibold">
              <th className="py-2.5 pl-3">Condition ID</th>
              <th className="py-2.5 px-3">Metric / Trigger Condition</th>
              <th className="py-2.5 px-3">Threshold</th>
              <th className="py-2.5 px-3">Audit Source</th>
              <th className="py-2.5 pr-3">Investment Risk Rationale</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-rose-100 text-zinc-800">
            {falsifiers.map((f) => {
              const audit = humanizeSourceRef(f.source_ref)
              return (
                <tr key={f.falsifier_id} className="hover:bg-rose-50/30 transition-colors">
                  <td className="py-3 pl-3 font-mono font-bold text-rose-700 whitespace-nowrap">
                    {f.falsifier_id}
                  </td>
                  <td className="py-3 px-3 font-medium">
                    <div>{f.metric_name}</div>
                    <div className="text-rose-600 font-normal text-[11px] mt-0.5">{f.condition}</div>
                  </td>
                  <td className="py-3 px-3 font-bold font-mono text-zinc-900 whitespace-nowrap">
                    {f.threshold_value != null ? `${f.threshold_value}` : 'N/A'}
                  </td>
                  <td className="py-3 px-3 whitespace-nowrap">
                    <span
                      className="px-2 py-0.5 rounded-md bg-zinc-100 border border-zinc-200 text-zinc-700 font-medium text-[11px] cursor-help inline-block"
                      title={audit.tooltip}
                    >
                      {audit.label}
                    </span>
                  </td>
                  <td className="py-3 pr-3 text-zinc-600 leading-relaxed max-w-md">
                    {f.narrative_explanation}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </section>
  )
}

export default KillSwitchProximityMeter
