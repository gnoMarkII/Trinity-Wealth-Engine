import React from 'react'
import type { SectorAnalysisDTO } from '../../../api/types'

interface SectorAiPanelProps {
  sectorAnalysis?: SectorAnalysisDTO | Record<string, any> | null
}

export const SectorAiPanel: React.FC<SectorAiPanelProps> = ({ sectorAnalysis }) => {
  if (!sectorAnalysis) {
    return (
      <div className="rounded-xl border border-edge bg-panel p-4 shadow-sm">
        <h3 className="text-sm font-semibold text-zinc-900 mb-1">
          มุมมองรายกลุ่มอุตสาหกรรม (AI Sector Rotation Analysis)
        </h3>
        <p className="text-xs text-zinc-500">
          ยังไม่มีบทวิเคราะห์กลุ่มอุตสาหกรรมจาก AI ในรอบนี้ (ข้อมูลคำนวณ RRG ดูได้ในแท็บสหรัฐอเมริกา)
        </p>
      </div>
    )
  }

  const analysis = sectorAnalysis as Record<string, any>
  const status = analysis.analysis_status || 'available'
  const summaryTh = analysis.summary_th || ''
  const resolvedMetrics = analysis.resolved_metrics || analysis.fact_claims || []
  const watchConditions = analysis.watch_conditions || []
  const snapshotId = analysis.snapshot_id || ''
  const asOfDate = analysis.as_of_date || ''

  return (
    <div className="rounded-xl border border-sky-100 bg-panel p-4 shadow-sm space-y-3">
      {/* Title & Snapshot Provenance */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 border-b border-sky-100/70 pb-2">
        <div>
          <h3 className="text-sm font-semibold text-zinc-900 flex items-center gap-1.5">
            <span>🔄</span>
            <span>มุมมองรายกลุ่มอุตสาหกรรม (AI Sector Rotation Analysis)</span>
          </h3>
          <p className="text-xs text-zinc-500">
            วิเคราะห์แนวโน้มการหมุนเวียนกลุ่มอุตสาหกรรม (S&P 500 11 Sectors) ร่วมกับสภาวะเศรษฐกิจ
          </p>
        </div>
        <div className="flex items-center gap-1.5 text-[11px] font-mono text-zinc-500 shrink-0">
          <span className="rounded bg-sky-50 text-sky-800 border border-sky-200 px-2 py-0.5 font-sans font-semibold">
            {status === 'available' ? 'วิเคราะห์แล้ว' : status}
          </span>
          {asOfDate && <span>ณ วันที่ {asOfDate}</span>}
        </div>
      </div>

      {/* Thai Narrative Summary */}
      {summaryTh ? (
        <div className="rounded-lg bg-sky-50/60 border border-sky-100/80 p-3 text-xs sm:text-[13px] leading-relaxed text-zinc-800">
          <div className="font-semibold text-sky-950 mb-1 text-xs">
            📊 บทสรุปการหมุนเวียนกลุ่มหุ้น (Sector Rotation Narrative):
          </div>
          <p>{summaryTh}</p>
        </div>
      ) : (
        <p className="text-xs text-zinc-600">
          กลุ่มอุตสาหกรรมถูกประเมินตามกรอบ Relative Trend และ Momentum เทียบกับ S&P 500
        </p>
      )}

      {/* Verified Claims / Resolved Metrics */}
      {resolvedMetrics.length > 0 && (
        <div className="space-y-1.5 pt-1">
          <span className="text-xs font-semibold text-zinc-700 block">
            ตัวชี้วัดที่ยืนยันแล้ว (Verified Sector Claims):
          </span>
          <div className="flex flex-wrap gap-1.5">
            {resolvedMetrics.map((m: any, idx: number) => {
              const label = typeof m === 'string' ? m : `${m.metric_ref || m.ticker || ''}: ${m.value_text || m.formatted_value || m.value || ''}`
              return (
                <span
                  key={idx}
                  className="rounded border border-slate-200 bg-white px-2 py-0.5 text-xs font-mono text-zinc-700 shadow-2xs"
                >
                  {label}
                </span>
              )
            })}
          </div>
        </div>
      )}

      {/* Watch Conditions */}
      {watchConditions.length > 0 && (
        <div className="rounded-lg bg-amber-50/70 border border-amber-200/80 p-2.5 text-xs text-amber-900 space-y-1">
          <div className="font-semibold flex items-center gap-1 text-[11px]">
            <span>👀</span>
            <span>เงื่อนไขเฝ้าระวังทางสถิติ (Watch Conditions):</span>
          </div>
          <ul className="list-disc list-inside space-y-0.5 text-[11px] text-amber-950">
            {watchConditions.map((wc: any, idx: number) => (
              <li key={idx}>
                {typeof wc === 'string' ? wc : `${wc.ticker || ''} ${wc.condition || wc.future_threshold || ''}`}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Snapshot Reference */}
      {snapshotId && (
        <div className="text-[10px] text-zinc-400 font-mono pt-1 flex items-center justify-between border-t border-slate-100">
          <span>อ้างอิง Canonical Snapshot: {snapshotId}</span>
          <span>ดูกราฟ RRG ได้ที่แท็บสหรัฐอเมริกา →</span>
        </div>
      )}
    </div>
  )
}
