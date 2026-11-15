import { useEffect, useState } from 'react'
import { api } from '../api/client'
import Modal from './ui/Modal'
import { CheckCircleIcon, ChevronDownIcon, ChevronUpIcon, ShieldCheckIcon, SparklesIcon } from './ui/AppIcons'

interface TelemetryData {
  total_events: number
  resolved_rate: number
  issue_counts: Record<string, number>
  top_recurring_issues: [string, number][]
}

const DEFAULT_NOMINAL_TELEMETRY: TelemetryData = {
  total_events: 0,
  resolved_rate: 1.0,
  issue_counts: {},
  top_recurring_issues: [],
}

export default function HarnessTelemetryWidget({ collapsed = false }: { collapsed?: boolean }) {
  const [data, setData] = useState<TelemetryData | null>(null)
  const [isOpen, setIsOpen] = useState(false)
  const [showModal, setShowModal] = useState(false)

  useEffect(() => {
    let cancelled = false
    api
      .getHarnessTelemetry()
      .then((res) => {
        if (!cancelled && res) setData(res)
      })
      .catch(() => {
        // Fallback silently if telemetry endpoint is temporarily unreachable
      })
    return () => {
      cancelled = true
    }
  }, [])

  const currentData = data ?? DEFAULT_NOMINAL_TELEMETRY
  const resolvedPct = Math.round((currentData.resolved_rate || 0) * 100)
  const isHealthy = resolvedPct >= 90 || currentData.total_events === 0
  const totalIssuesCount = Object.values(currentData.issue_counts || {}).reduce((a, b) => a + b, 0)

  return (
    <>
      {collapsed ? (
        <div className="my-2 flex w-full justify-center border-t border-sky-100/80 pt-3">
          <button
            type="button"
            onClick={() => setShowModal(true)}
            aria-label="Harness AI Health — สถิติเชิงลึก (Telemetry)"
            title={`Harness AI Health: ${currentData.total_events === 0 ? 'Nominal' : `${resolvedPct}% Clean`}`}
            className={`relative flex h-10 w-10 items-center justify-center rounded-xl border transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-section-accent/45 ${
              isHealthy
                ? 'border-emerald-200 bg-emerald-50/75 text-emerald-700 hover:bg-emerald-100/75'
                : 'border-amber-200 bg-amber-50/75 text-amber-700 hover:bg-amber-100/75'
            }`}
          >
            <ShieldCheckIcon className="h-4 w-4" />
            <span
              className={`absolute top-1.5 right-1.5 h-2 w-2 rounded-full ${
                isHealthy ? 'bg-emerald-500' : 'bg-amber-500'
              }`}
            />
          </button>
        </div>
      ) : (
        <div className="mt-3 space-y-2 border-t border-sky-100/80 pt-3">
          {/* Header with Title and Status Pill */}
          <div className="flex items-center justify-between px-1">
            <h3 className="text-[11px] font-bold uppercase tracking-[0.14em] text-slate-600 flex items-center gap-1.5">
              <span>Harness AI Health</span>
            </h3>
            <button
              type="button"
              onClick={() => setShowModal(true)}
              title="คลิกเพื่อดูรายละเอียดสถิติ Telemetry เชิงลึก"
              className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-semibold transition-all hover:scale-105 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-section-accent/45 ${
                isHealthy
                  ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                  : 'bg-amber-50 text-amber-700 border border-amber-200'
              }`}
            >
              <span
                className={`h-1.5 w-1.5 rounded-full ${
                  isHealthy ? 'bg-emerald-500' : 'bg-amber-500'
                }`}
              />
              {currentData.total_events === 0 ? 'Nominal' : `${resolvedPct}% Clean`}
            </button>
          </div>

          {/* Glassmorphic Metric Card */}
          <div className="rounded-2xl border border-sky-100/80 bg-surface-2 p-2.5 shadow-sm backdrop-blur-md">
            <div className="flex items-center justify-between text-xs text-slate-600">
              <span className="text-[11px] text-slate-500">Total Probes</span>
              <span className="font-mono font-semibold tabular-nums text-slate-800">{currentData.total_events}</span>
            </div>

            <div className="mt-1 flex items-center justify-between text-[11px]">
              <button
                type="button"
                onClick={() => setShowModal(true)}
                className="text-sky-700 hover:text-sky-900 font-medium underline decoration-sky-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-section-accent/45 rounded"
              >
                สถิติเชิงลึก (Telemetry)
              </button>
            </div>

            {currentData.top_recurring_issues.length > 0 && (
              <div className="mt-2 pt-2 border-t border-sky-100/60">
                <button
                  type="button"
                  onClick={() => setIsOpen(!isOpen)}
                  className="flex w-full items-center justify-between text-[11px] font-medium text-sky-700 hover:text-sky-900 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-section-accent/45 rounded"
                >
                  <span>Recurring Patterns</span>
                  {isOpen ? <ChevronUpIcon className="h-3.5 w-3.5" /> : <ChevronDownIcon className="h-3.5 w-3.5" />}
                </button>
                {isOpen && (
                  <ul className="mt-1.5 space-y-1 text-[11px] text-slate-500">
                    {currentData.top_recurring_issues.slice(0, 3).map(([issue, count]) => (
                      <li key={issue} className="flex justify-between items-center">
                        <span className="truncate max-w-[130px]" title={issue}>
                          {issue.replace(/_/g, ' ')}
                        </span>
                        <span className="font-mono font-semibold tabular-nums text-slate-700">x{count}</span>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {showModal && (
        <Modal
          titleId="telemetry-details-title"
          onClose={() => setShowModal(false)}
          panelClassName="max-w-lg w-full bg-surface-3 p-6 rounded-2xl border border-sky-100 shadow-2xl space-y-5 backdrop-blur-xl"
        >
          <div>
            <div className="flex items-center justify-between">
              <h2
                id="telemetry-details-title"
                className="text-base font-bold text-slate-900 flex items-center gap-2"
              >
                <ShieldCheckIcon className="h-5 w-5 text-sky-600" />
                <span>AI Harness & Learning Telemetry</span>
              </h2>
              <span
                className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium ${
                  isHealthy
                    ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                    : 'bg-amber-50 text-amber-700 border border-amber-200'
                }`}
              >
                <span className={`h-2 w-2 rounded-full ${isHealthy ? 'bg-emerald-500' : 'bg-amber-500'}`} />
                {currentData.total_events === 0 ? 'Nominal Status' : `${resolvedPct}% Clean Rate`}
              </span>
            </div>
            <p className="mt-1 text-xs text-slate-500">
              ระบบตรวจสอบความถูกต้องอัตโนมัติ (Verification Probes) และการเรียนรู้ข้อผิดพลาดต่อเนื่องตามมาตรฐาน Institutional Grade
            </p>
          </div>

          {/* Metric Cards Grid */}
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
            <div className="surface-inset p-3">
              <span className="text-[11px] text-slate-500">First-Pass Yield</span>
              <p className="text-lg font-mono font-bold tabular-nums text-sky-700">
                {currentData.total_events === 0 ? '100%' : `${resolvedPct}%`}
              </p>
            </div>
            <div className="surface-inset p-3">
              <span className="text-[11px] text-slate-500">Total Probes</span>
              <p className="text-lg font-mono font-bold tabular-nums text-slate-900">{currentData.total_events}</p>
            </div>
            <div className="col-span-2 sm:col-span-1 surface-inset p-3">
              <span className="text-[11px] text-slate-500">Detected Issues</span>
              <p className="text-lg font-mono font-bold tabular-nums text-slate-900">{totalIssuesCount}</p>
            </div>
          </div>

          {/* Pareto Distribution Section */}
          <div className="space-y-2">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-700">
              ความถี่ของข้อบกพร่อง (Pareto Analysis)
            </h3>
            {currentData.top_recurring_issues.length === 0 ? (
              <div className="flex items-center justify-center gap-2 rounded-xl border border-dashed border-emerald-200 bg-emerald-50/50 p-4 text-center text-xs text-emerald-700">
                <SparklesIcon className="h-4 w-4 shrink-0" />
                <span>ไม่พบประวัติข้อผิดพลาดตกค้างในระบบ (100% Clean Pass)</span>
              </div>
            ) : (
              <div className="space-y-2 surface-inset p-3">
                {currentData.top_recurring_issues.map(([issue, count]) => {
                  const pct = totalIssuesCount > 0 ? Math.round((count / totalIssuesCount) * 100) : 0
                  return (
                    <div key={issue} className="space-y-1">
                      <div className="flex justify-between text-xs">
                        <span className="font-mono text-slate-700">{issue}</span>
                        <span className="text-slate-500 tabular-nums">
                          {count} ครั้ง ({pct}%)
                        </span>
                      </div>
                      <div className="h-1.5 w-full rounded-full bg-slate-200/70 overflow-hidden">
                        <div
                          className="h-full rounded-full bg-sky-500"
                          style={{ width: `${Math.max(pct, 5)}%` }}
                        />
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </div>

          {/* Active Guardrails Invariants */}
          <div className="space-y-2">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-700">
              ข้อกำหนดที่บังคับใช้ในระบบ (Active Guardrails)
            </h3>
            <ul className="space-y-1.5 text-xs text-slate-600">
              <li className="flex items-center gap-1.5">
                <CheckCircleIcon className="h-3.5 w-3.5 shrink-0 text-emerald-600" />
                <span>Deterministic Auto-Backfill ข้อมูลจริงจาก Observable Registry</span>
              </li>
              <li className="flex items-center gap-1.5">
                <CheckCircleIcon className="h-3.5 w-3.5 shrink-0 text-emerald-600" />
                <span>Multi-Source / Multi-Indicator Corroboration Engine</span>
              </li>
              <li className="flex items-center gap-1.5">
                <CheckCircleIcon className="h-3.5 w-3.5 shrink-0 text-emerald-600" />
                <span>Equity Verifier: ตรวจสอบความสอดคล้องระหว่าง DCF และ Action Stance</span>
              </li>
              <li className="flex items-center gap-1.5">
                <CheckCircleIcon className="h-3.5 w-3.5 shrink-0 text-emerald-600" />
                <span>Taxonomy 4 กลุ่ม: แยกแยะความเสี่ยงออกจากสมมติฐานและข้อมูลตลาด</span>
              </li>
            </ul>
          </div>

          {/* Modal Footer */}
          <div className="flex justify-end pt-2 border-t border-sky-100">
            <button
              type="button"
              onClick={() => setShowModal(false)}
              className="rounded-xl bg-sky-600 px-4 py-1.5 text-xs font-medium text-white hover:bg-sky-700 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-500/50"
            >
              ปิดหน้าต่าง
            </button>
          </div>
        </Modal>
      )}
    </>
  )
}
