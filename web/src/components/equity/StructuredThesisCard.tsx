import React, { useEffect, useState } from 'react'
import { api } from '../../api/client'
import type { ThesisClaimDTO } from '../../api/types'

export interface StructuredThesisCardProps {
  ticker: string
  initialClaim?: ThesisClaimDTO | null
  className?: string
}

const getEvidenceTierConfig = (tier: string) => {
  switch (tier) {
    case 'DOCUMENTED':
      return {
        label: 'DOCUMENTED',
        thLabel: 'มีเอกสารยืนยัน',
        icon: '📑',
        style: 'border-emerald-200 bg-emerald-50 text-emerald-800',
        title: 'ตรวจสอบจากเอกสารจริงของทางการ (SEC 10-K/8-K Disclosure)',
      }
    case 'DIRECTLY_TESTED':
      return {
        label: 'TESTED',
        thLabel: 'ทดสอบด้วยระบบแล้ว',
        icon: '🧪',
        style: 'border-sky-200 bg-sky-50 text-sky-800',
        title: 'คำนวณและทดสอบทางสถิติด้วยระบบอัตโนมัติแล้ว',
      }
    case 'INFERRED':
      return {
        label: 'INFERRED',
        thLabel: 'จากการอนุมาน',
        icon: '💡',
        style: 'border-purple-200 bg-purple-50 text-purple-800',
        title: 'สมมติฐานการวิเคราะห์หรือการคาดการณ์ไปข้างหน้า',
      }
    case 'UNVERIFIED':
    default:
      return {
        label: 'UNVERIFIED',
        thLabel: 'ยังไม่ยืนยัน',
        icon: '⚠️',
        style: 'border-amber-200 bg-amber-50 text-amber-800',
        title: 'ยังไม่พบเอกสารยืนยันหลักฐาน',
      }
  }
}

const getMetricFriendlyName = (metricName: string) => {
  const normalized = metricName.toLowerCase()
  if (normalized.includes('gross_margin')) return 'อัตรากำไรขั้นต้น'
  if (normalized.includes('operating_margin') || normalized.includes('ebit_margin')) return 'อัตรากำไรจากการดำเนินงาน'
  if (normalized.includes('net_margin')) return 'อัตรากำไรสุทธิ'
  if (normalized.includes('revenue')) return 'ยอดขาย/รายได้'
  if (normalized.includes('price')) return 'ราคาหุ้น / จุดตัดขาดทุน'
  if (normalized.includes('free_cash_flow') || normalized.includes('fcf')) return 'กระแสเงินสดอิสระ'
  if (normalized.includes('eps')) return 'กำไรต่อหุ้น'
  return metricName.replace(/_/g, ' ')
}

const getInvalidationFriendlyNote = (impact: string) => {
  const norm = impact.toUpperCase()
  if (norm.includes('BREAKS_BUY_CASE') || norm.includes('BREAKS BUY CASE')) return 'จุดยอมแพ้ / ผิดแผนการลงทุน'
  if (norm.includes('FORCES_PORTFOLIO_TRIM') || norm.includes('FORCES PORTFOLIO TRIM')) return 'ต้องลดสัดส่วนถือครองลง'
  if (norm.includes('EXIT')) return 'ขายออกทั้งหมด'
  return 'กระทบต่อสมมติฐานการลงทุน'
}

export const StructuredThesisCard: React.FC<StructuredThesisCardProps> = ({
  ticker,
  initialClaim = null,
  className = '',
}) => {
  const [claim, setClaim] = useState<ThesisClaimDTO | null>(initialClaim)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    if (initialClaim) {
      setClaim(initialClaim)
      return
    }
    if (!ticker) {
      setClaim(null)
      return
    }
    const loadThesis = async () => {
      setLoading(true)
      setError(null)
      try {
        const data = await api.getActiveThesis(ticker)
        if (!cancelled) {
          setClaim(data)
        }
      } catch (err: any) {
        if (!cancelled) {
          setError(err?.message || 'Failed to load thesis')
        }
      } finally {
        if (!cancelled) {
          setLoading(false)
        }
      }
    }
    loadThesis()
    return () => {
      cancelled = true
    }
  }, [ticker, initialClaim])

  if (loading && !claim) {
    return (
      <div
        data-testid="structured-thesis-card-loading"
        className={`rounded-2xl border border-edge/80 bg-surface p-6 shadow-sm animate-pulse space-y-4 ${className}`}
      >
        <div className="h-6 w-48 bg-zinc-200 rounded" />
        <div className="h-16 bg-zinc-100 rounded-xl" />
        <div className="grid grid-cols-2 gap-3">
          <div className="h-24 bg-zinc-100 rounded-xl" />
          <div className="h-24 bg-zinc-100 rounded-xl" />
        </div>
      </div>
    )
  }

  if (error || !claim) {
    return null
  }

  return (
    <section
      data-testid="structured-thesis-card"
      className={`rounded-2xl border border-sky-200/90 bg-white/90 p-6 shadow-sm backdrop-blur-md space-y-5 ${className}`}
    >
      {/* 1. Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-sky-100/80 pb-4">
        <div className="flex items-start sm:items-center gap-3">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-sky-500 text-white shadow-2xs font-bold text-base">
            🎯
          </div>
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <h3 className="text-base font-bold text-sky-950">
                Institutional Investment Thesis (v{claim.version})
              </h3>
              {(() => {
                const badge = getEvidenceTierConfig(claim.evidence_tier)
                return (
                  <span
                    data-testid="evidence-tier-badge"
                    title={badge.title}
                    className={`inline-flex items-center gap-1 rounded-md border px-2 py-0.5 font-mono text-[11px] font-bold ${badge.style}`}
                  >
                    <span>{badge.icon}</span>
                    <span>{claim.evidence_tier}</span>
                    <span className="font-sans font-normal text-[10px] opacity-80 hidden sm:inline">({badge.thLabel})</span>
                  </span>
                )
              })()}
            </div>
            <p className="text-xs text-sky-800/80 mt-0.5">
              กรอบสมมติฐานการลงทุนและเกณฑ์เตือนภัย (เหตุผลการเข้าซื้อ &amp; จุดตัดขาดทุนเมื่อผิดแผน)
            </p>
            <div className="flex items-center gap-2 text-[11px] text-zinc-500 mt-1">
              <span>ข้อมูล ณ วันที่: <strong className="font-mono text-zinc-700">{claim.as_of_utc.split('T')[0]}</strong></span>
              <span>•</span>
              <span className="font-mono text-[10px] text-zinc-400">
                SHA: {claim.checksum_sha256.substring(0, 10)}...
              </span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2 self-start sm:self-auto">
          <span
            className="rounded-full bg-emerald-50 border border-emerald-200 px-3 py-1 font-mono text-xs font-bold text-emerald-800"
            title="สมมติฐานนี้กำลังมีผลบังคับใช้กับพอร์ตการลงทุน"
          >
            {claim.status}
          </span>
        </div>
      </div>

      {/* 2. Core Statement */}
      <div className="rounded-xl border border-sky-100 bg-sky-50/40 p-4 space-y-1">
        <div className="flex items-center justify-between">
          <span className="text-[10px] font-bold uppercase tracking-wider text-sky-900 block">
            Core Thesis Claim
          </span>
          <span className="text-[10px] text-sky-700 font-medium">🎯 เหตุผลหลักในการลงทุน</span>
        </div>
        <p className="text-sm font-medium leading-relaxed text-sky-950">
          {claim.claim_statement}
        </p>
        <p className="text-[11px] text-zinc-500 italic pt-1">
          💡 สรุปง่ายๆ: สิ่งที่ผู้จัดการกองทุนเชื่อมั่นว่าบริษัทจะทำได้สำเร็จ หากเป็นจริงราคาหุ้นจะเติบโตได้ต่อเนื่อง
        </p>
      </div>

      {/* 3. Falsifier Floor Rules */}
      <div className="space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1">
          <div>
            <h4 className="text-xs font-bold uppercase tracking-wider text-zinc-700 flex items-center gap-1.5">
              <span>🛑 Falsifier Floor Bounds ({claim.falsifiers.length} rules)</span>
            </h4>
            <p className="text-[11px] text-zinc-500">
              เกณฑ์เฝ้าระวังความเสี่ยง (จุดยอมแพ้): หากตัวเลขจริงหลุดต่ำกว่าเกณฑ์นี้ แสดงว่าสมมติฐานผิด ต้องหยุดซื้อหรือตัดขาดทุนทันที
            </p>
          </div>
          <span className="text-[10px] font-medium text-rose-700 bg-rose-50 border border-rose-200 px-2 py-0.5 rounded self-start sm:self-auto">
            Breach triggers immediate review
          </span>
        </div>

        {claim.falsifiers.length === 0 ? (
          <p className="text-xs text-zinc-400 italic">No quantitative falsifiers defined.</p>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {claim.falsifiers.map((f) => (
              <div
                key={f.falsifier_id}
                className="rounded-xl border border-edge/80 bg-surface/80 p-3.5 space-y-2"
              >
                <div className="flex items-center justify-between text-xs">
                  <div className="flex items-center gap-1.5">
                    <span className="font-bold text-zinc-900">{f.metric_name}</span>
                    <span className="text-[10px] text-zinc-500 font-normal">({getMetricFriendlyName(f.metric_name)})</span>
                  </div>
                  <span
                    className="rounded bg-rose-50 border border-rose-200 px-1.5 py-0.5 font-mono text-[10px] font-bold text-rose-700"
                    title={getInvalidationFriendlyNote(f.invalidation_impact)}
                  >
                    {f.invalidation_impact.replace(/_/g, ' ')}
                  </span>
                </div>
                <div className="font-mono text-xs font-semibold text-rose-600">
                  Floor: {f.operator} {f.threshold_value} {f.unit} ({f.period_basis})
                </div>
                <div className="text-[10px] text-rose-700 font-medium bg-rose-50/50 rounded px-2 py-0.5 border border-rose-100">
                  ⚠️ จุดผิดแผน: {getInvalidationFriendlyNote(f.invalidation_impact)}
                </div>
                {f.narrative_explanation && (
                  <p className="text-[11px] text-zinc-600 leading-normal pt-1 border-t border-edge/40">
                    {f.narrative_explanation}
                  </p>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* 4. Catalyst Timeline */}
      {claim.linked_catalysts.length > 0 && (
        <div className="space-y-3 pt-2 border-t border-sky-100/70">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1">
            <div>
              <h4 className="text-xs font-bold uppercase tracking-wider text-zinc-700 flex items-center gap-1.5">
                <span>🗓️ Catalyst Milestones</span>
              </h4>
              <p className="text-[11px] text-zinc-500">
                เหตุการณ์สำคัญที่รอการพิสูจน์ (เช่น วันประกาศงบการเงิน หรือเปิดตัวผลิตภัณฑ์ใหม่)
              </p>
            </div>
            <span className="text-[10px] text-zinc-400">Verifiable forward events</span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {claim.linked_catalysts.map((c) => (
              <div
                key={c.catalyst_id}
                className="rounded-xl border border-sky-100 bg-white/70 p-3 space-y-1 text-xs"
              >
                <div className="flex items-center justify-between">
                  <span className="font-bold text-sky-950">{c.event_name}</span>
                  <span className="font-mono text-[10px] font-bold rounded bg-zinc-100 px-1.5 py-0.5 text-zinc-600">
                    {c.status}
                  </span>
                </div>
                <div className="flex items-center gap-2 text-[11px] text-zinc-500">
                  <span>Target: <strong className="font-mono text-zinc-800">{c.expected_date_str}</strong></span>
                  <span>(±{c.confidence_window_days}d)</span>
                </div>
                {c.source_locator && (
                  <div className="text-[10px] text-sky-700 truncate font-mono">
                    Source: {c.source_locator}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  )
}

export default StructuredThesisCard

