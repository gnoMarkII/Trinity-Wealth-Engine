/**
 * Utility for evaluating Macro AI report age and freshness.
 * Rules:
 * - Fresh: within 24 hours (< 24h) -> Green
 * - Aging: >= 24 hours up to 3 business days -> Yellow/Amber
 * - Stale: > 3 business days -> Red (advises refresh)
 */

export type ReportAgeStatus = 'fresh' | 'aging' | 'stale'

export interface ReportAgeInfo {
  status: ReportAgeStatus
  label: string
  detail: string
  badgeClass: string
  dotClass: string
  businessDays: number
  hoursAgo: number
}

/**
 * Counts full elapsed Monday-Friday business days between two dates.
 */
export function countBusinessDays(start: Date, end: Date): number {
  if (end <= start) return 0
  let count = 0
  const cur = new Date(start)
  cur.setHours(0, 0, 0, 0)
  const target = new Date(end)
  target.setHours(0, 0, 0, 0)

  while (cur < target) {
    cur.setDate(cur.getDate() + 1)
    const day = cur.getDay()
    if (day !== 0 && day !== 6) {
      count++
    }
  }
  return count
}

export function getReportAgeInfo(evaluatedAt?: string | null, now: Date = new Date()): ReportAgeInfo {
  if (!evaluatedAt) {
    return {
      status: 'stale',
      label: 'ไม่มีข้อมูลเวลา',
      detail: 'ไม่พบวันและเวลาที่ประเมิน',
      badgeClass: 'border-rose-200 bg-rose-50 text-rose-700',
      dotClass: 'bg-rose-500',
      businessDays: 999,
      hoursAgo: 999,
    }
  }

  const evalDate = new Date(evaluatedAt)
  if (isNaN(evalDate.getTime())) {
    return {
      status: 'stale',
      label: 'เวลาไม่ถูกต้อง',
      detail: 'รูปแบบเวลา evaluated_at ผิดพลาด',
      badgeClass: 'border-rose-200 bg-rose-50 text-rose-700',
      dotClass: 'bg-rose-500',
      businessDays: 999,
      hoursAgo: 999,
    }
  }

  const diffMs = Math.max(0, now.getTime() - evalDate.getTime())
  const hoursAgo = Math.floor(diffMs / (1000 * 60 * 60))
  const businessDays = countBusinessDays(evalDate, now)

  // 1. Fresh (< 24h)
  if (hoursAgo < 24) {
    const timeLabel =
      hoursAgo === 0
        ? `${Math.max(1, Math.floor(diffMs / (1000 * 60)))} นาทีที่แล้ว`
        : `${hoursAgo} ชม. ที่แล้ว`
    return {
      status: 'fresh',
      label: `สดใหม่ (${timeLabel})`,
      detail: `อัปเดตเมื่อ ${timeLabel}`,
      badgeClass: 'border-emerald-200 bg-emerald-50 text-emerald-700',
      dotClass: 'bg-emerald-500',
      businessDays,
      hoursAgo,
    }
  }

  // 2. Aging (>= 24h and <= 3 business days)
  if (businessDays <= 3) {
    const dayLabel = businessDays <= 1 ? '1 วันก่อน' : `${businessDays} วันก่อน`
    return {
      status: 'aging',
      label: `รายงานเริ่มเก่า (${dayLabel})`,
      detail: `อัปเดตแล้ว ${dayLabel}`,
      badgeClass: 'border-amber-200 bg-amber-50 text-amber-700',
      dotClass: 'bg-amber-500',
      businessDays,
      hoursAgo,
    }
  }

  // 3. Stale (> 3 business days)
  return {
    status: 'stale',
    label: `ข้อมูลค้าง (${businessDays} วันทำการ)`,
    detail: `รายงานเก่าเกิน ${businessDays} วันทำการ แนะนำควรอัปเดต`,
    badgeClass: 'border-rose-200 bg-rose-50 text-rose-700',
    dotClass: 'bg-rose-500',
    businessDays,
    hoursAgo,
  }
}
