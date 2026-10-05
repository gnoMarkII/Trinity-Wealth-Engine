import { describe, it, expect } from 'vitest'
import { countBusinessDays, getReportAgeInfo } from './reportAge'

describe('reportAge utility', () => {
  it('correctly counts business days excluding weekends', () => {
    // Friday to Monday = 1 business day
    const friday = new Date('2026-10-02T15:00:00Z')
    const monday = new Date('2026-10-05T10:00:00Z')
    expect(countBusinessDays(friday, monday)).toBe(1)

    // Monday to Friday same week = 4 days
    const nextFriday = new Date('2026-10-09T10:00:00Z')
    expect(countBusinessDays(monday, nextFriday)).toBe(4)

    // Same day
    expect(countBusinessDays(monday, monday)).toBe(0)
  })

  it('evaluates reports less than 24h as fresh (green)', () => {
    const now = new Date('2026-10-03T18:00:00Z')
    const twoHoursAgo = new Date('2026-10-03T16:00:00Z').toISOString()
    const info = getReportAgeInfo(twoHoursAgo, now)
    expect(info.status).toBe('fresh')
    expect(info.badgeClass).toContain('emerald')
    expect(info.label).toContain('สดใหม่')
    expect(info.label).toContain('2 ชม. ที่แล้ว')
  })

  it('evaluates reports between 24h and 3 business days as aging (yellow)', () => {
    const now = new Date('2026-10-05T18:00:00Z') // Monday evening
    const fridayMorning = new Date('2026-10-02T08:00:00Z').toISOString() // Friday morning (1 business day passed)
    const info = getReportAgeInfo(fridayMorning, now)
    expect(info.status).toBe('aging')
    expect(info.badgeClass).toContain('amber')
    expect(info.label).toContain('เริ่มเก่า')
  })

  it('evaluates reports older than 3 business days as stale (red)', () => {
    const now = new Date('2026-10-09T18:00:00Z') // Friday evening
    const lastFriday = new Date('2026-10-02T08:00:00Z').toISOString() // 5 business days passed
    const info = getReportAgeInfo(lastFriday, now)
    expect(info.status).toBe('stale')
    expect(info.badgeClass).toContain('rose')
    expect(info.label).toContain('ข้อมูลค้าง')
  })

  it('handles null, undefined or invalid dates safely', () => {
    expect(getReportAgeInfo(null).status).toBe('stale')
    expect(getReportAgeInfo(undefined).status).toBe('stale')
    expect(getReportAgeInfo('invalid-date').status).toBe('stale')
  })
})
