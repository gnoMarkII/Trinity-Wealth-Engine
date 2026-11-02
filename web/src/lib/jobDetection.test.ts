import { describe, expect, it } from 'vitest'
import { extractAllJobIds, extractPrimaryJobId } from './jobDetection'

describe('jobDetection utility', () => {
  it('extracts Job ID from markdown formatted bold with backticks', () => {
    const text = '- **Job ID:** `a382a37f-10e8-4162-8014-74f05d5f7f4c` (อ้างอิงตามคำสั่งล่าสุด)'
    expect(extractPrimaryJobId(text)).toBe('a382a37f-10e8-4162-8014-74f05d5f7f4c')
  })

  it('extracts Job ID from standard plain text', () => {
    const text = 'ส่งคำสั่งเรียบร้อยแล้ว Job ID: a382a37f-10e8-4162-8014-74f05d5f7f4c'
    expect(extractPrimaryJobId(text)).toBe('a382a37f-10e8-4162-8014-74f05d5f7f4c')
  })

  it('extracts Job ID with job_mgr or sim prefix', () => {
    expect(extractPrimaryJobId('(Job ID: sim_macro_intel)')).toBe('sim_macro_intel')
    expect(extractPrimaryJobId('Job ID: job_mgr_88721_alpha')).toBe('job_mgr_88721_alpha')
  })

  it('extracts bare UUID without explicit label', () => {
    const text = 'ระบบกำลังประมวลผลงาน f2069643-c28a-4d88-b589-e583ae76028c สำหรับ Macro'
    expect(extractPrimaryJobId(text)).toBe('f2069643-c28a-4d88-b589-e583ae76028c')
  })

  it('returns null when no job ID is present', () => {
    expect(extractPrimaryJobId('สวัสดีครับ มีอะไรให้ช่วยไหม')).toBeNull()
    expect(extractPrimaryJobId('')).toBeNull()
  })

  it('extracts all unique Job IDs across multi-job message', () => {
    const text = `
    Job 1: a382a37f-10e8-4162-8014-74f05d5f7f4c
    Job 2: sim_macro_intel
    Job 3: 7792e177-f895-4d4b-bcbd-3c1c18ef89ea
    Duplicate: a382a37f-10e8-4162-8014-74f05d5f7f4c
    `
    const all = extractAllJobIds(text)
    expect(all).toContain('a382a37f-10e8-4162-8014-74f05d5f7f4c')
    expect(all).toContain('sim_macro_intel')
    expect(all).toContain('7792e177-f895-4d4b-bcbd-3c1c18ef89ea')
    expect(all.length).toBe(3)
  })
})

