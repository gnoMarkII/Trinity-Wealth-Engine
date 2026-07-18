// @vitest-environment node — test ล้วนๆ ไม่แตะ DOM ข้าม jsdom ให้รันเร็วขึ้น
import { afterEach, describe, expect, it, vi } from 'vitest'
import { api, ApiError, setUnauthorizedHandler } from './client'

function mockFetchOnce(response: Partial<Response> & { jsonBody?: unknown }) {
  const { jsonBody, ...rest } = response
  const fetchMock = vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    statusText: 'OK',
    json: jsonBody === undefined ? vi.fn().mockRejectedValue(new Error('no body')) : vi.fn().mockResolvedValue(jsonBody),
    ...rest,
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

afterEach(() => {
  vi.unstubAllGlobals()
  setUnauthorizedHandler(null)
})

describe('api client', () => {
  it('response สำเร็จคืน JSON ที่ parse แล้ว', async () => {
    mockFetchOnce({ jsonBody: { overall_regime: 'Reflation' } })
    const data = await api.getMacroDashboard()
    expect(data.overall_regime).toBe('Reflation')
  })

  it('ส่ง credentials: include และ Content-Type ทุก request (cookie auth)', async () => {
    const fetchMock = mockFetchOnce({ jsonBody: [] })
    await api.listKanbanCards()
    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit]
    expect(init.credentials).toBe('include')
    expect((init.headers as Record<string, string>)['Content-Type']).toBe('application/json')
  })

  it('error ที่มี detail ใน body → ApiError พร้อมข้อความจาก backend', async () => {
    mockFetchOnce({ ok: false, status: 422, statusText: 'Unprocessable', jsonBody: { detail: 'ชื่อการ์ดว่างไม่ได้' } })
    await expect(api.listKanbanCards()).rejects.toMatchObject({ status: 422, message: 'ชื่อการ์ดว่างไม่ได้' })
  })

  it('error ที่ไม่มี JSON body → fallback ไป statusText', async () => {
    mockFetchOnce({ ok: false, status: 502, statusText: 'Bad Gateway' })
    await expect(api.listKanbanCards()).rejects.toMatchObject({ status: 502, message: 'Bad Gateway' })
  })

  it('เจอ 401 กลาง session → เรียก unauthorizedHandler (บังคับกลับหน้า login)', async () => {
    const handler = vi.fn()
    setUnauthorizedHandler(handler)
    mockFetchOnce({ ok: false, status: 401, statusText: 'Unauthorized', jsonBody: { detail: 'session expired' } })
    await expect(api.listKanbanCards()).rejects.toBeInstanceOf(ApiError)
    expect(handler).toHaveBeenCalledOnce()
  })

  it('401 จากการ login (รหัสผ่านผิด) ไม่นับเป็น session หลุด — ไม่เรียก handler', async () => {
    const handler = vi.fn()
    setUnauthorizedHandler(handler)
    mockFetchOnce({ ok: false, status: 401, statusText: 'Unauthorized', jsonBody: { detail: 'รหัสผ่านไม่ถูกต้อง' } })
    await expect(api.login('wrong-password')).rejects.toBeInstanceOf(ApiError)
    expect(handler).not.toHaveBeenCalled()
  })

  it('dispatchJob ส่ง payload ตาม schema ของ backend', async () => {
    const fetchMock = mockFetchOnce({ jsonBody: { job_id: 'j1' } })
    await api.dispatchJob('วิเคราะห์พอร์ต', 'card-1', 'manager', 'both')
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit]
    expect(url).toBe('/api/agents/dispatch')
    expect(JSON.parse(init.body as string)).toEqual({
      instruction: 'วิเคราะห์พอร์ต',
      card_id: 'card-1',
      flow: 'manager',
      scope: 'both',
    })
  })

  it('indicator id ถูก encode ใน URL (กัน id ที่มีอักขระพิเศษพัง path)', async () => {
    const fetchMock = mockFetchOnce({ jsonBody: { points: [] } })
    await api.getMacroIndicatorSeries('obs/weird id', '3m')
    const [url] = fetchMock.mock.calls[0] as [string]
    expect(url).toBe('/api/macro/indicators/obs%2Fweird%20id/series?range=3m')
  })

  it('resumeJob ส่ง approved_pitch_ids ได้ถูกต้อง', async () => {
    const fetchMock = mockFetchOnce({ jsonBody: { job_id: 'j1', status: 'running' } })
    await api.resumeJob('j1', [], [], [], ['pitch-abc'])
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit]
    expect(url).toBe('/api/agents/jobs/j1/resume')
    expect(JSON.parse(init.body as string)).toEqual({
      approved_news_links: [],
      approved_youtube_links: [],
      approved_event_ids: [],
      approved_pitch_ids: ['pitch-abc'],
    })
  })
})
