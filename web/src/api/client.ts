import type {
  ActiveAgentStatusDTO,
  JobStatusDTO,
  KanbanCardDTO,
  MacroDashboardDTO,
  PortfolioDTO,
} from './types'

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

let unauthorizedHandler: (() => void) | null = null

export function setUnauthorizedHandler(handler: (() => void) | null): void {
  unauthorizedHandler = handler
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
  })
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = body.detail ?? detail
    } catch {
      // ignore — ไม่มี JSON body
    }
    // เจอ 401 กลางคัน (session หมดอายุ/ถูกลบ) — ไม่ใช่ตอน login เอง (401 = รหัสผ่านผิด เป็น
    // เรื่องปกติที่ฟอร์ม login จัดการเอง ไม่ใช่สัญญาณว่า session หลุด) → บังคับ logout ไปหน้า login
    if (res.status === 401 && path !== '/api/auth/login') {
      unauthorizedHandler?.()
    }
    throw new ApiError(res.status, detail)
  }
  if (res.status === 204) return undefined as T
  return (await res.json()) as T
}

export const api = {
  login: (password: string) =>
    request<{ ok: boolean }>('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify({ password }),
    }),

  logout: () => request<{ ok: boolean }>('/api/auth/logout', { method: 'POST' }),

  getPortfolio: () => request<PortfolioDTO>('/api/portfolio/latest'),

  getMacroDashboard: () => request<MacroDashboardDTO>('/api/macro/dashboard'),

  listKanbanCards: () => request<KanbanCardDTO[]>('/api/kanban/cards'),

  createKanbanCard: (title: string, flow: string = 'manager', prompt?: string, scope: string = 'both') =>
    request<{ card: KanbanCardDTO; created: boolean }>('/api/kanban/cards', {
      method: 'POST',
      body: JSON.stringify({ title, flow, prompt: prompt ?? null, scope }),
    }),

  updateKanbanCard: (cardId: string, title: string, prompt: string, flow: string, scope: string) =>
    request<KanbanCardDTO>(`/api/kanban/cards/${cardId}`, {
      method: 'PATCH',
      body: JSON.stringify({ title, prompt: prompt || null, flow, scope }),
    }),

  moveKanbanCard: (cardId: string, columnName: string, jobId?: string) =>
    request<KanbanCardDTO>('/api/kanban/move', {
      method: 'PUT',
      body: JSON.stringify({ card_id: cardId, column_name: columnName, job_id: jobId }),
    }),

  deleteKanbanCard: (cardId: string) =>
    request<{ ok: boolean }>(`/api/kanban/cards/${cardId}`, { method: 'DELETE' }),

  dispatchJob: (instruction: string, cardId?: string, flow: string = 'manager', scope: string = 'both') =>
    request<JobStatusDTO>('/api/agents/dispatch', {
      method: 'POST',
      body: JSON.stringify({ instruction, card_id: cardId, flow, scope }),
    }),

  getJobStatus: (jobId: string) => request<JobStatusDTO>(`/api/agents/jobs/${jobId}`),

  getActiveAgentStatus: () => request<ActiveAgentStatusDTO>('/api/agents/active'),

  resumeJob: (jobId: string, approvedNewsLinks: string[], approvedYoutubeLinks: string[]) =>
    request<JobStatusDTO>(`/api/agents/jobs/${jobId}/resume`, {
      method: 'POST',
      body: JSON.stringify({
        approved_news_links: approvedNewsLinks,
        approved_youtube_links: approvedYoutubeLinks,
      }),
    }),
}
