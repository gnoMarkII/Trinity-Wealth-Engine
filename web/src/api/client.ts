import type {
  ActiveAgentStatusDTO,
  JobOutputsDTO,
  JobStatusDTO,
  KanbanCardDTO,
  MacroDashboardDTO,
  ActualPortfolioStateDTO,
  BucketAllocationResponseDTO,
  ActualWatchlistStateDTO,
  ActualGoalsResponseDTO,
  PerformanceSnapshotDTO,
  JournalEntryDTO,
  UpsertAllocationTargetsPayload,
  AssignBucketPayload,
  BatchAssignBucketPayload,
  BatchRemoveHoldingsPayload,
  TradePayload,
  CashFlowPayload,
  IncomePayload,
  EditHoldingPayload,
  UpsertWatchlistItemPayload,
  UpsertGoalPayload,
  AppendJournalPayload,
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

  me: () => request<{ authenticated: boolean }>('/api/auth/me'),

  getMacroDashboard: () => request<MacroDashboardDTO>('/api/macro/dashboard'),

  getMacroIndicatorSeries: (indicatorId: string, range: '1m' | '3m' | '1y') =>
    request<import('./types').MacroIndicatorSeriesDTO>(
      `/api/macro/indicators/${encodeURIComponent(indicatorId)}/series?range=${range}`
    ),

  getNewsFunnelPending: () => request<import('./types').NewsFunnelPendingItem[]>('/api/macro/news_funnel/pending'),

  getNewsFunnelFiltered: () => request<import('./types').NewsFunnelFilteredItem[]>('/api/macro/news_funnel/filtered'),

  deleteNewsFunnelPending: (eventId: string) =>
    request<{ ok: boolean; remaining_count: number }>(`/api/macro/news_funnel/pending/${encodeURIComponent(eventId)}`, {
      method: 'DELETE',
    }),

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

  getJobOutputs: (jobId: string) => request<JobOutputsDTO>(`/api/agents/jobs/${jobId}/outputs`),

  getActiveAgentStatus: () => request<ActiveAgentStatusDTO>('/api/agents/active'),

  resumeJob: (
    jobId: string,
    approvedNewsLinks: string[] = [],
    approvedYoutubeLinks: string[] = [],
    approvedEventIds?: string[],
    approvedPitchIds?: string[]
  ) =>
    request<JobStatusDTO>(`/api/agents/jobs/${jobId}/resume`, {
      method: 'POST',
      body: JSON.stringify({
        approved_news_links: approvedNewsLinks,
        approved_youtube_links: approvedYoutubeLinks,
        approved_event_ids: approvedEventIds,
        approved_pitch_ids: approvedPitchIds,
      }),
    }),

  // ---------------------------------------------------------
  // Actual Portfolio Hub Endpoints (Phase 1)
  // ---------------------------------------------------------
  getActualPortfolioState: (refreshPrices: boolean = false, fetchFundamentals: boolean = false) =>
    request<ActualPortfolioStateDTO>(
      `/api/portfolio/actual/state?refresh_prices=${refreshPrices}&fetch_fundamentals=${fetchFundamentals}`
    ),

  getActualBucketAllocations: () =>
    request<BucketAllocationResponseDTO>('/api/portfolio/actual/allocations'),

  getActualWatchlist: () => request<ActualWatchlistStateDTO>('/api/portfolio/actual/watchlist'),

  getActualGoals: () => request<ActualGoalsResponseDTO>('/api/portfolio/actual/goals'),

  getActualPerformance: (days?: number) => {
    const params = days !== undefined ? `?days=${days}` : ''
    return request<PerformanceSnapshotDTO[]>(`/api/portfolio/actual/performance${params}`)
  },

  triggerPerformanceSnapshot: (refreshPrices: boolean = false) =>
    request<PerformanceSnapshotDTO[]>(`/api/portfolio/actual/performance/snapshot?refresh_prices=${refreshPrices}`, {
      method: 'POST',
    }),

  getActualJournal: (days: number = 365, keyword?: string, limit: number = 100) => {
    const params = new URLSearchParams({ days: days.toString(), limit: limit.toString() })
    if (keyword) params.append('keyword', keyword)
    return request<JournalEntryDTO[]>(`/api/portfolio/actual/journal?${params.toString()}`)
  },

  // ---------------------------------------------------------
  // Actual Portfolio Hub Mutation Endpoints (Phase 2.1 & 2.2)
  // ---------------------------------------------------------
  upsertAllocationTargets: (payload: UpsertAllocationTargetsPayload) =>
    request<ActualPortfolioStateDTO>('/api/portfolio/actual/allocations/targets', {
      method: 'PUT',
      body: JSON.stringify(payload),
    }),

  assignHoldingBucket: (symbol: string, payload: AssignBucketPayload) =>
    request<ActualPortfolioStateDTO>(`/api/portfolio/actual/holdings/${encodeURIComponent(symbol)}/bucket`, {
      method: 'PUT',
      body: JSON.stringify(payload),
    }),

  batchAssignHoldingBuckets: (payload: BatchAssignBucketPayload) =>
    request<ActualPortfolioStateDTO>('/api/portfolio/actual/holdings/batch-bucket', {
      method: 'PUT',
      body: JSON.stringify(payload),
    }),

  batchRemoveHoldings: (payload: BatchRemoveHoldingsPayload) =>
    request<ActualPortfolioStateDTO>('/api/portfolio/actual/holdings/batch-delete', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  resetPortfolioCleanSlate: () =>
    request<ActualPortfolioStateDTO>('/api/portfolio/actual/reset', {
      method: 'POST',
    }),

  executeTrade: (payload: TradePayload) =>
    request<ActualPortfolioStateDTO>('/api/portfolio/actual/trade', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  manageCashFlow: (payload: CashFlowPayload) =>
    request<ActualPortfolioStateDTO>('/api/portfolio/actual/cashflow', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  recordIncome: (payload: IncomePayload) =>
    request<ActualPortfolioStateDTO>('/api/portfolio/actual/income', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  editHolding: (symbol: string, payload: EditHoldingPayload) =>
    request<ActualPortfolioStateDTO>(`/api/portfolio/actual/holdings/${encodeURIComponent(symbol)}/edit`, {
      method: 'PUT',
      body: JSON.stringify(payload),
    }),

  removeHolding: (symbol: string) =>
    request<ActualPortfolioStateDTO>(`/api/portfolio/actual/holdings/${encodeURIComponent(symbol)}`, {
      method: 'DELETE',
    }),

  upsertWatchlistItem: (symbol: string, payload: UpsertWatchlistItemPayload) =>
    request<ActualWatchlistStateDTO>(`/api/portfolio/actual/watchlist/${encodeURIComponent(symbol)}`, {
      method: 'PUT',
      body: JSON.stringify(payload),
    }),

  removeWatchlistItem: (symbol: string) =>
    request<ActualWatchlistStateDTO>(`/api/portfolio/actual/watchlist/${encodeURIComponent(symbol)}`, {
      method: 'DELETE',
    }),

  upsertGoal: (name: string, payload: UpsertGoalPayload) =>
    request<ActualGoalsResponseDTO>(`/api/portfolio/actual/goals/${encodeURIComponent(name)}`, {
      method: 'PUT',
      body: JSON.stringify(payload),
    }),

  removeGoal: (name: string) =>
    request<ActualGoalsResponseDTO>(`/api/portfolio/actual/goals/${encodeURIComponent(name)}`, {
      method: 'DELETE',
    }),

  appendJournal: (payload: AppendJournalPayload) =>
    request<JournalEntryDTO[]>('/api/portfolio/actual/journal', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
}

