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
  NotebookLMAvailableSourceDTO,
  NotebookLMGenerateResponse,
  NotebookLMStatusDTO,
  FXRateResponseDTO,
  SyncDividendsResponseDTO,
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

  getEquityLatest: () => request<import('./types').EquitySummaryDTO[]>('/api/equity/latest'),

  getEquityDetail: (ticker: string) => request<import('./types').EquityDetailDTO>(`/api/equity/${encodeURIComponent(ticker)}`),

  getEquityNews: (ticker: string) => request<import('./types').EquityNewsDTO>(`/api/equity/${encodeURIComponent(ticker)}/news`),

  getEquityNotes: (ticker: string) => request<import('./types').EquityNotesDTO>(`/api/equity/${encodeURIComponent(ticker)}/notes`),

  getEquityNoteContent: (relPath: string) => request<import('./types').EquityNoteContentDTO>(`/api/equity/notes/content?rel_path=${encodeURIComponent(relPath)}`),

  getEquityOHLCV: (ticker: string, range: string = '6mo', interval: string = '1d', signal?: AbortSignal) =>
    request<import('./types').OHLCVResponseDTO>(
      `/api/equity/${encodeURIComponent(ticker)}/ohlcv?range=${encodeURIComponent(range)}&interval=${encodeURIComponent(interval)}`,
      { signal }
    ),

  getValuationTargets: (ticker: string, signal?: AbortSignal) =>
    request<import('./types').ValuationTargetsDTO>(
      `/api/equity/${encodeURIComponent(ticker)}/valuation-targets`,
      { signal }
    ),

  getInsiderFilings: (ticker: string, range: string = '1y', interval: string = '1d', signal?: AbortSignal) =>
    request<import('./types').InsiderFilingsResponseDTO>(
      `/api/equity/${encodeURIComponent(ticker)}/insider-filings?range=${encodeURIComponent(range)}&interval=${encodeURIComponent(interval)}`,
      { signal }
    ),

  getAnalystContext: (ticker: string, signal?: AbortSignal) =>
    request<import('./types').AnalystContextDTO>(
      `/api/equity/${encodeURIComponent(ticker)}/analyst-context`,
      { signal }
    ),




  getPortfolioCalendar: (portfolioId: string = 'default') =>
    request<import('./types').PortfolioCalendarDTO>(`/api/portfolio/calendar?portfolio_id=${encodeURIComponent(portfolioId)}`),

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

  toggleCardDiscord: (cardId: string, enabled: boolean) =>
    request<KanbanCardDTO>(`/api/kanban/cards/${cardId}/discord`, {
      method: 'PATCH',
      body: JSON.stringify({ enabled }),
    }),

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
    approvedPitchIds?: string[],
    action: 'approve' | 'refresh_sources' = 'approve',
    unverifiedDraftSelections?: import('./types').UnverifiedDraftSelection[],
    pitchPresentationStyles?: Record<string, string>
  ) =>
    request<JobStatusDTO>(`/api/agents/jobs/${jobId}/resume`, {
      method: 'POST',
      body: JSON.stringify({
        approved_news_links: approvedNewsLinks,
        approved_youtube_links: approvedYoutubeLinks,
        approved_event_ids: approvedEventIds,
        approved_pitch_ids: approvedPitchIds,
        unverified_draft_selections: unverifiedDraftSelections,
        pitch_presentation_styles: pitchPresentationStyles || {},
        action,
      }),
    }),

  // ---------------------------------------------------------
  // Actual Portfolio Hub Endpoints (Phase 1 & Multi-Portfolio)
  // ---------------------------------------------------------
  listPortfolios: () => request<import('./types').PortfolioMetaDTO[]>('/api/portfolio/list'),

  createPortfolio: (name: string, portfolioId?: string) =>
    request<import('./types').PortfolioMetaDTO>('/api/portfolio/create', {
      method: 'POST',
      body: JSON.stringify({ name, portfolio_id: portfolioId }),
    }),

  deletePortfolio: (portfolioId: string) =>
    request<{ status: string }>(`/api/portfolio/${encodeURIComponent(portfolioId)}`, {
      method: 'DELETE',
    }),

  renamePortfolio: (portfolioId: string, name: string) =>
    request<import('./types').PortfolioMetaDTO>(`/api/portfolio/${encodeURIComponent(portfolioId)}/rename`, {
      method: 'PUT',
      body: JSON.stringify({ name }),
    }),


  getActualPortfolioState: (refreshPrices: boolean = false, fetchFundamentals: boolean = false, portfolioId: string = 'default') =>
    request<ActualPortfolioStateDTO>(
      `/api/portfolio/actual/state?refresh_prices=${refreshPrices}&fetch_fundamentals=${fetchFundamentals}&portfolio_id=${encodeURIComponent(portfolioId)}`
    ),

  getActualBucketAllocations: (portfolioId: string = 'default') =>
    request<BucketAllocationResponseDTO>(`/api/portfolio/actual/allocations?portfolio_id=${encodeURIComponent(portfolioId)}`),

  getActualWatchlist: (portfolioId: string = 'default') =>
    request<ActualWatchlistStateDTO>(`/api/portfolio/actual/watchlist?portfolio_id=${encodeURIComponent(portfolioId)}`),

  getActualGoals: (portfolioId?: string) => {
    const params = portfolioId ? `?portfolio_id=${encodeURIComponent(portfolioId)}` : ''
    return request<ActualGoalsResponseDTO>(`/api/portfolio/actual/goals${params}`)
  },

  getActualPerformance: (days?: number, portfolioId: string = 'default') => {
    const params = new URLSearchParams({ portfolio_id: portfolioId })
    if (days !== undefined) params.append('days', days.toString())
    return request<PerformanceSnapshotDTO[]>(`/api/portfolio/actual/performance?${params.toString()}`)
  },

  triggerPerformanceSnapshot: (refreshPrices: boolean = false, portfolioId: string = 'default') =>
    request<PerformanceSnapshotDTO[]>(`/api/portfolio/actual/performance/snapshot?refresh_prices=${refreshPrices}&portfolio_id=${encodeURIComponent(portfolioId)}`, {
      method: 'POST',
    }),

  getActualJournal: (days: number = 365, keyword?: string, limit: number = 100, portfolioId: string = 'default') => {
    const params = new URLSearchParams({ days: days.toString(), limit: limit.toString(), portfolio_id: portfolioId })
    if (keyword) params.append('keyword', keyword)
    return request<JournalEntryDTO[]>(`/api/portfolio/actual/journal?${params.toString()}`)
  },

  // ---------------------------------------------------------
  // Actual Portfolio Hub Mutation Endpoints (Phase 2.1 & 2.2)
  // ---------------------------------------------------------
  upsertAllocationTargets: (payload: UpsertAllocationTargetsPayload, portfolioId: string) =>
    request<ActualPortfolioStateDTO>(`/api/portfolio/actual/allocations/targets?portfolio_id=${encodeURIComponent(portfolioId)}`, {
      method: 'PUT',
      body: JSON.stringify(payload),
    }),

  assignHoldingBucket: (symbol: string, payload: AssignBucketPayload, portfolioId: string) =>
    request<ActualPortfolioStateDTO>(`/api/portfolio/actual/holdings/${encodeURIComponent(symbol)}/bucket?portfolio_id=${encodeURIComponent(portfolioId)}`, {
      method: 'PUT',
      body: JSON.stringify(payload),
    }),

  batchAssignHoldingBuckets: (payload: BatchAssignBucketPayload, portfolioId: string) =>
    request<ActualPortfolioStateDTO>(`/api/portfolio/actual/holdings/batch-bucket?portfolio_id=${encodeURIComponent(portfolioId)}`, {
      method: 'PUT',
      body: JSON.stringify(payload),
    }),

  batchRemoveHoldings: (payload: BatchRemoveHoldingsPayload, portfolioId: string) =>
    request<ActualPortfolioStateDTO>(`/api/portfolio/actual/holdings/batch-delete?portfolio_id=${encodeURIComponent(portfolioId)}`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  resetPortfolioCleanSlate: (portfolioId: string) =>
    request<ActualPortfolioStateDTO>(`/api/portfolio/actual/reset?portfolio_id=${encodeURIComponent(portfolioId)}`, {
      method: 'POST',
    }),

  executeTrade: (payload: TradePayload, portfolioId: string) =>
    request<ActualPortfolioStateDTO>(`/api/portfolio/actual/trade?portfolio_id=${encodeURIComponent(portfolioId)}`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  manageCashFlow: (payload: CashFlowPayload, portfolioId: string) =>
    request<ActualPortfolioStateDTO>(`/api/portfolio/actual/cashflow?portfolio_id=${encodeURIComponent(portfolioId)}`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  recordIncome: (payload: IncomePayload, portfolioId: string) =>
    request<ActualPortfolioStateDTO>(`/api/portfolio/actual/income?portfolio_id=${encodeURIComponent(portfolioId)}`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  editHolding: (symbol: string, payload: EditHoldingPayload, portfolioId: string) =>
    request<ActualPortfolioStateDTO>(`/api/portfolio/actual/holdings/${encodeURIComponent(symbol)}/edit?portfolio_id=${encodeURIComponent(portfolioId)}`, {
      method: 'PUT',
      body: JSON.stringify(payload),
    }),

  removeHolding: (symbol: string, portfolioId: string) =>
    request<ActualPortfolioStateDTO>(`/api/portfolio/actual/holdings/${encodeURIComponent(symbol)}?portfolio_id=${encodeURIComponent(portfolioId)}`, {
      method: 'DELETE',
    }),

  upsertWatchlistItem: (symbol: string, payload: UpsertWatchlistItemPayload, portfolioId: string) =>
    request<ActualWatchlistStateDTO>(`/api/portfolio/actual/watchlist/${encodeURIComponent(symbol)}?portfolio_id=${encodeURIComponent(portfolioId)}`, {
      method: 'PUT',
      body: JSON.stringify(payload),
    }),

  removeWatchlistItem: (symbol: string, portfolioId: string) =>
    request<ActualWatchlistStateDTO>(`/api/portfolio/actual/watchlist/${encodeURIComponent(symbol)}?portfolio_id=${encodeURIComponent(portfolioId)}`, {
      method: 'DELETE',
    }),

  upsertGoal: (name: string, payload: UpsertGoalPayload) =>
    request<ActualGoalsResponseDTO>(`/api/portfolio/actual/goals/${encodeURIComponent(name)}`, {
      method: 'PUT',
      body: JSON.stringify(payload),
    }),

  removeGoal: (name: string, portfolioId?: string) => {
    const params = portfolioId ? `?portfolio_id=${encodeURIComponent(portfolioId)}` : ''
    return request<ActualGoalsResponseDTO>(`/api/portfolio/actual/goals/${encodeURIComponent(name)}${params}`, {
      method: 'DELETE',
    })
  },

  appendJournal: (payload: AppendJournalPayload, portfolioId: string) =>
    request<JournalEntryDTO[]>(`/api/portfolio/actual/journal?portfolio_id=${encodeURIComponent(portfolioId)}`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  getTransactions: (portfolioId: string = 'default', symbol?: string) => {
    const params = new URLSearchParams()
    params.set('portfolio_id', portfolioId)
    if (symbol) params.set('symbol', symbol)
    return request<import('./types').TransactionListResponseDTO>(`/api/portfolio/actual/transactions?${params.toString()}`)
  },

  updateTransactionNote: (txId: string, notes: string, portfolioId: string = 'default') =>
    request<import('./types').TransactionItemDTO>(`/api/portfolio/actual/transactions/${encodeURIComponent(txId)}/note?portfolio_id=${encodeURIComponent(portfolioId)}`, {
      method: 'PATCH',
      body: JSON.stringify({ notes }),
    }),

  editTransaction: (txId: string, payload: import('./types').EditTransactionPayload, portfolioId: string = 'default') =>
    request<import('./types').ActualPortfolioStateDTO>(`/api/portfolio/actual/transactions/${encodeURIComponent(txId)}?portfolio_id=${encodeURIComponent(portfolioId)}`, {
      method: 'PUT',
      body: JSON.stringify(payload),
    }),

  deleteTransaction: (txId: string, payload?: import('./types').DeleteTransactionPayload, portfolioId: string = 'default') => {
    const adjustCash = payload?.adjust_cash ?? true
    return request<import('./types').ActualPortfolioStateDTO>(`/api/portfolio/actual/transactions/${encodeURIComponent(txId)}?adjust_cash=${adjustCash}&portfolio_id=${encodeURIComponent(portfolioId)}`, {
      method: 'DELETE',
    })
  },

  getFxRate: (date?: string, portfolioId: string = 'default') => {
    const params = new URLSearchParams()
    if (date) params.set('date', date)
    params.set('portfolio_id', portfolioId)
    return request<FXRateResponseDTO>(`/api/portfolio/actual/fx-rate?${params.toString()}`)
  },

  syncDividends: (portfolioId: string = 'default') =>
    request<SyncDividendsResponseDTO>(`/api/portfolio/actual/sync-dividends?portfolio_id=${encodeURIComponent(portfolioId)}`, {
      method: 'POST',
    }),

  // ---------------------------------------------------------
  // NotebookLM Audio Overview (การ์ด flow="notebooklm" สร้างเองผ่าน Create Card ปกติ —
  // เลือกไฟล์ Briefing Book ทีหลังใน Drawer)
  // ---------------------------------------------------------
  getNotebookLMAvailableSources: () =>
    request<NotebookLMAvailableSourceDTO[]>('/api/notebooklm/available-sources'),

  generateNotebookLMAudio: (cardId: string, briefingFilePath?: string) =>
    request<NotebookLMGenerateResponse>('/api/notebooklm/generate', {
      method: 'POST',
      body: JSON.stringify({ card_id: cardId, briefing_file_path: briefingFilePath ?? null }),
    }),

  getNotebookLMStatus: (jobId: string) =>
    request<NotebookLMStatusDTO>(`/api/notebooklm/status/${encodeURIComponent(jobId)}`),
}
