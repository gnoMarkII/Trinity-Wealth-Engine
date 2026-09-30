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

  getEquityFinancials: (ticker: string, market?: 'US' | 'TH', forceRefresh = false, signal?: AbortSignal) => {
    const params = new URLSearchParams({ force_refresh: String(forceRefresh) })
    if (market) params.set('market', market)
    return request<import('./types').FinancialStatementsDTO>(
      `/api/equity/${encodeURIComponent(ticker)}/financials?${params.toString()}`,
      { signal }
    )
  },

  summarizeEarningsCall: (ticker: string, period: string, transcript: string, signal?: AbortSignal) =>
    request<import('./types').EarningsCallSummarizeResponse>(
      `/api/equity/${encodeURIComponent(ticker)}/earnings-call/summarize`,
      {
        method: 'POST',
        body: JSON.stringify({ period, transcript }),
        signal,
      }
    ),

  getEarningsCallRun: (ticker: string, runId: string, signal?: AbortSignal) =>
    request<import('./types').EarningsCallRunResponse>(
      `/api/equity/${encodeURIComponent(ticker)}/earnings-call/runs/${encodeURIComponent(runId)}`,
      { signal }
    ),

  retryEarningsCallRun: (ticker: string, runId: string, signal?: AbortSignal) =>
    request<import('./types').EarningsCallRunResponse>(
      `/api/equity/${encodeURIComponent(ticker)}/earnings-call/runs/${encodeURIComponent(runId)}/retry`,
      {
        method: 'POST',
        signal,
      }
    ),

  getEarningsCalls: (ticker: string, signal?: AbortSignal) =>
    request<import('./types').EarningsCallListResponse>(
      `/api/equity/${encodeURIComponent(ticker)}/earnings-calls`,
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

  voidTransaction: (txId: string, portfolioId: string = 'default') =>
    request<import('./types').ActualPortfolioStateDTO>(`/api/portfolio/actual/transactions/${encodeURIComponent(txId)}/void?portfolio_id=${encodeURIComponent(portfolioId)}`, {
      method: 'POST',
    }),

  deleteTransaction: (txId: string, _payload?: import('./types').DeleteTransactionPayload, portfolioId: string = 'default') => {
    // Void strictly replaces delete
    return request<import('./types').ActualPortfolioStateDTO>(`/api/portfolio/actual/transactions/${encodeURIComponent(txId)}/void?portfolio_id=${encodeURIComponent(portfolioId)}`, {
      method: 'POST',
    })
  },

  getDimeEmails: (query?: string, limit?: number) => {
    const params = new URLSearchParams()
    if (query) params.set('query', query)
    if (limit) params.set('limit', String(limit))
    return request<import('./types').DimeEmailListResponseDTO>(`/api/portfolio/dime/emails?${params.toString()}`)
  },

  scanDimeEmail: (messageId: string, attachmentId: string, password?: string) =>
    request<import('./types').DimeScanResponseDTO>('/api/portfolio/dime/scan/email', {
      method: 'POST',
      body: JSON.stringify({ message_id: messageId, attachment_id: attachmentId, password }),
    }),

  scanDimeUpload: async (pdfFile: File, password?: string) => {
    const formData = new FormData()
    formData.append('pdf_file', pdfFile)
    if (password) formData.append('password', password)
    const res = await fetch('/api/portfolio/dime/scan/upload', {
      method: 'POST',
      body: formData,
      credentials: 'same-origin',
    })
    if (!res.ok) {
      let msg = 'Upload failed'
      try {
        const err = await res.json()
        msg = err.detail || msg
      } catch {}
      throw new Error(msg)
    }
    return (await res.json()) as import('./types').DimeScanResponseDTO
  },

  getStagedDimeTrades: (scanId: string) =>
    request<import('./types').DimeScanResponseDTO>(`/api/portfolio/dime/staged/${encodeURIComponent(scanId)}`),

  commitDimeTrades: (scanId: string, portfolioId: string = 'default') =>
    request<import('./types').DimeCommitResponseDTO>(`/api/portfolio/dime/commit/${encodeURIComponent(scanId)}`, {
      method: 'POST',
      body: JSON.stringify({ portfolio_id: portfolioId }),
    }),

  getDimePdfUrl: (messageId: string, attachmentId: string, password?: string, decrypt: boolean = true) => {
    const params = new URLSearchParams()
    params.set('message_id', messageId)
    params.set('attachment_id', attachmentId)
    if (password) params.set('password', password)
    if (!decrypt) params.set('decrypt', 'false')
    return `/api/portfolio/dime/pdf?${params.toString()}`
  },

  getDimePdfText: (messageId: string, attachmentId: string, password?: string) => {
    const params = new URLSearchParams()
    params.set('message_id', messageId)
    params.set('attachment_id', attachmentId)
    if (password) params.set('password', password)
    return request<import('./types').DimePdfTextResponseDTO>(`/api/portfolio/dime/pdf-text?${params.toString()}`)
  },

  streamBatchDimeSync: async (
    payload: { password?: string; force_rescan?: boolean; portfolio_id?: string },
    callbacks: {
      onProgress?: (data: import('./types').DimeBatchScanProgressEvent) => void
      onWarning?: (data: import('./types').DimeBatchScanWarningEvent) => void
      onComplete?: (data: import('./types').DimeBatchScanCompleteEvent) => void
      onError?: (err: Error) => void
    },
    signal?: AbortSignal
  ) => {
    const res = await fetch('/api/portfolio/dime/scan/batch-stream', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
      credentials: 'same-origin',
      signal,
    })

    if (!res.ok) {
      let msg = 'Sync failed'
      try {
        const err = await res.json()
        msg = err.detail || msg
      } catch {}
      throw new Error(msg)
    }

    if (!res.body) {
      throw new Error('ReadableStream not supported')
    }

    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const blocks = buffer.split('\n\n')
        buffer = blocks.pop() || ''

        for (const block of blocks) {
          if (!block.trim() || block.startsWith(':')) continue // Ignore keep-alive heartbeats

          let eventType = 'message'
          let eventDataStr = ''

          const lines = block.split('\n')
          for (const line of lines) {
            if (line.startsWith('event:')) {
              eventType = line.replace('event:', '').trim()
            } else if (line.startsWith('data:')) {
              eventDataStr += line.replace('data:', '').trim()
            }
          }

          if (!eventDataStr) continue

          try {
            const data = JSON.parse(eventDataStr)
            if (eventType === 'progress') {
              callbacks.onProgress?.(data)
            } else if (eventType === 'warning') {
              callbacks.onWarning?.(data)
            } else if (eventType === 'complete') {
              callbacks.onComplete?.(data)
            } else if (eventType === 'error') {
              callbacks.onError?.(new Error(data.detail || 'Sync encountered an error'))
            }
          } catch (err) {
            console.error('Failed to parse SSE event:', err, block)
          }
        }
      }
    } finally {
      reader.releaseLock()
    }
  },

  getWealthXEmails: (query?: string, limit?: number) => {
    const params = new URLSearchParams()
    if (query) params.set('query', query)
    if (limit) params.set('limit', String(limit))
    return request<import('./types').WealthXEmailListResponseDTO>(`/api/portfolio/wealthx/emails?${params.toString()}`)
  },

  scanWealthXEmail: (messageId: string, attachmentId: string, password?: string) =>
    request<import('./types').WealthXScanResponseDTO>('/api/portfolio/wealthx/scan/email', {
      method: 'POST',
      body: JSON.stringify({ message_id: messageId, attachment_id: attachmentId, password }),
    }),

  scanWealthXUpload: async (pdfFile: File, password?: string) => {
    const formData = new FormData()
    formData.append('pdf_file', pdfFile)
    if (password) formData.append('password', password)
    const res = await fetch('/api/portfolio/wealthx/scan/upload', {
      method: 'POST',
      body: formData,
      credentials: 'same-origin',
    })
    if (!res.ok) {
      let msg = 'Upload failed'
      try {
        const err = await res.json()
        msg = err.detail || msg
      } catch {}
      throw new Error(msg)
    }
    return (await res.json()) as import('./types').WealthXScanResponseDTO
  },

  getStagedWealthXTrades: (scanId: string) =>
    request<import('./types').WealthXScanResponseDTO>(`/api/portfolio/wealthx/staged/${encodeURIComponent(scanId)}`),

  commitWealthXTrades: (scanId: string, portfolioId: string = 'default') =>
    request<import('./types').WealthXCommitResponseDTO>(`/api/portfolio/wealthx/commit/${encodeURIComponent(scanId)}`, {
      method: 'POST',
      body: JSON.stringify({ portfolio_id: portfolioId }),
    }),

  getWealthXPdfUrl: (messageId: string, attachmentId: string, password?: string, decrypt: boolean = true) => {
    const params = new URLSearchParams()
    params.set('message_id', messageId)
    params.set('attachment_id', attachmentId)
    if (password) params.set('password', password)
    if (!decrypt) params.set('decrypt', 'false')
    return `/api/portfolio/wealthx/pdf?${params.toString()}`
  },

  getWealthXPdfText: (messageId: string, attachmentId: string, password?: string) => {
    const params = new URLSearchParams()
    params.set('message_id', messageId)
    params.set('attachment_id', attachmentId)
    if (password) params.set('password', password)
    return request<import('./types').WealthXPdfTextResponseDTO>(`/api/portfolio/wealthx/pdf-text?${params.toString()}`)
  },

  streamBatchWealthXSync: async (
    payload: { password?: string; force_rescan?: boolean; portfolio_id?: string },
    callbacks: {
      onProgress?: (data: import('./types').WealthXBatchScanProgressEvent) => void
      onWarning?: (data: import('./types').WealthXBatchScanWarningEvent) => void
      onComplete?: (data: import('./types').WealthXBatchScanCompleteEvent) => void
      onError?: (err: Error) => void
    },
    signal?: AbortSignal
  ) => {
    const res = await fetch('/api/portfolio/wealthx/scan/batch-stream', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
      credentials: 'same-origin',
      signal,
    })

    if (!res.ok) {
      let msg = 'Sync failed'
      try {
        const err = await res.json()
        msg = err.detail || msg
      } catch {}
      throw new Error(msg)
    }

    if (!res.body) {
      throw new Error('ReadableStream not supported')
    }

    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n\n')
        buffer = lines.pop() || ''

        for (const block of lines) {
          if (!block.trim()) continue
          let eventType = 'message'
          let dataStr = ''

          for (const line of block.split('\n')) {
            if (line.startsWith('event: ')) {
              eventType = line.replace('event: ', '').trim()
            } else if (line.startsWith('data: ')) {
              dataStr = line.replace('data: ', '').trim()
            }
          }

          if (!dataStr) continue
          try {
            const data = JSON.parse(dataStr)
            if (eventType === 'progress') {
              callbacks.onProgress?.(data)
            } else if (eventType === 'warning') {
              callbacks.onWarning?.(data)
            } else if (eventType === 'complete') {
              callbacks.onComplete?.(data)
            } else if (eventType === 'error') {
              callbacks.onError?.(new Error(data.detail || 'Sync encountered an error'))
            }
          } catch (err) {
            console.error('Failed to parse SSE event:', err, block)
          }
        }
      }
    } finally {
      reader.releaseLock()
    }
  },

  getScbEmails: (query?: string, limit?: number) => {
    const params = new URLSearchParams()
    if (query) params.set('query', query)
    if (limit) params.set('limit', String(limit))
    return request<import('./types').SCBAMEmailListResponseDTO>(`/api/portfolio/scb/emails?${params.toString()}`)
  },

  scanScbEmail: (messageId: string, portfolioId: string = 'default') =>
    request<import('./types').SCBAMScanResponseDTO>('/api/portfolio/scb/scan/email', {
      method: 'POST',
      body: JSON.stringify({ message_id: messageId, portfolio_id: portfolioId }),
    }),

  getStagedScbTrades: (scanId: string) =>
    request<import('./types').SCBAMScanResponseDTO>(`/api/portfolio/scb/staged/${encodeURIComponent(scanId)}`),

  commitScbTrades: (scanId: string, portfolioId: string = 'default', selectedItemIds?: string[]) =>
    request<import('./types').SCBAMCommitResponseDTO>(`/api/portfolio/scb/commit/${encodeURIComponent(scanId)}`, {
      method: 'POST',
      body: JSON.stringify({ portfolio_id: portfolioId, selected_item_ids: selectedItemIds }),
    }),

  getScbEmailHtmlUrl: (messageId: string) => {
    return `/api/portfolio/scb/emails/${encodeURIComponent(messageId)}/html`
  },

  streamBatchScbSync: async (
    payload: { portfolio_id?: string; since_date?: string; limit?: number },
    callbacks: {
      onProgress?: (data: import('./types').SCBAMBatchScanProgressEvent) => void
      onWarning?: (data: import('./types').SCBAMBatchScanWarningEvent) => void
      onComplete?: (data: import('./types').SCBAMBatchScanCompleteEvent) => void
      onError?: (err: Error) => void
    },
    signal?: AbortSignal
  ) => {
    const res = await fetch('/api/portfolio/scb/scan/batch-stream', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
      credentials: 'same-origin',
      signal,
    })

    if (!res.ok) {
      let msg = 'Sync failed'
      try {
        const err = await res.json()
        msg = err.detail || msg
      } catch {}
      throw new Error(msg)
    }

    if (!res.body) {
      throw new Error('ReadableStream not supported')
    }

    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n\n')
        buffer = lines.pop() || ''

        for (const block of lines) {
          if (!block.trim() || block.startsWith(':')) continue
          let eventType = 'message'
          let dataStr = ''

          for (const line of block.split('\n')) {
            if (line.startsWith('event:')) {
              eventType = line.replace('event:', '').trim()
            } else if (line.startsWith('data:')) {
              dataStr = line.replace('data:', '').trim()
            }
          }

          if (!dataStr) continue
          try {
            const data = JSON.parse(dataStr)
            if (eventType === 'progress') {
              callbacks.onProgress?.(data)
            } else if (eventType === 'warning') {
              callbacks.onWarning?.(data)
            } else if (eventType === 'complete') {
              callbacks.onComplete?.(data)
            } else if (eventType === 'error') {
              callbacks.onError?.(new Error(data.message || data.detail || 'Sync encountered an error'))
            }
          } catch (err) {
            console.error('Failed to parse SSE event:', err, block)
          }
        }
      }
    } finally {
      reader.releaseLock()
    }
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

  // ---------------------------------------------------------
  // Terminal V2 Institutional Market Intelligence (Phase 1, 2, 3)
  // ---------------------------------------------------------
  getFinancialStress: () =>
    request<any>('/api/v2/market/macro/financial-stress'),

  getMetalsCot: (commodity: string = 'gold') =>
    request<any>(`/api/v2/market/commodities/metals/cot?commodity=${encodeURIComponent(commodity)}`),

  getGlobalPolicyRates: () =>
    request<any>('/api/v2/market/macro/global-policy-rates'),

  getNasdaqConsensus: (symbol: string) =>
    request<any>(`/api/v2/market/equity/consensus/${encodeURIComponent(symbol)}`),

  getOptionsChain: (symbol: string, expiry?: string) => {
    const params = expiry ? `?expiry=${encodeURIComponent(expiry)}` : ''
    return request<import('./types').OptionsChainResponseDTO>(
      `/api/v2/market/equity/options/${encodeURIComponent(symbol)}${params}`
    )
  },

  getOptionsMaxPain: (symbol: string, expiry?: string) => {
    const params = expiry ? `?expiry=${encodeURIComponent(expiry)}` : ''
    return request<import('./types').OptionsMaxPainResponseDTO>(
      `/api/v2/market/equity/max-pain/${encodeURIComponent(symbol)}${params}`
    )
  },

  // ---------------------------------------------------------
  // Terminal V2 Phase 4 Endpoints (Non-Crypto)
  // ---------------------------------------------------------
  getCommodityVolatility: (symbol?: string) => {
    const params = symbol ? `?symbol=${encodeURIComponent(symbol)}` : ''
    return request<import('./types').CommodityVolSnapshotDTO[]>(
      `/api/v2/market/commodities/volatility${params}`
    )
  },

  getTreasuryAuctionDemand: (securityType: string, securityTerm: string) =>
    request<import('./types').AuctionDemandSnapshotDTO>(
      `/api/v2/market/macro/treasury/auction-demand?security_type=${encodeURIComponent(securityType)}&security_term=${encodeURIComponent(securityTerm)}`
    ),

  getSecFinancials: (symbol: string) =>
    request<import('./types').SecCompanyFactsSnapshotDTO>(
      `/api/v2/market/equity/sec/financials/${encodeURIComponent(symbol)}`
    ),

  getSecInsiderTrades: (symbol: string, limit: number = 20) =>
    request<import('./types').SecInsiderTradeSnapshotDTO>(
      `/api/v2/market/equity/sec/insider-trades/${encodeURIComponent(symbol)}?limit=${limit}`
    ),

  getEquityNewsDiscovery: (symbol: string, limit: number = 15) =>
    request<import('./types').NewsDiscoverySnapshotDTO>(
      `/api/v2/market/equity/news/${encodeURIComponent(symbol)}?limit=${limit}`
    ),

  getNationalDebt: (limit: number = 30) =>
    request<import('./types').UsNationalDebtDTO[]>(
      `/api/v2/market/macro/treasury/debt?limit=${limit}`
    ),

  // ---------------------------------------------------------
  // Terminal V2 Thailand & Rates Radar
  // ---------------------------------------------------------
  getThaiInvestorFlow: (market: string = 'SET') =>
    request<import('./types').ThaiFundFlowDTO>(
      `/api/v2/market/thailand/flow?market=${encodeURIComponent(market)}`
    ),

  getThaiRetailGold: () =>
    request<import('./types').ThaiRetailGoldDTO>('/api/v2/market/thailand/gold'),

  getThaiMarketValuation: (market: string = 'SET') =>
    request<import('./types').MarketValuationDTO>(
      `/api/v2/market/thailand/valuation?market=${encodeURIComponent(market)}`
    ),

  getThaiMarketBreadth: (market: string = 'SET') =>
    request<import('./types').MarketBreadthDTO>(
      `/api/v2/market/thailand/breadth?market=${encodeURIComponent(market)}`
    ),

  getTreasuryYieldCurve: (month?: string) => {
    const params = month ? `?month=${encodeURIComponent(month)}` : ''
    return request<import('./types').TreasuryYieldCurveDTO>(
      `/api/v2/market/macro/treasury/yield-curve${params}`
    )
  },
}

