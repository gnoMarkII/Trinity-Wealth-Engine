import { act, cleanup, fireEvent, render, renderHook, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import InvestorPrinciplesTab from './InvestorPrinciplesTab'
import BucketApplyReview from './BucketApplyReview'
import InvestmentAxisEditor from './InvestmentAxisEditor'
import { useInvestorPrinciples } from '../../../hooks/useInvestorPrinciples'
import { api } from '../../../api/client'

vi.mock('../../../api/client', () => ({
  api: Object.fromEntries([
    'getInterviewConfig', 'getCurrentConfirmedEssence', 'getCurrentEssenceSession',
    'getEssenceSummary', 'getEssenceSession', 'getFinancialContext', 'getCurrentConfirmedAxis',
    'getLatestAxisDraft', 'getLatestBucketPlan', 'getActualGoals', 'previewBucketPlan',
    'updateFinancialContext', 'createAxisDraft', 'updateAxisDraft', 'confirmInvestmentAxis',
    'createBucketPlan', 'updateBucketPlan', 'applyBucketPlan', 'recordEssenceAnswer', 'summarizeEssence',
  ].map(name => [name, vi.fn()])),
}))

const axisA: any = {
  draft_id: 'axis-A', portfolio_id: 'A', basic_policy: 'CONFIRMED POLICY FOR A', risk_limits: {},
  investment_horizon: '10 years', rebalance_frequency: 'annual', non_actions: ['a', 'b', 'c'],
  invest_targets: ['equity'], exclude_targets: ['margin'], primary_methods: ['DCA'],
  secondary_methods: ['reinvest'], role_models: ['none'], allocation_rows: [], revision: 1,
}
const planA: any = {
  draft_id: 'plan-A', portfolio_id: 'A', portfolio_checkpoint: { nav_thb: '1000000' }, revision: 1,
  purpose_buckets: [{ bucket_id: 'growth', name: 'Growth', role: 'Long term', color: '#336699',
    target_percent: '100', source_value_ids: [], source_axis_allocation_ids: [] }],
  mapping_weights: [], remapping: [], constraints: [], validation_issues: [], is_valid: true,
}
const planB: any = {
  draft_id: 'plan-B', portfolio_id: 'B', portfolio_checkpoint: { nav_thb: '500000' }, revision: 1,
  purpose_buckets: [{ bucket_id: 'income', name: 'Income', role: 'Cash flow', color: '#10b981',
    target_percent: '100', source_value_ids: [], source_axis_allocation_ids: [] }],
  mapping_weights: [], remapping: [], constraints: [], validation_issues: [], is_valid: true,
}
const previewValid: any = { validated_targets: [], affected_holdings: [], issues: [], before_allocation: {}, after_allocation: {} }

beforeEach(() => {
  vi.resetAllMocks()
  vi.mocked(api.getInterviewConfig).mockResolvedValue({} as any)
  vi.mocked(api.getCurrentConfirmedEssence).mockResolvedValue({ scope: 'workspace', kind: 'investor_essence', pointer: 'vault:confirmed' } as any)
  vi.mocked(api.getCurrentEssenceSession).mockRejectedValue(new Error('not found'))
  vi.mocked(api.getFinancialContext).mockResolvedValue({ unknown_fields: [], readiness_issues: [], is_ready_for_numeric_policy: false } as any)
  vi.mocked(api.getCurrentConfirmedAxis).mockRejectedValue(new Error('not found'))
  vi.mocked(api.getLatestAxisDraft).mockResolvedValue(null)
  vi.mocked(api.getLatestBucketPlan).mockResolvedValue(null)
  vi.mocked(api.getActualGoals).mockResolvedValue({ goals: [] } as any)
  vi.mocked(api.previewBucketPlan).mockResolvedValue(previewValid)
})

afterEach(cleanup)

describe('Deliverable D1 (WP00 + WP01): State Isolation & Failure Propagation', () => {
  it('WP01-01: aborts before drafting Axis if Financial Context save fails, preserving user edits and showing error', async () => {
    vi.mocked(api.updateFinancialContext).mockRejectedValue(new Error('save financial context failed'))
    vi.mocked(api.createAxisDraft).mockResolvedValue(axisA)

    render(<InvestorPrinciplesTab portfolioId="A" />)
    fireEvent.click(await screen.findByText('สร้างแกนหลัก (Axis)'))

    const draftBtn = await screen.findByRole('button', { name: /ร่างแกนหลักครบ 8 หัวข้อ/ })
    fireEvent.click(draftBtn)

    // Wait and verify that createAxisDraft was NEVER called because save failed
    await waitFor(() => {
      expect(api.updateFinancialContext).toHaveBeenCalled()
    })
    expect(api.createAxisDraft).not.toHaveBeenCalled()
    expect(await screen.findByText(/save financial context failed/)).toBeInTheDocument()
  })

  it('WP01-02: aborts before going to Review if Bucket draft save fails, preserving edits', async () => {
    vi.mocked(api.getCurrentConfirmedAxis).mockResolvedValue(axisA)
    vi.mocked(api.getLatestBucketPlan).mockResolvedValue(planA)
    vi.mocked(api.updateBucketPlan).mockRejectedValue(new Error('save bucket draft failed'))

    render(<InvestorPrinciplesTab portfolioId="A" />)
    fireEvent.click(await screen.findByText('ออกแบบ Buckets'))

    const reviewBtn = await screen.findByRole('button', { name: /ตรวจสอบก่อนใช้กับพอร์ต/ })
    fireEvent.click(reviewBtn)

    await waitFor(() => {
      expect(api.updateBucketPlan).toHaveBeenCalled()
    })
    // Review screen should NOT appear
    expect(screen.queryByText('ตรวจสอบก่อนใช้กับพอร์ต (Preview & Safety Check)')).not.toBeInTheDocument()
    // Error alert must be shown
    expect(await screen.findByText(/save bucket draft failed/)).toBeInTheDocument()
  })

  it('WP01-03: resets portfolio state when switching from Portfolio A to B and does NOT send mutation to A', async () => {
    vi.mocked(api.getLatestBucketPlan).mockImplementation(async pid => pid === 'A' ? planA : null)
    vi.mocked(api.updateBucketPlan).mockResolvedValue(planA)

    const hook = renderHook(
      ({ portfolioId }) => useInvestorPrinciples({ portfolioId }),
      { initialProps: { portfolioId: 'A' } },
    )

    await waitFor(() => expect(hook.result.current.loading).toBe(false))
    expect(hook.result.current.bucketPlan?.portfolio_id).toBe('A')

    // Switch to Portfolio B
    hook.rerender({ portfolioId: 'B' })

    await waitFor(() => expect(api.getLatestBucketPlan).toHaveBeenCalledWith('B'))
    await waitFor(() => expect(hook.result.current.loading).toBe(false))

    expect(hook.result.current.portfolioId).toBe('B')
    // Portfolio A's draft must NOT linger in B
    expect(hook.result.current.bucketPlan).toBeNull()

    // Any attempt to update bucket plan for B when B has no plan must not call updateBucketPlan for A
    await act(async () => {
      try {
        await hook.result.current.updateBucketPlan({ purpose_buckets: planA.purpose_buckets })
      } catch {
        // Expected guard error or no-op
      }
    })
    expect(api.updateBucketPlan).not.toHaveBeenCalledWith('plan-A', expect.anything())
  })

  it('WP01-04: disallows Apply in BucketApplyReview when preview is null or missing', () => {
    const onApply = vi.fn().mockResolvedValue(undefined)
    render(
      <BucketApplyReview
        draft={planA}
        preview={null}
        applyReceipt={null}
        onApply={onApply}
        onBack={vi.fn()}
        onGoToOverview={vi.fn()}
        loading={false}
      />,
    )

    // Even if user checks checkbox, apply button must remain disabled
    const checkbox = screen.getByRole('checkbox')
    fireEvent.click(checkbox)

    const applyBtn = screen.getByRole('button', { name: /ยืนยันและนำไปใช้/ })
    expect(applyBtn).toBeDisabled()

    fireEvent.click(applyBtn)
    expect(onApply).not.toHaveBeenCalled()
  })

  it('WP01-05: disallows Apply in BucketApplyReview when preview has validation issues', () => {
    const onApply = vi.fn().mockResolvedValue(undefined)
    render(
      <BucketApplyReview
        draft={planA}
        preview={{ ...previewValid, issues: ['invalid matrix target mismatch'] }}
        applyReceipt={null}
        onApply={onApply}
        onBack={vi.fn()}
        onGoToOverview={vi.fn()}
        loading={false}
      />,
    )

    const checkbox = screen.getByRole('checkbox')
    fireEvent.click(checkbox)

    const applyBtn = screen.getByRole('button', { name: /ยืนยันและนำไปใช้/ })
    expect(applyBtn).toBeDisabled()

    fireEvent.click(applyBtn)
    expect(onApply).not.toHaveBeenCalled()
  })

  it('WP01-06: displays confirmed Axis in read-only mode, NOT unconfirmed draft with "ใช้งานจริง"', () => {
    render(
      <InvestmentAxisEditor
        axisDraft={{ ...axisA, basic_policy: 'UNCONFIRMED DRAFT POLICY' }}
        confirmedAxis={axisA}
        portfolioId="A"
        onUpdateDraft={vi.fn()}
        onConfirmAxis={vi.fn()}
        onProceedToBuckets={vi.fn()}
        loading={false}
      />,
    )

    // Read-only view with "ใช้งานจริง" must display the CONFIRMED policy
    expect(screen.getByText('ใช้งานจริง')).toBeInTheDocument()
    expect(screen.getByText(/CONFIRMED POLICY FOR A/)).toBeInTheDocument()
    expect(screen.queryByText(/UNCONFIRMED DRAFT POLICY/)).not.toBeInTheDocument()
  })

  it('WP01-07: handles late-arriving responses out of order when switching portfolios (generation token)', async () => {
    let resolvePortA: (val: any) => void
    const slowPortAPromise = new Promise((resolve) => {
      resolvePortA = resolve
    })

    vi.mocked(api.getLatestBucketPlan).mockImplementation((pid) => {
      if (pid === 'A') return slowPortAPromise as any
      if (pid === 'B') return Promise.resolve(planB)
      return Promise.resolve(null)
    })

    const hook = renderHook(
      ({ portfolioId }) => useInvestorPrinciples({ portfolioId }),
      { initialProps: { portfolioId: 'A' } },
    )

    // Quickly switch to Portfolio B before A resolves
    hook.rerender({ portfolioId: 'B' })

    await waitFor(() => {
      expect(hook.result.current.portfolioId).toBe('B')
      expect(hook.result.current.bucketPlan?.portfolio_id).toBe('B')
    })

    // Now late resolve A
    await act(async () => {
      resolvePortA!(planA)
    })

    // Hook must still hold B's data, NOT overridden by A's late response!
    expect(hook.result.current.portfolioId).toBe('B')
    expect(hook.result.current.bucketPlan?.portfolio_id).toBe('B')
  })
})
