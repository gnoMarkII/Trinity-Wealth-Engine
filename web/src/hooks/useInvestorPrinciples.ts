import { useState, useEffect, useCallback, useRef } from 'react'
import { api } from '../api/client'
import type {
  SessionResponseDTO,
  SummaryResponseDTO,
  ConfirmedEssenceDTO,
  FinancialContextDTO,
  UpdateFinancialContextRequestDTO,
  AxisDraftDTO,
  UpdateAxisDraftRequestDTO,
  ConfirmedAxisDTO,
  BucketPlanDraftDTO,
  UpdateBucketPlanRequestDTO,
  AllocationPreviewDTO,
  AllocationApplyReceiptDTO,
  InterviewConfigDTO,
} from '../api/types'

export type PrinciplesMilestone = 'essence' | 'axis' | 'buckets'

interface UseInvestorPrinciplesProps {
  portfolioId: string
  initialMilestone?: PrinciplesMilestone
  onPortfolioChangeNeeded?: () => void
}

export function useInvestorPrinciples({
  portfolioId,
  initialMilestone = 'essence',
  onPortfolioChangeNeeded,
}: UseInvestorPrinciplesProps) {
  const [activeMilestone, setActiveMilestone] = useState<PrinciplesMilestone>(initialMilestone)
  const [loading, setLoading] = useState(true)
  const [actionLoading, setActionLoading] = useState(false)
  const [actionStatusText, setActionStatusText] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Milestone 1: Essence
  const [interviewConfig, setInterviewConfig] = useState<InterviewConfigDTO | null>(null)
  const [session, setSession] = useState<SessionResponseDTO | null>(null)
  const [summary, setSummary] = useState<SummaryResponseDTO | null>(null)
  const [confirmedEssence, setConfirmedEssence] = useState<ConfirmedEssenceDTO | null>(null)

  // Milestone 2: Financial Context & Axis
  const [financialContext, setFinancialContext] = useState<FinancialContextDTO | null>(null)
  const [axisDraft, setAxisDraft] = useState<AxisDraftDTO | null>(null)
  const [confirmedAxis, setConfirmedAxis] = useState<ConfirmedAxisDTO | null>(null)

  // Milestone 3: Purpose Buckets
  const [bucketPlan, setBucketPlan] = useState<BucketPlanDraftDTO | null>(null)
  const [preview, setPreview] = useState<AllocationPreviewDTO | null>(null)
  const [applyReceipt, setApplyReceipt] = useState<AllocationApplyReceiptDTO | null>(null)

  const isMountedRef = useRef(true)
  useEffect(() => {
    isMountedRef.current = true
    return () => {
      isMountedRef.current = false
    }
  }, [])

  // ---------------------------------------------------------------------------
  // Initial Data Fetching
  // ---------------------------------------------------------------------------
  const refreshAll = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      // 1. Load interview config & current confirmed essence
      const [configRes, currEssence] = await Promise.allSettled([
        api.getInterviewConfig(),
        api.getCurrentConfirmedEssence(),
      ])

      if (configRes.status === 'fulfilled') {
        setInterviewConfig(configRes.value)
      }
      if (currEssence.status === 'fulfilled') {
        setConfirmedEssence(currEssence.value as ConfirmedEssenceDTO)
      } else {
        setConfirmedEssence(null)
      }

      // 2. Try loading active session
      try {
        const sessRes = await api.getCurrentEssenceSession('workspace')
        setSession(sessRes)
        if (sessRes.is_complete) {
          try {
            const sumRes = await api.getEssenceSummary(sessRes.session_id)
            setSummary(sumRes)
          } catch {
            setSummary(null)
          }
        }
      } catch {
        setSession(null)
        setSummary(null)
      }

      // 3. Try loading confirmed axis & financial context for this portfolio
      if (portfolioId) {
        try {
          const fc = await api.getFinancialContext(portfolioId)
          setFinancialContext(fc)
        } catch {
          setFinancialContext(null)
        }

        try {
          const ca = await api.getCurrentConfirmedAxis(portfolioId)
          setConfirmedAxis(ca as ConfirmedAxisDTO)
        } catch {
          setConfirmedAxis(null)
        }
      }
    } catch (err: any) {
      setError(err?.message || 'เกิดข้อผิดพลาดในการโหลดข้อมูลหลักการลงทุน')
    } finally {
      if (isMountedRef.current) {
        setLoading(false)
      }
    }
  }, [portfolioId])

  useEffect(() => {
    void refreshAll()
  }, [refreshAll])

  // ---------------------------------------------------------------------------
  // Milestone 1 Actions: Essence
  // ---------------------------------------------------------------------------
  const startNewSession = useCallback(async () => {
    setActionLoading(true)
    setActionStatusText('กำลังเตรียมชุดคำถามแรก...')
    setError(null)
    try {
      const newSess = await api.startEssenceSession({ scope: 'workspace', prompt_version: '1.0' })
      setSession(newSess)
      setSummary(null)
      setActiveMilestone('essence')
    } catch (err: any) {
      setError(err?.message || 'ไม่สามารถเริ่มการค้นหาแก่นแท้ได้')
    } finally {
      setActionLoading(false)
      setActionStatusText(null)
    }
  }, [])

  const recordAndNext = useCallback(
    async (
      answerKind: 'choice' | 'free_text' | 'unsure' | 'skipped',
      optionId?: string | null,
      freeText?: string | null,
    ) => {
      if (!session || !session.current_question) return
      const currentQ = session.current_question
      setActionLoading(true)
      setActionStatusText('บันทึกแล้ว กำลังเตรียมคำถามถัดไป...')
      setError(null)

      try {
        // Step 1: Record answer
        const updatedSess = await api.recordEssenceAnswer(session.session_id, currentQ.question_id, {
          answer_kind: answerKind,
          option_id: optionId ?? null,
          free_text: freeText ?? null,
          expected_revision: session.revision,
        })
        setSession(updatedSess)

        // Step 2: Next action based on sequence
        if (currentQ.sequence_no < 10) {
          const nextSess = await api.advanceEssenceQuestion(session.session_id, {
            expected_revision: updatedSess.revision,
          })
          setSession(nextSess)
        } else {
          // Final question (Q10) -> Summarize
          setActionStatusText('กำลังวิเคราะห์และสังเคราะห์แก่นแท้การลงทุน...')
          const sumRes = await api.summarizeEssence(session.session_id, {
            expected_revision: updatedSess.revision,
          })
          setSummary(sumRes)
          // Refresh session to reflect completion
          const finalizedSess = await api.getEssenceSession(session.session_id)
          setSession(finalizedSess)
        }
      } catch (err: any) {
        setError(err?.message || 'เกิดข้อผิดพลาดในการบันทึกคำตอบหรือสร้างคำถามถัดไป')
      } finally {
        setActionLoading(false)
        setActionStatusText(null)
      }
    },
    [session],
  )

  const rateClaim = useCallback(
    async (claimId: string, fitRating: 'exact' | 'partial' | 'rejected') => {
      if (!session || !summary) return
      setActionLoading(true)
      setError(null)
      try {
        const updatedSummary = await api.reviewEssenceClaim(session.session_id, claimId, {
          fit_rating: fitRating,
          expected_revision: summary.revision,
        })
        setSummary(updatedSummary)
      } catch (err: any) {
        setError(err?.message || 'ไม่สามารถบันทึกการประเมินข้อสรุปได้')
      } finally {
        setActionLoading(false)
      }
    },
    [session, summary],
  )

  const editClaim = useCallback(
    async (claimId: string, editedText: string) => {
      if (!session || !summary) return
      setActionLoading(true)
      setError(null)
      try {
        const updatedSummary = await api.reviewEssenceClaim(session.session_id, claimId, {
          edited_text: editedText,
          expected_revision: summary.revision,
        })
        setSummary(updatedSummary)
      } catch (err: any) {
        setError(err?.message || 'ไม่สามารถแก้ไขข้อความได้')
      } finally {
        setActionLoading(false)
      }
    },
    [session, summary],
  )

  const excludeClaim = useCallback(
    async (claimId: string) => {
      if (!session || !summary) return
      setActionLoading(true)
      setError(null)
      try {
        const updatedSummary = await api.reviewEssenceClaim(session.session_id, claimId, {
          is_excluded: true,
          expected_revision: summary.revision,
        })
        setSummary(updatedSummary)
      } catch (err: any) {
        setError(err?.message || 'ไม่สามารถตัดข้อสรุปออกได้')
      } finally {
        setActionLoading(false)
      }
    },
    [session, summary],
  )

  const confirmEssenceBatch = useCallback(
    async (proceedToAxis: boolean = false) => {
      if (!session || !summary) return
      setActionLoading(true)
      setActionStatusText('กำลังยืนยันและบันทึกแก่นแท้การลงทุน...')
      setError(null)

      try {
        // Find all accepted claim IDs (exact, user_edited, or not rejected)
        const acceptedIds = summary.claims
          .filter((c) => c.fit_rating !== 'rejected')
          .map((c) => c.claim_id)

        await api.confirmEssence(session.session_id, {
          accepted_claim_ids: acceptedIds,
          expected_summary_revision: summary.revision,
          idempotency_key: `confirm_essence_${session.session_id}_${summary.revision}`,
        })

        const refreshed = await api.getCurrentConfirmedEssence()
        setConfirmedEssence(refreshed as ConfirmedEssenceDTO)

        if (proceedToAxis) {
          setActiveMilestone('axis')
        }
      } catch (err: any) {
        setError(err?.message || 'เกิดข้อผิดพลาดในการยืนยันแก่นแท้')
      } finally {
        setActionLoading(false)
        setActionStatusText(null)
      }
    },
    [session, summary],
  )

  // ---------------------------------------------------------------------------
  // Milestone 2 Actions: Financial Context & Axis
  // ---------------------------------------------------------------------------
  const saveFinancialContext = useCallback(
    async (updates: UpdateFinancialContextRequestDTO) => {
      if (!portfolioId) return
      setActionLoading(true)
      setActionStatusText('กำลังบันทึกบริบททางการเงิน...')
      setError(null)
      try {
        const fc = await api.updateFinancialContext(portfolioId, updates)
        setFinancialContext(fc)
      } catch (err: any) {
        setError(err?.message || 'ไม่สามารถบันทึกบริบททางการเงินได้')
      } finally {
        setActionLoading(false)
        setActionStatusText(null)
      }
    },
    [portfolioId],
  )

  const generateAxisDraft = useCallback(async () => {
    if (!portfolioId) return
    setActionLoading(true)
    setActionStatusText('กำลังร่างแกนหลักการลงทุนครบ 8 หัวข้อตามคำสั่งสอง...')
    setError(null)
    try {
      const draft = await api.createAxisDraft(portfolioId)
      setAxisDraft(draft)
    } catch (err: any) {
      setError(err?.message || 'ไม่สามารถสร้างร่างแกนหลักการลงทุนได้')
    } finally {
      setActionLoading(false)
      setActionStatusText(null)
    }
  }, [portfolioId])

  const updateAxis = useCallback(
    async (updates: UpdateAxisDraftRequestDTO) => {
      if (!axisDraft) return
      setActionLoading(true)
      setError(null)
      try {
        const updated = await api.updateAxisDraft(axisDraft.draft_id, {
          ...updates,
          expected_revision: axisDraft.revision,
        })
        setAxisDraft(updated)
      } catch (err: any) {
        setError(err?.message || 'ไม่สามารถแก้ไขร่างแกนหลักได้')
      } finally {
        setActionLoading(false)
      }
    },
    [axisDraft],
  )

  const confirmAxis = useCallback(
    async (proceedToBuckets: boolean = false) => {
      if (!axisDraft) return
      setActionLoading(true)
      setActionStatusText('กำลังตรวจสอบความครบถ้วนและยืนยันแกนหลักการลงทุน...')
      setError(null)
      try {
        await api.confirmInvestmentAxis(axisDraft.draft_id, {
          idempotency_key: `confirm_axis_${axisDraft.draft_id}_${axisDraft.revision}`,
        })
        const refreshed = await api.getCurrentConfirmedAxis(portfolioId)
        setConfirmedAxis(refreshed as ConfirmedAxisDTO)

        if (proceedToBuckets) {
          setActiveMilestone('buckets')
        }
      } catch (err: any) {
        setError(err?.message || 'เกิดข้อผิดพลาดในการยืนยันแกนหลัก')
      } finally {
        setActionLoading(false)
        setActionStatusText(null)
      }
    },
    [axisDraft, portfolioId],
  )

  // ---------------------------------------------------------------------------
  // Milestone 3 Actions: Purpose Buckets
  // ---------------------------------------------------------------------------
  const generateBucketPlan = useCallback(async () => {
    if (!portfolioId) return
    setActionLoading(true)
    setActionStatusText('กำลังออกแบบ Purpose Buckets และสัดส่วนจากแกนหลัก...')
    setError(null)
    try {
      const plan = await api.createBucketPlan(portfolioId)
      setBucketPlan(plan)
      // Automatically load preview
      const prev = await api.previewBucketPlan(plan.draft_id)
      setPreview(prev)
    } catch (err: any) {
      setError(err?.message || 'ไม่สามารถสร้างร่าง Purpose Buckets ได้')
    } finally {
      setActionLoading(false)
      setActionStatusText(null)
    }
  }, [portfolioId])

  const updateBucketPlan = useCallback(
    async (updates: UpdateBucketPlanRequestDTO) => {
      if (!bucketPlan) return
      setActionLoading(true)
      setError(null)
      try {
        const updated = await api.updateBucketPlan(bucketPlan.draft_id, {
          ...updates,
          expected_revision: bucketPlan.revision,
        })
        setBucketPlan(updated)
        // Refresh preview with new values
        const prev = await api.previewBucketPlan(bucketPlan.draft_id)
        setPreview(prev)
      } catch (err: any) {
        setError(err?.message || 'ไม่สามารถปรับเปลี่ยน Purpose Buckets ได้')
      } finally {
        setActionLoading(false)
      }
    },
    [bucketPlan],
  )

  const loadPreview = useCallback(async () => {
    if (!bucketPlan) return
    setActionLoading(true)
    setError(null)
    try {
      const prev = await api.previewBucketPlan(bucketPlan.draft_id)
      setPreview(prev)
    } catch (err: any) {
      setError(err?.message || 'ไม่สามารถดูตัวอย่างการจัดสัดส่วนได้')
    } finally {
      setActionLoading(false)
    }
  }, [bucketPlan])

  const applyBucketPlan = useCallback(
    async (idempotencyKey?: string) => {
      if (!bucketPlan) return
      setActionLoading(true)
      setActionStatusText('กำลังนำ Purpose Buckets ไปใช้กับพอร์ตการลงทุนอย่างปลอดภัย...')
      setError(null)
      try {
        const key = idempotencyKey || `apply_plan_${bucketPlan.draft_id}_${bucketPlan.revision}`
        const receipt = await api.applyBucketPlan(bucketPlan.draft_id, key)
        setApplyReceipt(receipt)
        onPortfolioChangeNeeded?.()
      } catch (err: any) {
        setError(err?.message || 'เกิดข้อผิดพลาดในการนำ Buckets ไปใช้กับพอร์ต')
      } finally {
        setActionLoading(false)
        setActionStatusText(null)
      }
    },
    [bucketPlan, onPortfolioChangeNeeded],
  )

  return {
    // State
    portfolioId,
    activeMilestone,
    setActiveMilestone,
    loading,
    actionLoading,
    actionStatusText,
    error,
    clearError: () => setError(null),

    // Milestone 1 (Essence)
    interviewConfig,
    session,
    summary,
    confirmedEssence,
    startNewSession,
    recordAndNext,
    rateClaim,
    editClaim,
    excludeClaim,
    confirmEssenceBatch,

    // Milestone 2 (Financial Context & Axis)
    financialContext,
    axisDraft,
    confirmedAxis,
    saveFinancialContext,
    generateAxisDraft,
    updateAxis,
    confirmAxis,

    // Milestone 3 (Purpose Buckets)
    bucketPlan,
    preview,
    applyReceipt,
    generateBucketPlan,
    updateBucketPlan,
    loadPreview,
    applyBucketPlan,
    resetApplyReceipt: () => setApplyReceipt(null),

    // Refresh
    refreshAll,
  }
}
