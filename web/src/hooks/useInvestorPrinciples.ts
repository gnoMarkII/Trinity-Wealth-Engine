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
  ActualGoalItemDTO,
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

  // Portfolio Goals
  const [portfolioGoals, setPortfolioGoals] = useState<ActualGoalItemDTO[]>([])

  const isMountedRef = useRef(true)
  const prevPortfolioIdRef = useRef<string>(portfolioId)
  const portfolioFetchGenRef = useRef<number>(0)

  useEffect(() => {
    isMountedRef.current = true
    return () => {
      isMountedRef.current = false
    }
  }, [])

  // Immediately isolate and reset portfolio-scoped state when portfolioId changes
  useEffect(() => {
    if (prevPortfolioIdRef.current !== portfolioId) {
      prevPortfolioIdRef.current = portfolioId
      portfolioFetchGenRef.current += 1
      setFinancialContext(null)
      setAxisDraft(null)
      setConfirmedAxis(null)
      setBucketPlan(null)
      setPreview(null)
      setApplyReceipt(null)
      setPortfolioGoals([])
    }
  }, [portfolioId])

  // ---------------------------------------------------------------------------
  // Initial Data Fetching
  // ---------------------------------------------------------------------------
  const refreshAll = useCallback(async () => {
    const currentGen = ++portfolioFetchGenRef.current
    const targetPortfolioId = portfolioId

    setLoading(true)
    setError(null)
    try {
      // 1. Load interview config & current confirmed essence (workspace-scoped)
      const [configRes, currEssence] = await Promise.allSettled([
        api.getInterviewConfig(),
        api.getCurrentConfirmedEssence(),
      ])

      if (isMountedRef.current) {
        if (configRes.status === 'fulfilled') {
          setInterviewConfig(configRes.value)
        }
        if (currEssence.status === 'fulfilled') {
          setConfirmedEssence(currEssence.value as ConfirmedEssenceDTO)
        } else {
          setConfirmedEssence(null)
        }
      }

      // 2. Try loading active session (workspace-scoped)
      let activeSession: SessionResponseDTO | null = null
      try {
        const sessRes = await api.getCurrentEssenceSession('workspace')
        if (isMountedRef.current) {
          activeSession = sessRes
          setSession(sessRes)
          if (sessRes.is_complete) {
            try {
              const sumRes = await api.getEssenceSummary(sessRes.session_id)
              if (isMountedRef.current) setSummary(sumRes)
            } catch {
              if (isMountedRef.current) setSummary(null)
            }
          }
        }
      } catch {
        if (isMountedRef.current) {
          setSession(null)
          setSummary(null)
        }
      }

      // If an interview is currently in progress, downstream planning data must NOT be loaded or shown
      if (activeSession && !activeSession.is_complete) {
        if (isMountedRef.current && currentGen === portfolioFetchGenRef.current) {
          setConfirmedEssence(null)
          setConfirmedAxis(null)
          setAxisDraft(null)
          setBucketPlan(null)
          setPreview(null)
          setApplyReceipt(null)
        }
        return
      }

      // 3. Try loading portfolio-scoped data for this portfolio
      if (targetPortfolioId) {
        try {
          const fc = await api.getFinancialContext(targetPortfolioId)
          if (isMountedRef.current && currentGen === portfolioFetchGenRef.current) {
            setFinancialContext(fc || null)
          }
        } catch {
          if (isMountedRef.current && currentGen === portfolioFetchGenRef.current) {
            setFinancialContext(null)
          }
        }

        try {
          const ca = await api.getCurrentConfirmedAxis(targetPortfolioId)
          if (isMountedRef.current && currentGen === portfolioFetchGenRef.current) {
            setConfirmedAxis((ca as ConfirmedAxisDTO) || null)
          }
        } catch {
          if (isMountedRef.current && currentGen === portfolioFetchGenRef.current) {
            setConfirmedAxis(null)
          }
        }

        try {
          const draft = await api.getLatestAxisDraft(targetPortfolioId)
          if (isMountedRef.current && currentGen === portfolioFetchGenRef.current) {
            setAxisDraft(draft || null)
          }
        } catch {
          if (isMountedRef.current && currentGen === portfolioFetchGenRef.current) {
            setAxisDraft(null)
          }
        }

        try {
          const bp = await api.getLatestBucketPlan(targetPortfolioId)
          if (isMountedRef.current && currentGen === portfolioFetchGenRef.current) {
            setBucketPlan(bp || null)
            if (bp) {
              try {
                const prev = await api.previewBucketPlan(bp.draft_id)
                if (isMountedRef.current && currentGen === portfolioFetchGenRef.current) {
                  setPreview(prev || null)
                }
              } catch {
                if (isMountedRef.current && currentGen === portfolioFetchGenRef.current) {
                  setPreview(null)
                }
              }
            } else {
              setPreview(null)
            }
          }
        } catch {
          if (isMountedRef.current && currentGen === portfolioFetchGenRef.current) {
            setBucketPlan(null)
            setPreview(null)
          }
        }

        try {
          const goalsRes = await api.getActualGoals()
          if (isMountedRef.current && currentGen === portfolioFetchGenRef.current) {
            if (goalsRes?.goals && goalsRes.goals.length > 0) {
              setPortfolioGoals(goalsRes.goals)
            } else {
              const fallbackRes = await api.getActualGoals(targetPortfolioId)
              setPortfolioGoals(fallbackRes?.goals || [])
            }
          }
        } catch {
          if (isMountedRef.current && currentGen === portfolioFetchGenRef.current) {
            setPortfolioGoals([])
          }
        }
      }
    } catch (err: any) {
      if (isMountedRef.current && currentGen === portfolioFetchGenRef.current) {
        setError(err?.message || 'เกิดข้อผิดพลาดในการโหลดข้อมูลหลักการลงทุน')
      }
    } finally {
      if (isMountedRef.current && currentGen === portfolioFetchGenRef.current) {
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
      const newSess = await api.startEssenceSession({
        scope: 'workspace',
        prompt_version: '1.0',
        portfolio_id: portfolioId,
        clear_downstream: true,
      })
      setSession(newSess)
      setSummary(null)
      setConfirmedEssence(null)
      setAxisDraft(null)
      setConfirmedAxis(null)
      setBucketPlan(null)
      setPreview(null)
      setApplyReceipt(null)
      setActiveMilestone('essence')
    } catch (err: any) {
      setError(err?.message || 'ไม่สามารถเริ่มการค้นหาแก่นแท้ได้')
    } finally {
      setActionLoading(false)
      setActionStatusText(null)
    }
  }, [portfolioId])

  const resetPrinciples = useCallback(async () => {
    setActionLoading(true)
    setActionStatusText('กำลังล้างข้อมูลแกนหลักและ Buckets...')
    setError(null)
    try {
      await api.resetPrinciples(portfolioId)
      setConfirmedAxis(null)
      setAxisDraft(null)
      setBucketPlan(null)
      setPreview(null)
      setApplyReceipt(null)
    } catch (err: any) {
      setError(err?.message || 'ไม่สามารถล้างข้อมูลแกนหลักและ Buckets ได้')
    } finally {
      setActionLoading(false)
      setActionStatusText(null)
    }
  }, [portfolioId])

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

  const advanceNextQuestion = useCallback(async () => {
    if (!session) return
    const nextSeq = (session.answers_count || 0) + 1
    setActionLoading(true)
    setActionStatusText(`กำลังเตรียมคำถามที่ ${nextSeq}...`)
    setError(null)
    try {
      const nextSess = await api.advanceEssenceQuestion(session.session_id, {
        expected_revision: session.revision,
      })
      setSession(nextSess)
    } catch (err: any) {
      setError(err?.message || 'เกิดข้อผิดพลาดในการสร้างคำถามถัดไป')
    } finally {
      setActionLoading(false)
      setActionStatusText(null)
    }
  }, [session])

  const generateSummary = useCallback(async () => {
    if (!session) return
    setActionLoading(true)
    setActionStatusText('กำลังวิเคราะห์และสังเคราะห์แก่นแท้การลงทุน...')
    setError(null)
    try {
      const sumRes = await api.summarizeEssence(session.session_id, {
        expected_revision: session.revision,
      })
      setSummary(sumRes)
      const finalizedSess = await api.getEssenceSession(session.session_id)
      setSession(finalizedSess)
    } catch (err: any) {
      setError(err?.message || 'เกิดข้อผิดพลาดในการสังเคราะห์แก่นแท้')
    } finally {
      setActionLoading(false)
      setActionStatusText(null)
    }
  }, [session])

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
      if (!portfolioId) {
        const err = new Error('กรุณาเลือกพอร์ตการลงทุน')
        setError(err.message)
        throw err
      }
      setActionLoading(true)
      setActionStatusText('กำลังบันทึกบริบททางการเงิน...')
      setError(null)
      try {
        const fc = await api.updateFinancialContext(portfolioId, updates)
        setFinancialContext(fc)
        return fc
      } catch (err: any) {
        const msg = err?.message || 'ไม่สามารถบันทึกบริบททางการเงินได้'
        setError(msg)
        throw err
      } finally {
        setActionLoading(false)
        setActionStatusText(null)
      }
    },
    [portfolioId],
  )

  const syncFinancialContextFromGoals = useCallback(async () => {
    if (!portfolioId) {
      const err = new Error('กรุณาเลือกพอร์ตการลงทุน')
      setError(err.message)
      throw err
    }
    setActionLoading(true)
    setActionStatusText('กำลังซิงค์ข้อมูลจากเป้าหมาย (Goals)...')
    setError(null)
    try {
      const fc = await api.syncFinancialContextFromGoals(portfolioId)
      setFinancialContext(fc)
      return fc
    } catch (err: any) {
      const msg = err?.message || 'ไม่สามารถซิงค์ข้อมูลจากเป้าหมายได้'
      setError(msg)
      throw err
    } finally {
      setActionLoading(false)
      setActionStatusText(null)
    }
  }, [portfolioId])

  const generateAxisDraft = useCallback(async () => {
    if (!portfolioId) {
      const err = new Error('กรุณาเลือกพอร์ตการลงทุน')
      setError(err.message)
      throw err
    }
    setActionLoading(true)
    setActionStatusText('กำลังร่างแกนหลักการลงทุนครบ 8 หัวข้อตามคำสั่งสอง...')
    setError(null)
    try {
      const draft = await api.createAxisDraft(portfolioId)
      setAxisDraft(draft)
      return draft
    } catch (err: any) {
      const msg = err?.message || 'ไม่สามารถสร้างร่างแกนหลักการลงทุนได้'
      setError(msg)
      throw err
    } finally {
      setActionLoading(false)
      setActionStatusText(null)
    }
  }, [portfolioId])

  const updateAxis = useCallback(
    async (updates: UpdateAxisDraftRequestDTO) => {
      let draft = axisDraft
      if (portfolioId) {
        try {
          const latest = await api.getLatestAxisDraft(portfolioId)
          if (latest) {
            draft = latest
            setAxisDraft(latest)
          }
        } catch {
          // ignore
        }
      }
      if (!draft) {
        const err = new Error('ไม่พบแบบร่างแกนหลักสำหรับพอร์ตนี้')
        setError(err.message)
        throw err
      }
      if (draft.portfolio_id !== portfolioId) {
        const err = new Error(`ข้อมูลร่างแกนหลักไม่ตรงกับพอร์ตปัจจุบัน (${draft.portfolio_id} !== ${portfolioId})`)
        setError(err.message)
        throw err
      }
      setActionLoading(true)
      setError(null)
      try {
        const updated = await api.updateAxisDraft(draft.draft_id, {
          ...updates,
          expected_revision: draft.revision,
        })
        setAxisDraft(updated)
        return updated
      } catch (err: any) {
        if (portfolioId) {
          try {
            const latest = await api.getLatestAxisDraft(portfolioId)
            if (latest) setAxisDraft(latest)
          } catch {
            // ignore
          }
        }
        const msg = err?.message || 'ไม่สามารถแก้ไขร่างแกนหลักได้'
        setError(msg)
        throw err
      } finally {
        setActionLoading(false)
      }
    },
    [axisDraft, portfolioId],
  )

  const confirmAxis = useCallback(
    async (proceedToBuckets: boolean = false, targetDraft?: AxisDraftDTO) => {
      let draft = targetDraft || axisDraft
      if (portfolioId) {
        try {
          const latest = await api.getLatestAxisDraft(portfolioId)
          if (latest) {
            draft = latest
            setAxisDraft(latest)
          }
        } catch {
          // ignore
        }
      }
      if (!draft) {
        const err = new Error('ไม่พบแบบร่างแกนหลักสำหรับพอร์ตนี้')
        setError(err.message)
        throw err
      }
      if (draft.portfolio_id !== portfolioId) {
        const err = new Error(`ข้อมูลร่างแกนหลักไม่ตรงกับพอร์ตปัจจุบัน (${draft.portfolio_id} !== ${portfolioId})`)
        setError(err.message)
        throw err
      }
      setActionLoading(true)
      setActionStatusText('กำลังตรวจสอบความครบถ้วนและยืนยันแกนหลักการลงทุน...')
      setError(null)
      try {
        await api.confirmInvestmentAxis(draft.draft_id, {
          idempotency_key: `confirm_axis_${draft.draft_id}_${draft.revision}`,
          expected_draft_revision: draft.revision,
        })
        const refreshed = await api.getCurrentConfirmedAxis(portfolioId)
        setConfirmedAxis(refreshed as ConfirmedAxisDTO)

        if (proceedToBuckets) {
          setActiveMilestone('buckets')
        }
        return refreshed as ConfirmedAxisDTO
      } catch (err: any) {
        if (portfolioId) {
          try {
            const latest = await api.getLatestAxisDraft(portfolioId)
            if (latest) setAxisDraft(latest)
          } catch {
            // ignore
          }
        }
        const msg = err?.message || 'เกิดข้อผิดพลาดในการยืนยันแกนหลัก'
        setError(msg)
        throw err
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
    if (!portfolioId) {
      const err = new Error('กรุณาเลือกพอร์ตการลงทุน')
      setError(err.message)
      throw err
    }
    setActionLoading(true)
    setActionStatusText('กำลังออกแบบ Purpose Buckets และสัดส่วนจากแกนหลัก...')
    setError(null)
    setApplyReceipt(null)
    try {
      const plan = await api.createBucketPlan(portfolioId)
      setBucketPlan(plan)
      // Automatically load preview
      try {
        const prev = await api.previewBucketPlan(plan.draft_id)
        setPreview(prev)
      } catch {
        setPreview(null)
      }
      return plan
    } catch (err: any) {
      const msg = err?.message || 'ไม่สามารถสร้างร่าง Purpose Buckets ได้'
      setError(msg)
      throw err
    } finally {
      setActionLoading(false)
      setActionStatusText(null)
    }
  }, [portfolioId])

  const updateBucketPlan = useCallback(
    async (updates: UpdateBucketPlanRequestDTO) => {
      if (!bucketPlan) {
        const err = new Error('ไม่พบแบบร่าง Purpose Buckets สำหรับพอร์ตนี้')
        setError(err.message)
        throw err
      }
      if (bucketPlan.portfolio_id !== portfolioId) {
        const err = new Error(`ไม่สามารถแก้ไข Buckets: ข้อมูลแบบร่างเป็นของพอร์ต ${bucketPlan.portfolio_id} ไม่ตรงกับพอร์ตปัจจุบัน ${portfolioId}`)
        setError(err.message)
        throw err
      }
      setActionLoading(true)
      setError(null)
      try {
        const updated = await api.updateBucketPlan(bucketPlan.draft_id, {
          ...updates,
          expected_revision: bucketPlan.revision,
        })
        setBucketPlan(updated)
        // Refresh preview with new values
        try {
          const prev = await api.previewBucketPlan(bucketPlan.draft_id)
          setPreview(prev)
        } catch {
          setPreview(null)
        }
        return updated
      } catch (err: any) {
        const msg = err?.message || 'ไม่สามารถปรับเปลี่ยน Purpose Buckets ได้'
        setError(msg)
        throw err
      } finally {
        setActionLoading(false)
      }
    },
    [bucketPlan, portfolioId],
  )

  const loadPreview = useCallback(async () => {
    if (!bucketPlan) return
    if (bucketPlan.portfolio_id !== portfolioId) {
      const err = new Error('แบบร่างไม่ตรงกับพอร์ตปัจจุบัน')
      setError(err.message)
      throw err
    }
    setActionLoading(true)
    setError(null)
    try {
      const prev = await api.previewBucketPlan(bucketPlan.draft_id)
      setPreview(prev)
      return prev
    } catch (err: any) {
      const msg = err?.message || 'ไม่สามารถดูตัวอย่างการจัดสัดส่วนได้'
      setError(msg)
      throw err
    } finally {
      setActionLoading(false)
    }
  }, [bucketPlan, portfolioId])

  const applyBucketPlan = useCallback(
    async (idempotencyKey?: string) => {
      if (!bucketPlan) {
        const err = new Error('ไม่พบแบบร่าง Purpose Buckets')
        setError(err.message)
        throw err
      }
      if (bucketPlan.portfolio_id !== portfolioId) {
        const err = new Error(`ไม่สามารถนำไปใช้: แบบร่างเป็นของพอร์ต ${bucketPlan.portfolio_id} ไม่ตรงกับ ${portfolioId}`)
        setError(err.message)
        throw err
      }
      setActionLoading(true)
      setActionStatusText('กำลังนำ Purpose Buckets ไปใช้กับพอร์ตการลงทุนอย่างปลอดภัย...')
      setError(null)
      try {
        const key = idempotencyKey || `apply_plan_${bucketPlan.draft_id}_${bucketPlan.revision}`
        const receipt = await api.applyBucketPlan(bucketPlan.draft_id, {
          idempotency_key: key,
          expected_draft_revision: bucketPlan.revision,
          expected_checkpoint_sequence: preview?.checkpoint_sequence,
          expected_checkpoint_state_hash: preview?.checkpoint_state_hash,
          preview_id: preview?.preview_id,
          payload_hash: preview?.payload_hash,
        })
        setApplyReceipt(receipt)
        onPortfolioChangeNeeded?.()
        return receipt
      } catch (err: any) {
        const msg = err?.message || 'เกิดข้อผิดพลาดในการนำ Buckets ไปใช้กับพอร์ต'
        setError(msg)
        throw err
      } finally {
        setActionLoading(false)
        setActionStatusText(null)
      }
    },
    [bucketPlan, portfolioId, preview, onPortfolioChangeNeeded],
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
    resetPrinciples,
    recordAndNext,
    advanceNextQuestion,
    generateSummary,
    rateClaim,

    editClaim,
    excludeClaim,
    confirmEssenceBatch,

    // Milestone 2 (Financial Context & Axis)
    financialContext,
    axisDraft,
    confirmedAxis,
    saveFinancialContext,
    syncFinancialContextFromGoals,
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

    // Portfolio Goals
    portfolioGoals,

    // Refresh
    refreshAll,
  }
}
