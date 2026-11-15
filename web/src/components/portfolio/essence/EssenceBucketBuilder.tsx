import { useState, useEffect, useMemo } from 'react'
import AutoResizeTextarea from '../../ui/AutoResizeTextarea'
import {
  AlertTriangleIcon,
  BoltStrategyIcon,
  BriefcasePortfolioIcon,
  BulbIcon,
  CheckCircleIcon,
  CheckSmallIcon,
  CloseIcon,
  CpuIcon,
  FolderIcon,
  ShieldCheckIcon,
  SlidersIcon,
  SparklesIcon,
  TargetIcon,
  TrashIcon,
} from '../../ui/AppIcons'
import type {
  BucketPlanDraftDTO,
  UpdateBucketPlanRequestDTO,
  PurposeBucketDTO,
} from '../../../api/types'
import { VIBRANT_BUCKET_PALETTE } from '../../../lib/bucketColors'

interface EssenceBucketBuilderProps {
  draft: BucketPlanDraftDTO
  portfolioId: string
  portfolioName?: string
  onUpdateDraft: (updates: UpdateBucketPlanRequestDTO) => Promise<any>
  onProceedToReview: () => void
  onRegenerateBuckets?: () => Promise<any>
  loading: boolean
  statusText?: string | null
}

export default function EssenceBucketBuilder({
  draft,
  portfolioId,
  portfolioName,
  onUpdateDraft,
  onProceedToReview,
  onRegenerateBuckets,
  loading,
  statusText,
}: EssenceBucketBuilderProps) {
  const displayName = portfolioName || (portfolioId === 'default' ? 'พอร์ตลงทุนหลัก' : portfolioId)
  const [buckets, setBuckets] = useState<PurposeBucketDTO[]>(draft.purpose_buckets || [])

  // Input states for adding new asset or action rule per bucket
  const [newAssetInputs, setNewAssetInputs] = useState<Record<string, string>>({})
  const [newRuleInputs, setNewRuleInputs] = useState<Record<string, string>>({})

  useEffect(() => {
    setBuckets(draft.purpose_buckets || [])
  }, [draft])

  const totalPercent = useMemo(() => {
    return buckets.reduce((sum, b) => sum + (parseFloat(b.target_percent) || 0), 0)
  }, [buckets])

  const is100Percent = Math.abs(totalPercent - 100) < 0.05

  const navThb = parseFloat(draft.portfolio_checkpoint?.nav_thb || '0')

  const handleUpdateBucket = (
    index: number,
    field: keyof PurposeBucketDTO,
    value: any,
  ) => {
    const updated = [...buckets]
    const current = updated[index]
    if (!current) return
    updated[index] = { ...current, [field]: value }
    setBuckets(updated)
  }

  const handleAddBucket = () => {
    const nextIdx = buckets.length + 1
    const color = VIBRANT_BUCKET_PALETTE[buckets.length % VIBRANT_BUCKET_PALETTE.length] || '#3b82f6'
    const newB: PurposeBucketDTO = {
      bucket_id: `b_purpose_${Date.now()}`,
      name: `พอร์ตย่อย ${nextIdx}`,
      role: 'เพื่อสร้างผลตอบแทนและบริหารความเสี่ยงตามเป้าหมาย',
      color,
      target_percent: '0',
      source_value_ids: [],
      source_axis_allocation_ids: [],
      linked_goal_name: '',
      eligible_assets: ['สินทรัพย์คุณภาพดีตามแกนหลัก'],
      cash_inflow_routing_pct: '0',
      action_rules: ['ปฏิบัติตามกฎเหล็กในแกนหลักการลงทุน'],
    }
    setBuckets([...buckets, newB])
  }

  const handleRemoveBucket = (index: number) => {
    if (buckets.length <= 1) return
    setBuckets(buckets.filter((_, i) => i !== index))
  }

  const handleAddAsset = (bucketIdx: number, bucketId: string) => {
    const text = (newAssetInputs[bucketId] || '').trim()
    if (!text) return
    const b = buckets[bucketIdx]
    if (!b) return
    const currentAssets = b.eligible_assets || []
    if (!currentAssets.includes(text)) {
      handleUpdateBucket(bucketIdx, 'eligible_assets', [...currentAssets, text])
    }
    setNewAssetInputs({ ...newAssetInputs, [bucketId]: '' })
  }

  const handleRemoveAsset = (bucketIdx: number, assetIdx: number) => {
    const b = buckets[bucketIdx]
    if (!b) return
    const currentAssets = b.eligible_assets || []
    handleUpdateBucket(
      bucketIdx,
      'eligible_assets',
      currentAssets.filter((_, i) => i !== assetIdx),
    )
  }

  const handleAddRule = (bucketIdx: number, bucketId: string) => {
    const text = (newRuleInputs[bucketId] || '').trim()
    if (!text) return
    const b = buckets[bucketIdx]
    if (!b) return
    const currentRules = b.action_rules || []
    if (!currentRules.includes(text)) {
      handleUpdateBucket(bucketIdx, 'action_rules', [...currentRules, text])
    }
    setNewRuleInputs({ ...newRuleInputs, [bucketId]: '' })
  }

  const handleRemoveRule = (bucketIdx: number, ruleIdx: number) => {
    const b = buckets[bucketIdx]
    if (!b) return
    const currentRules = b.action_rules || []
    handleUpdateBucket(
      bucketIdx,
      'action_rules',
      currentRules.filter((_, i) => i !== ruleIdx),
    )
  }

  const handleCycleColor = (index: number) => {
    const current = buckets[index]
    if (!current) return
    const currentIdx = VIBRANT_BUCKET_PALETTE.indexOf(current.color)
    const nextColor =
      VIBRANT_BUCKET_PALETTE[(currentIdx + 1) % VIBRANT_BUCKET_PALETTE.length] || '#3b82f6'
    handleUpdateBucket(index, 'color', nextColor)
  }

  const applyPreset = (presetType: '3_buckets_standard' | '4_buckets_growth') => {
    const resolveId = (prefix: string, fallback: string) => {
      const existing = buckets.find((b) => b.bucket_id.startsWith(prefix) || b.bucket_id === fallback)
      return existing ? existing.bucket_id : fallback
    }

    if (presetType === '3_buckets_standard') {
      setBuckets([
        {
          bucket_id: resolveId('b_emergency', 'b_emergency'),
          name: 'เสบียงสำรอง & สภาพคล่อง',
          role: 'รองรับค่าใช้จ่ายฉุกเฉิน 6-12 เดือน ปกป้องเงินต้น ไม่ขาดทุน สภาพคล่องสูง',
          color: '#10B981',
          target_percent: '15',
          linked_goal_name: 'เงินสำรองฉุกเฉิน',
          eligible_assets: ['บัญชีเงินฝากดิจิทัลดอกเบี้ยสูง', 'กองทุนรวมตลาดเงิน (MMF)', 'ตราสารหนี้ระยะสั้นมาก'],
          cash_inflow_routing_pct: '20',
          action_rules: [
            'ห้ามนำเงินก้อนนี้ไปลงทุนในหุ้นหรือสินทรัพย์เสี่ยงเด็ดขาด',
            'เป็น Watermark แรกในการถอนเมื่อเกิดเหตุฉุกเฉิน',
            'เมื่อสะสมครบเป้าหมาย 300,000 บาทแล้ว ส่งต่อเงินออมใหม่ส่วนเกินไปที่ Bucket เติบโตความมั่งคั่ง',
          ],
          source_value_ids: [],
          source_axis_allocation_ids: [],
        },
        {
          bucket_id: resolveId('b_income', 'b_income'),
          name: 'ฐานรากกระแสเงินสด & กันชน',
          role: 'สร้างกระแสเงินสดสม่ำเสมอ ชนะเงินเฟ้อ และเป็นกันชนลดความผันผวนของพอร์ต',
          color: '#8B5CF6',
          target_percent: '25',
          linked_goal_name: 'กระแสเงินสดและกันชนความเสี่ยง',
          eligible_assets: ['กองทุนตราสารหนี้คุณภาพดี', 'หุ้นกู้ Investment Grade', 'กองทุนอสังหาริมทรัพย์ / REITs ปันผลสูง'],
          cash_inflow_routing_pct: '20',
          action_rules: [
            'นำกระแสเงินสดหรือเงินปันผลที่ได้กลับไป Reinvest ใน Bucket เติบโตระยะยาวอัตโนมัติ',
            'รักษาสภาพคล่องเพื่อพร้อมปรับสมดุล Rebalance สู่สินทรัพย์เติบโตเมื่อตลาดปรับฐาน',
          ],
          source_value_ids: [],
          source_axis_allocation_ids: [],
        },
        {
          bucket_id: resolveId('b_growth', 'b_growth'),
          name: 'กองหน้าขับเคลื่อน 10 ล้าน',
          role: 'เร่งผลตอบแทนทบต้นในระยะยาว 7-10 ปี เพื่อบรรลุเป้าหมายพอร์ต 10 ล้านบาท',
          color: '#3B82F6',
          target_percent: '60',
          linked_goal_name: 'พอร์ต 10 ล้านบาท',
          eligible_assets: ['ดัชนีหุ้นโลก (เช่น S&P 500, MSCI World)', 'กองทุนรวมหุ้นเติบโตคุณภาพสูง', 'หุ้นรายตัวตามเกณฑ์แกนหลัก'],
          cash_inflow_routing_pct: '60',
          action_rules: [
            'เป็นท่อหลักในการรับเงินลงทุน DCA รายเดือน',
            'เน้นถือครองระยะยาว ไม่ขายตัดขาดทุนเพราะความตื่นตระหนกยามตลาดปรับฐาน',
            'ตรวจสอบสัดส่วน Rebalance ทุก 6 เดือน หรือเมื่อสัดส่วนเบี่ยงเบนเกิน ±5%',
          ],
          source_value_ids: [],
          source_axis_allocation_ids: [],
        },
      ])
    } else {
      setBuckets([
        {
          bucket_id: resolveId('b_emergency', 'b_emergency'),
          name: 'เสบียงสำรอง & สภาพคล่อง',
          role: 'รองรับค่าใช้จ่ายฉุกเฉิน 6 เดือน สภาพคล่องสูงสุด',
          color: '#10B981',
          target_percent: '10',
          linked_goal_name: 'เงินสำรองฉุกเฉิน',
          eligible_assets: ['เงินฝากดอกเบี้ยสูง', 'กองทุนรวมตลาดเงิน MMF'],
          cash_inflow_routing_pct: '10',
          action_rules: ['ห้ามนำไปเสี่ยงในหุ้นเด็ดขาด', 'สำรองเฉพาะค่าใช้จ่ายจำเป็น'],
          source_value_ids: [],
          source_axis_allocation_ids: [],
        },
        {
          bucket_id: resolveId('b_income', 'b_income'),
          name: 'ฐานรากกระแสเงินสด & กันชน',
          role: 'กันชนลดความผันผวนและสร้างกระแสเงินสด',
          color: '#8B5CF6',
          target_percent: '20',
          linked_goal_name: 'กระแสเงินสดและกันชนความเสี่ยง',
          eligible_assets: ['กองทุนตราสารหนี้คุณภาพดี', 'หุ้นกู้ Investment Grade'],
          cash_inflow_routing_pct: '20',
          action_rules: ['นำปันผลไป Reinvest ในสินทรัพย์เติบโต'],
          source_value_ids: [],
          source_axis_allocation_ids: [],
        },
        {
          bucket_id: resolveId('b_growth', 'b_growth'),
          name: 'กองหน้าขับเคลื่อน 10 ล้าน',
          role: 'เร่งผลตอบแทนทบต้นในระยะยาวเพื่อพิชิตพอร์ต 10 ล้านบาท',
          color: '#3B82F6',
          target_percent: '60',
          linked_goal_name: 'พอร์ต 10 ล้านบาท',
          eligible_assets: ['ดัชนีหุ้นโลก S&P 500', 'กองทุนรวมหุ้นเติบโต'],
          cash_inflow_routing_pct: '60',
          action_rules: ['รับเงิน DCA รายเดือนเป็นหลัก', 'ถือครองระยะยาว Rebalance ทุก 6 เดือน'],
          source_value_ids: [],
          source_axis_allocation_ids: [],
        },
        {
          bucket_id: resolveId('b_tactical', 'b_tactical'),
          name: 'สำรวจโอกาส & เทรนด์เติบโตสูง',
          role: 'ลงทุนตามธีมเทคโนโลยีหรือหุ้นรายตัวที่มีศักยภาพสูงเพื่อผลตอบแทนส่วนเพิ่ม (Alpha)',
          color: '#F59E0B',
          target_percent: '10',
          linked_goal_name: 'ผลตอบแทนส่วนเพิ่ม (Alpha)',
          eligible_assets: ['หุ้นเทคโนโลยีรายตัว', 'หุ้น Growth / Momentum คุณภาพสูงตามแกนหลัก'],
          cash_inflow_routing_pct: '10',
          action_rules: ['จำกัดขาดทุนต่อไม้ไม่เกิน 2% ตามกฎเหล็กแกนหลัก', 'ขนาดรวมห้ามเกิน 10% ของพอร์ต'],
          source_value_ids: [],
          source_axis_allocation_ids: [],
        },
      ])
    }
  }

  const handleSaveDraft = async () => {
    await onUpdateDraft({
      purpose_buckets: buckets,
    })
  }

  const handleSaveAndReview = async () => {
    try {
      await handleSaveDraft()
      onProceedToReview()
    } catch {
      // Do not proceed to review if saving draft failed; preserve user edits in buckets state
    }
  }

  // Check if current buckets have empty or missing linked goals
  const hasEmptyLinkedGoals = buckets.some((b) => !b.linked_goal_name || b.linked_goal_name.trim() === '')

  return (
    <div className="space-y-4 max-w-5xl mx-auto w-full pb-6">
      {/* Top Header Banner & Strategy Summary */}
      <div className="rounded-2xl border border-sky-100 bg-gradient-to-br from-sky-50/80 via-white to-blue-50/30 p-4 sm:p-5 shadow-sm space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-sky-100/80 pb-2.5">
          <div className="flex items-center gap-2.5">
            <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-tr from-blue-600 to-sky-500 text-white shadow-sm">
              <FolderIcon className="h-5 w-5" />
            </span>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-base sm:text-lg font-black text-slate-900 tracking-tight">
                  ออกแบบ Purpose Buckets ตามเป้าหมายชีวิต
                </h1>
                <span className="inline-flex items-center gap-1 rounded-full bg-sky-100 px-2.5 py-0.5 text-[11px] font-bold text-blue-800 border border-sky-200">
                  {buckets.length} Buckets
                </span>
              </div>
              <p className="text-[11px] text-slate-600">
                พอร์ต: <span className="font-bold text-slate-800">{displayName}</span>
                {navThb > 0 && (
                  <span className="ml-2 pl-2 border-l border-slate-300">
                    มูลค่าพอร์ต: <strong className="text-blue-700">{navThb.toLocaleString('th-TH')} บาท</strong>
                  </span>
                )}
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {/* Live Total Percentage Badge */}
            <span
              className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-extrabold border shadow-sm ${
                is100Percent
                  ? 'bg-emerald-50 text-emerald-800 border-emerald-300'
                  : 'bg-rose-50 text-rose-800 border-rose-300 animate-pulse'
              }`}
            >
              <span>รวมสัดส่วน:</span>
              <span className="text-sm font-black">{totalPercent.toFixed(1)}%</span>
              <span>/ 100%</span>
            </span>

            {/* AI Suggest / Regenerate Button */}
            {onRegenerateBuckets && (
              <button
                type="button"
                onClick={onRegenerateBuckets}
                disabled={loading}
                className="inline-flex items-center gap-1.5 rounded-xl border border-blue-500 bg-blue-600 hover:bg-blue-700 px-3.5 py-1.5 text-xs font-extrabold text-white shadow-sm active:scale-98 transition-all disabled:opacity-50"
                title="สั่งให้ AI วิเคราะห์แกนหลักและเป้าหมายล่าสุดเพื่อแนะนำ Purpose Buckets ให้โดยอัตโนมัติ"
              >
                <SparklesIcon className="h-3.5 w-3.5 shrink-0" />
                <span>ให้ AI แนะนำ Buckets จากแกนหลัก</span>
              </button>
            )}
          </div>
        </div>

        {/* Quick Suggestion Presets */}
        <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="font-bold text-slate-600 flex items-center gap-1 text-[11px]">
              <BoltStrategyIcon className="h-3.5 w-3.5 text-sky-600" />
              <span>เทมเพลตแนะนำด่วน:</span>
            </span>
            <button
              type="button"
              onClick={() => applyPreset('3_buckets_standard')}
              className="rounded-lg bg-white border border-sky-200 hover:bg-sky-50 hover:border-sky-300 px-2.5 py-1 text-[11px] font-bold text-blue-900 shadow-sm transition-all active:scale-98"
            >
              3-Buckets มาตรฐาน (เสบียง 15% + กันชน 25% + เติบโต 60%)
            </button>
            <button
              type="button"
              onClick={() => applyPreset('4_buckets_growth')}
              className="rounded-lg bg-white border border-sky-200 hover:bg-sky-50 hover:border-sky-300 px-2.5 py-1 text-[11px] font-bold text-blue-900 shadow-sm transition-all active:scale-98"
            >
              4-Buckets เติบโตรุก (เสบียง 10% + กันชน 20% + กองหน้า 60% + โอกาสพิเศษ 10%)
            </button>
          </div>
        </div>

        {/* Visual Stacked Bar Chart & Cash Inflow Routing summary */}
        <div className="space-y-1">
          <div className="flex justify-between text-[11px] text-slate-500 font-semibold">
            <span>การกระจายสัดส่วนพอร์ต (Target Allocation):</span>
            <span className="inline-flex items-center gap-1">
              {is100Percent ? (
                <>
                  <CheckCircleIcon className="h-3.5 w-3.5 text-emerald-600" />
                  <span>สัดส่วนสมบูรณ์ครบ 100%</span>
                </>
              ) : (
                <>
                  <AlertTriangleIcon className="h-3.5 w-3.5 text-amber-600" />
                  <span>ผลรวมต้องได้ 100%</span>
                </>
              )}
            </span>
          </div>
          <div className="h-3 w-full rounded-full bg-slate-100 overflow-hidden flex shadow-inner">
            {buckets.map((b) => {
              const pct = Math.max(0, parseFloat(b.target_percent) || 0)
              if (pct <= 0) return null
              return (
                <div
                  key={b.bucket_id}
                  className="h-full transition-all"
                  style={{ width: `${pct}%`, backgroundColor: b.color || '#3b82f6' }}
                  title={`${b.name}: ${pct}%`}
                />
              )
            })}
          </div>

          {/* Cash Inflow Routing Pills */}
          <div className="flex flex-wrap items-center gap-1.5 pt-0.5 text-[11px]">
            <span className="font-semibold text-slate-500 flex items-center gap-1">
              <BulbIcon className="h-3.5 w-3.5 text-sky-600" />
              <span>สัดส่วนเงินเข้าใหม่ (DCA):</span>
            </span>
            {buckets.map((b) => {
              const routePct = b.cash_inflow_routing_pct || b.target_percent || '0'
              return (
                <span
                  key={b.bucket_id}
                  className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-blue-50/70 border border-blue-100 text-slate-800"
                >
                  <span className="w-2 h-2 rounded-full" style={{ backgroundColor: b.color || '#3b82f6' }} />
                  <span className="font-medium truncate max-w-[120px]">{b.name}:</span>
                  <strong className="text-blue-700">{routePct}%</strong>
                </span>
              )
            })}
          </div>
        </div>
      </div>

      {/* Purpose Buckets Cards List */}
      <div className="space-y-4">
        {/* Informative Guidance Banner if buckets need suggestion */}
        {hasEmptyLinkedGoals && (
          <div className="rounded-xl border border-sky-200 bg-gradient-to-r from-sky-50 via-white to-blue-50/60 p-3 text-xs text-blue-950 shadow-sm flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div className="flex items-start gap-2.5">
              <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-sky-100 text-blue-700">
                <CpuIcon className="h-4 w-4" />
              </span>
              <div>
                <span className="font-extrabold block text-blue-900">ไม่ต้องพิมพ์ออกแบบเองจากศูนย์!</span>
                <span className="text-blue-800/80 leading-relaxed text-[11px]">
                  คลิกปุ่ม <strong>"ให้ AI แนะนำ Buckets จากแกนหลัก"</strong> ด้านบน หรือเลือกเทมเพลตแนะนำด่วน เพื่อให้ระบบวางโครงสร้างที่ผูกกับเป้าหมาย สินทรัพย์ที่อนุญาต และกฎปฏิบัติให้โดยอัตโนมัติ
                </span>
              </div>
            </div>
            {onRegenerateBuckets && (
              <button
                type="button"
                onClick={onRegenerateBuckets}
                disabled={loading}
                className="inline-flex items-center gap-1.5 rounded-xl bg-blue-700 px-3.5 py-1.5 text-xs font-bold text-white hover:bg-blue-800 shrink-0 shadow-sm active:scale-98"
              >
                <SparklesIcon className="h-3.5 w-3.5 shrink-0" />
                <span>ให้ AI แนะนำทันที</span>
              </button>
            )}
          </div>
        )}

        {/* Purpose Buckets List Header */}
        <div className="flex items-center justify-between px-1">
          <h2 className="text-sm sm:text-base font-extrabold text-slate-900">
            รายละเอียด Purpose Buckets ({buckets.length} รายการ)
          </h2>
          <button
            type="button"
            onClick={handleAddBucket}
            disabled={loading}
            className="rounded-xl border border-sky-200 bg-sky-50 px-3.5 py-1.5 text-xs font-bold text-blue-700 hover:bg-sky-100 transition-all flex items-center gap-1.5 shadow-sm"
          >
            <span>+ เพิ่ม Bucket</span>
          </button>
        </div>

        {buckets.map((b, idx) => {
          const targetPct = parseFloat(b.target_percent) || 0
          const approxThb = navThb > 0 ? (navThb * targetPct) / 100 : 0
          const bucketId = b.bucket_id

          return (
            <div
              key={bucketId}
              className="rounded-2xl border border-slate-200/90 bg-white p-5 sm:p-6 shadow-sm hover:border-sky-200 transition-all space-y-4"
            >
              {/* Card Header: Color, Name, Linked Goal, Target %, THB */}
              <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-3 border-b border-slate-100 pb-4">
                <div className="flex items-center gap-2.5 flex-1 min-w-0">
                  <button
                    type="button"
                    onClick={() => handleCycleColor(idx)}
                    className="flex h-9 w-9 items-center justify-center rounded-xl border border-black/10 shadow-sm shrink-0 cursor-pointer hover:scale-105 transition-transform"
                    style={{ backgroundColor: b.color }}
                    title="คลิกเพื่อสลับสี"
                  >
                    <SlidersIcon className="h-4 w-4 text-white drop-shadow" />
                  </button>

                  <div className="flex-1 min-w-0">
                    <label htmlFor={`b-name-${bucketId}`} className="sr-only">
                      ชื่อ Bucket
                    </label>
                    <input
                      id={`b-name-${bucketId}`}
                      type="text"
                      value={b.name}
                      onChange={(e) => handleUpdateBucket(idx, 'name', e.target.value)}
                      placeholder="ชื่อ Bucket เช่น ขับเคลื่อนพอร์ต 10 ล้าน"
                      className="w-full rounded-xl border border-slate-200 px-3.5 py-1.5 text-sm sm:text-base font-extrabold text-slate-900 focus:outline-none focus:ring-2 focus:ring-sky-400"
                    />
                  </div>
                </div>

                <div className="flex flex-wrap items-center gap-2 sm:gap-3">
                  {/* Linked Financial Goal */}
                  <div className="flex items-center gap-1.5 bg-sky-50/70 border border-sky-200/80 px-2.5 py-1 rounded-xl">
                    <TargetIcon className="h-3.5 w-3.5 text-sky-700 shrink-0" />
                    <input
                      type="text"
                      value={b.linked_goal_name || ''}
                      onChange={(e) => handleUpdateBucket(idx, 'linked_goal_name', e.target.value)}
                      placeholder="ระบุเป้าหมายที่ผูก..."
                      className="rounded bg-transparent px-1 text-xs font-bold text-blue-900 focus:outline-none focus-visible:ring-2 focus-visible:ring-section-accent/45 w-32 sm:w-40"
                      title="เป้าหมายทางการเงินที่เงินก้อนนี้รองรับโดยตรง"
                    />
                  </div>

                  {/* Target Percent & Calculated THB */}
                  <div className="flex items-center gap-1.5 bg-slate-50 border border-slate-200 px-2.5 sm:px-3 py-1 rounded-xl">
                    <span className="text-xs font-semibold text-slate-500">สัดส่วน:</span>
                    <input
                      id={`b-target-${bucketId}`}
                      type="number"
                      min="0"
                      max="100"
                      step="1"
                      value={b.target_percent}
                      onChange={(e) => handleUpdateBucket(idx, 'target_percent', e.target.value)}
                      className="w-14 sm:w-16 rounded border border-slate-300 px-1 py-0.5 text-xs font-black text-blue-700 text-right focus:outline-none focus:ring-1 focus:ring-sky-400"
                    />
                    <span className="text-xs font-bold text-slate-600">%</span>
                    {approxThb > 0 && (
                      <span className="text-[11px] font-bold text-slate-600 pl-1 border-l border-slate-300 whitespace-nowrap">
                        ≈ {Math.round(approxThb).toLocaleString('th-TH')} บ.
                      </span>
                    )}
                  </div>

                  {buckets.length > 1 && (
                    <button
                      type="button"
                      onClick={() => handleRemoveBucket(idx)}
                      className="rounded-xl p-2 text-slate-500 hover:text-rose-600 hover:bg-rose-50 transition-colors ml-auto sm:ml-0"
                      title="ลบ Bucket นี้"
                      aria-label={`ลบ ${b.name}`}
                    >
                      <TrashIcon className="h-4 w-4" />
                    </button>
                  )}
                </div>
              </div>

              {/* Role & Money Purpose */}
              <div>
                <label htmlFor={`b-role-${bucketId}`} className="block text-xs font-bold text-slate-700 mb-1">
                  บทบาทและเหตุผลของเงินก้อนนี้:
                </label>
                <AutoResizeTextarea
                  id={`b-role-${bucketId}`}
                  minRows={2}
                  value={b.role}
                  onChange={(e) => handleUpdateBucket(idx, 'role', e.target.value)}
                  placeholder="เช่น สร้างกระแสเงินสดปันผล หรือ เร่งพลังผลตอบแทนทบต้นเพื่อพอร์ต 10 ล้าน..."
                  className="rounded-xl border border-slate-200 bg-slate-50/50 p-2.5 text-xs text-slate-800 focus:outline-none focus:ring-2 focus:ring-sky-400 leading-relaxed shadow-sm"
                />
              </div>

              {/* Two Column Grid: Eligible Assets & Action Rules */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-1">
                {/* Eligible Assets */}
                <div className="rounded-xl border border-emerald-100 bg-emerald-50/30 p-3.5 space-y-2.5">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-extrabold text-emerald-950 flex items-center gap-1.5">
                      <BriefcasePortfolioIcon className="h-3.5 w-3.5 text-emerald-700" />
                      <span>สินทรัพย์ที่อนุญาต (Eligible Assets)</span>
                    </span>
                    <span className="text-[11px] font-bold text-emerald-700">
                      {(b.eligible_assets || []).length} รายการ
                    </span>
                  </div>

                  <div className="flex flex-wrap gap-1.5 min-h-[32px]">
                    {(b.eligible_assets || []).map((asset, aIdx) => (
                      <span
                        key={aIdx}
                        className="inline-flex items-center gap-1 rounded-lg bg-white border border-emerald-200 px-2 py-1 text-[11px] font-medium text-emerald-950 shadow-sm"
                      >
                        <span>• {asset}</span>
                        <button
                          type="button"
                          onClick={() => handleRemoveAsset(idx, aIdx)}
                          className="text-slate-500 hover:text-rose-600 p-0.5"
                          title="ลบสินทรัพย์นี้"
                          aria-label={`ลบ ${asset}`}
                        >
                          <CloseIcon className="h-3 w-3" />
                        </button>
                      </span>
                    ))}
                  </div>

                  <div className="flex gap-1.5 pt-1">
                    <input
                      type="text"
                      value={newAssetInputs[bucketId] || ''}
                      onChange={(e) => setNewAssetInputs({ ...newAssetInputs, [bucketId]: e.target.value })}
                      onKeyDown={(e) => e.key === 'Enter' && handleAddAsset(idx, bucketId)}
                      placeholder="เพิ่มสินทรัพย์ เช่น หุ้นดัชนี S&P500..."
                      className="w-full rounded-lg border border-slate-200 bg-white px-2.5 py-1 text-xs text-slate-800 focus:outline-none focus:ring-1 focus:ring-emerald-400"
                    />
                    <button
                      type="button"
                      onClick={() => handleAddAsset(idx, bucketId)}
                      className="rounded-lg bg-emerald-600 px-3 py-1 text-xs font-bold text-white hover:bg-emerald-700 shrink-0"
                    >
                      + เพิ่ม
                    </button>
                  </div>
                </div>

                {/* Action Rules & Safeguards */}
                <div className="rounded-xl border border-sky-100 bg-sky-50/30 p-3.5 space-y-2.5">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-extrabold text-sky-950 flex items-center gap-1.5">
                      <ShieldCheckIcon className="h-3.5 w-3.5 text-sky-700" />
                      <span>กฎปฏิบัติและแนวทาง (Action Rules)</span>
                    </span>
                    <span className="text-[11px] font-bold text-sky-700">
                      {(b.action_rules || []).length} ข้อ
                    </span>
                  </div>

                  <div className="space-y-1.5 min-h-[32px]">
                    {(b.action_rules || []).map((rule, rIdx) => (
                      <div
                        key={rIdx}
                        className="flex items-start justify-between gap-1.5 rounded-lg bg-white border border-sky-200/80 p-1.5 text-[11px] text-slate-800 shadow-sm"
                      >
                        <span className="leading-tight">• {rule}</span>
                        <button
                          type="button"
                          onClick={() => handleRemoveRule(idx, rIdx)}
                          className="text-slate-500 hover:text-rose-600 p-0.5 shrink-0"
                          title="ลบข้อนี้"
                          aria-label="ลบกฎข้อนี้"
                        >
                          <CloseIcon className="h-3 w-3" />
                        </button>
                      </div>
                    ))}
                  </div>

                  <div className="flex gap-1.5 pt-1">
                    <AutoResizeTextarea
                      minRows={1}
                      value={newRuleInputs[bucketId] || ''}
                      onChange={(e) => setNewRuleInputs({ ...newRuleInputs, [bucketId]: e.target.value })}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                          e.preventDefault()
                          handleAddRule(idx, bucketId)
                        }
                      }}
                      placeholder="เพิ่มกฎปฏิบัติ เช่น เป็นท่อหลักรับเงิน DCA..."
                      className="w-full rounded-lg border border-slate-200 bg-white px-2.5 py-1 text-xs text-slate-800 focus:outline-none focus:ring-1 focus:ring-sky-400"
                    />
                    <button
                      type="button"
                      onClick={() => handleAddRule(idx, bucketId)}
                      className="rounded-lg bg-sky-600 px-3 py-1 text-xs font-bold text-white hover:bg-sky-700 shrink-0"
                    >
                      + เพิ่ม
                    </button>
                  </div>
                </div>
              </div>

              {/* Funding & Inflow Configuration for this Bucket (WP09) */}
              <div className="pt-2 border-t border-slate-100 flex flex-wrap items-center justify-between gap-3 text-xs">
                <div className="flex flex-wrap items-center gap-3">
                  <div className="flex items-center gap-1.5">
                    <span className="font-bold text-slate-600">เงินออมใหม่ (Inflow):</span>
                    <input
                      type="number"
                      min="0"
                      max="100"
                      step="1"
                      value={b.cash_inflow_routing_pct || b.target_percent || '0'}
                      onChange={(e) => handleUpdateBucket(idx, 'cash_inflow_routing_pct', e.target.value)}
                      className="w-16 rounded border border-slate-300 px-2 py-0.5 text-xs font-bold text-blue-700 text-right focus:outline-none focus:ring-1 focus:ring-blue-400"
                    />
                    <span className="font-bold text-slate-500">%</span>
                  </div>

                  <div className="flex items-center gap-1.5">
                    <span className="font-bold text-slate-600 flex items-center gap-1">
                      <TargetIcon className="h-3.5 w-3.5 text-sky-700" />
                      <span>ยอดเป้าหมาย (บาท):</span>
                    </span>
                    <input
                      type="number"
                      min="0"
                      step="1000"
                      value={b.target_amount || ''}
                      onChange={(e) => handleUpdateBucket(idx, 'target_amount', e.target.value || null)}
                      placeholder="ไม่ระบุ"
                      className="w-28 rounded border border-slate-300 px-2 py-0.5 text-xs font-bold text-slate-800 text-right focus:outline-none focus:ring-1 focus:ring-sky-400"
                    />
                  </div>

                  <div className="flex items-center gap-1.5">
                    <span className="font-bold text-slate-600">โหมด:</span>
                    <select
                      value={b.inflow_route_mode || 'fixed_ratio'}
                      onChange={(e) => handleUpdateBucket(idx, 'inflow_route_mode', e.target.value as any)}
                      className="rounded border border-slate-300 px-1.5 py-0.5 text-xs font-medium text-slate-700 bg-white"
                    >
                      <option value="fixed_ratio">Fixed Ratio (%)</option>
                      <option value="priority_fill">Priority Fill</option>
                    </select>
                  </div>
                </div>

                <span className="text-[11px] text-slate-500">
                  {b.inflow_route_mode === 'priority_fill' ? 'เติมเงินให้เต็มเป้าหมายตามลำดับก่อน redirect' : 'สัดส่วนเงินที่จะนำไปซื้อสะสม (DCA) รายเดือน'}
                </span>
              </div>
            </div>
          )
        })}
      </div>

      {/* Action Footer Bar */}
      <div className="rounded-2xl border border-sky-100 bg-white p-4 sm:p-5 shadow-sm flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="text-xs">
          {statusText ? (
            <span className="flex items-center gap-2 text-blue-700 font-bold animate-pulse">
              <span className="h-2 w-2 rounded-full bg-blue-500" />
              {statusText}
            </span>
          ) : !is100Percent ? (
            <span className="text-rose-700 font-extrabold flex items-center gap-1.5">
              <AlertTriangleIcon className="h-4 w-4 shrink-0" />
              <span>ผลรวมสัดส่วนต้องเท่ากับ 100% (ปัจจุบัน {totalPercent.toFixed(1)}%)</span>
            </span>
          ) : (
            <span className="text-emerald-700 font-bold flex items-center gap-1.5">
              <CheckSmallIcon className="h-4 w-4 shrink-0" />
              <span>สัดส่วนรวมครบ 100% พร้อมสำหรับการตรวจสอบและนำไปใช้กับพอร์ต</span>
            </span>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-3 w-full sm:w-auto">
          <button
            type="button"
            onClick={async () => {
              try {
                await handleSaveDraft()
              } catch {
                // Handled
              }
            }}
            disabled={loading}
            className="inline-flex items-center gap-1.5 rounded-xl border border-sky-300 bg-sky-50 px-5 py-2.5 text-xs font-bold text-blue-700 hover:bg-sky-100 active:scale-98 transition-all disabled:opacity-50"
          >
            <CheckCircleIcon className="h-3.5 w-3.5" />
            <span>บันทึกแบบร่าง Buckets</span>
          </button>

          <button
            type="button"
            onClick={handleSaveAndReview}
            disabled={!is100Percent || loading}
            className={`px-6 py-2.5 rounded-xl font-bold text-xs text-white shadow-md transition-all flex items-center justify-center gap-2 ${
              is100Percent && !loading
                ? 'bg-gradient-to-r from-blue-600 to-blue-600 hover:from-blue-700 hover:to-blue-700 shadow-sky-500/20 active:scale-98 cursor-pointer'
                : 'bg-slate-300 text-slate-500 cursor-not-allowed shadow-none'
            }`}
          >
            <span>ตรวจสอบก่อนใช้กับพอร์ต →</span>
          </button>
        </div>
      </div>
    </div>
  )
}
