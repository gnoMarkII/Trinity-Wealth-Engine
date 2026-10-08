import { useState } from 'react'
import { useInvestorPrinciples } from '../../../hooks/useInvestorPrinciples'
import EssenceInterview from './EssenceInterview'
import EssenceResult from './EssenceResult'
import FinancialContextForm from './FinancialContextForm'
import InvestmentAxisEditor from './InvestmentAxisEditor'
import EssenceBucketBuilder from './EssenceBucketBuilder'
import BucketApplyReview from './BucketApplyReview'

interface InvestorPrinciplesTabProps {
  portfolioId: string
  onPortfolioChangeNeeded?: () => void
  onNavigateToOverview?: () => void
}

export default function InvestorPrinciplesTab({
  portfolioId,
  onPortfolioChangeNeeded,
  onNavigateToOverview,
}: InvestorPrinciplesTabProps) {
  const {
    activeMilestone,
    setActiveMilestone,
    loading,
    actionLoading,
    actionStatusText,
    error,
    clearError,

    // Milestone 1 (Essence)
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
    applyBucketPlan,
  } = useInvestorPrinciples({ portfolioId, onPortfolioChangeNeeded })

  const [bucketSubStep, setBucketSubStep] = useState<'editor' | 'review'>('editor')
  const [axisSubStep, setAxisSubStep] = useState<'context' | 'axis'>('axis')

  if (loading) {
    return (
      <div className="rounded-2xl border border-sky-100 bg-white p-12 text-center shadow-xs space-y-3">
        <div className="mx-auto h-8 w-8 animate-spin rounded-full border-4 border-sky-200 border-t-sky-600" />
        <p className="text-sm font-semibold text-zinc-600">กำลังโหลดข้อมูลหลักการลงทุน...</p>
      </div>
    )
  }

  // Stepper badges
  const essenceBadge = confirmedEssence
    ? { label: '✓ ยืนยันแล้ว', style: 'bg-emerald-100 text-emerald-800' }
    : session
    ? { label: `ตอบแล้ว ${session.answers_count}/10`, style: 'bg-sky-100 text-sky-800' }
    : { label: 'ยังไม่เริ่ม', style: 'bg-zinc-100 text-zinc-600' }

  const axisBadge = confirmedAxis
    ? { label: '✓ ยืนยันแล้ว', style: 'bg-emerald-100 text-emerald-800' }
    : axisDraft
    ? { label: 'ร่างนโยบาย', style: 'bg-amber-100 text-amber-800' }
    : { label: 'รอเริ่ม', style: 'bg-zinc-100 text-zinc-500' }

  const bucketsBadge = applyReceipt
    ? { label: '✓ นำไปใช้แล้ว', style: 'bg-emerald-100 text-emerald-800' }
    : bucketPlan
    ? { label: 'ร่าง Buckets', style: 'bg-purple-100 text-purple-800' }
    : { label: 'ทางเลือก', style: 'bg-zinc-100 text-zinc-500' }

  return (
    <div className="space-y-6">
      {/* Error alert */}
      {error && (
        <div className="flex items-center justify-between rounded-xl border border-rose-200 bg-rose-50 p-4 text-xs font-semibold text-rose-800 shadow-xs">
          <span>⚠️ {error}</span>
          <button
            type="button"
            onClick={clearError}
            className="text-rose-500 hover:text-rose-700 underline text-xs ml-4"
          >
            ปิด
          </button>
        </div>
      )}

      {/* Milestone Stepper Bar */}
      <div className="rounded-2xl border border-sky-100 bg-white p-3 shadow-xs">
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
          {/* Milestone 1 */}
          <button
            type="button"
            onClick={() => setActiveMilestone('essence')}
            className={`p-3 rounded-xl text-left transition-all flex items-center justify-between gap-2 ${
              activeMilestone === 'essence'
                ? 'bg-sky-50/80 border border-sky-300 ring-2 ring-sky-400/20'
                : 'hover:bg-zinc-50 border border-transparent'
            }`}
          >
            <div className="flex items-center gap-2.5 min-w-0">
              <span
                className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-lg text-xs font-extrabold ${
                  activeMilestone === 'essence' ? 'bg-sky-600 text-white' : 'bg-zinc-100 text-zinc-600'
                }`}
              >
                1
              </span>
              <div className="truncate">
                <span className="block text-xs font-bold text-zinc-900 truncate">
                  ค้นหาแก่นแท้ (Essence)
                </span>
                <span className="text-[11px] text-zinc-500 truncate block">สำรวจเป้าหมายและค่านิยม</span>
              </div>
            </div>
            <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full shrink-0 ${essenceBadge.style}`}>
              {essenceBadge.label}
            </span>
          </button>

          {/* Milestone 2 */}
          <button
            type="button"
            onClick={() => setActiveMilestone('axis')}
            className={`p-3 rounded-xl text-left transition-all flex items-center justify-between gap-2 ${
              activeMilestone === 'axis'
                ? 'bg-sky-50/80 border border-sky-300 ring-2 ring-sky-400/20'
                : 'hover:bg-zinc-50 border border-transparent'
            }`}
          >
            <div className="flex items-center gap-2.5 min-w-0">
              <span
                className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-lg text-xs font-extrabold ${
                  activeMilestone === 'axis' ? 'bg-sky-600 text-white' : 'bg-zinc-100 text-zinc-600'
                }`}
              >
                2
              </span>
              <div className="truncate">
                <span className="block text-xs font-bold text-zinc-900 truncate">
                  สร้างแกนหลัก (Axis)
                </span>
                <span className="text-[11px] text-zinc-500 truncate block">นโยบาย 8 หัวข้อ + สิ่งที่ไม่ทำ</span>
              </div>
            </div>
            <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full shrink-0 ${axisBadge.style}`}>
              {axisBadge.label}
            </span>
          </button>

          {/* Milestone 3 */}
          <button
            type="button"
            onClick={() => setActiveMilestone('buckets')}
            className={`p-3 rounded-xl text-left transition-all flex items-center justify-between gap-2 ${
              activeMilestone === 'buckets'
                ? 'bg-sky-50/80 border border-sky-300 ring-2 ring-sky-400/20'
                : 'hover:bg-zinc-50 border border-transparent'
            }`}
          >
            <div className="flex items-center gap-2.5 min-w-0">
              <span
                className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-lg text-xs font-extrabold ${
                  activeMilestone === 'buckets' ? 'bg-sky-600 text-white' : 'bg-zinc-100 text-zinc-600'
                }`}
              >
                3
              </span>
              <div className="truncate">
                <span className="block text-xs font-bold text-zinc-900 truncate">
                  ออกแบบ Buckets
                </span>
                <span className="text-[11px] text-zinc-500 truncate block">จัดกลุ่มพอร์ตตามบทบาทเงิน</span>
              </div>
            </div>
            <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full shrink-0 ${bucketsBadge.style}`}>
              {bucketsBadge.label}
            </span>
          </button>
        </div>
      </div>

      {/* Main Content Area */}
      <div>
        {/* MILESTONE 1: ESSENCE */}
        {activeMilestone === 'essence' && (
          <div>
            {!session && !confirmedEssence ? (
              // Welcome Card
              <div className="max-w-2xl mx-auto rounded-2xl border border-sky-100 bg-white p-8 text-center shadow-sm space-y-6">
                <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl bg-sky-100 text-3xl text-sky-600 shadow-xs">
                  🧭
                </div>
                <div>
                  <h2 className="text-xl sm:text-2xl font-extrabold text-zinc-900 tracking-tight">
                    ค้นหาแก่นแท้ในการลงทุนของคุณ
                  </h2>
                  <p className="text-xs sm:text-sm text-zinc-600 mt-2 max-w-md mx-auto leading-relaxed">
                    ตอบ 10 คำถามเพื่อค้นหาว่าคุณลงทุนเพื่ออะไรและให้ความสำคัญกับอะไร AI
                    จะสร้างคำถามทีละข้ออย่างเป็นกลางจากคำตอบก่อนหน้าของคุณ
                  </p>
                </div>

                <div className="rounded-xl bg-sky-50/60 p-4 border border-sky-100 text-left text-xs space-y-1.5 text-zinc-700">
                  <p className="font-bold text-sky-900">✨ ประสบการณ์ที่จะได้รับ:</p>
                  <p>• 10 คำถามแบบ 4 ตัวเลือก เริ่มจากเรื่องชีวิตที่ตอบง่าย</p>
                  <p>• ตอบไม่แน่ใจได้ พักไว้ก่อนได้ และกลับมาทำต่อได้เสมอ</p>
                  <p>• สรุปเป็นแก่นแท้ในหน้าเดียว พร้อมให้คุณตรวจความตรงรายข้อ</p>
                </div>

                <button
                  type="button"
                  onClick={startNewSession}
                  disabled={actionLoading}
                  className="rounded-xl bg-gradient-to-r from-sky-600 to-blue-600 px-8 py-3.5 text-sm font-bold text-white shadow-md shadow-sky-500/25 hover:from-sky-700 hover:to-blue-700 active:scale-98 transition-all"
                >
                  {actionLoading ? 'กำลังเตรียมคำถาม...' : '🚀 เริ่มค้นหาแก่นแท้ (10 คำถาม)'}
                </button>
              </div>
            ) : session && !session.is_complete ? (
              // Active adaptive interview
              <EssenceInterview
                session={session}
                onRecordAndNext={recordAndNext}
                onPause={() => {}}
                loading={actionLoading}
                statusText={actionStatusText}
              />
            ) : (
              // Result & Review Screen
              <EssenceResult
                summary={summary}
                confirmedEssence={confirmedEssence}
                onRateClaim={rateClaim}
                onEditClaim={editClaim}
                onExcludeClaim={excludeClaim}
                onConfirmBatch={confirmEssenceBatch}
                onRestart={startNewSession}
                loading={actionLoading}
                statusText={actionStatusText}
              />
            )}
          </div>
        )}

        {/* MILESTONE 2: INVESTMENT AXIS */}
        {activeMilestone === 'axis' && (
          <div>
            {!confirmedEssence ? (
              <div className="max-w-xl mx-auto rounded-2xl border border-amber-200 bg-amber-50/70 p-8 text-center space-y-4">
                <span className="text-3xl">🧭</span>
                <h3 className="text-lg font-bold text-amber-900">
                  กรุณาค้นหาและยืนยันแก่นแท้การลงทุนก่อน
                </h3>
                <p className="text-xs text-amber-800 leading-relaxed">
                  แกนหลักการลงทุน (Investment Axis) ต้องอ้างอิงแก่นแท้ที่คุณได้ตรวจและยืนยันแล้ว
                </p>
                <button
                  type="button"
                  onClick={() => setActiveMilestone('essence')}
                  className="rounded-xl bg-amber-600 px-5 py-2 text-xs font-bold text-white hover:bg-amber-700"
                >
                  ไปที่ขั้นค้นหาแก่นแท้
                </button>
              </div>
            ) : axisSubStep === 'context' || (!axisDraft && !confirmedAxis) ? (
              <FinancialContextForm
                financialContext={financialContext}
                portfolioId={portfolioId}
                onSave={saveFinancialContext}
                onProceedToDraft={async () => {
                  await generateAxisDraft()
                  setAxisSubStep('axis')
                }}
                loading={actionLoading}
                statusText={actionStatusText}
              />
            ) : (
              <div className="space-y-4">
                <div className="flex justify-end max-w-4xl mx-auto">
                  <button
                    type="button"
                    onClick={() => setAxisSubStep('context')}
                    className="text-xs font-semibold text-sky-600 hover:text-sky-800 underline"
                  >
                    ✏️ ทบทวนบริบททางการเงินของพอร์ตนี้
                  </button>
                </div>
                <InvestmentAxisEditor
                  axisDraft={axisDraft}
                  confirmedAxis={confirmedAxis}
                  portfolioId={portfolioId}
                  onUpdateDraft={updateAxis}
                  onConfirmAxis={confirmAxis}
                  onProceedToBuckets={() => setActiveMilestone('buckets')}
                  loading={actionLoading}
                  statusText={actionStatusText}
                />
              </div>
            )}
          </div>
        )}

        {/* MILESTONE 3: PURPOSE BUCKETS */}
        {activeMilestone === 'buckets' && (
          <div>
            {!confirmedAxis ? (
              <div className="max-w-xl mx-auto rounded-2xl border border-amber-200 bg-amber-50/70 p-8 text-center space-y-4">
                <span className="text-3xl">🔒</span>
                <h3 className="text-lg font-bold text-amber-900">
                  ต้องยืนยันแกนหลักการลงทุนของพอร์ตนี้ก่อน
                </h3>
                <p className="text-xs text-amber-800 leading-relaxed">
                  ปุ่มสร้าง Buckets จากแกนหลักจะเปิดใช้เมื่อมีแกนหลักฉบับยืนยันครบ 8 หัวข้อแล้ว
                </p>
                <button
                  type="button"
                  onClick={() => setActiveMilestone('axis')}
                  className="rounded-xl bg-amber-600 px-5 py-2 text-xs font-bold text-white hover:bg-amber-700"
                >
                  ไปที่ขั้นแกนหลักการลงทุน
                </button>
              </div>
            ) : !bucketPlan ? (
              <div className="max-w-2xl mx-auto rounded-2xl border border-sky-100 bg-white p-8 text-center shadow-sm space-y-6">
                <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl bg-purple-100 text-3xl text-purple-600 shadow-xs">
                  🗂️
                </div>
                <div>
                  <h2 className="text-xl sm:text-2xl font-extrabold text-zinc-900 tracking-tight">
                    สร้าง Purpose Buckets จากแกนหลักฉบับยืนยัน
                  </h2>
                  <p className="text-xs sm:text-sm text-zinc-600 mt-2 max-w-md mx-auto leading-relaxed">
                    ระบบจะร่าง Purpose Buckets 3–5 รายการ พร้อมสัดส่วนและสีที่สอดคล้องกับแผนจัดสรรที่คุณยืนยันไว้
                  </p>
                </div>

                <button
                  type="button"
                  onClick={generateBucketPlan}
                  disabled={actionLoading}
                  className="rounded-xl bg-gradient-to-r from-purple-600 to-indigo-600 px-8 py-3.5 text-sm font-bold text-white shadow-md shadow-purple-500/25 hover:from-purple-700 hover:to-indigo-700 active:scale-98 transition-all"
                >
                  {actionLoading ? 'กำลังร่าง Purpose Buckets...' : '✨ สร้าง Buckets จากแกนหลัก'}
                </button>
              </div>
            ) : bucketSubStep === 'editor' ? (
              <EssenceBucketBuilder
                draft={bucketPlan}
                portfolioId={portfolioId}
                onUpdateDraft={updateBucketPlan}
                onProceedToReview={() => setBucketSubStep('review')}
                loading={actionLoading}
                statusText={actionStatusText}
              />
            ) : (
              <BucketApplyReview
                draft={bucketPlan}
                preview={preview}
                applyReceipt={applyReceipt}
                onApply={() => applyBucketPlan()}
                onBack={() => setBucketSubStep('editor')}
                onGoToOverview={() => onNavigateToOverview?.()}
                loading={actionLoading}
                statusText={actionStatusText}
              />
            )}
          </div>
        )}
      </div>
    </div>
  )
}
