import { useState } from 'react'
import type { SessionResponseDTO } from '../../../api/types'

interface EssenceInterviewProps {
  session: SessionResponseDTO
  onRecordAndNext: (
    answerKind: 'choice' | 'free_text' | 'unsure' | 'skipped',
    optionId?: string | null,
    freeText?: string | null,
  ) => Promise<void>
  onPause: () => void
  loading: boolean
  statusText?: string | null
}

const TOPIC_NAMES: Record<string, string> = {
  life_goals: 'เป้าหมายชีวิต',
  priorities: 'ลำดับความสำคัญ',
  horizon_liquidity: 'กรอบเวลาและสภาพคล่อง',
  risk_experience: 'ความเสี่ยงและประสบการณ์',
  portfolio_habits: 'การดูแลพอร์ตที่ทำต่อเนื่องได้',
  constraints_beliefs: 'ข้อจำกัดและความเชื่อ',
}

export default function EssenceInterview({
  session,
  onRecordAndNext,
  onPause,
  loading,
  statusText,
}: EssenceInterviewProps) {
  const currentQ = session.current_question

  const [selectedOptionId, setSelectedOptionId] = useState<string | null>(null)
  const [answerKind, setAnswerKind] = useState<'choice' | 'free_text' | 'unsure' | 'skipped'>('choice')
  const [freeText, setFreeText] = useState<string>('')
  const [showFreeTextInput, setShowFreeTextInput] = useState<boolean>(false)

  if (!currentQ) {
    return (
      <div className="rounded-2xl border border-sky-100 bg-white p-8 text-center shadow-xs">
        <p className="text-zinc-600">ไม่มีคำถามที่พร้อมแสดง กรุณาลองใหม่อีกครั้ง</p>
      </div>
    )
  }

  const isFinalQuestion = currentQ.sequence_no >= 10
  const topicLabel = currentQ.coverage_topics?.[0]
    ? TOPIC_NAMES[currentQ.coverage_topics[0]] || currentQ.coverage_topics[0]
    : 'การสำรวจตัวตน'

  const handleSelectOption = (optionId: string) => {
    setSelectedOptionId(optionId)
    setAnswerKind('choice')
  }

  const handleSelectUnsure = () => {
    setSelectedOptionId(null)
    setAnswerKind('unsure')
  }

  const handleSubmit = async () => {
    if (loading) return
    if (answerKind === 'choice' && !selectedOptionId && !freeText.trim()) return

    const effectiveKind = freeText.trim() && answerKind === 'free_text' ? 'free_text' : answerKind
    await onRecordAndNext(
      effectiveKind,
      effectiveKind === 'choice' ? selectedOptionId : null,
      freeText.trim() ? freeText.trim() : null,
    )

    // Reset local selection for next question
    setSelectedOptionId(null)
    setAnswerKind('choice')
    setFreeText('')
    setShowFreeTextInput(false)
  }

  const canSubmit =
    (answerKind === 'choice' && selectedOptionId !== null) ||
    answerKind === 'unsure' ||
    (answerKind === 'free_text' && freeText.trim().length > 0)

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      {/* Progress & Header */}
      <div className="rounded-2xl border border-sky-100 bg-white p-6 shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-3 pb-4 border-b border-sky-50">
          <div className="flex items-center gap-2.5">
            <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-sky-100 text-sky-700 font-extrabold text-sm">
              {currentQ.sequence_no}
            </span>
            <span className="text-xs sm:text-sm font-bold text-zinc-700">
              คำถามที่ {currentQ.sequence_no} จาก 10
            </span>
            <span className="rounded-full bg-sky-50 px-2.5 py-0.5 text-xs font-semibold text-sky-700 border border-sky-100">
              {topicLabel}
            </span>
          </div>

          <div className="flex items-center gap-3">
            <span className="text-xs font-semibold text-zinc-400">
              ตอบแล้ว {session.answers_count}/10
            </span>
            <button
              type="button"
              onClick={onPause}
              disabled={loading}
              className="rounded-lg border border-zinc-200 px-3 py-1 text-xs font-semibold text-zinc-600 hover:bg-zinc-50 active:scale-95 transition-all"
              title="บันทึกร่างและกลับมาตอบต่อได้ตลอดเวลา"
            >
              พักไว้ก่อน
            </button>
          </div>
        </div>

        {/* Progress bar */}
        <div className="w-full bg-sky-50 h-2 rounded-full mt-4 overflow-hidden">
          <div
            className="bg-gradient-to-r from-sky-400 to-blue-600 h-full rounded-full transition-all duration-300"
            style={{ width: `${Math.min(100, (currentQ.sequence_no / 10) * 100)}%` }}
          />
        </div>

        {/* Question Text */}
        <div className="pt-6 pb-2">
          <h2 className="text-lg sm:text-xl font-bold text-zinc-900 leading-relaxed">
            {currentQ.text}
          </h2>
          <p className="text-xs text-zinc-500 mt-1">
            เลือกคำตอบที่ตรงกับตัวคุณที่สุด หรือพิมพ์อธิบายเพิ่มเติมด้วยคำของคุณเอง
          </p>
        </div>

        {/* 4 Choices */}
        <div className="grid grid-cols-1 gap-3 pt-4">
          {currentQ.options.map((option) => {
            const isSelected = answerKind === 'choice' && selectedOptionId === option.option_id
            return (
              <button
                key={option.option_id}
                type="button"
                onClick={() => handleSelectOption(option.option_id)}
                disabled={loading}
                className={`w-full text-left p-4 rounded-xl border transition-all flex items-start gap-3.5 ${
                  isSelected
                    ? 'border-sky-500 bg-sky-50/70 text-zinc-900 shadow-xs ring-2 ring-sky-400/30'
                    : 'border-zinc-200/80 bg-white hover:border-sky-300 hover:bg-sky-50/30 text-zinc-800'
                }`}
              >
                <span
                  className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-xs font-bold transition-colors ${
                    isSelected ? 'bg-sky-600 text-white' : 'bg-zinc-100 text-zinc-600'
                  }`}
                >
                  {option.option_key}
                </span>
                <span className="text-sm sm:text-base font-medium leading-snug">
                  {option.text}
                </span>
              </button>
            )
          })}
        </div>

        {/* Secondary options: Unsure & Free Text */}
        <div className="pt-4 flex flex-wrap items-center justify-between gap-3 border-t border-sky-50 mt-6">
          <button
            type="button"
            onClick={handleSelectUnsure}
            disabled={loading}
            className={`rounded-xl px-3.5 py-2 text-xs sm:text-sm font-semibold transition-all border ${
              answerKind === 'unsure'
                ? 'border-amber-400 bg-amber-50 text-amber-900 ring-2 ring-amber-300/40'
                : 'border-zinc-200 text-zinc-600 hover:bg-zinc-50'
            }`}
          >
            🤔 ยังไม่แน่ใจ / ยังไม่พร้อมตอบ
          </button>

          <button
            type="button"
            onClick={() => {
              setShowFreeTextInput(!showFreeTextInput)
              if (!showFreeTextInput) {
                setAnswerKind('free_text')
              }
            }}
            className="text-xs font-semibold text-sky-600 hover:text-sky-800 underline underline-offset-2"
          >
            {showFreeTextInput ? 'ซ่อนคำอธิบายเพิ่มเติม' : '+ อธิบายด้วยคำของฉันเอง'}
          </button>
        </div>

        {/* Free Text Input Box */}
        {showFreeTextInput && (
          <div className="mt-4 p-4 rounded-xl bg-sky-50/50 border border-sky-100 space-y-2 animate-fade-in">
            <label htmlFor="free-text-input" className="block text-xs font-semibold text-zinc-700">
              อธิบายเพิ่มเติมหรือตอบด้วยคำของฉันเอง:
            </label>
            <textarea
              id="free-text-input"
              rows={3}
              value={freeText}
              onChange={(e) => {
                setFreeText(e.target.value)
                setAnswerKind('free_text')
              }}
              placeholder="เช่น มีภาระต้องใช้เงินก้อนในอีก 2 ปี หรืออยากเน้นเงินปันผลเพื่อค่าใช้จ่ายประจำเดือน..."
              className="w-full rounded-lg border border-sky-200 bg-white p-3 text-sm text-zinc-800 focus:outline-none focus:ring-2 focus:ring-sky-400"
            />
          </div>
        )}

        {/* Action Button: Next / View Essence */}
        <div className="pt-6 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="text-xs text-zinc-500">
            {statusText ? (
              <span className="flex items-center gap-2 text-sky-700 font-semibold animate-pulse">
                <span className="inline-block w-2 h-2 rounded-full bg-sky-500" />
                {statusText}
              </span>
            ) : (
              <span>กดถัดไปเพื่อบันทึกคำตอบและสร้างคำถามข้อถัดไป</span>
            )}
          </div>

          <button
            type="button"
            onClick={handleSubmit}
            disabled={!canSubmit || loading}
            className={`w-full sm:w-auto px-7 py-3 rounded-xl font-bold text-sm text-white shadow-md transition-all flex items-center justify-center gap-2 ${
              canSubmit && !loading
                ? isFinalQuestion
                  ? 'bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-700 hover:to-teal-700 shadow-emerald-500/25 active:scale-98 cursor-pointer'
                  : 'bg-gradient-to-r from-sky-600 to-blue-600 hover:from-sky-700 hover:to-blue-700 shadow-sky-500/25 active:scale-98 cursor-pointer'
                : 'bg-zinc-300 text-zinc-500 cursor-not-allowed shadow-none'
            }`}
          >
            {loading ? (
              <span>กำลังประมวลผล...</span>
            ) : isFinalQuestion ? (
              <span>✨ ดูแก่นแท้ของฉัน</span>
            ) : (
              <span>ถัดไป →</span>
            )}
          </button>
        </div>
      </div>
    </div>
  )
}
