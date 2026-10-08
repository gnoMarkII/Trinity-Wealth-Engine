import { useState } from 'react'
import type { SummaryResponseDTO, ConfirmedEssenceDTO } from '../../../api/types'

interface EssenceResultProps {
  summary: SummaryResponseDTO | null
  confirmedEssence: ConfirmedEssenceDTO | null
  onRateClaim: (claimId: string, rating: 'exact' | 'partial' | 'rejected') => Promise<void>
  onEditClaim: (claimId: string, text: string) => Promise<void>
  onExcludeClaim: (claimId: string) => Promise<void>
  onConfirmBatch: (proceedToAxis: boolean) => Promise<void>
  onRestart: () => void
  loading: boolean
  statusText?: string | null
}

const SOURCE_LABELS: Record<string, { label: string; style: string }> = {
  user_stated: { label: 'คุณบอกไว้', style: 'bg-emerald-50 text-emerald-700 border-emerald-200' },
  ai_interpreted: { label: 'AI ตีความ', style: 'bg-sky-50 text-sky-700 border-sky-200' },
  user_edited: { label: 'คุณแก้ไข', style: 'bg-purple-50 text-purple-700 border-purple-200' },
}

export default function EssenceResult({
  summary,
  confirmedEssence,
  onRateClaim,
  onEditClaim,
  onExcludeClaim,
  onConfirmBatch,
  onRestart,
  loading,
  statusText,
}: EssenceResultProps) {
  const [editingClaimId, setEditingClaimId] = useState<string | null>(null)
  const [editText, setEditText] = useState<string>('')
  const [expandedEvidenceId, setExpandedEvidenceId] = useState<string | null>(null)

  // Determine whether we're showing confirmed view or review draft
  const isConfirmed = !!confirmedEssence && (!summary || confirmedEssence.artifact_id)

  const claims = summary?.claims ?? []
  const statement = confirmedEssence?.statement || summary?.statement || ''
  const unresolvedTopics = summary?.unresolved_topics ?? []

  const handleStartEdit = (claimId: string, currentText: string) => {
    setEditingClaimId(claimId)
    setEditText(currentText)
  }

  const handleSaveEdit = async (claimId: string) => {
    if (!editText.trim()) return
    await onEditClaim(claimId, editText.trim())
    setEditingClaimId(null)
  }

  const handleCancelEdit = () => {
    setEditingClaimId(null)
    setEditText('')
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header Banner */}
      <div className="rounded-2xl border border-sky-100 bg-gradient-to-br from-sky-50/70 via-white to-blue-50/30 p-6 sm:p-8 shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-sky-100 pb-4">
          <div className="flex items-center gap-2.5">
            <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-tr from-sky-500 to-blue-600 text-white shadow-sm font-bold text-lg">
              ✨
            </span>
            <div>
              <h1 className="text-xl sm:text-2xl font-extrabold text-zinc-900 tracking-tight">
                แก่นแท้การลงทุนของฉัน (Investor Essence)
              </h1>
              <p className="text-xs text-zinc-500 mt-0.5">
                ค้นพบจากคำตอบ 10 ข้อของคุณ เพื่อสะท้อนเป้าหมายและค่านิยมที่แท้จริง
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {isConfirmed ? (
              <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-50 px-3 py-1 text-xs font-bold text-emerald-700 border border-emerald-200">
                <span className="h-2 w-2 rounded-full bg-emerald-500" />
                ยืนยันแล้ว
              </span>
            ) : (
              <span className="inline-flex items-center gap-1.5 rounded-full bg-amber-50 px-3 py-1 text-xs font-bold text-amber-700 border border-amber-200">
                <span className="h-2 w-2 rounded-full bg-amber-500 animate-pulse" />
                ร่างรอการทบทวน
              </span>
            )}
          </div>
        </div>

        {/* Core Statement Card */}
        {statement && (
          <div className="mt-6 rounded-xl bg-white p-5 border border-sky-200/80 shadow-xs">
            <span className="text-[11px] font-bold uppercase tracking-wider text-sky-600 block mb-1">
              คำประกาศแก่นแท้ (Core Identity)
            </span>
            <p className="text-base sm:text-lg font-semibold text-zinc-800 leading-relaxed italic">
              "{statement}"
            </p>
          </div>
        )}
      </div>

      {/* Review in Single Page: Bulleted Claims */}
      <div className="rounded-2xl border border-sky-100 bg-white p-6 sm:p-8 shadow-sm space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-sky-50 pb-4">
          <div>
            <h2 className="text-base sm:text-lg font-bold text-zinc-900">
              ข้อสรุปแก่นแท้การลงทุน (Bulleted List)
            </h2>
            <p className="text-xs text-zinc-500 mt-0.5">
              ตรวจความตรงของแต่ละข้อ หากตรงกด “ตรงกับฉัน” หากต้องการปรับกด “ตรงบางส่วน” เพื่อแก้ไข
            </p>
          </div>
          <span className="text-xs font-semibold text-zinc-400">
            {claims.length} ข้อสรุป
          </span>
        </div>

        {/* Claims List */}
        <div className="space-y-4">
          {claims.map((claim, idx) => {
            const isEditing = editingClaimId === claim.claim_id
            const isExcluded = claim.fit_rating === 'rejected'
            const isPartial = claim.fit_rating === 'partial'
            const isExact = claim.fit_rating === 'exact'
            const sourceInfo = SOURCE_LABELS[claim.source_kind] || {
              label: claim.source_kind,
              style: 'bg-zinc-100 text-zinc-700 border-zinc-200',
            }

            return (
              <div
                key={claim.claim_id}
                className={`rounded-xl border p-4 transition-all ${
                  isExcluded
                    ? 'border-zinc-200 bg-zinc-50 opacity-60'
                    : isExact
                    ? 'border-emerald-200 bg-emerald-50/30'
                    : isPartial
                    ? 'border-purple-200 bg-purple-50/20'
                    : 'border-zinc-200/90 bg-white hover:border-sky-200'
                }`}
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="flex items-start gap-3 flex-1 min-w-0">
                    <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-sky-100 text-sky-800 text-xs font-bold mt-0.5">
                      {idx + 1}
                    </span>

                    <div className="flex-1 min-w-0 space-y-2">
                      {isEditing ? (
                        <div className="space-y-2">
                          <textarea
                            rows={2}
                            value={editText}
                            onChange={(e) => setEditText(e.target.value)}
                            className="w-full rounded-lg border border-purple-300 bg-white p-2.5 text-sm text-zinc-800 focus:outline-none focus:ring-2 focus:ring-purple-400"
                          />
                          <div className="flex items-center gap-2">
                            <button
                              type="button"
                              onClick={() => handleSaveEdit(claim.claim_id)}
                              disabled={loading}
                              className="rounded-lg bg-purple-600 px-3 py-1 text-xs font-bold text-white hover:bg-purple-700"
                            >
                              บันทึกการแก้ไข
                            </button>
                            <button
                              type="button"
                              onClick={handleCancelEdit}
                              className="rounded-lg border border-zinc-200 px-3 py-1 text-xs font-semibold text-zinc-600 hover:bg-zinc-50"
                            >
                              ยกเลิก
                            </button>
                          </div>
                        </div>
                      ) : (
                        <div>
                          <p
                            className={`text-sm sm:text-base font-semibold leading-relaxed ${
                              isExcluded ? 'line-through text-zinc-400' : 'text-zinc-800'
                            }`}
                          >
                            {claim.effective_text}
                          </p>

                          <div className="flex flex-wrap items-center gap-2 mt-2">
                            <span
                              className={`rounded-full border px-2 py-0.5 text-[11px] font-semibold ${sourceInfo.style}`}
                            >
                              {sourceInfo.label}
                            </span>

                            {claim.evidence_refs?.length > 0 && (
                              <button
                                type="button"
                                onClick={() =>
                                  setExpandedEvidenceId(
                                    expandedEvidenceId === claim.claim_id ? null : claim.claim_id,
                                  )
                                }
                                className="text-[11px] font-semibold text-sky-600 hover:text-sky-800 underline"
                              >
                                {expandedEvidenceId === claim.claim_id
                                  ? 'ซ่อนหลักฐาน'
                                  : `มาจากคำตอบใด (${claim.evidence_refs.length})`}
                              </button>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Actions: Exact / Partial / Rejected */}
                  {!isEditing && !isConfirmed && (
                    <div className="flex items-center gap-1.5 shrink-0">
                      <button
                        type="button"
                        onClick={() => onRateClaim(claim.claim_id, 'exact')}
                        disabled={loading}
                        className={`rounded-lg px-2.5 py-1 text-xs font-bold transition-all ${
                          isExact
                            ? 'bg-emerald-600 text-white shadow-xs'
                            : 'border border-zinc-200 text-zinc-600 hover:bg-emerald-50 hover:text-emerald-700 hover:border-emerald-200'
                        }`}
                        title="ตรงกับเป้าหมายและค่านิยมของฉัน"
                      >
                        ✓ ตรงกับฉัน
                      </button>

                      <button
                        type="button"
                        onClick={() => {
                          onRateClaim(claim.claim_id, 'partial')
                          handleStartEdit(claim.claim_id, claim.effective_text)
                        }}
                        disabled={loading}
                        className={`rounded-lg px-2.5 py-1 text-xs font-bold transition-all ${
                          isPartial
                            ? 'bg-purple-600 text-white shadow-xs'
                            : 'border border-zinc-200 text-zinc-600 hover:bg-purple-50 hover:text-purple-700 hover:border-purple-200'
                        }`}
                        title="ตรงบางส่วน และต้องการปรับถ้อยคำ"
                      >
                        ✏️ ปรับแก้
                      </button>

                      <button
                        type="button"
                        onClick={() => {
                          if (isExcluded) {
                            onRateClaim(claim.claim_id, 'exact')
                          } else {
                            onExcludeClaim(claim.claim_id)
                          }
                        }}
                        disabled={loading}
                        className={`rounded-lg px-2.5 py-1 text-xs font-bold transition-all ${
                          isExcluded
                            ? 'bg-zinc-600 text-white'
                            : 'border border-zinc-200 text-zinc-400 hover:bg-rose-50 hover:text-rose-600 hover:border-rose-200'
                        }`}
                        title="ไม่ตรงกับฉัน ไม่ส่งต่อข้อนี้ไปสร้างแกนหลัก"
                      >
                        {isExcluded ? 'กู้คืน' : '✕ ไม่ตรง'}
                      </button>
                    </div>
                  )}
                </div>

                {/* Evidence Disclosure Dropdown */}
                {expandedEvidenceId === claim.claim_id && claim.evidence_refs?.length > 0 && (
                  <div className="mt-3 pt-3 border-t border-sky-100 bg-sky-50/50 p-3 rounded-lg text-xs space-y-1 text-zinc-600">
                    <span className="font-bold text-sky-800 block mb-1">หลักฐานจากการตอบ:</span>
                    {claim.evidence_refs.map((ev, i) => (
                      <p key={i} className="italic text-zinc-700">
                        • "{ev.quote}" ({ev.evidence_type === 'hypothetical' ? 'สถานการณ์สมมติ' : 'คำตอบจริง'})
                      </p>
                    ))}
                  </div>
                )}
              </div>
            )
          })}
        </div>

        {/* Unresolved Topics Alert */}
        {unresolvedTopics.length > 0 && (
          <div className="rounded-xl border border-amber-200 bg-amber-50/70 p-4 text-xs text-amber-900 space-y-1">
            <span className="font-bold">⚠️ ประเด็นที่ยังรอการทบทวนเพิ่มเติม:</span>
            <ul className="list-disc list-inside">
              {unresolvedTopics.map((t, idx) => (
                <li key={idx}>{t}</li>
              ))}
            </ul>
          </div>
        )}

        {/* Action Bar: Save for now vs Proceed to Axis */}
        <div className="pt-6 border-t border-sky-100 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="text-xs text-zinc-500">
            {statusText ? (
              <span className="flex items-center gap-2 text-sky-700 font-semibold animate-pulse">
                <span className="h-2 w-2 rounded-full bg-sky-500" />
                {statusText}
              </span>
            ) : isConfirmed ? (
              <span className="text-emerald-700 font-semibold">
                ✓ บันทึกแก่นแท้ฉบับยืนยันเรียบร้อยแล้ว คุณสามารถกลับมาสร้างแกนหลักต่อได้ทุกเมื่อ
              </span>
            ) : (
              <span>การกดบันทึกจะยืนยันเฉพาะข้อสรุปที่คุณตรวจแล้ว และไม่รวมข้อที่ไม่ตรง</span>
            )}
          </div>

          <div className="flex flex-wrap items-center gap-3 w-full sm:w-auto">
            <button
              type="button"
              onClick={onRestart}
              disabled={loading}
              className="rounded-xl border border-zinc-200 px-4 py-2.5 text-xs font-semibold text-zinc-600 hover:bg-zinc-50 transition-all"
            >
              เริ่มทำแบบสอบถามใหม่
            </button>

            {!isConfirmed && (
              <button
                type="button"
                onClick={() => onConfirmBatch(false)}
                disabled={loading}
                className="rounded-xl border border-sky-300 bg-sky-50 px-5 py-2.5 text-xs sm:text-sm font-bold text-sky-700 hover:bg-sky-100 active:scale-98 transition-all"
              >
                💾 บันทึกไว้ก่อน (จบงานวันนี้)
              </button>
            )}

            <button
              type="button"
              onClick={() => onConfirmBatch(true)}
              disabled={loading}
              className="rounded-xl bg-gradient-to-r from-sky-600 to-blue-600 px-6 py-2.5 text-xs sm:text-sm font-bold text-white shadow-md shadow-sky-500/20 hover:from-sky-700 hover:to-blue-700 active:scale-98 transition-all flex items-center gap-1.5"
            >
              <span>สร้างแกนหลักต่อ →</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
