import React from 'react'
import type { PromiseItemDTO } from '../../../api/types'
import { CloseIcon } from '../../ui/AppIcons'
import { InPlaceDialog } from '../../ui/Modal'

interface VerbatimQuoteModalProps {
  promise: PromiseItemDTO | null
  onClose: () => void
}

export const VerbatimQuoteModal: React.FC<VerbatimQuoteModalProps> = ({ promise, onClose }) => {
  if (!promise) return null

  const isResolved = promise.status !== 'Pending'

  return (
    <InPlaceDialog
      titleId="verbatim-quote-modal-title"
      onClose={onClose}
      backdropClassName="absolute inset-0 bg-slate-900/60 backdrop-blur-sm"
      panelClassName="w-full max-w-2xl rounded-2xl border border-edge bg-surface shadow-2xl p-6 space-y-5 max-h-[90vh] overflow-y-auto"
    >
        {/* Modal Header */}
        <div className="flex items-start justify-between border-b border-edge/60 pb-3">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono font-bold px-2 py-0.5 rounded bg-slate-100 text-slate-700">
                {promise.id}
              </span>
              <span className="text-xs font-semibold text-slate-500">
                แถลงใน {promise.made_in} ({promise.section === 'prepared' ? 'Prepared Remarks' : 'Q&A Session'})
              </span>
            </div>
            <h3 id="verbatim-quote-modal-title" className="text-base font-bold text-slate-900 mt-1">
              {promise.metric_th || promise.metric}
            </h3>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="ปิดหน้าต่าง"
            className="p-2 rounded-lg text-slate-500 hover:text-slate-700 hover:bg-surface-muted transition-colors"
          >
            <CloseIcon className="h-4 w-4" />
          </button>
        </div>

        {/* Commitment Metadata */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
          <div className="p-2.5 rounded-lg bg-surface-muted/40 border border-edge/40">
            <span className="text-slate-500 block text-[11px]">ผู้บริหาร</span>
  <span className="font-semibold text-slate-800 truncate block">
              {promise.speaker}
            </span>
          </div>
          <div className="p-2.5 rounded-lg bg-surface-muted/40 border border-edge/40">
            <span className="text-slate-500 block text-[11px]">หมวดหมู่ / ประเภท</span>
  <span className="font-semibold text-slate-800 capitalize">
              {promise.category} ({promise.type === 'A_numeric' ? 'ตัวเลข' : 'Milestone'})
            </span>
          </div>
          <div className="p-2.5 rounded-lg bg-surface-muted/40 border border-edge/40">
            <span className="text-slate-500 block text-[11px]">กำหนดส่งมอบ</span>
  <span className="font-semibold text-slate-800">
              {promise.due} ({promise.due_precision})
            </span>
          </div>
          <div className="p-2.5 rounded-lg bg-surface-muted/40 border border-edge/40">
            <span className="text-slate-500 block text-[11px]">สถานะปัจจุบัน</span>
  <span className="font-bold text-slate-800">
              {promise.status}
            </span>
          </div>
        </div>

        {/* 1. Original Verbatim Quote */}
        <div className="space-y-2">
  <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-700">
            <span>🗣️</span>
            <span>คำพูดต้นฉบับจากบทสนทนา (Original Verbatim Quote):</span>
          </div>
  <blockquote className="p-3.5 rounded-xl border-l-4 border-sky-500 bg-sky-500/5 text-xs italic text-slate-800 leading-relaxed font-serif">
            "{promise.quote}"
          </blockquote>
        </div>

        {/* 2. Resolving Evidence Quote (if resolved) */}
        {isResolved && (
          <div className="space-y-2 pt-2 border-t border-edge/60">
  <div className="flex items-center justify-between text-xs font-semibold text-slate-700">
              <div className="flex items-center gap-1.5">
                <span>🔍</span>
                <span>หลักฐานการตรวจสอบในไตรมาส {promise.resolved_in || promise.due}:</span>
              </div>
              {promise.delta_pct != null && (
                <span className={`px-2 py-0.5 rounded text-[11px] font-bold ${
  promise.delta_pct > 0 ?'bg-emerald-100 text-emerald-800':
  promise.delta_pct < 0 ?'bg-rose-100 text-rose-800':
                  'bg-slate-100 text-slate-800'
                }`}>
                  {promise.delta_pct > 0 ? '+' : ''}{(promise.delta_pct * 100).toFixed(1)}% vs Target
                </span>
              )}
            </div>

            {promise.evidence_quote ? (
  <blockquote className="p-3.5 rounded-xl border-l-4 border-emerald-500 bg-emerald-500/5 text-xs italic text-slate-800 leading-relaxed font-serif">
                "{promise.evidence_quote}"
              </blockquote>
            ) : (
  <div className="p-3 rounded-xl bg-slate-100 text-xs text-slate-500 italic">
                ไม่มีหลักฐานระบุชัดเจน หรือถูกละทิ้งโดยไม่มีการแถลงต่อสาธารณะ (Quietly dropped)
              </div>
            )}
          </div>
        )}

        {/* Notes & Audit Explanation */}
        {(promise.notes_th || promise.notes) && (
          <div className="p-3 rounded-xl bg-surface-muted/30 border border-edge/50 space-y-1 text-xs">
  <span className="font-semibold text-slate-600 block text-[11px]">
              หมายเหตุการตรวจสอบ (Audit Notes):
            </span>
  <p className="text-slate-800 leading-relaxed">
              {promise.notes_th || promise.notes}
            </p>
          </div>
        )}

        {/* Modal Actions */}
        <div className="flex justify-end pt-2">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-xl bg-section-accent text-white text-xs font-semibold hover:bg-section-accent/90 transition-colors shadow-sm"
          >
            ปิดหน้าต่าง
          </button>
        </div>
    </InPlaceDialog>
  )
}
