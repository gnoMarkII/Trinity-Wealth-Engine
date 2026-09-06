import { useState, useEffect } from 'react'
import Modal from '../../ui/Modal'
import { api } from '../../../api/client'
import type { DimeBatchScanWarningEvent, DimePdfTextResponseDTO } from '../../../api/types'

interface Props {
  warning: DimeBatchScanWarningEvent
  password?: string
  source?: 'dime' | 'wealthx' | 'scb'
  onClose: () => void
}

export default function DimePdfViewerModal({ warning, password, source = 'dime', onClose }: Props) {
  const [activeTab, setActiveTab] = useState<'pdf' | 'text'>('pdf')
  const [textData, setTextData] = useState<DimePdfTextResponseDTO | null>(null)
  const [loadingText, setLoadingText] = useState(false)
  const [textError, setTextError] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)

  const messageId = warning.message_id || ''
  const attachmentId = warning.attachment_id || ''

  const pdfInlineUrl = source === 'scb' && messageId
    ? api.getScbEmailHtmlUrl(messageId)
    : messageId && attachmentId
      ? (source === 'wealthx'
          ? api.getWealthXPdfUrl(messageId, attachmentId, password, true)
          : api.getDimePdfUrl(messageId, attachmentId, password, true))
      : ''

  const pdfDownloadUrl = source === 'scb'
    ? ''
    : messageId && attachmentId
      ? (source === 'wealthx'
          ? api.getWealthXPdfUrl(messageId, attachmentId, password, false)
          : api.getDimePdfUrl(messageId, attachmentId, password, false))
      : ''

  // Load raw text on demand when switching to 'text' tab
  useEffect(() => {
    if (activeTab === 'text' && !textData && !loadingText && messageId && attachmentId) {
      setLoadingText(true)
      setTextError(null)
      const fetchText = source === 'wealthx'
        ? api.getWealthXPdfText(messageId, attachmentId, password)
        : api.getDimePdfText(messageId, attachmentId, password)
      fetchText
        .then((res) => {
          setTextData(res)
        })
        .catch((err: any) => {
          setTextError(err?.message || 'ไม่สามารถสกัดข้อความจาก PDF ได้')
        })
        .finally(() => {
          setLoadingText(false)
        })
    }
  }, [activeTab, textData, loadingText, messageId, attachmentId, password, source])

  const handleCopyText = () => {
    if (!textData) return
    const allText = textData.pages.map((p) => `--- หน้า ${p.page_number} ---\n${p.text}`).join('\n\n')
    navigator.clipboard.writeText(allText)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <Modal
      titleId="dime-pdf-viewer-title"
      onClose={onClose}
      zIndexClassName="z-[70]"
      panelClassName="max-w-4xl rounded-3xl border border-amber-200 bg-white/95 p-6 sm:p-7 shadow-2xl shadow-amber-900/10 backdrop-blur-xl"
    >
      {/* Header */}
      <div className="flex items-start justify-between border-b border-amber-100 pb-4">
        <div className="flex items-center gap-3">
          <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-amber-50 text-amber-700 border border-amber-200/70 text-xl shadow-2xs">
            📄
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 id="dime-pdf-viewer-title" className="text-base sm:text-lg font-bold text-zinc-900">
                {warning.filename || 'เอกสารยืนยันการซื้อขาย'}
              </h3>
              <span className="rounded-full bg-amber-100 px-2.5 py-0.5 text-xs font-semibold text-amber-800 border border-amber-300/60">
                พักรายการ (Quarantined)
              </span>
            </div>
            <p className="text-xs text-zinc-500 mt-0.5">
              {warning.subject || 'Confirmation Note'} {warning.received_at ? `• ได้รับเมื่อ ${warning.received_at}` : ''}
            </p>
          </div>
        </div>
        <button
          type="button"
          onClick={onClose}
          aria-label="ปิดหน้าต่าง"
          className="rounded-xl p-2 text-zinc-400 hover:bg-zinc-100 hover:text-zinc-600 transition-colors cursor-pointer"
        >
          ✕
        </button>
      </div>

      {/* Quarantine Reason Callout */}
      <div className="mt-4 rounded-2xl bg-amber-50 border border-amber-200/80 p-4 shadow-2xs">
        <div className="flex items-start gap-2.5">
          <span className="text-base leading-none mt-0.5">⚠️</span>
          <div>
            <div className="text-xs font-bold text-amber-900">สาเหตุที่ระบบไม่นำเข้าอัตโนมัติ (Quarantine Reason):</div>
            <div className="text-xs font-mono text-amber-800 mt-1 break-words leading-relaxed">
              {warning.reason}
            </div>
          </div>
        </div>
      </div>

      {/* Toolbar: Tabs & Actions */}
      <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-b border-zinc-100 pb-3">
        {/* Tabs */}
        <div className="flex items-center gap-1 rounded-2xl bg-zinc-100 p-1">
          <button
            type="button"
            onClick={() => setActiveTab('pdf')}
            className={`rounded-xl px-3.5 py-1.5 text-xs font-semibold transition-all cursor-pointer ${
              activeTab === 'pdf'
                ? 'bg-white text-zinc-900 shadow-xs'
                : 'text-zinc-600 hover:text-zinc-900'
            }`}
          >
            {source === 'scb' ? '📧 ดูอีเมลฉบับเต็ม (HTML Email)' : '📄 ดูเอกสาร PDF'}
          </button>
          {source !== 'scb' && (
            <button
              type="button"
              onClick={() => setActiveTab('text')}
              className={`rounded-xl px-3.5 py-1.5 text-xs font-semibold transition-all cursor-pointer ${
                activeTab === 'text'
                  ? 'bg-white text-zinc-900 shadow-xs'
                  : 'text-zinc-600 hover:text-zinc-900'
              }`}
            >
              📝 ข้อความในเอกสาร (Raw Text)
            </button>
          )}
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-2">
          {activeTab === 'text' && textData && (
            <button
              type="button"
              onClick={handleCopyText}
              className="inline-flex items-center gap-1.5 rounded-xl border border-zinc-200 bg-white px-3 py-1.5 text-xs font-medium text-zinc-700 hover:bg-zinc-50 shadow-2xs transition-colors cursor-pointer"
            >
              {copied ? '✓ คัดลอกแล้ว' : '📋 คัดลอกข้อความ'}
            </button>
          )}
          {pdfInlineUrl && (
            <a
              href={pdfInlineUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 rounded-xl border border-sky-200 bg-sky-50 px-3 py-1.5 text-xs font-medium text-sky-700 hover:bg-sky-100 shadow-2xs transition-colors cursor-pointer"
            >
              ↗️ เปิดในแท็บใหม่
            </a>
          )}
          {pdfDownloadUrl && (
            <a
              href={pdfDownloadUrl}
              download={warning.filename || 'confirmation.pdf'}
              className="inline-flex items-center gap-1.5 rounded-xl border border-zinc-200 bg-white px-3 py-1.5 text-xs font-medium text-zinc-700 hover:bg-zinc-50 shadow-2xs transition-colors cursor-pointer"
            >
              📥 ดาวน์โหลด
            </a>
          )}
        </div>
      </div>

      {/* Main Tab Content */}
      <div className="mt-4">
        {activeTab === 'pdf' ? (
          <div>
            {pdfInlineUrl ? (
              <div className="relative overflow-hidden rounded-2xl border border-sky-100 bg-slate-50 shadow-inner">
                <iframe
                  src={pdfInlineUrl}
                  title={`${source === 'scb' ? 'SCBAM Email' : 'Dime PDF'} - ${warning.filename || warning.subject || 'Confirmation'}`}
                  className="w-full h-[540px] border-0 bg-white"
                />
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center p-12 rounded-2xl bg-zinc-50 text-zinc-500 text-xs">
                <span className="text-3xl mb-2">{source === 'scb' ? '📧' : '📁'}</span>
                {source === 'scb' ? 'ไม่พบข้อมูลเนื้อหาอีเมลสำหรับรายการนี้' : 'ไม่พบข้อมูลไฟล์แนบ PDF สำหรับรายการนี้'}
              </div>
            )}
            <p className="mt-2 text-[11px] text-zinc-400 text-center">
              {source === 'scb'
                ? '* แสดงเนื้อหาอีเมลยืนยันจาก SCBAM Fund Click โดยตรง สามารถกดปุ่ม "เปิดในแท็บใหม่" เพื่อเปิดดูเต็มหน้าจอ'
                : '* เอกสารถูกถอดรหัสชั่วคราวเพื่อให้เปิดอ่านได้สะดวก หากหน้าต่างไม่แสดงตัวอย่าง สามารถกดปุ่ม "เปิดในแท็บใหม่" หรือ "ดาวน์โหลด"'}
            </p>
          </div>
        ) : (
          <div className="max-h-[540px] overflow-y-auto space-y-4 pr-1">
            {loadingText ? (
              <div className="flex flex-col items-center justify-center p-16 text-zinc-500 text-xs gap-3">
                <div className="h-6 w-6 animate-spin rounded-full border-2 border-sky-500 border-t-transparent" />
                กำลังสกัดข้อความจากเอกสาร PDF...
              </div>
            ) : textError ? (
              <div className="rounded-2xl bg-red-50 border border-red-200 p-4 text-xs text-red-700">
                <div className="font-semibold">เกิดข้อผิดพลาดในการอ่านข้อความ:</div>
                <div className="mt-1">{textError}</div>
              </div>
            ) : textData && textData.pages.length > 0 ? (
              textData.pages.map((p) => (
                <div key={p.page_number} className="rounded-2xl border border-zinc-200 bg-slate-50 p-4 shadow-2xs">
                  <div className="flex items-center justify-between border-b border-zinc-200/80 pb-2 mb-3">
                    <span className="text-xs font-bold text-zinc-700">
                      📄 หน้า {p.page_number} จาก {textData.page_count}
                    </span>
                    <span className="text-[11px] text-zinc-400">
                      {p.text.length} ตัวอักษร
                    </span>
                  </div>
                  <pre className="font-mono text-xs text-slate-800 whitespace-pre-wrap leading-relaxed overflow-x-auto select-text">
                    {p.text.trim() || '(ไม่มีข้อความที่สกัดได้ในหน้านี้)'}
                  </pre>
                </div>
              ))
            ) : (
              <div className="p-12 text-center text-zinc-400 text-xs">
                ไม่มีข้อความในเอกสาร
              </div>
            )}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="mt-6 flex items-center justify-end border-t border-zinc-100 pt-4">
        <button
          type="button"
          onClick={onClose}
          className="rounded-xl bg-zinc-100 px-5 py-2 text-xs font-semibold text-zinc-700 hover:bg-zinc-200 transition-colors cursor-pointer"
        >
          ปิดหน้าต่าง
        </button>
      </div>
    </Modal>
  )
}
