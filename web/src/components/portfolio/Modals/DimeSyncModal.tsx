import { useState, useRef, useEffect } from 'react'
import Modal from '../../ui/Modal'
import DimePdfViewerModal from './DimePdfViewerModal'
import { api } from '../../../api/client'
import type {
  ActualPortfolioStateDTO,
  DimeBatchScanWarningEvent,
  DimeEmailMetadataDTO,
  DimeScanResponseDTO,
  DimeStagedItemDTO,
} from '../../../api/types'

interface Props {
  portfolioId: string
  onClose: () => void
  onSuccess: (state: ActualPortfolioStateDTO) => void
}

export default function DimeSyncModal({ portfolioId, onClose, onSuccess }: Props) {
  const [activeSource, setActiveSource] = useState<'dime' | 'wealthx' | 'scb'>('dime')
  const [activeTab, setActiveTab] = useState<'upload' | 'email'>('upload')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [file, setFile] = useState<File | null>(null)

  // Email scan state
  const [emailQuery, setEmailQuery] = useState('')
  const [emails, setEmails] = useState<DimeEmailMetadataDTO[]>([])
  const [loadingEmails, setLoadingEmails] = useState(false)
  const [selectedEmail, setSelectedEmail] = useState<DimeEmailMetadataDTO | null>(null)

  // Staging / Analysis state
  const [analyzing, setAnalyzing] = useState(false)
  const [stagedResult, setStagedResult] = useState<DimeScanResponseDTO | null>(null)
  const [selectedItemIds, setSelectedItemIds] = useState<string[]>([])
  const [committing, setCommitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [successMessage, setSuccessMessage] = useState<string | null>(null)

  useEffect(() => {
    if (stagedResult?.items) {
      setSelectedItemIds(stagedResult.items.map((it: DimeStagedItemDTO) => it.item_id))
    } else {
      setSelectedItemIds([])
    }
  }, [stagedResult])

  const toggleSelectAll = () => {
    if (!stagedResult?.items) return
    if (selectedItemIds.length === stagedResult.items.length) {
      setSelectedItemIds([])
    } else {
      setSelectedItemIds(stagedResult.items.map((it: DimeStagedItemDTO) => it.item_id))
    }
  }

  const toggleSelectItem = (itemId: string) => {
    setSelectedItemIds((prev) =>
      prev.includes(itemId) ? prev.filter((id) => id !== itemId) : [...prev, itemId]
    )
  }

  // Batch Sync state
  const [batchSyncing, setBatchSyncing] = useState(false)
  const [forceRescan, setForceRescan] = useState(false)
  const [batchProgress, setBatchProgress] = useState<{
    current: number
    total: number
    percent: number
    items_found: number
    subject?: string
  } | null>(null)
  const [warnings, setWarnings] = useState<DimeBatchScanWarningEvent[]>([])
  const [inspectingWarning, setInspectingWarning] = useState<DimeBatchScanWarningEvent | null>(null)

  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleSourceChange = (src: 'dime' | 'wealthx' | 'scb') => {
    if (src === activeSource) return
    setActiveSource(src)
    if (src === 'scb') {
      setActiveTab('email')
    }
    setEmails([])
    setSelectedEmail(null)
    setFile(null)
    setStagedResult(null)
    setWarnings([])
    setError(null)
    setSuccessMessage(null)
    setBatchProgress(null)
  }

  const handleSearchEmails = async () => {
    setLoadingEmails(true)
    setError(null)
    try {
      if (activeSource === 'wealthx') {
        const res = await api.getWealthXEmails(emailQuery.trim() || undefined, 15)
        setEmails(res.emails || [])
        if (res.emails.length === 0) {
          setError('ไม่พบอีเมลใบยืนยันการซื้อขายจาก WealthX (noreply@wealthx.co) ในกล่องข้อความ')
        }
      } else if (activeSource === 'scb') {
        const res = await api.getScbEmails(emailQuery.trim() || undefined, 15)
        setEmails(res.emails || [])
        if (res.emails.length === 0) {
          setError('ไม่พบอีเมลยืนยันการทำรายการซื้อกองทุนจาก SCBAM (fundclick.scbam@scb.co.th) ในกล่องข้อความ')
        }
      } else {
        const res = await api.getDimeEmails(emailQuery.trim() || undefined, 15)
        setEmails(res.emails || [])
        if (res.emails.length === 0) {
          setError('ไม่พบอีเมล Trade Confirmation จาก Dime ในกล่องข้อความ')
        }
      }
    } catch (err: any) {
      setError(err?.message || 'ไม่สามารถค้นหาอีเมลได้ กรุณาตรวจสอบการตั้งค่า Gmail IMAP')
    } finally {
      setLoadingEmails(false)
    }
  }

  const handleAnalyzeUpload = async () => {
    if (!file) {
      setError('กรุณาเลือกไฟล์ PDF Trade Confirmation')
      return
    }
    setAnalyzing(true)
    setError(null)
    setWarnings([])
    setStagedResult(null)
    try {
      const res = activeSource === 'wealthx'
        ? await api.scanWealthXUpload(file, password.trim() || undefined)
        : await api.scanDimeUpload(file, password.trim() || undefined)
      setStagedResult(res)
    } catch (err: any) {
      setError(err?.message || 'ไม่สามารถวิเคราะห์ไฟล์ PDF ได้')
    } finally {
      setAnalyzing(false)
    }
  }

  const handleAnalyzeEmail = async (em: DimeEmailMetadataDTO) => {
    setSelectedEmail(em)
    setAnalyzing(true)
    setError(null)
    setWarnings([])
    setStagedResult(null)
    try {
      const res = activeSource === 'wealthx'
        ? await api.scanWealthXEmail(em.message_id, em.attachment_id, password.trim() || undefined)
        : activeSource === 'scb'
        ? await api.scanScbEmail(em.message_id, portfolioId)
        : await api.scanDimeEmail(em.message_id, em.attachment_id, password.trim() || undefined)
      setStagedResult(res)
    } catch (err: any) {
      setError(err?.message || 'ไม่สามารถดาวน์โหลดและวิเคราะห์ไฟล์แนบจากอีเมลได้')
    } finally {
      setAnalyzing(false)
    }
  }

  const handleBatchSyncAll = async () => {
    setBatchSyncing(true)
    setError(null)
    setWarnings([])
    setStagedResult(null)
    setSuccessMessage(null)
    setBatchProgress({ current: 0, total: 0, percent: 0, items_found: 0, subject: 'กำลังเชื่อมต่อ Gmail...' })

    const callbacks = {
      onProgress: (data: any) => {
        setBatchProgress(data)
      },
      onWarning: (data: any) => {
        setWarnings((prev) => [...prev, data])
      },
      onComplete: (data: any) => {
        setStagedResult({
          scan_id: data.scan_id,
          item_count: data.item_count,
          items: data.items,
        })
        if (data.warnings && data.warnings.length > 0) {
          setWarnings(data.warnings)
        }
        if (data.item_count === 0 && data.skipped_synced_count > 0) {
          setSuccessMessage(`ซิงค์ข้อมูลเรียบร้อยแล้ว: เอกสารทั้งหมด (${data.skipped_synced_count} ฉบับ) ได้รับการบันทึกลงสมุดบัญชีแล้ว`)
        }
      },
      onError: (err: any) => {
        setError(err.message || 'เกิดข้อผิดพลาดขณะทำการซิงค์')
      },
    }

    try {
      if (activeSource === 'wealthx') {
        await api.streamBatchWealthXSync(
          {
            password: password.trim() || undefined,
            force_rescan: forceRescan,
            portfolio_id: portfolioId,
          },
          callbacks
        )
      } else if (activeSource === 'scb') {
        await api.streamBatchScbSync(
          {
            portfolio_id: portfolioId,
          },
          callbacks
        )
      } else {
        await api.streamBatchDimeSync(
          {
            password: password.trim() || undefined,
            force_rescan: forceRescan,
            portfolio_id: portfolioId,
          },
          callbacks
        )
      }
    } catch (err: any) {
      setError(err?.message || 'ไม่สามารถเริ่มการซิงค์ได้')
    } finally {
      setBatchSyncing(false)
    }
  }

  const handleCommit = async () => {
    if (!stagedResult) return
    setCommitting(true)
    setError(null)
    try {
      const res = activeSource === 'wealthx'
        ? await api.commitWealthXTrades(stagedResult.scan_id, portfolioId)
        : activeSource === 'scb'
        ? await api.commitScbTrades(stagedResult.scan_id, portfolioId, selectedItemIds)
        : await api.commitDimeTrades(stagedResult.scan_id, portfolioId)
      setSuccessMessage(`นำเข้ารายการสำเร็จเรียบร้อยแล้ว จำนวน ${res.imported_count} รายการ`)
      setTimeout(() => {
        onSuccess(res.state)
        onClose()
      }, 1200)
    } catch (err: any) {
      setError(err?.message || 'การบันทึกรายการเทรดลงสมุดบัญชีไม่สำเร็จ')
    } finally {
      setCommitting(false)
    }
  }


  return (
    <Modal
      titleId="transaction-sync-title"
      onClose={onClose}
      panelClassName="max-w-3xl rounded-3xl border border-sky-100 bg-white/95 p-6 sm:p-7 shadow-2xl shadow-sky-900/10 backdrop-blur-xl"
    >
      {/* Modal Header */}
      <div className="flex items-center justify-between border-b border-sky-100 pb-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-sky-50 text-flow-blue border border-sky-200/60 text-lg shadow-2xs">
            📥
          </div>
          <div>
            <h3 id="transaction-sync-title" className="text-base sm:text-lg font-bold text-zinc-900">
              Sync ข้อมูลรายการเทรด (Transaction Sync)
            </h3>
            <p className="text-xs text-sky-700">
              นำเข้าประวัติการซื้อขายจากโบรกเกอร์เข้าสู่ Ledger พอร์ตโฟลิโออัตโนมัติ
            </p>
          </div>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="rounded-xl p-1.5 text-zinc-400 hover:bg-sky-50 hover:text-zinc-600 transition-colors"
          aria-label="Close"
        >
          ✕
        </button>
      </div>

      <div className="space-y-4 p-1 text-zinc-800 mt-4">
        {/* Step 1: Provider / Broker Section */}
        {!stagedResult && (
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <div className="text-xs font-bold text-zinc-700 flex items-center gap-1.5">
                <span>🏛️</span>
                <span>เลือกโบรกเกอร์ / ผู้ให้บริการ (Select Broker Source)</span>
              </div>
              <span className="text-[11px] text-purple-700 font-semibold flex items-center gap-1">
                <span className="inline-block h-1.5 w-1.5 rounded-full bg-purple-600"></span>
                <span>
                  {activeSource === 'wealthx'
                    ? 'WealthX กองทุนรวมไทย'
                    : activeSource === 'scb'
                    ? 'SCB (Thai Funds)'
                    : 'Dime! สหรัฐฯ/กองทุน'}
                </span>
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              {/* Dime Card */}
              <div
                onClick={() => handleSourceChange('dime')}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault()
                    handleSourceChange('dime')
                  }
                }}
                role="button"
                tabIndex={0}
                aria-pressed={activeSource === 'dime'}
                aria-label="Select Dime broker source"
                className={`p-3.5 rounded-2xl border transition-all cursor-pointer ${
                  activeSource === 'dime'
                    ? 'border-violet-300 bg-violet-50/70 shadow-sm ring-2 ring-violet-400/30'
                    : 'border-zinc-200 bg-white/70 hover:bg-zinc-50 opacity-75'
                }`}
              >
                <div className="flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-white shadow-2xs text-lg border border-violet-100">
                    🟣
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center justify-between">
                      <h4 className="text-xs sm:text-sm font-bold text-zinc-900">Dime!</h4>
                      <span
                        className={`text-[10px] font-bold px-2 py-0.5 rounded-md ${
                          activeSource === 'dime'
                            ? 'bg-violet-100 text-violet-800'
                            : 'bg-zinc-100 text-zinc-600'
                        }`}
                      >
                        {activeSource === 'dime' ? 'กำลังเลือก' : 'เลือก'}
                      </span>
                    </div>
                    <p className="text-[11px] text-zinc-500 truncate">
                      KKP Securities • หุ้น US & กองทุน Dime
                    </p>
                  </div>
                </div>
              </div>

              {/* WealthX Card */}
              <div
                onClick={() => handleSourceChange('wealthx')}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault()
                    handleSourceChange('wealthx')
                  }
                }}
                role="button"
                tabIndex={0}
                aria-pressed={activeSource === 'wealthx'}
                aria-label="Select WealthX broker source"
                className={`p-3.5 rounded-2xl border transition-all cursor-pointer ${
                  activeSource === 'wealthx'
                    ? 'border-emerald-300 bg-emerald-50/70 shadow-sm ring-2 ring-emerald-400/30'
                    : 'border-zinc-200 bg-white/70 hover:bg-zinc-50 opacity-75'
                }`}
              >
                <div className="flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-white shadow-2xs text-lg border border-emerald-100">
                    🟢
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center justify-between">
                      <h4 className="text-xs sm:text-sm font-bold text-zinc-900">WealthX</h4>
                      <span
                        className={`text-[10px] font-bold px-2 py-0.5 rounded-md ${
                          activeSource === 'wealthx'
                            ? 'bg-emerald-100 text-emerald-800'
                            : 'bg-zinc-100 text-zinc-600'
                        }`}
                      >
                        {activeSource === 'wealthx' ? 'กำลังเลือก' : 'เลือก'}
                      </span>
                    </div>
                    <p className="text-[11px] text-zinc-500 truncate">
                      บล. เวลธ์ เอกซ์ • กองทุนรวมไทย
                    </p>
                  </div>
                </div>
              </div>

              {/* SCB Card */}
              <div
                onClick={() => handleSourceChange('scb')}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault()
                    handleSourceChange('scb')
                  }
                }}
                role="button"
                tabIndex={0}
                aria-pressed={activeSource === 'scb'}
                aria-label="Select SCB broker source"
                className={`p-3.5 rounded-2xl border transition-all cursor-pointer ${
                  activeSource === 'scb'
                    ? 'border-purple-300 bg-purple-50/70 shadow-sm ring-2 ring-purple-400/30'
                    : 'border-zinc-200 bg-white/70 hover:bg-zinc-50 opacity-75'
                }`}
              >
                <div className="flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-white shadow-2xs text-lg border border-purple-100">
                    🟣
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center justify-between">
                      <h4 className="text-xs sm:text-sm font-bold text-zinc-900">SCB (Thai Funds)</h4>
                      <span
                        className={`text-[10px] font-bold px-2 py-0.5 rounded-md ${
                          activeSource === 'scb'
                            ? 'bg-purple-100 text-purple-800'
                            : 'bg-zinc-100 text-zinc-600'
                        }`}
                      >
                        {activeSource === 'scb' ? 'กำลังเลือก' : 'เลือก'}
                      </span>
                    </div>
                    <p className="text-[11px] text-zinc-500 truncate">
                      SCBAM Fund Click • กองทุนรวมไทย
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Reassurance Banner: Institutional Integrity (Clean Flow theme) */}
        {!stagedResult && (
          <div className="rounded-2xl border border-sky-200/80 bg-gradient-to-r from-sky-50/90 via-white to-sky-50/60 p-3.5 text-xs text-sky-950 shadow-2xs">
            <div className="flex items-start gap-3">
              <span className="text-xl">🛡️</span>
              <div className="space-y-0.5">
                <p className="font-bold text-sky-950">
                  ระบบตรวจสอบและจัดสรรต้นทุนอัตโนมัติ (Automated Reconciliation)
                </p>
                <p className="text-[11px] leading-relaxed text-sky-800">
                  ระบบจะตรวจสอบยอดเงินรวม ค่าธรรมเนียมตามจริง และคัดกรองรายการซ้ำซ้อน (Deduplication) อัตโนมัติก่อนบันทึกลง Ledger เพื่อให้ข้อมูลพอร์ตแม่นยำ 100%
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Error Feedback */}
        {error && (
          <div className="rounded-2xl border border-rose-200 bg-rose-50/90 p-3.5 text-xs text-rose-800 shadow-2xs flex items-start gap-2 animate-fade-in">
            <span className="text-base">❌</span>
            <div className="flex-1">
              <strong className="font-bold">เกิดข้อผิดพลาด:</strong> {error}
            </div>
          </div>
        )}

        {/* Success Feedback */}
        {successMessage && (
          <div className="rounded-2xl border border-emerald-200 bg-emerald-50/90 p-3.5 text-xs text-emerald-800 shadow-2xs flex items-start gap-2 animate-fade-in">
            <span className="text-base">✅</span>
            <div className="flex-1 font-semibold">{successMessage}</div>
          </div>
        )}

        {/* Review Stage: Staged Items Display */}
        {stagedResult ? (
          <div className="space-y-4 animate-fade-in">
            {/* Warning / Quarantine Banner */}
            {warnings.length > 0 && (
              <div className="rounded-2xl border border-amber-200 bg-amber-50/90 p-4 text-xs text-amber-900 shadow-2xs space-y-2.5 animate-fade-in">
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2 font-bold text-amber-950">
                    <span className="text-base">⚠️</span>
                    <span>พบข้อควรระวัง / รายการที่ถูกพัก (Quarantined) จำนวน {warnings.length} ฉบับ</span>
                  </div>
                  <span className="text-[11px] text-amber-700 bg-amber-100/70 px-2 py-0.5 rounded-lg border border-amber-200">
                    คลิกเพื่อเปิดดูและตรวจสอบเอกสาร PDF
                  </span>
                </div>
                <p className="text-[11px] text-amber-800 leading-relaxed">
                  รายการเหล่านี้มีข้อขัดแย้ง (Conflict) กับ Ledger หรือรูปแบบตัวเลขไม่ตรงเกณฑ์ ระบบจึงพักรายการไว้โดยอัตโนมัติ ไม่นำเข้ารายการที่ไม่ปลอดภัย เพื่อรักษาความถูกต้องของสมุดบัญชี
                </p>
                <div className="max-h-56 overflow-y-auto space-y-2 pt-1">
                  {warnings.map((w, idx) => {
                    const canView = Boolean(w.message_id && (w.attachment_id || w.can_preview))
                    return (
                      <div
                        key={idx}
                        className="bg-white/95 p-3 rounded-2xl border border-amber-200/80 shadow-2xs flex flex-col sm:flex-row sm:items-center justify-between gap-2.5 transition-all hover:border-amber-300"
                      >
                        <div className="space-y-1 flex-1 min-w-0">
                          <div className="flex items-center gap-2 flex-wrap">
                            <span className="font-bold text-zinc-900 text-xs truncate max-w-sm">
                              {w.filename || w.subject || 'Confirmation Note'}
                            </span>
                            {w.received_at && (
                              <span className="text-[10px] text-zinc-400">({w.received_at})</span>
                            )}
                          </div>
                          <div className="font-mono text-[11px] text-amber-900 break-words leading-relaxed">
                            {w.reason}
                          </div>
                        </div>
                        {canView && (
                          <div className="flex items-center gap-1.5 shrink-0">
                            <button
                              type="button"
                              onClick={() => setInspectingWarning(w)}
                              className="inline-flex items-center gap-1 rounded-xl bg-amber-100/90 hover:bg-amber-200 text-amber-950 font-bold px-3 py-1.5 text-xs transition-all cursor-pointer border border-amber-300/60 shadow-2xs active:scale-95"
                            >
                              {activeSource === 'scb' ? '📧 ดูอีเมล' : '📄 ดูเอกสาร PDF'}
                            </button>
                            {activeSource !== 'scb' && w.message_id && w.attachment_id && (
                              <a
                                href={
                                  activeSource === 'wealthx'
                                    ? api.getWealthXPdfUrl(w.message_id, w.attachment_id, password.trim() || undefined, false)
                                    : api.getDimePdfUrl(w.message_id, w.attachment_id, password.trim() || undefined, false)
                                }
                                download={w.filename || 'confirmation.pdf'}
                                className="inline-flex items-center rounded-xl bg-zinc-100 hover:bg-zinc-200 text-zinc-700 px-2.5 py-1.5 text-xs transition-colors cursor-pointer border border-zinc-200 shadow-2xs"
                                title="ดาวน์โหลด PDF ต้นฉบับ"
                              >
                                📥
                              </a>
                            )}
                          </div>
                        )}
                      </div>
                    )
                  })}
                </div>
              </div>
            )}

            {/* Staged Items Table */}
            <div className="flex items-center justify-between">
              <div>
                <h4 className="text-sm font-bold text-sky-950">
                  ตรวจสอบรายการที่พบ ({stagedResult.item_count} รายการ)
                </h4>
                <div className="flex items-center gap-2 text-xs text-sky-700 mt-0.5">
                  <span>
                    แหล่งข้อมูล:{' '}
                    <strong className="text-zinc-800 font-semibold">
                      {activeSource === 'wealthx'
                        ? 'WealthX Confirmation (กองทุนรวมไทย)'
                        : activeSource === 'scb'
                        ? 'SCBAM Fund Click (กองทุนรวมไทย)'
                        : 'Dime! Confirmation (KKP Securities)'}
                    </strong>
                  </span>
                  <span>•</span>
                  <span>
                    เลือก <strong className="text-zinc-800 font-semibold">{selectedItemIds.length}</strong> จาก {stagedResult.item_count} รายการ
                  </span>
                  <span>•</span>
                  <span>Scan ID: <code className="font-mono text-[11px] bg-white px-1.5 py-0.5 rounded border border-sky-200">{stagedResult.scan_id}</code></span>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setStagedResult(null)}
                className="rounded-xl border border-sky-200 bg-white px-3 py-1.5 text-xs font-semibold text-sky-800 hover:bg-sky-50 transition-colors cursor-pointer shadow-2xs"
              >
                ← เลือกไฟล์ใหม่
              </button>
            </div>

            <div className="max-h-80 overflow-y-auto rounded-2xl border border-sky-100 shadow-sm bg-white">
              <table className="w-full text-left text-xs">
                <thead className="bg-sky-50/80 text-sky-950 sticky top-0 backdrop-blur-md">
                  <tr>
                    <th className="py-2.5 px-3 w-8">
                      <input
                        type="checkbox"
                        checked={stagedResult.items.length > 0 && selectedItemIds.length === stagedResult.items.length}
                        onChange={toggleSelectAll}
                        className="rounded border-sky-300 text-flow-blue focus:ring-sky-500 cursor-pointer"
                        aria-label="Select all staged trades"
                        title="เลือกทั้งหมด"
                      />
                    </th>
                    <th className="py-2.5 px-3">วันที่</th>
                    <th className="py-2.5 px-3">สินทรัพย์</th>
                    <th className="py-2.5 px-3">Action</th>
                    <th className="py-2.5 px-3 text-right">จำนวน</th>
                    <th className="py-2.5 px-3 text-right">ราคา</th>
                    <th className="py-2.5 px-3 text-right">Gross</th>
                    <th className="py-2.5 px-3 text-right">ค่าธรรมเนียม</th>
                    <th className="py-2.5 px-3 text-right">ยอดสุทธิ (Net)</th>
                    <th className="py-2.5 px-3">Confirm No.</th>
                    <th className="py-2.5 px-3">Order ID</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-sky-50 bg-white">
                  {stagedResult.items.map((it: DimeStagedItemDTO) => {
                    const isSelected = selectedItemIds.includes(it.item_id)
                    return (
                      <tr key={it.item_id} className={`hover:bg-sky-50/50 transition-colors ${isSelected ? 'bg-sky-50/20' : 'opacity-60'}`}>
                        <td className="py-2.5 px-3">
                          <input
                            type="checkbox"
                            checked={isSelected}
                            onChange={() => toggleSelectItem(it.item_id)}
                            className="rounded border-sky-300 text-flow-blue focus:ring-sky-500 cursor-pointer"
                            aria-label={`Select ${it.symbol} ${it.trade_date}`}
                          />
                        </td>
                        <td className="py-2.5 px-3 font-mono text-zinc-600">{it.trade_date}</td>
                        <td className="py-2.5 px-3 font-bold text-zinc-900">
                          <div className="flex items-center gap-1.5">
                            <span>{it.symbol}</span>
                            {it.asset_type === 'Fund' ? (
                              <span className="rounded bg-violet-50 border border-violet-200 px-1.5 py-0.5 text-[10px] font-semibold text-violet-700">
                                Fund
                              </span>
                            ) : (
                              <span className="rounded bg-sky-50 border border-sky-200 px-1.5 py-0.5 text-[10px] font-semibold text-sky-700">
                                Stock
                              </span>
                            )}
                          </div>
                        </td>
                        <td className="py-2.5 px-3">
                          <span
                            className={`rounded-lg px-2 py-0.5 text-[10px] font-bold ${
                              it.action === 'BUY'
                                ? 'bg-emerald-100 text-emerald-800'
                                : 'bg-rose-100 text-rose-800'
                            }`}
                          >
                            {it.action}
                          </span>
                        </td>
                      <td className="py-2.5 px-3 text-right font-mono text-zinc-700">{it.units}</td>
                      <td className="py-2.5 px-3 text-right font-mono text-zinc-700">{it.price}</td>
                      <td className="py-2.5 px-3 text-right font-mono text-zinc-500">{it.gross_amount}</td>
                      <td className="py-2.5 px-3 text-right font-mono text-zinc-500">
                        {it.fees.commission} ({it.fees.fee_currency})
                      </td>
                      <td className="py-2.5 px-3 text-right font-mono font-bold text-zinc-900">
                        {it.net_amount} {it.currency}
                      </td>
                      <td className="py-2.5 px-3 font-mono text-[11px] text-zinc-500">{it.confirmation_no}</td>
                      <td className="py-2.5 px-3 font-mono text-[11px] text-sky-800 font-semibold">{it.order_id || '-'}</td>
                    </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>

            <div className="flex items-center justify-end gap-2.5 pt-3 border-t border-sky-100">
              <button
                type="button"
                onClick={onClose}
                disabled={committing}
                className="rounded-xl border border-zinc-200 bg-white px-4 py-2 text-xs font-semibold text-zinc-700 hover:bg-zinc-50 transition-colors cursor-pointer"
              >
                ยกเลิก
              </button>
              <button
                type="button"
                onClick={handleCommit}
                disabled={committing}
                className="rounded-xl bg-flow-blue px-6 py-2 text-xs font-bold text-white shadow-md hover:bg-sky-600 active:scale-95 transition-all disabled:opacity-50 cursor-pointer"
              >
                {committing ? 'กำลังบันทึกลง Ledger และ Replay...' : `ยืนยันนำเข้า ${stagedResult.item_count} รายการ`}
              </button>
            </div>
          </div>
        ) : (
          /* Selection / Input Stage */
          <div className="space-y-4">
            {/* Flow-Themed Segmented Tabs */}
            {activeSource !== 'scb' && (
              <div className="flex rounded-2xl bg-sky-100/60 p-1 border border-sky-200/60">
                <button
                  type="button"
                  onClick={() => setActiveTab('upload')}
                  className={`flex-1 py-2 px-4 text-xs font-bold rounded-xl transition-all cursor-pointer ${
                    activeTab === 'upload'
                      ? 'bg-white text-flow-blue shadow-xs'
                      : 'text-sky-800 hover:text-sky-950 font-medium'
                  }`}
                >
                  📤 อัปโหลดไฟล์ PDF
                </button>
                <button
                  type="button"
                  onClick={() => setActiveTab('email')}
                  className={`flex-1 py-2 px-4 text-xs font-bold rounded-xl transition-all cursor-pointer ${
                    activeTab === 'email'
                      ? 'bg-white text-flow-blue shadow-xs'
                      : 'text-sky-800 hover:text-sky-950 font-medium'
                  }`}
                >
                  📧 ค้นหาจาก Gmail
                </button>
              </div>
            )}

            {/* Password input for protected PDFs */}
            {activeSource !== 'scb' ? (
              <div className="rounded-2xl border border-sky-100 bg-sky-50/40 p-3.5">
                <div className="flex items-center justify-between mb-1.5">
                  <label htmlFor="dime-pdf-password" className="block text-xs font-bold text-sky-950">
                    🔑 รหัสผ่านเปิดไฟล์ PDF (PDF Password)
                  </label>
                  <span className="text-[11px] text-sky-700">
                    เว้นว่างไว้เพื่อใช้ค่าเริ่มต้น (.env:{' '}
                    {activeSource === 'wealthx' ? 'WEALTHX_PDF_PASSWORD' : 'DIME_PDF_PASSWORD'})
                  </span>
                </div>
                <div className="relative">
                  <input
                    id="dime-pdf-password"
                    type={showPassword ? 'text' : 'password'}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="เว้นว่างไว้เพื่อใช้รหัสเริ่มต้นจากระบบ"
                    className="w-full rounded-xl border border-sky-200 bg-white px-3.5 py-2 pr-10 text-xs text-zinc-900 focus:ring-2 focus:ring-flow-blue/20 focus:border-flow-blue focus:outline-none transition-all"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-2.5 top-1/2 -translate-y-1/2 text-xs text-zinc-400 hover:text-sky-600 p-1 cursor-pointer"
                    title={showPassword ? 'ซ่อนรหัสผ่าน' : 'แสดงรหัสผ่าน'}
                  >
                    {showPassword ? '👁️' : '🔒'}
                  </button>
                </div>
              </div>
            ) : (
              <div className="rounded-2xl border border-purple-200 bg-purple-50/60 p-3.5 text-xs text-purple-900 flex items-center gap-2.5 shadow-2xs">
                <span className="text-base">💡</span>
                <div>
                  <span className="font-semibold">SCBAM Fund Click (Direct Email Confirmation):</span>{' '}
                  <span className="text-purple-800">
                    คำสั่งซื้อส่งตรงจาก <code className="font-mono text-purple-900 bg-white/80 px-1 py-0.5 rounded border border-purple-200">fundclick.scbam@scb.co.th</code> เป็นอีเมล HTML จึงไม่ต้องใช้รหัสผ่าน PDF
                  </span>
                </div>
              </div>
            )}

            {/* Upload Tab */}
            {activeTab === 'upload' && activeSource !== 'scb' && (
              <div className="space-y-3 animate-fade-in">
                <div
                  onClick={() => fileInputRef.current?.click()}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' || event.key === ' ') {
                      event.preventDefault()
                      fileInputRef.current?.click()
                    }
                  }}
                  role="button"
                  tabIndex={0}
                  aria-label="Choose a PDF file to upload"
                  className="flex flex-col items-center justify-center p-7 border-2 border-dashed border-sky-200 rounded-3xl cursor-pointer hover:border-flow-blue hover:bg-sky-50/70 transition-all bg-sky-50/30 text-center"
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="application/pdf"
                    className="hidden"
                    onChange={(e) => {
                      if (e.target.files && e.target.files[0]) {
                        setFile(e.target.files[0])
                      }
                    }}
                  />
                  <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-white text-flow-blue text-2xl mb-2 shadow-xs border border-sky-100">
                    📄
                  </div>
                  <p className="text-xs font-bold text-zinc-800">
                    {file ? file.name : 'คลิกเพื่อเลือกไฟล์ PDF หรือลากไฟล์มาวางที่นี่'}
                  </p>
                  {file ? (
                    <span className="inline-flex items-center gap-1.5 mt-2 rounded-lg bg-sky-100 px-2.5 py-1 text-[11px] font-mono text-sky-900 border border-sky-200">
                      ขนาด {(file.size / 1024).toFixed(1)} KB
                    </span>
                  ) : (
                    <p className="text-[11px] text-sky-600 mt-1">
                      {activeSource === 'wealthx'
                        ? 'รองรับไฟล์ใบยืนยันการซื้อขายกองทุนรวมจาก WealthX (noreply@wealthx.co) ขนาดไม่เกิน 10MB'
                        : 'รองรับไฟล์ Trade Confirmation Note จาก Dime ขนาดไม่เกิน 10MB'}
                    </p>
                  )}
                </div>

                <div className="flex justify-end pt-1">
                  <button
                    type="button"
                    onClick={handleAnalyzeUpload}
                    disabled={!file || analyzing}
                    className="rounded-xl bg-flow-blue px-6 py-2.5 text-xs font-bold text-white shadow-md hover:bg-sky-600 active:scale-95 transition-all disabled:opacity-50 cursor-pointer"
                  >
                    {analyzing ? 'กำลังวิเคราะห์ PDF...' : 'วิเคราะห์เอกสาร (Analyze)'}
                  </button>
                </div>
              </div>
            )}

            {/* Email Tab */}
            {activeTab === 'email' && (
              <div className="space-y-4 animate-fade-in">
                {/* Hero Batch Sync Card */}
                <div className="rounded-2xl border border-sky-200/80 bg-gradient-to-br from-sky-50/90 via-white to-sky-100/40 p-4 shadow-xs space-y-3">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="space-y-0.5">
                      <h4 className="text-xs sm:text-sm font-bold text-sky-950 flex items-center gap-1.5">
                        <span>⚡</span>
                        <span>
                          {activeSource === 'wealthx'
                            ? 'ซิงค์ใบยืนยัน WealthX ทั้งหมดอัตโนมัติ (Sync All)'
                            : activeSource === 'scb'
                            ? 'ซิงค์คำสั่งซื้อ SCBAM ทั้งหมดอัตโนมัติ (Sync All)'
                            : 'ซิงค์ข้อมูลทั้งหมดอัตโนมัติ (Sync All Confirmation Notes)'}
                        </span>
                      </h4>
                      <p className="text-[11px] text-sky-800">
                        {activeSource === 'wealthx'
                          ? 'สแกนและดึงใบยืนยันการซื้อขายกองทุนรวมทั้งหมดจาก WealthX (noreply@wealthx.co) ใน Gmail'
                          : activeSource === 'scb'
                          ? 'สแกนและดึงคำสั่งซื้อกองทุนรวมทั้งหมดจาก SCBAM Fund Click (fundclick.scbam@scb.co.th) ใน Gmail พร้อมคำนวณหน่วยลงทุนจาก NAV ย้อนหลังอัตโนมัติ'
                          : 'สแกนและดึงเอกสารยืนยันการซื้อขายทั้งหมดจาก Dime ใน Gmail ด้วย Smart Incremental Sync'}
                      </p>
                    </div>
                    <label className="flex items-center gap-1.5 text-xs text-zinc-600 cursor-pointer select-none bg-white/70 px-2.5 py-1 rounded-xl border border-sky-200/60 shadow-2xs">
                      <input
                        type="checkbox"
                        checked={forceRescan}
                        onChange={(e) => setForceRescan(e.target.checked)}
                        disabled={batchSyncing}
                        className="rounded text-flow-blue focus:ring-flow-blue cursor-pointer"
                      />
                      <span className="text-[11px] text-zinc-700 font-medium">บังคับสแกนซ้ำทั้งหมด (Force Rescan)</span>
                    </label>
                  </div>

                  {/* Real-time Streaming Progress Bar */}
                  {batchSyncing && batchProgress && (
                    <div className="space-y-2 rounded-xl bg-white/95 p-3.5 border border-sky-200/70 shadow-2xs animate-fade-in">
                      <div className="flex items-center justify-between text-xs font-semibold text-zinc-800">
                        <span>
                          {batchProgress.total > 0
                            ? `กำลังประมวลผลฉบับที่ ${batchProgress.current} จาก ${batchProgress.total} (${batchProgress.percent}%)`
                            : 'กำลังเชื่อมต่อ Gmail และจัดเรียงเอกสาร...'}
                        </span>
                        <span className="text-flow-blue font-bold">
                          พบรายการเทรด {batchProgress.items_found} รายการ
                        </span>
                      </div>
                      <div className="w-full h-2.5 bg-sky-100 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-flow-blue rounded-full transition-all duration-300 ease-out"
                          style={{ width: `${Math.max(batchProgress.percent, batchSyncing ? 5 : 0)}%` }}
                        />
                      </div>
                      {batchProgress.subject && (
                        <p className="text-[11px] text-zinc-500 truncate">
                          📄 {batchProgress.subject}
                        </p>
                      )}
                    </div>
                  )}

                  <div className="flex items-center justify-end pt-1">
                    <button
                      type="button"
                      onClick={handleBatchSyncAll}
                      disabled={batchSyncing || analyzing}
                      className="rounded-xl bg-flow-blue px-5 py-2 text-xs font-bold text-white shadow-md hover:bg-sky-600 active:scale-95 transition-all disabled:opacity-50 cursor-pointer flex items-center gap-2"
                    >
                      {batchSyncing ? (
                        <>
                          <span className="inline-block animate-spin">⏳</span>
                          <span>กำลังซิงค์ข้อมูล ({batchProgress?.percent || 0}%)...</span>
                        </>
                      ) : (
                        <>
                          <span>🚀</span>
                          <span>ดึงและนำเข้าข้อมูลทั้งหมด (Sync All)</span>
                        </>
                      )}
                    </button>
                  </div>
                </div>

                {/* Divider to Individual Search */}
                <div className="relative flex py-1 items-center">
                  <div className="flex-grow border-t border-sky-100"></div>
                  <span className="flex-shrink mx-3 text-[11px] font-semibold text-sky-700 bg-sky-50/80 px-2.5 py-0.5 rounded-full border border-sky-200/50">
                    หรือค้นหาและตรวจสอบทีละฉบับ (Individual Search)
                  </span>
                  <div className="flex-grow border-t border-sky-100"></div>
                </div>

                <div className="flex gap-2">
                  <input
                    type="text"
                    value={emailQuery}
                    onChange={(e) => setEmailQuery(e.target.value)}
                    placeholder={
                      activeSource === 'wealthx'
                        ? 'คำค้นหา เช่น ใบยืนยัน หรือชื่อกองทุน เช่น TLWORLD-X (เว้นว่างเพื่อค้นหาล่าสุด)'
                        : activeSource === 'scb'
                        ? 'คำค้นหา เช่น ยืนยันการทำรายการซื้อ หรือชื่อกองทุน เช่น SCBS&P500E (เว้นว่างเพื่อค้นหาล่าสุด)'
                        : 'คำค้นหา เช่น Dime Confirmation หรือชื่อหุ้น (เว้นว่างเพื่อค้นหาล่าสุด)'
                    }
                    className="flex-1 rounded-xl border border-sky-200 bg-white px-3.5 py-2 text-xs text-zinc-900 focus:ring-2 focus:ring-flow-blue/20 focus:border-flow-blue focus:outline-none transition-all"
                  />
                  <button
                    type="button"
                    onClick={handleSearchEmails}
                    disabled={loadingEmails || batchSyncing}
                    className="rounded-xl bg-flow-blue text-white px-5 py-2 text-xs font-bold shadow-sm hover:bg-sky-600 active:scale-95 transition-all disabled:opacity-50 cursor-pointer"
                  >
                    {loadingEmails ? 'กำลังค้นหา...' : 'ค้นหา'}
                  </button>
                </div>

                {emails.length > 0 && (
                  <div className="max-h-64 overflow-y-auto rounded-2xl border border-sky-100 divide-y divide-sky-100 shadow-sm bg-white">
                    {emails.map((em, idx) => (
                      <div
                        key={em.message_id || em.attachment_id || idx}
                        className="p-3.5 flex items-center justify-between hover:bg-sky-50/50 transition-colors text-xs"
                      >
                        <div className="space-y-1 max-w-[70%]">
                          <p className="font-bold text-zinc-900 truncate">{em.subject}</p>
                          <div className="flex items-center gap-2 text-[11px] text-zinc-500">
                            {em.filename ? (
                              <span>ไฟล์: <code className="font-mono text-sky-800 bg-sky-50 px-1 py-0.5 rounded">{em.filename}</code></span>
                            ) : (
                              <span>ประเภท: <code className="font-mono text-purple-800 bg-purple-50 px-1 py-0.5 rounded">HTML Email</code></span>
                            )}
                            {em.size_bytes > 0 && <span>({(em.size_bytes / 1024).toFixed(1)} KB)</span>}
                          </div>
                          <p className="text-[10px] text-zinc-400">{em.received_at}</p>
                        </div>
                        <button
                          type="button"
                          onClick={() => handleAnalyzeEmail(em)}
                          disabled={analyzing || batchSyncing}
                          className="rounded-xl bg-flow-blue px-4 py-1.5 text-xs font-bold text-white hover:bg-sky-600 active:scale-95 transition-all disabled:opacity-50 cursor-pointer shadow-2xs"
                        >
                          {analyzing && (selectedEmail?.message_id === em.message_id || (selectedEmail?.attachment_id && selectedEmail?.attachment_id === em.attachment_id))
                            ? 'กำลังวิเคราะห์...'
                            : 'ดึงและวิเคราะห์'}
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

          </div>
        )}
      </div>

      {inspectingWarning && (
        <DimePdfViewerModal
          warning={inspectingWarning}
          password={password.trim() || undefined}
          source={activeSource}
          onClose={() => setInspectingWarning(null)}
        />
      )}
    </Modal>
  )
}
