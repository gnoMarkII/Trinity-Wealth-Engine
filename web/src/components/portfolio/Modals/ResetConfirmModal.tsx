import { useState } from 'react'
import Modal from '../../ui/Modal'
import { api } from '../../../api/client'
import type { ActualPortfolioStateDTO } from '../../../api/types'

interface Props {
  onClose: () => void
  onSuccess: (state: ActualPortfolioStateDTO) => void
}

export default function ResetConfirmModal({ onClose, onSuccess }: Props) {
  const [confirmText, setConfirmText] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleReset = async (e: React.FormEvent) => {
    e.preventDefault()
    if (confirmText.trim().toUpperCase() !== 'RESET') {
      setError('กรุณาพิมพ์ RESET เพื่อยืนยันการล้างข้อมูลทั้งหมด')
      return
    }
    setLoading(true)
    setError(null)
    try {
      const state = await api.resetPortfolioCleanSlate()
      onSuccess(state)
      onClose()
    } catch (err: any) {
      setError(err?.message || 'ล้างข้อมูลพอร์ตไม่สำเร็จ')
    } finally {
      setLoading(false)
    }
  }

  return (
    <Modal titleId="reset-modal-title" onClose={onClose} panelClassName="max-w-md rounded-2xl border border-rose-200 bg-white p-6 shadow-2xl">
      <div className="flex items-center justify-between border-b border-rose-100 pb-3">
        <h3 id="reset-modal-title" className="text-base font-extrabold text-rose-700 flex items-center gap-2">
          🚨 ยืนยันการล้างข้อมูลพอร์ต (Clean Slate Reset)
        </h3>
        <button type="button" onClick={onClose} className="text-zinc-400 hover:text-zinc-600">
          ✕
        </button>
      </div>

      <form onSubmit={handleReset} className="mt-4 space-y-4 text-xs">
        {error && (
          <div className="rounded-xl bg-rose-50 p-3 font-semibold text-rose-800 border border-rose-200">
            ⚠️ {error}
          </div>
        )}

        <div className="rounded-xl bg-rose-50/70 p-3.5 text-zinc-700 leading-relaxed space-y-2 border border-rose-100">
          <p className="font-bold text-rose-900">
            ⚠️ คำเตือน: คุณกำลังจะล้างข้อมูลพอร์ตการลงทุนจริง (Actual Portfolio) ทั้งหมดกลับเป็นค่าเริ่มต้น!
          </p>
          <ul className="list-disc pl-4 space-y-1 text-zinc-600">
            <li>ลบรายการ Holding ในพอร์ตทั้งหมดให้เหลือเพียง CASH_THB (0 บาท) และ CASH_USD ($0)</li>
            <li>รีเซ็ตสัดส่วนเป้าหมาย (Allocation Targets) กลับเป็น 3 กลยุทธ์มาตรฐาน (60/20/20)</li>
            <li>ลบไฟล์ Obsidian Sidecars ของสินทรัพย์ที่เคยมีออกจากสารบบเพื่อความสะอาด</li>
          </ul>
          <p className="font-semibold text-zinc-800 pt-1">
            * ประวัติ Trading Journal, Watchlist/Goals และ Performance History จะไม่ถูกลบไปด้วย
          </p>
        </div>

        <div>
          <label className="block font-bold text-zinc-700 mb-1">
            พิมพ์คำว่า <span className="font-mono text-rose-600 font-extrabold">RESET</span> เพื่อยืนยัน:
          </label>
          <input
            type="text"
            value={confirmText}
            onChange={(e) => setConfirmText(e.target.value)}
            placeholder="RESET"
            className="w-full rounded-xl border border-zinc-300 px-3 py-2 font-mono font-bold text-zinc-900 uppercase focus:border-rose-500 focus:outline-none"
            required
          />
        </div>

        <div className="flex justify-end gap-2 border-t border-zinc-100 pt-4">
          <button
            type="button"
            onClick={onClose}
            className="rounded-xl border border-zinc-300 bg-white px-4 py-2 font-semibold text-zinc-700 hover:bg-zinc-50"
          >
            ยกเลิก
          </button>
          <button
            type="submit"
            disabled={loading || confirmText.trim().toUpperCase() !== 'RESET'}
            className="rounded-xl bg-rose-600 px-5 py-2 font-bold text-white shadow-md hover:bg-rose-700 disabled:opacity-40"
          >
            {loading ? 'กำลังล้างข้อมูล...' : '💥 ยืนยันล้างข้อมูลพอร์ตทั้งหมด'}
          </button>
        </div>
      </form>
    </Modal>
  )
}
