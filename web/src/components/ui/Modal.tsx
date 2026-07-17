import { type ReactNode } from 'react'
import { useFocusTrap } from '../../hooks/useFocusTrap'

interface Props {
  titleId: string
  onClose: () => void
  children: ReactNode
  /** override สไตล์ของกล่อง dialog ทั้งชุด (ค่าเริ่มต้น = ฟอร์มมาตรฐาน max-w-lg มี padding)
   * ใช้กรณี dialog ที่ layout ต่างออกไป เช่น YouTube preview ที่วิดีโอชนขอบ */
  panelClassName?: string
  /** ปรับ z-index สำหรับกรณีเปิด modal ซ้อนกัน (เช่น JournalModal ซ้อน All Entries Modal) ค่าเริ่มต้น = z-50 */
  zIndexClassName?: string
}

const DEFAULT_PANEL_CLASS =
  'max-w-lg rounded-2xl border border-sky-100 bg-white/95 p-5 shadow-2xl shadow-sky-900/10 backdrop-blur-xl'

export default function Modal({ titleId, onClose, children, panelClassName, zIndexClassName = 'z-50' }: Props) {
  const dialogRef = useFocusTrap<HTMLDivElement>(true, onClose)

  return (
    <div className={`animate-fade-in fixed inset-0 flex items-center justify-center p-4 ${zIndexClassName}`}>
      {/* backdrop เป็น sibling แยกจาก dialog (ไม่ใช่ parent) — ปิดด้วยคลิกได้โดยไม่ต้อง
          stopPropagation และใส่ aria-hidden ได้เพราะเป็นแค่ฉากหลังตกแต่ง (คีย์บอร์ดใช้ Escape) */}
      <div aria-hidden="true" onClick={onClose} className="absolute inset-0 bg-black/40" />
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        className={`animate-modal-in relative w-full ${panelClassName ?? DEFAULT_PANEL_CLASS}`}
      >
        {children}
      </div>
    </div>
  )
}
