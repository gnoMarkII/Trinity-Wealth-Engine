import { type ReactNode } from 'react'
import { createPortal } from 'react-dom'
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

export interface InPlaceDialogProps {
  titleId: string
  onClose: () => void
  children: ReactNode
  wrapperClassName?: string
  backdropClassName?: string
  panelClassName?: string
}

/**
 * กล่อง Dialog มาตรฐานที่ render ในต้นไม้ DOM เดิม (ไม่ portal ออกนอก root)
 * ใช้สำหรับโมดอลเฉพาะกิจหรือ slide-over drawer เพื่อให้ได้ role="dialog", aria-modal,
 * backdrop aria-hidden, และ useFocusTrap ร่วมกันโดยไม่เปลี่ยนตำแหน่ง DOM เดิม
 */
export function InPlaceDialog({
  titleId,
  onClose,
  children,
  wrapperClassName = 'animate-fade-in fixed inset-0 z-50 flex items-center justify-center p-4',
  backdropClassName = 'absolute inset-0 bg-slate-900/40 backdrop-blur-sm',
  panelClassName,
}: InPlaceDialogProps) {
  const dialogRef = useFocusTrap<HTMLDivElement>(true, onClose)

  return (
    <div className={wrapperClassName}>
      <div aria-hidden="true" onClick={onClose} className={backdropClassName} />
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        className={`relative ${panelClassName ?? `w-full ${DEFAULT_PANEL_CLASS}`}`}
      >
        {children}
      </div>
    </div>
  )
}

export default function Modal({ titleId, onClose, children, panelClassName, zIndexClassName = 'z-50' }: Props) {
  return createPortal(
    <InPlaceDialog
      titleId={titleId}
      onClose={onClose}
      wrapperClassName={`animate-fade-in fixed inset-0 flex items-center justify-center p-4 ${zIndexClassName}`}
      panelClassName={`animate-modal-in w-full ${panelClassName ?? DEFAULT_PANEL_CLASS}`}
    >
      {children}
    </InPlaceDialog>,
    document.body
  )
}

