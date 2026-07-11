import { useEffect, useRef, type RefObject } from 'react'

const FOCUSABLE_SELECTOR =
  'a[href], button:not([disabled]), textarea:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])'

/**
 * Focus management ร่วมของ dialog ทุกแบบ (Modal กลางจอ / slide-over drawer):
 * จำ focus เดิม → โฟกัส element แรกใน dialog → ขัง Tab ให้วนใน dialog → Escape ปิด →
 * คืน focus เมื่อปิด — แยกออกมาจาก ui/Modal.tsx เพื่อให้ MacroReferenceDrawer ใช้ร่วมได้
 * (เดิม drawer มีแค่ Escape + initial focus แต่ Tab หลุดไป background ได้)
 *
 * `active` เป็น dependency เดียวของ effect — onClose อ่านผ่าน ref เพื่อไม่ให้ parent
 * re-render (สร้าง closure ใหม่) มา re-run effect แล้วขโมย focus ซ้ำ (ดู fc73932)
 */
export function useFocusTrap<T extends HTMLElement>(active: boolean, onClose: () => void): RefObject<T | null> {
  const containerRef = useRef<T>(null)
  const onCloseRef = useRef(onClose)
  onCloseRef.current = onClose

  useEffect(() => {
    if (!active) return
    const previouslyFocused = document.activeElement as HTMLElement | null
    const container = containerRef.current
    container?.querySelector<HTMLElement>(FOCUSABLE_SELECTOR)?.focus()

    function onKeyDown(e: KeyboardEvent) {
      if (e.key === 'Escape') {
        onCloseRef.current()
        return
      }
      if (e.key !== 'Tab' || !container) return
      const focusable = container.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR)
      const first = focusable[0]
      const last = focusable[focusable.length - 1]
      if (!first || !last) return
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault()
        last.focus()
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault()
        first.focus()
      }
    }

    window.addEventListener('keydown', onKeyDown)
    return () => {
      window.removeEventListener('keydown', onKeyDown)
      previouslyFocused?.focus()
    }
  }, [active])

  return containerRef
}
