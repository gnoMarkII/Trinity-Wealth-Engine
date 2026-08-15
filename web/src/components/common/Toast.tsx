import React, { useCallback, useEffect, useState } from 'react'

export interface ToastProps {
  message: string
  type?: 'info' | 'success' | 'error'
  actionLabel?: string
  onAction?: () => void
  onClose?: () => void
  durationMs?: number
}

export const Toast: React.FC<ToastProps> = ({
  message,
  type = 'info',
  actionLabel,
  onAction,
  onClose,
  durationMs = 4000,
}) => {
  const [leaving, setLeaving] = useState(false)

  const handleClose = useCallback(() => {
    setLeaving(true)
    setTimeout(() => {
      onClose?.()
    }, 200)
  }, [onClose])

  useEffect(() => {
    if (durationMs <= 0) return
    const timer = setTimeout(() => {
      handleClose()
    }, durationMs)
    return () => clearTimeout(timer)
  }, [durationMs, handleClose])

  const typeStyles = {
    info: 'bg-zinc-900/90 text-white border-zinc-700 shadow-xl',
    success: 'bg-emerald-950/90 text-emerald-100 border-emerald-800 shadow-emerald-950/20',
    error: 'bg-rose-950/90 text-rose-100 border-rose-800 shadow-rose-950/20',
  }

  return (
    <div
      role="status"
      className={`fixed bottom-6 right-6 z-50 flex items-center gap-3 rounded-xl border px-4 py-3 text-sm backdrop-blur-md transition-all duration-200 ${
        typeStyles[type]
      } ${leaving ? 'opacity-0 translate-y-2 scale-95' : 'animate-notice-in opacity-100 translate-y-0 scale-100'}`}
    >
      <span className="font-medium">{message}</span>
      {actionLabel && onAction && (
        <button
          onClick={() => {
            onAction()
            handleClose()
          }}
          className="ml-2 rounded-lg bg-sky-500/20 px-2.5 py-1 text-xs font-semibold text-sky-300 hover:bg-sky-500/30 hover:text-sky-200 transition-colors"
        >
          {actionLabel}
        </button>
      )}
      <button
        onClick={handleClose}
        aria-label="Close Toast"
        className="ml-1 rounded-md p-1 text-zinc-400 hover:text-white transition-colors"
      >
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
    </div>
  )
}

export default Toast
