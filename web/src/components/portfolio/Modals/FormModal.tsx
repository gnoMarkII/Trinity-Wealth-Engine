import React, { type ReactNode, type InputHTMLAttributes, type SelectHTMLAttributes, type TextareaHTMLAttributes } from 'react'
import Modal from '../../ui/Modal'

export interface FormModalProps {
  titleId: string
  title: ReactNode
  icon?: ReactNode
  onClose: () => void
  onSubmit: (e: React.FormEvent) => void
  error?: string | null
  loading?: boolean
  submitText: ReactNode
  loadingText?: ReactNode
  cancelText?: ReactNode
  submitDisabled?: boolean
  submitClassName?: string
  panelClassName?: string
  zIndexClassName?: string
  children: ReactNode
}

export default function FormModal({
  titleId,
  title,
  icon,
  onClose,
  onSubmit,
  error,
  loading = false,
  submitText,
  loadingText = 'กำลังดำเนินการ...',
  cancelText = 'ยกเลิก',
  submitDisabled = false,
  submitClassName = 'bg-flow-blue hover:bg-sky-600',
  panelClassName = 'max-w-lg rounded-2xl border border-sky-100 bg-white p-6 shadow-2xl',
  zIndexClassName = 'z-50',
  children,
}: FormModalProps) {
  return (
    <Modal
      titleId={titleId}
      onClose={onClose}
      zIndexClassName={zIndexClassName}
      panelClassName={panelClassName}
    >
      <div className="flex items-center justify-between border-b border-zinc-100 pb-3">
        <h3 id={titleId} className="flex items-center gap-2 text-base font-bold text-zinc-900">
          {icon && <span className="flex items-center justify-center flex-shrink-0 text-flow-blue">{icon}</span>}
          <span>{title}</span>
        </h3>
        <button
          type="button"
          onClick={onClose}
          className="rounded-lg p-1 text-zinc-400 hover:bg-zinc-100 hover:text-zinc-600 transition-colors"
          aria-label="Close"
        >
          ✕
        </button>
      </div>

      <form onSubmit={onSubmit} className="mt-4 space-y-4 text-xs">
        {error && (
          <div className="flex items-start gap-2 rounded-xl border border-rose-200 bg-rose-50 p-3 font-semibold text-rose-800">
            <span className="flex-shrink-0 mt-0.5">⚠️</span>
            <span>{error}</span>
          </div>
        )}

        <div className="space-y-4">
          {children}
        </div>

        <div className="flex justify-end gap-2 border-t border-zinc-100 pt-4">
          <button
            type="button"
            onClick={onClose}
            className="rounded-xl border border-zinc-300 bg-white px-4 py-2 font-semibold text-zinc-700 hover:bg-zinc-50 transition-colors"
          >
            {cancelText}
          </button>
          <button
            type="submit"
            disabled={loading || submitDisabled}
            className={`rounded-xl px-5 py-2 font-bold text-white shadow-md transition-colors disabled:opacity-50 ${submitClassName}`}
          >
            {loading ? loadingText : submitText}
          </button>
        </div>
      </form>
    </Modal>
  )
}

// Shared Form Field Wrapper
export interface FormFieldProps {
  label: ReactNode
  hint?: ReactNode
  required?: boolean
  className?: string
  children: ReactNode
}

export function FormField({ label, hint, required, className = '', children }: FormFieldProps) {
  return (
    <div className={className}>
      <label className="block font-bold text-zinc-700 mb-1">
        {label}
        {required && <span className="ml-1 text-rose-500">*</span>}
      </label>
      {children}
      {hint && <div className="mt-1 text-[11px] text-zinc-400">{hint}</div>}
    </div>
  )
}

// Shared Input
export function FormInput({ className = '', ...props }: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      className={`w-full rounded-xl border border-zinc-300 px-3 py-2 text-zinc-900 disabled:bg-zinc-100 disabled:text-zinc-500 focus:border-flow-blue focus:outline-none transition-colors ${className}`}
      {...props}
    />
  )
}

// Shared Select
export function FormSelect({ className = '', ...props }: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select
      className={`w-full rounded-xl border border-zinc-300 bg-white px-3 py-2 font-medium text-zinc-800 disabled:bg-zinc-100 disabled:text-zinc-500 focus:border-flow-blue focus:outline-none transition-colors ${className}`}
      {...props}
    />
  )
}

// Shared Textarea
export function FormTextarea({ className = '', ...props }: TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return (
    <textarea
      className={`w-full rounded-xl border border-zinc-300 px-3 py-2 text-zinc-800 disabled:bg-zinc-100 disabled:text-zinc-500 focus:border-flow-blue focus:outline-none transition-colors ${className}`}
      {...props}
    />
  )
}
