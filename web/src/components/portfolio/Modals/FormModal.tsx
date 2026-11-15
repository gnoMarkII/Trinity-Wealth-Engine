import React, { type ReactNode, type InputHTMLAttributes, type SelectHTMLAttributes, type TextareaHTMLAttributes } from 'react'
import { AlertTriangleIcon, CloseIcon } from '../../ui/AppIcons'
import Modal from '../../ui/Modal'

export interface FormModalProps {
  titleId: string
  title: ReactNode
  icon?: ReactNode
  onClose: () => void
  onSubmit: (e: React.FormEvent) => void
  error?: ReactNode
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
      <div className="flex items-center justify-between border-b border-slate-100 pb-3">
        <h3 id={titleId} className="flex items-center gap-2 text-base font-bold text-slate-900">
          {icon && <span className="flex items-center justify-center flex-shrink-0 text-flow-blue">{icon}</span>}
          <span>{title}</span>
        </h3>
        <button
          type="button"
          onClick={onClose}
          className="rounded-lg p-2 sm:p-1.5 text-slate-500 hover:bg-slate-100 hover:text-slate-600 transition-colors"
          aria-label="Close"
        >
          <CloseIcon className="h-4 w-4" />
        </button>
      </div>

      <form onSubmit={onSubmit} className="mt-4 space-y-4 text-xs">
        {error && (
          <div className="flex flex-col gap-2 rounded-xl border border-rose-200 bg-rose-50 p-3 font-semibold text-rose-800">
            <div className="flex items-start gap-2">
              <AlertTriangleIcon className="h-4 w-4 shrink-0 text-rose-600 mt-0.5" />
              <div className="flex-1">{error}</div>
            </div>
          </div>
        )}

        <div className="space-y-4">
          {children}
        </div>

        <div className="flex justify-end gap-2 border-t border-slate-100 pt-4">
          <button
            type="button"
            onClick={onClose}
            className="rounded-xl border border-slate-300 bg-white px-4 py-2 font-semibold text-slate-700 hover:bg-slate-50 transition-colors"
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
      <label className="block font-bold text-slate-700 mb-1">
        {label}
        {required && <span className="ml-1 text-rose-500">*</span>}
      </label>
      {children}
      {hint && <div className="mt-1 text-[11px] text-slate-500">{hint}</div>}
    </div>
  )
}

// Shared Input
export function FormInput({ className = '', ...props }: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      className={`w-full rounded-xl border border-slate-300 px-3 py-2 text-slate-900 disabled:bg-slate-100 disabled:text-slate-500 focus:border-flow-blue focus:outline-none transition-colors ${className}`}
      {...props}
    />
  )
}

// Shared Select
export function FormSelect({ className = '', ...props }: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select
      className={`w-full rounded-xl border border-slate-300 bg-white px-3 py-2 font-medium text-slate-800 disabled:bg-slate-100 disabled:text-slate-500 focus:border-flow-blue focus:outline-none transition-colors ${className}`}
      {...props}
    />
  )
}

// Shared Textarea
export function FormTextarea({ className = '', ...props }: TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return (
    <textarea
      className={`w-full rounded-xl border border-slate-300 px-3 py-2 text-slate-800 disabled:bg-slate-100 disabled:text-slate-500 focus:border-flow-blue focus:outline-none transition-colors ${className}`}
      {...props}
    />
  )
}
