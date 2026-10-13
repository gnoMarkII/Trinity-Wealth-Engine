import type { ReactNode } from 'react'

interface Option {
  key: string
  label: ReactNode
}

interface Props {
  options: Option[]
  value: string
  onChange: (value: string) => void
  /** id ของ element ที่เป็นชื่อกลุ่ม (แทน <label> ซึ่งใช้ไม่ได้เพราะปุ่มไม่ใช่ form control) */
  ariaLabelledby?: string
  wrap?: boolean
}

export default function SegmentedControl({
  options,
  value,
  onChange,
  ariaLabelledby,
  wrap = true,
}: Props) {
  return (
    <div
      role="group"
      aria-labelledby={ariaLabelledby}
      className={`flex ${
        wrap ? 'flex-wrap' : 'overflow-x-auto custom-scrollbar flex-nowrap'
      } gap-1 sm:gap-1.5 rounded-2xl border border-sky-200/80 bg-panel p-1 sm:p-1.5 shadow-sm shadow-sky-100/60 max-w-full overflow-x-auto custom-scrollbar`}
    >
      {options.map((opt) => (
        <button
          key={opt.key}
          type="button"
          onClick={() => onChange(opt.key)}
          aria-pressed={value === opt.key}
          className={`rounded-xl px-2.5 sm:px-3.5 py-1.5 sm:py-2 text-xs sm:text-sm font-semibold transition-all duration-150 flex items-center gap-1.5 shrink-0 whitespace-nowrap ${
            value === opt.key
              ? 'bg-gradient-to-r from-sky-600 to-blue-600 text-white shadow-md shadow-sky-500/25 font-bold'
              : 'text-zinc-600 hover:bg-sky-50 hover:text-zinc-900'
          }`}
        >
          {opt.label}
        </button>
      ))}
    </div>
  )
}
