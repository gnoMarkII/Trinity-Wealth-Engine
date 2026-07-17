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
}

export default function SegmentedControl({ options, value, onChange, ariaLabelledby }: Props) {
  return (
    <div
      role="group"
      aria-labelledby={ariaLabelledby}
      className="flex w-fit gap-1 rounded-xl border border-sky-200 bg-panel p-1 shadow-sm shadow-sky-100/60"
    >
      {options.map((opt) => (
        <button
          key={opt.key}
          type="button"
          onClick={() => onChange(opt.key)}
          aria-pressed={value === opt.key}
          className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
            value === opt.key ? 'bg-flow-cyan/10 text-sky-700' : 'text-zinc-500 hover:bg-sky-50 hover:text-zinc-800'
          }`}
        >
          {opt.label}
        </button>
      ))}
    </div>
  )
}
