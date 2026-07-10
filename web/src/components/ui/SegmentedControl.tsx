interface Option {
  key: string
  label: string
}

interface Props {
  options: Option[]
  value: string
  onChange: (value: string) => void
}

export default function SegmentedControl({ options, value, onChange }: Props) {
  return (
    <div className="flex w-fit gap-1 rounded-lg border border-zinc-200 bg-white p-1">
      {options.map((opt) => (
        <button
          key={opt.key}
          type="button"
          onClick={() => onChange(opt.key)}
          aria-pressed={value === opt.key}
          className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
            value === opt.key ? 'bg-terra/10 text-terra' : 'text-zinc-500 hover:text-zinc-800'
          }`}
        >
          {opt.label}
        </button>
      ))}
    </div>
  )
}
