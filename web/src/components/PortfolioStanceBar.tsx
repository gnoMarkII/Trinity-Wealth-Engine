import type { AssetAllocationDTO } from '../api/types'
import { stanceCategory } from '../lib/stance'

interface Props {
  allocations: AssetAllocationDTO[]
}

const LEGEND: { key: 'overweight' | 'neutral' | 'underweight'; label: string; dotClass: string; barClass: string }[] = [
  { key: 'overweight', label: 'Overweight', dotClass: 'bg-emerald-500', barClass: 'bg-emerald-500' },
  { key: 'neutral', label: 'Neutral', dotClass: 'bg-zinc-300', barClass: 'bg-zinc-300' },
  { key: 'underweight', label: 'Underweight', dotClass: 'bg-red-500', barClass: 'bg-red-500' },
]

export default function PortfolioStanceBar({ allocations }: Props) {
  const counts = { overweight: 0, neutral: 0, underweight: 0 }
  for (const a of allocations) {
    counts[stanceCategory(a.stance)] += 1
  }

  return (
    <div className="space-y-2">
      <div className="flex h-3 overflow-hidden rounded-full bg-sky-50">
        {LEGEND.map((item, i) => (
          <div
            key={item.key}
            className={`animate-bar-grow ${item.barClass}`}
            style={{ flex: counts[item.key] || 0, animationDelay: `${i * 60}ms` }}
          />
        ))}
      </div>
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-zinc-500">
        {LEGEND.map((item) => (
          <span key={item.key} className="flex items-center gap-1.5">
            <span className={`h-1.5 w-1.5 rounded-full ${item.dotClass}`} />
            {item.label} ({counts[item.key]})
          </span>
        ))}
      </div>
    </div>
  )
}
