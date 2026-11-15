interface Props {
  probabilities: Record<string, number | string>
}

// สีต่อชื่อ regime แบบ fixed (ไม่ cycle ตามค่า/อันดับ) — ผ่าน validator แล้วตอน dark theme
// (dataviz skill: CVD separation ΔE ≥ 15.7 ทุกคู่ที่ติดกัน) ยังไม่ได้ re-validate contrast
// บน surface สว่างหลังเปลี่ยนเป็น Studio Light — ถ้าพบว่าอ่านยากให้รัน dataviz skill ใหม่
const REGIME_COLOR: Record<string, string> = {
  goldilocks: '#199e70',
  reflation: '#3987e5',
  stagflation: '#c98500',
  recession: '#e66767',
}

const DEFAULT_ORDER = ['Goldilocks', 'Reflation', 'Stagflation', 'Recession']

function colorFor(name: string): string {
  return REGIME_COLOR[name.toLowerCase()] ?? '#898781' // muted fallback สำหรับชื่อ regime ที่ไม่รู้จัก
}

function parseProbability(val: unknown): number {
  if (typeof val === 'string') {
    val = val.replace('%', '').trim()
  }
  const num = typeof val === 'number' ? val : parseFloat(String(val ?? 0))
  if (isNaN(num) || num <= 0) return 0
  // ถ้า num > 1 (เช่น 15 หรือ 47.5) ถือว่าเป็น percentage 0-100 อยู่แล้ว
  // ถ้า num <= 1 (เช่น 0.15 หรือ 0.475) ถือว่าเป็น fraction 0-1 ให้คูณ 100
  const pct = num <= 1 ? num * 100 : num
  return Math.min(100, Math.max(0, Math.round(pct)))
}

export default function RegimeProbabilityChart({ probabilities }: Props) {
  const names = DEFAULT_ORDER.filter((n) => n in probabilities).concat(
    Object.keys(probabilities).filter((n) => !DEFAULT_ORDER.includes(n)),
  )

  return (
    <div className="space-y-3 rounded-xl border border-sky-100 bg-panel p-4 shadow-[0_8px_26px_rgba(14,165,233,0.05)] backdrop-blur-sm">
      {names.map((name, i) => {
        const pct = parseProbability(probabilities[name])
        return (
          <div key={name} className="flex items-center gap-3">
            <span className="w-28 shrink-0 text-sm text-slate-700">{name}</span>
            <div className="h-4 flex-1 overflow-hidden rounded-full bg-sky-50">
              {/* square ที่ baseline (0%), โค้งแค่ data-end (ปลายขวา) ตาม mark spec */}
              <div
                className="animate-bar-grow h-4 rounded-r-full"
                style={{ width: `${pct}%`, backgroundColor: colorFor(name), animationDelay: `${i * 60}ms` }}
              />
            </div>
            <span className="w-12 shrink-0 text-right font-mono text-sm text-slate-500">{pct}%</span>
          </div>
        )
      })}
    </div>
  )
}
