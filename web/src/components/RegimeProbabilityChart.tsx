interface Props {
  probabilities: Record<string, number>
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

export default function RegimeProbabilityChart({ probabilities }: Props) {
  const names = DEFAULT_ORDER.filter((n) => n in probabilities).concat(
    Object.keys(probabilities).filter((n) => !DEFAULT_ORDER.includes(n)),
  )

  return (
    <div className="space-y-3 rounded-xl border border-zinc-200 bg-white p-4 shadow-sm shadow-black/5">
      {names.map((name, i) => {
        const value = probabilities[name] ?? 0
        const pct = Math.round(value * 100)
        return (
          <div key={name} className="flex items-center gap-3">
            <span className="w-28 shrink-0 text-sm text-zinc-700">{name}</span>
            <div className="h-4 flex-1 bg-zinc-100">
              {/* square ที่ baseline (0%), โค้งแค่ data-end (ปลายขวา) ตาม mark spec */}
              <div
                className="animate-bar-grow h-4 rounded-r-full"
                style={{ width: `${pct}%`, backgroundColor: colorFor(name), animationDelay: `${i * 60}ms` }}
              />
            </div>
            <span className="w-12 shrink-0 text-right font-mono text-sm text-zinc-500">{pct}%</span>
          </div>
        )
      })}
    </div>
  )
}
