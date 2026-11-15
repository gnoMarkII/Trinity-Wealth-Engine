import { useEffect, useState } from 'react'

interface ScoreRingProps {
  score: number | null
  size?: number
  textSizeClass?: string
}

export default function ScoreRing({ score, size = 88, textSizeClass = 'text-2xl' }: ScoreRingProps) {
  const isNull = score === null
  const clamped = isNull ? 0 : Math.max(0, Math.min(100, score))
  const strokeWidth = 8
  const radius = (size - strokeWidth) / 2
  const circumference = 2 * Math.PI * radius
  const targetOffset = circumference * (1 - clamped / 100)

  // เริ่มจากวงว่าง (offset เต็มเส้นรอบวง) แล้วค่อย animate ไปหาค่าจริงในเฟรมถัดไป
  // เพื่อให้ CSS transition ของ stroke-dashoffset เล่นตอน mount
  const [offset, setOffset] = useState(circumference)
  useEffect(() => {
    const id = requestAnimationFrame(() => setOffset(targetOffset))
    return () => cancelAnimationFrame(id)
  }, [targetOffset])

  let ringClass = 'stroke-slate-300'
  let textClass = 'text-slate-500'
  if (!isNull) {
    if (score < 40) {
      ringClass = 'stroke-red-400'
      textClass = 'text-red-600'
    } else if (score < 70) {
      ringClass = 'stroke-amber-400'
      textClass = 'text-amber-600'
    } else {
      ringClass = 'stroke-emerald-400'
      textClass = 'text-emerald-600'
    }
  }

  return (
    <div className="relative inline-flex shrink-0 items-center justify-center" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle cx={size / 2} cy={size / 2} r={radius} strokeWidth={strokeWidth} className="fill-none stroke-surface-strong" />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          className={`fill-none transition-[stroke-dashoffset] duration-700 ease-out ${ringClass}`}
        />
      </svg>
      <div className="absolute flex flex-col items-center">
        <span className={`font-serif font-semibold ${textClass} ${textSizeClass}`}>{isNull ? 'N/A' : Math.round(score)}</span>
      </div>
    </div>
  )
}
