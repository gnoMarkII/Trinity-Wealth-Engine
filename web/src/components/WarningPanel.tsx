import type { WarningDTO } from '../api/types'

interface Props {
  warnings: WarningDTO[]
  compact?: boolean
}

/**
 * Render warning แบบ generic — ไม่ hardcode switch-case ตาม warning ID ที่มีอยู่วันนี้
 * (schemas/warning_registry.py เพิ่ม ID ใหม่เรื่อยๆ) ใช้แค่ keyword heuristic บน code
 * เพื่อจัดสี ถ้าไม่เข้าเงื่อนไขไหนเลยก็ fallback เป็น badge สีกลาง ไม่มีทาง "หาย" หรือ crash
 */
function severityClass(code: string | null): string {
  const c = (code ?? '').toUpperCase()
  if (c.includes('CONTRADICTION') || c.includes('MISMATCH') || c.includes('INVALID')) {
    return 'border-red-200 bg-red-50 text-red-700'
  }
  if (c.includes('STALE') || c.includes('DEFENSIVE') || c.includes('PENALTY') || c.includes('DOWNGRADE')) {
    return 'border-amber-200 bg-amber-50 text-amber-700'
  }
  return 'border-zinc-200 bg-surface text-zinc-600'
}

export default function WarningPanel({ warnings, compact }: Props) {
  if (warnings.length === 0) return null

  return (
    <div className={compact ? 'space-y-1' : 'space-y-2 rounded-xl border border-zinc-200 bg-white p-4 shadow-sm shadow-black/5'}>
      {!compact && (
        <h3 className="text-sm font-semibold text-zinc-500">
          ⚠️ Guardrail Warnings ({warnings.length})
        </h3>
      )}
      <ul className="space-y-1">
        {warnings.map((w, i) => (
          <li
            key={`${w.code ?? 'unknown'}-${i}`}
            className={`rounded-lg border px-2 py-1 text-xs ${severityClass(w.code)}`}
          >
            {w.code && <span className="mr-1 font-mono opacity-70">[{w.code}]</span>}
            {w.message}
          </li>
        ))}
      </ul>
    </div>
  )
}
