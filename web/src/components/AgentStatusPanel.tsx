import { useEffect, useState } from 'react'
import { api } from '../api/client'
import type { ActiveAgentStatusDTO } from '../api/types'

const POLL_INTERVAL_MS = 4000

// รายชื่อ agent ฝั่ง flow "manager" เท่านั้น (ตัด gate/ingest ของ news_youtube ออกตามที่ตกลง)
// nodeKeys หลายค่าต่อ agent ไว้รองรับ node ภายในที่ map มาเป็นชื่อเดียวกัน เช่น
// prepare_archivist กับ archivist ใน web/src/lib/nodeDisplayNames.ts
const AGENT_ROSTER: { label: string; nodeKeys: string[] }[] = [
  { label: 'Manager', nodeKeys: ['supervisor'] },
  { label: 'Macro Quant', nodeKeys: ['macro_quant'] },
  { label: 'Macro Economist', nodeKeys: ['macro_economist'] },
  { label: 'Strategic Allocator', nodeKeys: ['strategic_allocator'] },
  { label: 'Researcher', nodeKeys: ['researcher'] },
  { label: 'Archivist', nodeKeys: ['archivist', 'prepare_archivist'] },
  { label: 'Bookkeeper', nodeKeys: ['bookkeeper'] },
]

const FLOW_LABEL: Record<string, string> = {
  news_youtube: 'News/YouTube',
}

export default function AgentStatusPanel() {
  const [status, setStatus] = useState<ActiveAgentStatusDTO | null>(null)

  useEffect(() => {
    let cancelled = false

    function poll() {
      api
        .getActiveAgentStatus()
        .then((s) => {
          if (!cancelled) setStatus(s)
        })
        .catch(() => {
          // เงียบไว้ — เดี๋ยว poll รอบถัดไปลองใหม่ ไม่ต้องโชว์ error ใน sidebar
        })
    }

    poll()
    const timer = window.setInterval(poll, POLL_INTERVAL_MS)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [])

  const running = status?.running ?? false
  const isOtherFlow = running && status?.flow !== 'manager'

  return (
    <div className="space-y-1.5 border-t border-zinc-200 pt-3">
      <h3 className="px-1 text-[11px] font-semibold uppercase tracking-wide text-zinc-400">Agents</h3>

      <div className="rounded-lg border border-zinc-200 bg-white p-2 shadow-sm shadow-black/5">
        {isOtherFlow && (
          <div className="mb-1.5 flex items-center gap-1.5 rounded-md border border-amber-200 bg-amber-50 px-2 py-1 text-[11px] text-amber-800">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-amber-500" />
            Running: {FLOW_LABEL[status?.flow ?? ''] ?? status?.flow}
          </div>
        )}

        <ul className="space-y-0.5">
          {AGENT_ROSTER.map((agent) => {
            const active = running && !isOtherFlow && agent.nodeKeys.includes(status?.node ?? '')
            return (
              <li key={agent.label} className="flex items-center gap-2 px-1 py-0.5 text-sm text-zinc-600">
                <span className={`h-1.5 w-1.5 rounded-full ${active ? 'animate-pulse bg-emerald-500' : 'bg-zinc-300'}`} />
                {agent.label}
              </li>
            )
          })}
        </ul>
      </div>
    </div>
  )
}
