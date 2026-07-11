import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { api, ApiError } from '../../api/client'
import type { JobOutputsDTO } from '../../api/types'
import { nodeDisplayName } from '../../lib/nodeDisplayNames'

interface Props {
  jobId: string
  refreshVersion: number
  onStatusChange?: (status: JobOutputsDTO['status']) => void
}

function OutputMarkdown({ content }: { content: string }) {
  return (
    <ReactMarkdown
      components={{
        a: ({ children, href }) => (
          <a href={href} target="_blank" rel="noreferrer" className="font-medium text-sky-700 underline underline-offset-2 hover:text-sky-900">
            {children}
          </a>
        ),
        code: ({ children }) => <code className="rounded bg-zinc-100 px-1 py-0.5 font-mono text-[0.84em] text-zinc-700">{children}</code>,
        h1: ({ children }) => <h4 className="mt-4 text-sm font-semibold text-zinc-900 first:mt-0">{children}</h4>,
        h2: ({ children }) => <h4 className="mt-4 text-sm font-semibold text-zinc-900 first:mt-0">{children}</h4>,
        h3: ({ children }) => <h5 className="mt-3 text-xs font-semibold text-zinc-800 first:mt-0">{children}</h5>,
        li: ({ children }) => <li className="ml-4 list-disc pl-1">{children}</li>,
        p: ({ children }) => <p className="mt-2 first:mt-0">{children}</p>,
        ul: ({ children }) => <ul className="mt-2 space-y-1">{children}</ul>,
      }}
    >
      {content}
    </ReactMarkdown>
  )
}

export default function KanbanCardOutputs({ jobId, refreshVersion, onStatusChange }: Props) {
  const [outputs, setOutputs] = useState<JobOutputsDTO | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [activeNode, setActiveNode] = useState<string | null>(null)
  const onStatusChangeRef = useRef(onStatusChange)
  onStatusChangeRef.current = onStatusChange

  useEffect(() => {
    setOutputs(null)
    setActiveNode(null)
    setError(null)
  }, [jobId])

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    api
      .getJobOutputs(jobId)
      .then((next) => {
        if (cancelled) return
        setOutputs(next)
        setError(null)
        setActiveNode((current) => current ?? next.specialists[0]?.node_name ?? null)
        onStatusChangeRef.current?.(next.status)
      })
      .catch((requestError) => {
        if (cancelled) return
        setError(requestError instanceof ApiError ? requestError.message : 'โหลดผลลัพธ์งานไม่สำเร็จ')
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [jobId, refreshVersion])

  if (loading && !outputs) {
    return <div className="animate-shimmer h-36 rounded-xl border border-zinc-200/80" />
  }
  if (error && !outputs) {
    return <p className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">{error}</p>
  }
  if (!outputs) return null

  const activeOutput = outputs.specialists.find((item) => item.node_name === activeNode) ?? outputs.specialists[0]

  return (
    <section className="space-y-3" aria-live="polite">
      <div className="flex items-center justify-between gap-3">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-zinc-500">Card Outputs</h3>
        {loading && <span className="text-[10px] text-zinc-400">Updating…</span>}
      </div>

      {outputs.executive_summary ? (
        <article className="rounded-xl border border-orange-200/80 bg-orange-50/60 p-3 shadow-[0_8px_22px_rgba(251,140,0,0.06)]">
          <p className="text-[10px] font-semibold uppercase tracking-wide text-orange-800">Manager Summary</p>
          <div className="mt-2 text-xs leading-relaxed text-zinc-700">
            <OutputMarkdown content={outputs.executive_summary} />
          </div>
        </article>
      ) : outputs.status === 'error' ? (
        <p className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
          {outputs.error_message || 'งานหยุดก่อนที่ Manager Summary จะถูกสร้าง'}
        </p>
      ) : (
        <p className="rounded-lg border border-zinc-200 bg-surface p-3 text-xs text-zinc-500">
          Manager Summary จะปรากฏเมื่อการทำงานเสร็จสมบูรณ์
        </p>
      )}

      {outputs.specialists.length > 0 && activeOutput && (
        <section className="rounded-xl border border-sky-100 bg-white/80 p-3 shadow-[0_8px_24px_rgba(14,165,233,0.05)] backdrop-blur-sm">
          <p className="text-[10px] font-semibold uppercase tracking-wide text-sky-800">Specialist Breakdown</p>
          <div className="mt-2 flex gap-1 overflow-x-auto pb-1">
            {outputs.specialists.map((item) => (
              <button
                key={item.node_name}
                type="button"
                onClick={() => setActiveNode(item.node_name)}
                className={`shrink-0 rounded-md px-2 py-1 text-[11px] font-medium transition-colors ${
                  item.node_name === activeOutput.node_name
                    ? 'bg-flow-cyan/10 text-sky-700'
                    : 'text-zinc-500 hover:bg-sky-50 hover:text-zinc-800'
                }`}
              >
                {nodeDisplayName(item.node_name)}
              </button>
            ))}
          </div>
          <div className="mt-2 border-t border-zinc-100 pt-2 text-xs leading-relaxed text-zinc-600">
            <p className="mb-2 text-[10px] font-medium uppercase tracking-wide text-zinc-400">{activeOutput.label}</p>
            <OutputMarkdown content={activeOutput.content} />
          </div>
        </section>
      )}
    </section>
  )
}
