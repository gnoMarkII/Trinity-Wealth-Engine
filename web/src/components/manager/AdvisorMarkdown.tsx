import ReactMarkdown from 'react-markdown'
import type { Components } from 'react-markdown'

export interface TableData {
  headers: string[]
  alignments: ('left' | 'center' | 'right')[]
  rows: string[][]
}

export type ContentSegment =
  | { type: 'markdown'; content: string }
  | { type: 'table'; data: TableData }

export function parseTableRow(line: string): string[] {
  const trimmed = line.trim().replace(/^\|/, '').replace(/\|$/, '')
  return trimmed.split('|').map((c) => c.trim())
}

export function parseAlignment(cell: string): 'left' | 'center' | 'right' {
  const c = cell.trim()
  if (c.startsWith(':') && c.endsWith(':')) return 'center'
  if (c.endsWith(':')) return 'right'
  return 'left'
}

export function splitMarkdownAndTables(text: string): ContentSegment[] {
  const lines = text.split('\n')
  const segments: ContentSegment[] = []
  let currentMarkdownLines: string[] = []
  let i = 0

  while (i < lines.length) {
    const line = lines[i] ?? ''
    const trimmed = line.trim()
    const isTableLine = trimmed.startsWith('|') && trimmed.endsWith('|')

    if (isTableLine && i + 1 < lines.length) {
      const nextLine = (lines[i + 1] ?? '').trim()
      const isSeparator = /^\|(?:\s*:?-+:?\s*\|)+\s*$/.test(nextLine)

      if (isSeparator) {
        if (currentMarkdownLines.length > 0) {
          segments.push({
            type: 'markdown',
            content: currentMarkdownLines.join('\n'),
          })
          currentMarkdownLines = []
        }

        const headers = parseTableRow(line)
        const alignments = parseTableRow(nextLine).map(parseAlignment)
        const rows: string[][] = []
        i += 2

        while (i < lines.length) {
          const rowLine = (lines[i] ?? '').trim()
          if (!rowLine.startsWith('|') || !rowLine.endsWith('|')) {
            break
          }
          rows.push(parseTableRow(rowLine))
          i++
        }

        segments.push({
          type: 'table',
          data: { headers, alignments, rows },
        })
        continue
      }
    }

    currentMarkdownLines.push(line)
    i++
  }

  if (currentMarkdownLines.length > 0) {
    segments.push({
      type: 'markdown',
      content: currentMarkdownLines.join('\n'),
    })
  }

  return segments
}

function renderCellContent(cell: string) {
  const clean = cell.trim()

  // 1. Positive Percentage (e.g. +71.86%, +29.92%, +15%)
  if (/^\+\d+(?:\.\d+)?%?$/.test(clean) || /^\d+(?:\.\d+)?%$/.test(clean)) {
    return (
      <span className="inline-flex items-center rounded-md border border-emerald-200 bg-emerald-50 px-2 py-0.5 font-mono text-[11px] font-bold text-emerald-700 shadow-sm">
        {clean.startsWith('+') ? clean : `+${clean}`}
      </span>
    )
  }

  // 2. Negative Percentage (e.g. -20.70%, -15.12%)
  if (/^-\d+(?:\.\d+)?%?$/.test(clean)) {
    return (
      <span className="inline-flex items-center rounded-md border border-rose-200 bg-rose-50 px-2 py-0.5 font-mono text-[11px] font-bold text-rose-700 shadow-sm">
        {clean}
      </span>
    )
  }

  // 3. Ticker in bold (e.g. **FTNT**, **MSFT**)
  const tickerMatch = clean.match(/^\*\*([A-Z0-9_\-\.]+)\*\*$/)
  if (tickerMatch && tickerMatch[1]) {
    return (
      <span className="inline-flex items-center rounded-lg border border-sky-200/90 bg-sky-50/80 px-2 py-0.5 font-mono text-xs font-bold text-sky-950 shadow-sm">
        ${tickerMatch[1]}
      </span>
    )
  }

  // 4. Currency or numeric values (e.g. 13,065.73, 26,999.72 THB)
  if (/^[\d,]+(?:\.\d+)?(?:\s*[A-Z]{3})?$/.test(clean)) {
    return <span className="font-mono text-xs font-semibold text-zinc-900 tabular-nums">{clean}</span>
  }

  // 5. Default markdown rendering for complex text inside cell
  return (
    <span className="text-xs leading-normal text-zinc-800">
      <ReactMarkdown>{clean}</ReactMarkdown>
    </span>
  )
}

function AdvisorTable({ data }: { data: TableData }) {
  const { headers, alignments, rows } = data

  return (
    <div className="my-4 overflow-hidden rounded-2xl border border-sky-200/90 bg-white/80 shadow-[0_4px_20px_rgba(14,165,233,0.06)] backdrop-blur-md">
      <div className="overflow-x-auto custom-scrollbar">
        <table className="w-full text-left text-xs border-collapse">
          <thead>
            <tr className="border-b border-sky-100 bg-gradient-to-r from-sky-50/90 via-cyan-50/50 to-sky-50/90 text-sky-950">
              {headers.map((h, i) => (
                <th
                  key={i}
                  className={`px-3.5 py-2.5 font-bold tracking-tight text-sky-950 ${
                    alignments[i] === 'right'
                      ? 'text-right'
                      : alignments[i] === 'center'
                        ? 'text-center'
                        : 'text-left'
                  }`}
                >
                  <ReactMarkdown>{h}</ReactMarkdown>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-sky-100/60 bg-white/40">
            {rows.map((row, rIdx) => (
              <tr key={rIdx} className="hover:bg-sky-50/40 transition-colors">
                {row.map((cell, cIdx) => (
                  <td
                    key={cIdx}
                    className={`px-3.5 py-2.5 ${
                      alignments[cIdx] === 'right'
                        ? 'text-right'
                        : alignments[cIdx] === 'center'
                          ? 'text-center'
                          : 'text-left'
                    }`}
                  >
                    {renderCellContent(cell)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

const markdownComponents: Components = {
  h1: ({ children }) => (
    <h3 className="mt-4 mb-2 text-base font-bold text-sky-950 first:mt-0">
      {children}
    </h3>
  ),
  h2: ({ children }) => (
    <h4 className="mt-3.5 mb-1.5 text-sm font-bold text-sky-950 first:mt-0">
      {children}
    </h4>
  ),
  h3: ({ children }) => (
    <h5 className="mt-3 mb-1 text-xs font-bold text-sky-900 tracking-wide first:mt-0">
      {children}
    </h5>
  ),
  p: ({ children }) => (
    <p className="mt-2 text-sm leading-relaxed text-zinc-800 first:mt-0">
      {children}
    </p>
  ),
  strong: ({ children }) => (
    <strong className="font-bold text-sky-950">
      {children}
    </strong>
  ),
  em: ({ children }) => (
    <em className="italic text-zinc-600">
      {children}
    </em>
  ),
  ul: ({ children }) => (
    <ul className="my-2 space-y-1.5 pl-1">
      {children}
    </ul>
  ),
  ol: ({ children }) => (
    <ol className="my-2 space-y-1.5 pl-4 list-decimal text-zinc-800">
      {children}
    </ol>
  ),
  li: ({ children }) => (
    <li className="ml-4 list-disc pl-1 text-sm leading-relaxed text-zinc-800 marker:text-sky-400">
      {children}
    </li>
  ),
  hr: () => (
    <hr className="my-3.5 border-t border-sky-100" />
  ),
  code: ({ children }) => {
    const isMultiLine = typeof children === 'string' && children.includes('\n')
    if (isMultiLine) {
      return (
        <pre className="my-2.5 overflow-x-auto rounded-xl border border-sky-100 bg-sky-50/50 p-3 font-mono text-xs text-sky-950 custom-scrollbar">
          <code>{children}</code>
        </pre>
      )
    }
    return (
      <code className="rounded-md border border-sky-100 bg-sky-50/80 px-1.5 py-0.5 font-mono text-[0.85em] font-semibold text-sky-900">
        {children}
      </code>
    )
  },
  blockquote: ({ children }) => (
    <blockquote className="my-2.5 rounded-r-xl border-l-3 border-sky-400 bg-sky-50/40 py-1.5 pl-3 pr-2 text-xs italic text-zinc-700">
      {children}
    </blockquote>
  ),
  a: ({ href, children }) => (
    <a
      href={href}
      target="_blank"
      rel="noreferrer"
      className="font-medium text-sky-700 underline underline-offset-2 hover:text-sky-900 transition-colors"
    >
      {children}
    </a>
  ),
}

export default function AdvisorMarkdown({
  content,
  isStreaming,
}: {
  content: string
  isStreaming?: boolean
}) {
  const segments = splitMarkdownAndTables(content)

  return (
    <div className="advisor-markdown break-words">
      {segments.map((seg, idx) => {
        if (seg.type === 'table') {
          return <AdvisorTable key={idx} data={seg.data} />
        }
        return (
          <ReactMarkdown key={idx} components={markdownComponents}>
            {seg.content}
          </ReactMarkdown>
        )
      })}
      {isStreaming && (
        <span className="inline-block h-4 w-1.5 translate-y-0.5 rounded-sm bg-flow-blue animate-stream-caret ml-1" />
      )}
    </div>
  )
}

