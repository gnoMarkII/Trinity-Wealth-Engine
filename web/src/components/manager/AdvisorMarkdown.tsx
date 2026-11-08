import React from 'react'
import ReactMarkdown from 'react-markdown'
import type { Components } from 'react-markdown'

export function renderWithEvidenceBadges(node: React.ReactNode): React.ReactNode {
  if (typeof node !== 'string') return node
  const pattern = /\[(DOCUMENTED|FACT|TESTED|MATH|INFERRED|UNVERIFIED)(?::\s*([^\]]+))?\]/g
  if (!pattern.test(node)) return node

  const parts: React.ReactNode[] = []
  let lastIndex = 0
  pattern.lastIndex = 0
  let match: RegExpExecArray | null

  while ((match = pattern.exec(node)) !== null) {
    if (match.index > lastIndex) {
      parts.push(node.substring(lastIndex, match.index))
    }
    const kind = match[1]?.toUpperCase()
    const detail = match[2]?.trim()

    if (kind === 'DOCUMENTED' || kind === 'FACT') {
      parts.push(
        <span
          key={match.index}
          className="inline-flex items-center gap-1 mx-1 px-2 py-0.5 rounded-md border border-emerald-200 bg-emerald-50 text-xs font-semibold text-emerald-800 shadow-2xs"
          title={detail ? `Verified Primary Document: ${detail}` : 'Verified Primary Document'}
        >
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 shrink-0" />
          {detail ? `DOC: ${detail}` : 'DOCUMENTED'}
        </span>
      )
    } else if (kind === 'TESTED' || kind === 'MATH') {
      parts.push(
        <span
          key={match.index}
          className="inline-flex items-center gap-1 mx-1 px-2 py-0.5 rounded-md border border-sky-200 bg-sky-50 text-xs font-semibold text-sky-800 shadow-2xs"
          title={detail ? `Deterministic Python Code: ${detail}` : 'Deterministic Python Code'}
        >
          <span className="w-1.5 h-1.5 rounded-full bg-sky-500 shrink-0" />
          {detail ? `MATH: ${detail}` : 'TESTED'}
        </span>
      )
    } else if (kind === 'INFERRED') {
      parts.push(
        <span
          key={match.index}
          className="inline-flex items-center gap-1 mx-1 px-2 py-0.5 rounded-md border border-purple-200 bg-purple-50 text-xs font-semibold text-purple-800 shadow-2xs"
          title={detail ? `Analytical Deduction: ${detail}` : 'Analytical Deduction'}
        >
          <span className="w-1.5 h-1.5 rounded-full bg-purple-500 shrink-0" />
          {detail ? `INFERRED: ${detail}` : 'INFERRED'}
        </span>
      )
    } else {
      parts.push(
        <span
          key={match.index}
          className="inline-flex items-center gap-1 mx-1 px-2 py-0.5 rounded-md border border-amber-200 bg-amber-50 text-xs font-semibold text-amber-800 shadow-2xs"
          title={detail ? `Unverified / Rumor: ${detail}` : 'Unverified Report'}
        >
          <span className="w-1.5 h-1.5 rounded-full bg-amber-500 shrink-0" />
          {detail ? `UNVERIFIED: ${detail}` : 'UNVERIFIED'}
        </span>
      )
    }
    lastIndex = pattern.lastIndex
  }

  if (lastIndex < node.length) {
    parts.push(node.substring(lastIndex))
  }

  return parts
}

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
      <span className="inline-flex items-center rounded-md border border-emerald-200 bg-emerald-50 px-2 py-0.5 font-mono text-xs font-bold text-emerald-700 shadow-sm">
        {clean.startsWith('+') ? clean : `+${clean}`}
      </span>
    )
  }

  // 2. Negative Percentage (e.g. -20.70%, -15.12%)
  if (/^-\d+(?:\.\d+)?%?$/.test(clean)) {
    return (
      <span className="inline-flex items-center rounded-md border border-rose-200 bg-rose-50 px-2 py-0.5 font-mono text-xs font-bold text-rose-700 shadow-sm">
        {clean}
      </span>
    )
  }

  // 3. Ticker in bold (e.g. **FTNT**, **MSFT**)
  const tickerMatch = clean.match(/^\*\*([A-Z0-9_\-\.]+)\*\*$/)
  if (tickerMatch && tickerMatch[1]) {
    return (
      <span className="inline-flex items-center rounded-lg border border-sky-200/90 bg-sky-50/80 px-2.5 py-0.5 font-mono text-xs font-bold text-sky-950 shadow-sm">
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
    <span className="text-sm leading-relaxed text-zinc-800">
      <ReactMarkdown>{clean}</ReactMarkdown>
    </span>
  )
}

function AdvisorTable({ data }: { data: TableData }) {
  const { headers, alignments, rows } = data

  return (
    <div className="my-4 overflow-hidden rounded-2xl border border-sky-200/90 bg-white/80 shadow-[0_4px_20px_rgba(14,165,233,0.06)] backdrop-blur-md">
      <div className="overflow-x-auto custom-scrollbar">
        <table className="w-full text-left text-sm border-collapse">
          <thead>
            <tr className="border-b border-sky-100 bg-gradient-to-r from-sky-50/90 via-cyan-50/50 to-sky-50/90 text-sky-950">
              {headers.map((h, i) => (
                <th
                  key={i}
                  className={`px-4 py-3 font-bold tracking-tight text-sky-950 text-sm ${
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
                    className={`px-4 py-3 text-sm ${
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
    <h3 className="mt-5 mb-2.5 text-lg sm:text-xl font-bold text-sky-950 first:mt-0">
      {children}
    </h3>
  ),
  h2: ({ children }) => (
    <h4 className="mt-4 mb-2 text-base sm:text-lg font-bold text-sky-950 first:mt-0">
      {children}
    </h4>
  ),
  h3: ({ children }) => (
    <h5 className="mt-3.5 mb-1.5 text-sm sm:text-base font-bold text-sky-900 tracking-wide first:mt-0">
      {children}
    </h5>
  ),
  p: ({ children }) => (
    <p className="mt-2.5 text-sm sm:text-[15px] leading-relaxed text-zinc-800 first:mt-0">
      {React.Children.map(children, renderWithEvidenceBadges)}
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
    <ul className="my-2.5 space-y-2 pl-1">
      {children}
    </ul>
  ),
  ol: ({ children }) => (
    <ol className="my-2.5 space-y-2 pl-4 list-decimal text-zinc-800 text-sm sm:text-[15px]">
      {children}
    </ol>
  ),
  li: ({ children }) => (
    <li className="ml-4 list-disc pl-1 text-sm sm:text-[15px] leading-relaxed text-zinc-800 marker:text-sky-400">
      {React.Children.map(children, renderWithEvidenceBadges)}
    </li>
  ),
  hr: () => (
    <hr className="my-4 border-t border-sky-100" />
  ),
  code: ({ children }) => {
    const isMultiLine = typeof children === 'string' && children.includes('\n')
    if (isMultiLine) {
      return (
        <pre className="my-3 overflow-x-auto rounded-xl border border-sky-100 bg-sky-50/50 p-3.5 font-mono text-xs sm:text-sm text-sky-950 custom-scrollbar">
          <code>{children}</code>
        </pre>
      )
    }
    return (
      <code className="rounded-md border border-sky-100 bg-sky-50/80 px-1.5 py-0.5 font-mono text-[0.88em] font-semibold text-sky-900">
        {children}
      </code>
    )
  },
  blockquote: ({ children }) => (
    <blockquote className="my-3 rounded-r-xl border-l-3 border-sky-400 bg-sky-50/40 py-2 pl-3.5 pr-2.5 text-sm italic leading-relaxed text-zinc-700">
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

