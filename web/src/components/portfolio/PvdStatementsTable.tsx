import { useEffect, useMemo, useState } from 'react'
import type { ActualPortfolioStateDTO, TransactionItemDTO } from '../../api/types'
import { api } from '../../api/client'
import { formatMoneyCents, type PvdFundComparison, type PvdHistory, type PvdStatementGroup } from '../../lib/pvdStatementHistory'
import PvdStatementModal from './Modals/PvdStatementModal'

interface Props {
  portfolioId: string
  transactions: TransactionItemDTO[]
  history: PvdHistory
  active: boolean
  loading: boolean
  initialSymbol?: string | null
  onClearSymbolFilter?: () => void
  onRefresh: () => Promise<void>
  onSuccess?: (state: ActualPortfolioStateDTO) => void
}

const dateLabel = (date: string | null) => date
  ? new Date(`${date}T00:00:00.000Z`).toLocaleDateString('th-TH', { year: 'numeric', month: 'short', day: 'numeric', timeZone: 'UTC' })
  : 'ไม่ทราบวันที่'

function formatBaht(cents: bigint | null, showSign = false) {
  if (cents === null) return '—'
  const text = formatMoneyCents(cents, showSign)
  const negative = text.startsWith('-')
  const positive = text.startsWith('+')
  const unsigned = negative || positive ? text.slice(1) : text
  const [whole = '0', fraction = '00'] = unsigned.split('.')
  const grouped = BigInt(whole).toLocaleString('th-TH')
  return `${negative ? '-' : positive ? '+' : ''}฿${grouped}.${fraction}`
}

function statementContribution(statement: PvdStatementGroup, transactions: TransactionItemDTO[]): string | null {
  const row = transactions.find((tx) => tx.confirmation_no?.toLowerCase() === statement.id.toLowerCase())
  if (!row?.pvd_metadata_json) return null
  try {
    const metadata = JSON.parse(row.pvd_metadata_json) as { total_contribution?: unknown }
    return typeof metadata.total_contribution === 'string' ? metadata.total_contribution : null
  } catch {
    return null
  }
}

function statusLabel(status: PvdFundComparison['status']) {
  const labels: Record<PvdFundComparison['status'], string> = {
    baseline: 'งวดฐาน', new: 'กองทุนใหม่', zero: 'ยอดเป็นศูนย์', zero_both: 'ศูนย์ทั้งสองงวด',
    increase: 'เพิ่มขึ้น', decrease: 'ลดลง', unchanged: 'ไม่เปลี่ยน', missing: 'ข้อมูลกองทุนขาด', unavailable: 'เทียบไม่ได้',
  }
  return labels[status]
}

function IssueList({ issues }: { issues: PvdHistory['unresolvedIssues'] | PvdStatementGroup['issues'] }) {
  if (!issues.length) return null
  return (
    <ul className="mt-3 space-y-1 text-xs text-amber-900">
      {issues.map((issue) => (
        <li key={`${issue.code}-${issue.transactionIds.join(',')}`}>
          <strong>{issue.code}:</strong> {issue.message}
          {issue.transactionIds.length > 0 && <span className="ml-1 text-amber-700">({issue.transactionIds.join(', ')})</span>}
        </li>
      ))}
    </ul>
  )
}

function StatementCard({
  statement, previousStatement, transactions, expanded, onToggle, portfolioId,
}: {
  statement: PvdStatementGroup
  previousStatement: PvdStatementGroup | null
  transactions: TransactionItemDTO[]
  expanded: boolean
  onToggle: () => void
  portfolioId: string
}) {
  const contribution = statementContribution(statement, transactions)
  const comparisons = new Map(statement.comparisons.map((comparison) => [comparison.code, comparison]))
  const rowCount = statement.funds.length
  const summary = statement.isValid
    ? `ตรวจสอบยอดรวมแล้ว · ${rowCount} กองทุน`
    : 'ข้อมูลมีข้อขัดแย้ง ต้องตรวจสอบก่อนใช้เป็นยอดล่าสุด'

  return (
    <article className={`overflow-hidden rounded-2xl border bg-white shadow-sm ${statement.isValid ? 'border-zinc-200' : 'border-amber-300'}`}>
      <div className="flex flex-wrap items-center justify-between gap-4 p-4 sm:p-5">
        <div className="min-w-[180px] flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="text-base font-extrabold text-zinc-900">Statement · {dateLabel(statement.asOfDate)}</h3>
            <span className={`rounded-full px-2.5 py-1 text-[11px] font-bold ${statement.isValid ? 'bg-emerald-50 text-emerald-800' : 'bg-amber-100 text-amber-900'}`}>
              {statement.isValid ? 'ตรวจสอบแล้ว' : 'มีข้อผิดพลาด'}
            </span>
          </div>
          <p className="mt-1 text-xs text-zinc-500">{summary}{statement.previousId ? ` · เทียบกับ ${dateLabel(previousStatement?.asOfDate ?? null)}` : ''}</p>
          {contribution && <p className="mt-1 text-xs text-zinc-600">เงินสมทบสะสม: <span className="font-mono">฿{contribution}</span></p>}
        </div>
        <div className="flex items-center gap-2">
          <div className="text-right">
            <div className="text-[11px] font-semibold text-zinc-500">มูลค่ารวมตาม Statement</div>
            <div className="font-mono text-lg font-extrabold tabular-nums text-zinc-900">{formatBaht(statement.totalNavCents)}</div>
          </div>
          {statement.previousId && (
            <div className="border-l border-zinc-200 pl-3 text-right">
              <div className="text-[11px] font-semibold text-zinc-500">เปลี่ยนจากงวดก่อน</div>
              <div className={`font-mono text-sm font-bold tabular-nums ${statement.deltaNavCents === null ? 'text-zinc-400' : statement.deltaNavCents >= 0n ? 'text-emerald-700' : 'text-rose-700'}`}>
                {formatBaht(statement.deltaNavCents, true)}
              </div>
            </div>
          )}
          {statement.archiveId ? (
            <a
              href={api.getPvdDocumentUrl(statement.archiveId, portfolioId)}
              target="_blank"
              rel="noreferrer"
              className="rounded-lg border border-teal-200 bg-teal-50 px-3 py-2 text-xs font-bold text-teal-800 hover:bg-teal-100"
              aria-label={`เปิด PDF Statement ${dateLabel(statement.asOfDate)}`}
            >PDF</a>
          ) : (
            <button type="button" disabled title="ไม่พบ archive ID หรือ PDF hash ที่ผ่านการตรวจสอบ" className="rounded-lg border border-zinc-200 px-3 py-2 text-xs font-bold text-zinc-400">PDF</button>
          )}
          <button
            type="button"
            aria-expanded={expanded}
            aria-controls={`pvd-details-${statement.id}`}
            onClick={onToggle}
            className="rounded-lg border border-zinc-200 px-3 py-2 text-xs font-bold text-zinc-700 hover:bg-zinc-50"
          >{expanded ? 'ซ่อนรายละเอียด' : 'ดูรายละเอียด'}</button>
        </div>
      </div>

      {expanded && (
        <div id={`pvd-details-${statement.id}`} className="border-t border-zinc-100">
          {statement.issues.length > 0 && <div role="alert" className="bg-amber-50 px-4 py-3"><IssueList issues={statement.issues} /></div>}
          <div className="overflow-x-auto">
            <table className="w-full min-w-[920px] text-left text-xs">
              <thead className="border-b border-zinc-200 bg-zinc-50 text-zinc-600">
                <tr>
                  <th className="px-4 py-3">กองทุนย่อย</th>
                  <th className="px-4 py-3 text-right">หน่วย</th>
                  <th className="px-4 py-3 text-right">NAV</th>
                  <th className="px-4 py-3 text-right">มูลค่าดิบ</th>
                  <th className="px-4 py-3 text-right">ปรับกระทบยอด</th>
                  <th className="px-4 py-3 text-right">มูลค่าหลังปรับ</th>
                  <th className="px-4 py-3 text-right">เปลี่ยนจากงวดก่อน</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-100">
                {statement.funds.map((fund) => {
                  const comparison = comparisons.get(fund.code)
                  return (
                    <tr key={fund.code}>
                      <th scope="row" className="px-4 py-3 font-mono font-bold text-zinc-800">{fund.code}</th>
                      <td className="px-4 py-3 text-right font-mono tabular-nums">{fund.units.toLocaleString('en-US', { maximumFractionDigits: 6 })}</td>
                      <td className="px-4 py-3 text-right font-mono tabular-nums">{fund.nav.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 4 })}</td>
                      <td className="px-4 py-3 text-right font-mono tabular-nums">{formatBaht(fund.rawValueCents)}</td>
                      <td className="px-4 py-3 text-right font-mono tabular-nums">{formatBaht(fund.adjustmentCents, true)}</td>
                      <td className="px-4 py-3 text-right font-mono font-bold tabular-nums">{formatBaht(fund.valueCents)}</td>
                      <td className="px-4 py-3 text-right">
                        {comparison ? (
                          <span className="inline-flex flex-col items-end gap-0.5">
                            <span className="font-mono tabular-nums">{comparison.deltaCents === null ? '—' : formatBaht(comparison.deltaCents, true)}</span>
                            <span className={`rounded-full px-2 py-0.5 text-[10px] font-bold ${comparison.status === 'increase' ? 'bg-emerald-50 text-emerald-700' : comparison.status === 'decrease' ? 'bg-rose-50 text-rose-700' : comparison.status === 'missing' || comparison.status === 'unavailable' ? 'bg-amber-50 text-amber-800' : 'bg-zinc-100 text-zinc-600'}`}>{statusLabel(comparison.status)}</span>
                          </span>
                        ) : <span className="text-zinc-400">{statement.previousId ? statusLabel('unavailable') : statusLabel('baseline')}</span>}
                      </td>
                    </tr>
                  )
                })}
                {rowCount === 0 && <tr><td colSpan={7} className="px-4 py-8 text-center text-zinc-500">Statement นี้ไม่มีแถวกองทุนให้แสดง</td></tr>}
                {statement.comparisons.filter((item) => item.status === 'missing').map((item) => (
                  <tr key={`missing-${item.code}`} className="bg-amber-50/50">
                    <th scope="row" className="px-4 py-3 font-mono font-bold text-amber-900">{item.code}</th>
                    <td colSpan={5} className="px-4 py-3 text-amber-800">ไม่มีแถวกองทุนนี้ใน Statement ปัจจุบัน จึงไม่ถือว่ายอดเป็นศูนย์</td>
                    <td className="px-4 py-3 text-right"><span className="rounded-full bg-amber-100 px-2 py-0.5 text-[10px] font-bold text-amber-900">ข้อมูลกองทุนขาด</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </article>
  )
}

export default function PvdStatementsTable({
  portfolioId, transactions, history, active, loading, initialSymbol, onClearSymbolFilter, onRefresh, onSuccess,
}: Props) {
  const [query, setQuery] = useState(initialSymbol || '')
  const [pageSize, setPageSize] = useState<10 | 20 | 50>(10)
  const [currentPage, setCurrentPage] = useState(1)
  const [expandedIds, setExpandedIds] = useState<Set<string>>(() => new Set())
  const [modalOpen, setModalOpen] = useState(false)

  useEffect(() => {
    if (initialSymbol) {
      setQuery(initialSymbol)
      setCurrentPage(1)
    }
  }, [initialSymbol])
  useEffect(() => {
    if (!active) setModalOpen(false)
  }, [active])
  useEffect(() => {
    const validIds = new Set(history.statements.map((statement) => statement.id))
    setExpandedIds((previous) => new Set([...previous].filter((id) => validIds.has(id))))
  }, [history.statements])

  const filteredStatements = useMemo(() => {
    const term = query.trim().toUpperCase()
    if (!term) return history.statements
    const canonicalTerm = term.startsWith('PVD_') ? term.slice(4) : term
    return history.statements.filter((statement) =>
      statement.id.toUpperCase().includes(term)
      || (statement.asOfDate || '').includes(term)
      || statement.funds.some((fund) => fund.code.toUpperCase().includes(term) || fund.symbol.toUpperCase().includes(term) || fund.code.toUpperCase().includes(canonicalTerm))
    )
  }, [history.statements, query])
  const totalPages = Math.max(1, Math.ceil(filteredStatements.length / pageSize))
  const pageStatements = filteredStatements.slice((currentPage - 1) * pageSize, currentPage * pageSize)
  const latest = history.statements.find((statement) => statement.id === history.latestStatementId) ?? null
  const currentPageLabel = pageStatements.length
    ? `${(currentPage - 1) * pageSize + 1}–${(currentPage - 1) * pageSize + pageStatements.length} จาก ${filteredStatements.length} งวด`
    : `0 จาก ${filteredStatements.length} งวด`

  useEffect(() => { if (currentPage > totalPages) setCurrentPage(totalPages) }, [currentPage, totalPages])

  const toggleExpanded = (id: string) => setExpandedIds((previous) => {
    const next = new Set(previous)
    if (next.has(id)) next.delete(id)
    else next.add(id)
    return next
  })

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-teal-100 bg-white p-4 shadow-sm">
        <div>
          <h2 className="text-lg font-extrabold text-zinc-900">ประวัติ Statement PVD</h2>
          <p className="mt-0.5 text-xs text-zinc-500">แสดงยอดตาม Statement และรายการกระทบยอด โดยไม่สร้างรายการซื้อขาย</p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <label className="sr-only" htmlFor="pvd-statement-search">ค้นหา Statement หรือกองทุน</label>
          <input
            id="pvd-statement-search"
            type="search"
            value={query}
            onChange={(event) => { setQuery(event.target.value); setCurrentPage(1) }}
            placeholder="ค้นหา Statement / กองทุน"
            className="w-52 rounded-xl border border-zinc-200 px-3 py-2 text-sm focus:border-teal-600 focus:outline-none"
          />
          {query && <button type="button" onClick={() => { setQuery(''); setCurrentPage(1); onClearSymbolFilter?.() }} className="rounded-lg px-2 py-2 text-xs font-semibold text-zinc-600 hover:bg-zinc-100">ล้างตัวกรอง</button>}
          <button type="button" onClick={() => setModalOpen(true)} className="rounded-xl bg-teal-700 px-4 py-2 text-sm font-bold text-white shadow-sm hover:bg-teal-800">อัปเดตจาก Statement</button>
        </div>
      </div>

      {latest && (
        <div className={`rounded-2xl border p-4 sm:p-5 ${history.summaryUnavailableReason ? 'border-amber-300 bg-amber-50' : 'border-teal-200 bg-teal-50/70'}`}>
          <div className="flex flex-wrap items-end justify-between gap-4">
            <div>
              <div className="text-xs font-bold uppercase tracking-wide text-teal-900">ยอดล่าสุด · {dateLabel(latest.asOfDate)}</div>
              {history.summaryUnavailableReason ? (
                <p role="status" className="mt-1 max-w-3xl text-sm font-semibold text-amber-900">{history.summaryUnavailableReason}</p>
              ) : (
                <div className="mt-1 font-mono text-2xl font-extrabold tabular-nums text-zinc-950">{formatBaht(latest.totalNavCents)}</div>
              )}
            </div>
            {latest.previousId && (
              <div className="text-right">
                <div className="text-xs font-semibold text-zinc-600">เปลี่ยนจาก Statement ก่อนหน้า</div>
                <div className={`font-mono text-lg font-bold tabular-nums ${latest.deltaNavCents === null ? 'text-zinc-400' : latest.deltaNavCents >= 0n ? 'text-emerald-700' : 'text-rose-700'}`}>
                  {history.summaryUnavailableReason ? '—' : formatBaht(latest.deltaNavCents, true)}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {history.unresolvedIssues.length > 0 && (
        <div role="alert" className="rounded-xl border border-amber-300 bg-amber-50 px-4 py-3">
          <div className="text-sm font-bold text-amber-950">ยังจัดลำดับข้อมูล PVD บางรายการไม่ได้</div>
          <IssueList issues={history.unresolvedIssues} />
        </div>
      )}

      {loading && <div role="status" className="rounded-xl border border-zinc-200 bg-white p-4 text-sm text-zinc-500">กำลังโหลด Statement…</div>}
      {!loading && filteredStatements.length === 0 && (
        <div className="rounded-2xl border border-dashed border-zinc-300 bg-white px-4 py-12 text-center">
          <h3 className="font-bold text-zinc-800">{history.statements.length === 0 ? 'ยังไม่มี Statement PVD ในพอร์ตนี้' : 'ไม่พบ Statement ตามตัวกรอง'}</h3>
          <p className="mt-1 text-sm text-zinc-500">{history.statements.length === 0 ? 'นำเข้า Statement เพื่อบันทึกยอดและประวัติรายงวด' : 'ลองค้นด้วยรหัสกองทุน เช่น SINSA_GTO หรือ PVD_SINSA_GTO'}</p>
          {history.statements.length === 0 && <button type="button" onClick={() => setModalOpen(true)} className="mt-4 rounded-xl bg-teal-700 px-4 py-2 text-sm font-bold text-white">นำเข้า Statement แรก</button>}
        </div>
      )}

      <div className="space-y-3">
        {pageStatements.map((statement) => (
          <StatementCard
            key={statement.id}
            statement={statement}
            previousStatement={history.statements.find((item) => item.id === statement.previousId) ?? null}
            transactions={transactions}
            expanded={expandedIds.has(statement.id)}
            onToggle={() => toggleExpanded(statement.id)}
            portfolioId={portfolioId}
          />
        ))}
      </div>

      {filteredStatements.length > 0 && (
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-zinc-200 bg-white px-4 py-3 text-xs">
          <div className="flex items-center gap-2 text-zinc-600">
            <span>{currentPageLabel}</span>
            <label htmlFor="pvd-page-size" className="ml-2">งวดต่อหน้า</label>
            <select id="pvd-page-size" value={pageSize} onChange={(event) => { setPageSize(Number(event.target.value) as 10 | 20 | 50); setCurrentPage(1) }} className="rounded-md border border-zinc-200 bg-white px-2 py-1">
              <option value={10}>10</option><option value={20}>20</option><option value={50}>50</option>
            </select>
          </div>
          <div className="flex items-center gap-2">
            <button type="button" disabled={currentPage <= 1} onClick={() => setCurrentPage((page) => page - 1)} className="rounded-lg border border-zinc-200 px-3 py-1.5 font-semibold disabled:opacity-40">ก่อนหน้า</button>
            <span className="min-w-16 text-center text-zinc-600">{currentPage} / {totalPages}</span>
            <button type="button" disabled={currentPage >= totalPages} onClick={() => setCurrentPage((page) => page + 1)} className="rounded-lg border border-zinc-200 px-3 py-1.5 font-semibold disabled:opacity-40">ถัดไป</button>
          </div>
        </div>
      )}

      {modalOpen && (
        <PvdStatementModal
          portfolioId={portfolioId}
          onClose={() => setModalOpen(false)}
          onSuccess={(state) => {
            onSuccess?.(state)
            void onRefresh()
          }}
        />
      )}
    </div>
  )
}
