import { useEffect, useMemo, useState } from 'react'
import Modal from '../../ui/Modal'
import { api } from '../../../api/client'
import type { ActualPortfolioStateDTO, PvdAnalyzeResponseDTO } from '../../../api/types'
import { formatTHB } from '../../../utils/formatters'

interface Props {
  portfolioId: string
  onClose: () => void
  onSuccess: (state: ActualPortfolioStateDTO) => void
}

const passwordStorageKey = (portfolioId: string) => `pvd-statement-password:${portfolioId}`

function toCents(value: unknown): bigint | null {
  if (typeof value !== 'string') return null
  const match = value.trim().match(/^(-?)(\d+)(?:\.(\d{1,2}))?$/)
  if (!match) return null
  const cents = BigInt(match[2]!) * 100n + BigInt((match[3] || '').padEnd(2, '0') || '0')
  return match[1] === '-' ? -cents : cents
}

function fromCents(value: bigint): string {
  const absolute = value < 0n ? -value : value
  const formatted = `${absolute / 100n}.${String(absolute % 100n).padStart(2, '0')}`
  return value < 0n ? `-${formatted}` : formatted
}

export default function PvdStatementModal({ portfolioId, onClose, onSuccess }: Props) {
  const [file, setFile] = useState<File | null>(null)
  const [password, setPassword] = useState(() => {
    try { return sessionStorage.getItem(passwordStorageKey(portfolioId)) || '' } catch { return '' }
  })
  const [rememberPassword, setRememberPassword] = useState(() => {
    try { return sessionStorage.getItem(passwordStorageKey(portfolioId)) !== null } catch { return false }
  })
  const [vestingPercent, setVestingPercent] = useState(100)
  const [scan, setScan] = useState<PvdAnalyzeResponseDTO | null>(null)
  const [analyzing, setAnalyzing] = useState(false)
  const [committing, setCommitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  const [archiveId, setArchiveId] = useState<string | null>(null)
  const [idempotencyKey, setIdempotencyKey] = useState('')

  useEffect(() => {
    try {
      const key = passwordStorageKey(portfolioId)
      if (rememberPassword) sessionStorage.setItem(key, password)
      else if (!rememberPassword) sessionStorage.removeItem(key)
    } catch {
      // Session storage can be disabled by browser policy; password remains in component state.
    }
  }, [password, rememberPassword, portfolioId])

  const statement = scan?.statement || null
  const vestedAmount = useMemo(() => {
    if (!statement) return scan?.vested_amount || null
    const memberPrincipal = toCents(statement.member_principal)
    const memberBenefit = toCents(statement.member_benefit)
    const employerPrincipal = toCents(statement.employer_principal)
    const employerBenefit = toCents(statement.employer_benefit)
    if (memberPrincipal === null || memberBenefit === null || employerPrincipal === null || employerBenefit === null) {
      return scan?.vested_amount || null
    }
    const memberCents = memberPrincipal + memberBenefit
    const employerCents = employerPrincipal + employerBenefit
    const weightedEmployer = (employerCents * BigInt(vestingPercent) + 50n) / 100n
    return fromCents(memberCents + weightedEmployer)
  }, [statement, scan?.vested_amount, vestingPercent])
  const totalNavCents = toCents(statement?.total_nav)
  const totalContributionCents = toCents(statement?.total_contribution)
  const totalBenefit = totalNavCents !== null && totalContributionCents !== null
    ? fromCents(totalNavCents - totalContributionCents)
    : null
  const formatAmount = (value: string | null | undefined) => {
    const cents = toCents(value)
    return cents === null ? 'ไม่พร้อมคำนวณ' : formatTHB(Number(fromCents(cents)))
  }

  const handleAnalyze = async () => {
    if (!file) {
      setError('เลือกไฟล์ Statement PDF ก่อน')
      return
    }
    setAnalyzing(true)
    setError(null)
    setSuccess(null)
    setArchiveId(null)
    setScan(null)
    try {
      const result = await api.analyzePvdUpload(file, portfolioId, password, vestingPercent)
      setScan(result)
      if (result.can_commit && result.scan_id) setIdempotencyKey(crypto.randomUUID())
      else setIdempotencyKey('')
      if (!result.can_commit) setError(result.message)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'อ่าน Statement ไม่สำเร็จ')
    } finally {
      setAnalyzing(false)
    }
  }

  const handleCommit = async () => {
    if (!scan?.scan_id || !idempotencyKey) return
    setCommitting(true)
    setError(null)
    try {
      const result = await api.commitPvdStatement({
        scan_id: scan.scan_id,
        idempotency_key: idempotencyKey,
        portfolio_id: portfolioId,
        vesting_percent: vestingPercent,
      })
      setSuccess(result.message)
      setArchiveId(result.archive_id)
      onSuccess(result.state)
      setScan(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'นำเข้า Statement ไม่สำเร็จ')
    } finally {
      setCommitting(false)
    }
  }

  return (
    <Modal
      titleId="pvd-statement-title"
      onClose={onClose}
      panelClassName="max-w-5xl max-h-[92vh] overflow-y-auto rounded-2xl border border-teal-100 bg-white p-0 shadow-2xl"
    >
      <div className="sticky top-0 z-10 flex items-center justify-between border-b border-teal-100 bg-white/95 px-5 py-4 backdrop-blur">
        <div>
          <h2 id="pvd-statement-title" className="text-lg font-extrabold text-zinc-900">อัปเดต PVD จาก Statement</h2>
          <p className="mt-0.5 text-xs text-zinc-500">BBLAM · นำเข้ายอดหน่วยและมูลค่าปลายงวดแบบ snapshot</p>
        </div>
        <button type="button" onClick={onClose} className="rounded-lg p-2 text-zinc-400 hover:bg-zinc-100" aria-label="ปิดหน้าต่าง">✕</button>
      </div>

      <div className="space-y-5 p-5">
        {success && (
          <div role="status" className="rounded-xl border border-emerald-200 bg-emerald-50 p-4 text-sm font-semibold text-emerald-800">
            <div>{success}</div>
            {archiveId && <a className="mt-2 inline-block underline" href={api.getPvdDocumentUrl(archiveId, portfolioId)} target="_blank" rel="noreferrer">ดาวน์โหลด Statement PDF ที่เก็บไว้</a>}
          </div>
        )}
        {error && <div role="alert" className="rounded-xl border border-rose-200 bg-rose-50 p-3 text-sm font-semibold text-rose-800">{error}</div>}

        <div className="grid gap-4 md:grid-cols-[1.4fr_1fr]">
          <div className="space-y-3 rounded-xl border border-zinc-200 bg-zinc-50/70 p-4">
            <label className="block text-xs font-bold text-zinc-700" htmlFor="pvd-statement-file">Statement PDF (สูงสุด 10 MB)</label>
            <input
              id="pvd-statement-file"
              type="file"
              accept="application/pdf,.pdf"
              onChange={(event) => { setFile(event.target.files?.[0] || null); setScan(null); setError(null) }}
              className="block w-full text-sm text-zinc-600 file:mr-3 file:rounded-lg file:border-0 file:bg-teal-700 file:px-3 file:py-2 file:font-bold file:text-white hover:file:bg-teal-800"
            />
            {file && <p className="truncate text-xs text-zinc-500">{file.name} · {(file.size / 1024 / 1024).toFixed(2)} MB</p>}
            <label className="block text-xs font-bold text-zinc-700" htmlFor="pvd-statement-password">รหัสผ่าน PDF (ถ้ามี)</label>
            <input
              id="pvd-statement-password"
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              autoComplete="off"
              className="w-full rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm focus:border-teal-600 focus:outline-none"
            />
            <label className="inline-flex items-center gap-2 text-xs text-zinc-600">
              <input type="checkbox" checked={rememberPassword} onChange={(event) => setRememberPassword(event.target.checked)} />
              จำรหัสผ่านใน session ของเบราว์เซอร์นี้
            </label>
            <button type="button" onClick={handleAnalyze} disabled={!file || analyzing || committing} className="w-full rounded-xl bg-teal-700 px-4 py-2.5 text-sm font-bold text-white shadow hover:bg-teal-800 disabled:cursor-not-allowed disabled:opacity-50">
              {analyzing ? 'กำลังอ่านและตรวจสอบ…' : 'ตรวจสอบ Statement'}
            </button>
          </div>

          <div className="space-y-3 rounded-xl border border-teal-100 bg-teal-50/60 p-4">
            <div className="flex items-center justify-between text-xs font-bold text-teal-950">
              <label htmlFor="pvd-vesting">สิทธิเงินสมทบนายจ้าง</label>
              <span>{vestingPercent}%</span>
            </div>
            <input id="pvd-vesting" aria-label="เปอร์เซ็นต์สิทธิเงินสมทบนายจ้าง" type="range" min={0} max={100} step={1} value={vestingPercent} onChange={(event) => { setVestingPercent(Number(event.target.value)); setIdempotencyKey(crypto.randomUUID()) }} className="w-full accent-teal-700" />
            <div className="flex justify-between text-[11px] text-zinc-500"><span>0%</span><span>50%</span><span>100%</span></div>
            {scan?.statement && (
              <div className="space-y-2 border-t border-teal-100 pt-3 text-sm">
                <div className="flex justify-between gap-2"><span>วันที่ Statement</span><strong>{String(scan.statement.as_of_date || '-')}</strong></div>
                <div className="flex justify-between gap-2"><span>สมาชิก</span><strong>{String(scan.statement.masked_member_id || '-')}</strong></div>
                <div className="flex justify-between gap-2"><span>ยอด PVD ล่าสุดที่บันทึกในระบบ</span><strong>{formatAmount(scan.current_nav)}</strong></div>
                <div className="flex justify-between gap-2"><span>มูลค่าตาม Statement ฉบับใหม่</span><strong>{formatAmount(scan.new_nav)}</strong></div>
                <div className="flex justify-between gap-2"><span>ยอดเปลี่ยนจากที่บันทึก</span><strong>{formatAmount(scan.delta_nav)}</strong></div>
                <div className="flex justify-between gap-2 border-t border-teal-100 pt-2"><span>เงินต้นสะสมตาม Statement</span><strong>{formatAmount(typeof statement?.total_contribution === 'string' ? statement.total_contribution : null)}</strong></div>
                <div className="flex justify-between gap-2"><span>ผลประโยชน์รวมตาม Statement</span><strong>{totalBenefit === null ? 'ไม่พร้อมคำนวณ' : formatTHB(Number(totalBenefit))}</strong></div>
                <div className="flex justify-between gap-2 border-t border-teal-100 pt-2 font-bold"><span>ยอดที่มีสิทธิได้รับ</span><strong>{vestedAmount === null ? 'ไม่พร้อมคำนวณ' : formatTHB(Number(vestedAmount))}</strong></div>
              </div>
            )}
          </div>
        </div>

        {scan?.holdings && scan.holdings.length > 0 && (
          <div className="overflow-hidden rounded-xl border border-zinc-200">
            <div className="flex items-center justify-between bg-zinc-50 px-4 py-3">
              <div>
                <h3 className="text-sm font-bold text-zinc-900">กองทุนย่อย ({scan.holdings.length})</h3>
                <p className="text-[11px] text-zinc-500">เงินต้นจัดสรรจากยอดรวมตามสัดส่วนมูลค่าเพื่อกระทบยอด ไม่ใช่ต้นทุนซื้อจริงรายกอง</p>
              </div>
              <span className="font-mono text-xs font-semibold text-zinc-600">{scan.status}</span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[760px] text-left text-xs">
                <thead className="border-y border-zinc-200 bg-white text-zinc-500">
                  <tr><th className="px-3 py-2">กองทุน</th><th className="px-3 py-2 text-right">หน่วย</th><th className="px-3 py-2 text-right">NAV</th><th className="px-3 py-2 text-right">มูลค่า</th><th className="px-3 py-2 text-right">ปรับยอด</th><th className="px-3 py-2 text-right">เงินต้นจัดสรร</th></tr>
                </thead>
                <tbody className="divide-y divide-zinc-100">
                  {scan.holdings.map((holding) => (
                    <tr key={holding.fund_code}>
                      <td className="px-3 py-2"><strong className="font-mono">{holding.fund_code}</strong><span className="ml-2 text-zinc-500">{holding.name}</span></td>
                      <td className="px-3 py-2 text-right font-mono">{Number(holding.units).toLocaleString(undefined, { maximumFractionDigits: 6 })}</td>
                      <td className="px-3 py-2 text-right font-mono">{holding.statement_nav}</td>
                      <td className="px-3 py-2 text-right font-mono">{formatTHB(Number(holding.statement_market_value))}</td>
                      <td className="px-3 py-2 text-right font-mono">{formatTHB(Number(holding.reconciliation_adjustment))}</td>
                      <td className="px-3 py-2 text-right font-mono">{formatTHB(Number(holding.allocated_cost))}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        <div className="flex flex-wrap justify-end gap-2 border-t border-zinc-100 pt-4">
          <button type="button" onClick={onClose} className="rounded-xl border border-zinc-300 bg-white px-4 py-2 text-sm font-semibold text-zinc-700 hover:bg-zinc-50">ปิด</button>
          {scan?.can_commit && scan.scan_id && (
            <button type="button" onClick={handleCommit} disabled={committing || analyzing} className="rounded-xl bg-teal-700 px-5 py-2 text-sm font-bold text-white shadow hover:bg-teal-800 disabled:opacity-50">
              {committing ? 'กำลังบันทึก…' : 'ยืนยันนำเข้าทั้งชุด'}
            </button>
          )}
        </div>
      </div>
    </Modal>
  )
}
