import React, { useState, useEffect } from 'react'
import type { PivotLevelsDTO } from '../../api/types'

interface SupportResistanceTableProps {
  pivotLevels: PivotLevelsDTO | null
  pivotPeriod?: string | null
  pivotAsOf?: string | null
  currency: 'USD' | 'THB'
  currentPrice: number | null
}

export const SupportResistanceTable: React.FC<SupportResistanceTableProps> = ({
  pivotLevels,
  pivotPeriod,
  pivotAsOf,
  currency,
  currentPrice,
}) => {
  const defaultCapital = currency === 'THB' ? 10000 : 1000
  const storageKey = `sr_calc_capital:${currency}`

  const [capital, setCapital] = useState<number>(() => {
    try {
      const saved = localStorage.getItem(storageKey)
      if (saved) {
        const parsed = parseFloat(saved)
        if (!isNaN(parsed) && parsed > 0) return parsed
      }
    } catch {
      // ignore
    }
    return defaultCapital
  })

  const [rememberCapital, setRememberCapital] = useState<boolean>(() => {
    try {
      return localStorage.getItem(storageKey) !== null
    } catch {
      return false
    }
  })

  // Update capital when currency changes if not remembered
  useEffect(() => {
    try {
      const saved = localStorage.getItem(storageKey)
      if (saved) {
        const parsed = parseFloat(saved)
        if (!isNaN(parsed) && parsed > 0) {
          setCapital(parsed)
          setRememberCapital(true)
          return
        }
      }
    } catch {
      // ignore
    }
    setCapital(currency === 'THB' ? 10000 : 1000)
    setRememberCapital(false)
  }, [currency, storageKey])

  const handleCapitalChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value)
    const newCapital = isNaN(val) ? 0 : val
    setCapital(newCapital)
    if (rememberCapital && newCapital > 0) {
      try {
        localStorage.setItem(storageKey, newCapital.toString())
      } catch {
        // ignore
      }
    }
  }

  const handleRememberToggle = (e: React.ChangeEvent<HTMLInputElement>) => {
    const checked = e.target.checked
    setRememberCapital(checked)
    try {
      if (checked && capital > 0) {
        localStorage.setItem(storageKey, capital.toString())
      } else {
        localStorage.removeItem(storageKey)
      }
    } catch {
      // ignore
    }
  }

  const currSign = currency === 'THB' ? '฿' : '$'

  if (!pivotLevels) {
    return (
      <div className="flex h-full flex-col justify-center items-center rounded-2xl border border-edge bg-panel p-6 text-center shadow-sm">
        <span className="text-3xl mb-2">🎯</span>
        <h4 className="font-medium text-zinc-800">ไม่มีข้อมูลแนวรับ-แนวต้าน</h4>
        <p className="text-xs text-zinc-400 mt-1 max-w-xs">
          ยังไม่มีข้อมูลราคาในอดีตเพียงพอสำหรับการคำนวณ Pivot Point
        </p>
      </div>
    )
  }

  const rLevels = [
    { label: 'R1', price: pivotLevels.r1 },
    { label: 'R2', price: pivotLevels.r2 },
    { label: 'R3', price: pivotLevels.r3 },
  ]

  const sLevels = [
    { label: 'S1', price: pivotLevels.s1 },
    { label: 'S2', price: pivotLevels.s2 },
    { label: 'S3', price: pivotLevels.s3 },
    { label: 'S4', price: pivotLevels.s4 },
  ]

  const getProfitCellColor = (pct: number) => {
    if (pct < 0) return 'bg-rose-50 text-rose-700'
    if (pct >= 45) return 'bg-emerald-200/90 text-emerald-950 font-bold'
    if (pct >= 30) return 'bg-emerald-100 text-emerald-900 font-semibold'
    if (pct >= 15) return 'bg-emerald-50 text-emerald-800 font-medium'
    return 'bg-emerald-50/60 text-emerald-700'
  }

  return (
    <div className="flex flex-col rounded-2xl border border-edge bg-panel p-5 shadow-sm">
      {/* Header Title & Controls */}
      <div className="border-b border-edge/60 pb-4">
        <div className="flex items-center justify-between">
          <h3 className="text-base font-bold text-zinc-900 flex items-center gap-1.5">
            <span>🎯</span>
            <span>ตารางคำนวณ แนวรับ-แนวต้าน</span>
          </h3>
          <span className="text-xs font-mono text-zinc-500 bg-surface px-2 py-0.5 rounded border border-edge">
            Pivot: {currSign}{pivotLevels.pivot.toFixed(2)}
          </span>
        </div>

        {/* Capital Input & Remember Checkbox */}
        <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <label htmlFor="capital-input" className="text-xs font-semibold text-zinc-700">
              ใส่เงินลงทุนเพิ่ม ({currency}):
            </label>
            <div className="relative">
              <span className="absolute left-2.5 top-1/2 -translate-y-1/2 text-xs text-zinc-400 font-medium">
                {currSign}
              </span>
              <input
                id="capital-input"
                type="number"
                min="1"
                step="100"
                value={capital === 0 ? '' : capital}
                onChange={handleCapitalChange}
                className="w-28 rounded-lg border border-edge bg-surface pl-6 pr-2.5 py-1 text-xs font-mono font-bold text-zinc-900 focus:border-sky-500 focus:outline-none focus:ring-1 focus:ring-sky-500"
              />
            </div>
          </div>

          <label className="flex items-center gap-1.5 text-xs text-zinc-500 cursor-pointer hover:text-zinc-800 select-none">
            <input
              type="checkbox"
              checked={rememberCapital}
              onChange={handleRememberToggle}
              className="h-3.5 w-3.5 rounded border-zinc-300 text-sky-600 focus:ring-sky-500"
            />
            <span>จดจำนวนเงินนี้ไว้</span>
          </label>
        </div>
      </div>

      {/* S/R Matrix Table */}
      <div className="mt-4 overflow-x-auto">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="border-b border-edge text-[11px] font-semibold uppercase text-zinc-600 bg-surface/70">
              <th className="py-2.5 px-3 rounded-l-lg">ซื้อที่</th>
              {rLevels.map((r) => (
                <th key={r.label} className="py-2.5 px-3 text-right">
                  <div>{r.label}</div>
                  <div className="font-mono text-[10px] text-zinc-500 font-normal">
                    {currSign}{r.price.toFixed(2)}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-edge/40 text-xs font-mono">
            {sLevels.map((s) => {
              const isBelowCurrent = currentPrice !== null && s.price < currentPrice
              return (
                <tr key={s.label} className="hover:bg-surface/50 transition-colors">
                  {/* Row Header (Sx) */}
                  <td className="py-2.5 px-3">
                    <div className="font-bold text-zinc-900 flex items-center gap-1">
                      <span>{s.label}</span>
                      {isBelowCurrent && (
                        <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" title="ต่ำกว่าราคาปัจจุบัน" />
                      )}
                    </div>
                    <div className="text-[11px] text-zinc-500">
                      {currSign}{s.price.toFixed(2)}
                    </div>
                  </td>

                  {/* Calculations for R1, R2, R3 */}
                  {rLevels.map((r) => {
                    const shares = capital > 0 && s.price > 0 ? capital / s.price : 0
                    const profit = shares * (r.price - s.price)
                    const profitPct = s.price > 0 ? ((r.price - s.price) / s.price) * 100 : 0
                    const isProfit = profit >= 0

                    return (
                      <td
                        key={r.label}
                        className={`py-2 px-3 text-right transition-colors rounded-md ${getProfitCellColor(
                          profitPct
                        )}`}
                      >
                        <div className="font-bold">
                          {isProfit ? '+' : ''}
                          {currSign}
                          {profit.toLocaleString('en-US', {
                            minimumFractionDigits: 2,
                            maximumFractionDigits: 2,
                          })}
                        </div>
                        <div className="text-[10px] opacity-85">
                          ({isProfit ? '+' : ''}
                          {profitPct.toFixed(2)}%)
                        </div>
                      </td>
                    )
                  })}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {/* Footnote Metadata */}
      <div className="mt-4 flex flex-wrap items-center justify-between gap-2 border-t border-edge/60 pt-3 text-[11px] text-zinc-400">
        <div>
          <span>สูตร Classic Pivot ({pivotPeriod ?? 'monthly'})</span>
          {pivotAsOf && (
            <span className="ml-1.5">
              • ฐานคำนวณ: แท่งสิ้นเดือน <strong className="text-zinc-600">{pivotAsOf}</strong>
            </span>
          )}
        </div>
        {currentPrice !== null && (
          <div className="text-zinc-500">
            ราคาปัจจุบัน: <span className="font-mono font-semibold text-zinc-700">{currSign}{currentPrice.toFixed(2)}</span>
          </div>
        )}
      </div>
    </div>
  )
}
