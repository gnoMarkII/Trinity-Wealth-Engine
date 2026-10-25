import { useEffect, useState } from 'react'
import { api } from '../../api/client'
import type { PositionReviewResponseDTO } from '../../api/types'
import PriceReturnDecompositionBar from './PriceReturnDecompositionBar'

interface PositionReviewModalProps {
  isOpen: boolean
  onClose: () => void
  ticker: string
  market?: string
  portfolioId?: string
}

export function PositionReviewModal({ isOpen, onClose, ticker, market = 'US', portfolioId = 'default' }: PositionReviewModalProps) {
  const [data, setData] = useState<PositionReviewResponseDTO | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const currencySymbol = market === 'TH' ? '฿' : '$'

  useEffect(() => {
    if (!isOpen || !ticker) {
      setData(null)
      return
    }

    let isMounted = true
    setLoading(true)
    setError(null)

    api.getPositionReview(ticker, market, portfolioId)
      .then((res) => {
        if (isMounted) {
          setData(res)
          setLoading(false)
        }
      })
      .catch((err) => {
        if (isMounted) {
          setError(err?.message || 'Failed to load position review')
          setLoading(false)
        }
      })

    return () => {
      isMounted = false
    }
  }, [isOpen, ticker, market, portfolioId])

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4 overflow-y-auto">
      <div className="relative w-full max-w-4xl max-h-[90vh] flex flex-col bg-zinc-900 border border-zinc-700/80 rounded-2xl shadow-2xl overflow-hidden text-zinc-100">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-800 bg-zinc-900/80">
          <div className="flex items-center gap-3">
            <h2 className="text-xl font-bold tracking-tight text-white">
              {ticker} Position Review
            </h2>
            {data?.action_stance && (
              <span className="px-2.5 py-0.5 text-xs font-semibold rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                {data.action_stance}
              </span>
            )}
          </div>
          <button
            type="button"
            aria-label="Close"
            onClick={onClose}
            className="p-2 text-zinc-400 hover:text-white rounded-lg hover:bg-zinc-800 transition"
          >
            ✕
          </button>
        </div>

        {/* Content Body */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {loading && (
            <div className="py-12 flex flex-col items-center justify-center text-zinc-400 gap-2">
              <div className="w-6 h-6 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
              <span>กำลังคำนวณการแยกตัวประกอบราคา (Price Move Decomposition) และจำลองการประเมินสถานะ...</span>
            </div>
          )}

          {error && (
            <div className="p-4 bg-rose-500/10 border border-rose-500/20 rounded-xl text-rose-400 text-sm">
              {error}
            </div>
          )}

          {!loading && data && (
            <>
              {/* Holding Summary */}
              {data.holding_summary && (
                <div className="p-4 rounded-xl bg-zinc-800/60 border border-zinc-700/60 flex flex-wrap items-center gap-6 text-sm">
                  <div>
                    <span className="text-zinc-400 text-xs block">สถานะการถือครอง</span>
                    <span className="font-semibold text-white">
                      {data.holding_summary.is_held ? 'ถือครองอยู่ (Held)' : 'ยังไม่ได้เปิดสถานะ'}
                    </span>
                  </div>
                  {data.holding_summary.current_weight_pct != null && (
                    <div>
                      <span className="text-zinc-400 text-xs block">สัดส่วนพอร์ต</span>
                      <span className="font-semibold text-white">
                        {data.holding_summary.current_weight_pct.toFixed(2)}%
                      </span>
                    </div>
                  )}
                  {data.holding_summary.avg_cost != null && (
                    <div>
                      <span className="text-zinc-400 text-xs block">ต้นทุนเฉลี่ย</span>
                      <span className="font-semibold text-white">
                        {currencySymbol}{data.holding_summary.avg_cost.toFixed(2)}
                      </span>
                    </div>
                  )}
                  {data.holding_summary.unrealized_pnl_pct != null && (
                    <div>
                      <span className="text-zinc-400 text-xs block">กำไร/ขาดทุนทางบัญชี</span>
                      <span className={`font-semibold ${data.holding_summary.unrealized_pnl_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                        {data.holding_summary.unrealized_pnl_pct >= 0 ? '+' : ''}
                        {data.holding_summary.unrealized_pnl_pct.toFixed(2)}%
                      </span>
                    </div>
                  )}
                </div>
              )}

              {/* 1. Price Move Decomposition */}
              <div className="space-y-3">
                <h3 className="text-base font-semibold text-zinc-200">
                  1. Price Move Decomposition
                </h3>
                {data.decompositions && data.decompositions.length > 0 ? (
                  <div className="space-y-3">
                    {data.decompositions.map((decomp, idx) => (
                      <PriceReturnDecompositionBar key={idx} decomposition={decomp} />
                    ))}
                  </div>
                ) : (
                  <p className="text-xs text-zinc-500">ไม่มีข้อมูล Decomposition</p>
                )}
              </div>

              {/* 2. Then vs Now */}
              {data.then_vs_now && (
                <div className="space-y-3">
                  <h3 className="text-base font-semibold text-zinc-200">
                    2. Historical Thesis Reconstruction (Then vs Now)
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="p-4 rounded-xl bg-zinc-800/40 border border-zinc-700/50">
                      <span className="text-xs font-semibold text-zinc-400 uppercase tracking-wider block mb-2">
                        สมมติฐานเดิมเมื่อเข้าซื้อ (Then)
                      </span>
                      <p className="text-sm text-zinc-300">{data.then_vs_now.what_was_true_then}</p>
                    </div>
                    <div className="p-4 rounded-xl bg-zinc-800/40 border border-zinc-700/50">
                      <span className="text-xs font-semibold text-zinc-400 uppercase tracking-wider block mb-2">
                        ความเป็นจริงในปัจจุบัน (Now)
                      </span>
                      <p className="text-sm text-zinc-300">{data.then_vs_now.what_is_true_now}</p>
                    </div>
                  </div>
                  {data.then_vs_now.thesis_reflection_question && (
                    <div className="p-3.5 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-200 text-xs">
                      <span className="font-semibold block mb-1">💡 Anti-Anchoring Reflection Question:</span>
                      {data.then_vs_now.thesis_reflection_question}
                    </div>
                  )}
                </div>
              )}

              {/* 3. Decision Triggers */}
              <div className="space-y-3">
                <h3 className="text-base font-semibold text-zinc-200">
                  3. Portfolio-Aware Decision Triggers (Worth Watching)
                </h3>
                {data.triggers && data.triggers.length > 0 ? (
                  <div className="space-y-2">
                    {data.triggers.map((trig) => (
                      <div key={trig.trigger_id} className="p-3 rounded-lg bg-zinc-800/40 border border-zinc-700/50 flex items-start justify-between gap-4">
                        <div className="space-y-1">
                          <div className="font-medium text-sm text-zinc-200">{trig.title}</div>
                          <div className="text-xs text-zinc-400">{trig.condition_text}</div>
                        </div>
                        <div className="text-right shrink-0">
                          <span className="text-xs px-2 py-0.5 rounded bg-zinc-700 text-zinc-300 font-medium">
                            {trig.case_impact.replace(/_/g, ' ').toLowerCase().replace(/\b\w/g, (c) => c.toUpperCase())}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-xs text-zinc-500">ไม่มี Decision Triggers ที่ตั้งไว้</p>
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
