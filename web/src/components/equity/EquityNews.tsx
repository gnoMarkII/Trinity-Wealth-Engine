import React, { useEffect, useState } from 'react'
import { api } from '../../api/client'
import type { EquityNewsDTO, NewsDiscoverySnapshotDTO } from '../../api/types'

interface EquityNewsProps {
  ticker: string
}

export const EquityNews: React.FC<EquityNewsProps> = ({ ticker }) => {
  const [news, setNews] = useState<EquityNewsDTO | null>(null)
  const [discoveryNews, setDiscoveryNews] = useState<NewsDiscoverySnapshotDTO | null>(null)
  const [loading, setLoading] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)
  const [is404, setIs404] = useState<boolean>(false)

  useEffect(() => {
    let isMounted = true
    const fetchNews = async () => {
      setLoading(true)
      setError(null)
      setIs404(false)
      try {
        const data = await api.getEquityNews(ticker)
        if (isMounted) {
          setNews(data)
        }
      } catch (err: any) {
        if (isMounted) {
          if (err.status === 404 || err.message?.includes('404')) {
            setIs404(true)
          } else {
            setError(err.message || 'ไม่สามารถโหลดข้อมูลข่าวได้')
          }
        }
      } finally {
        if (isMounted) {
          setLoading(false)
        }
      }
    }

    fetchNews()

    // Secondary RSS discovery candidate aggregation
    api.getEquityNewsDiscovery?.(ticker)
      .then((res) => {
        if (isMounted) setDiscoveryNews(res)
      })
      .catch(() => {})

    return () => {
      isMounted = false
    }
  }, [ticker])

  if (loading) {
    return (
      <div className="flex justify-center items-center h-48 text-zinc-500" aria-live="polite">
        กำลังโหลดข่าวสาร...
      </div>
    )
  }

  if (is404 || (!news && !error)) {
    return (
      <div className="flex flex-col justify-center items-center h-64 bg-surface rounded-xl border border-dashed border-edge p-8 text-center">
        <svg className="w-12 h-12 text-zinc-400 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1m2 13a2 2 0 01-2-2V7m2 13a2 2 0 002-2V9a2 2 0 00-2-2h-2m-4-3H9M7 16h6M7 8h6v4H7V8z" />
        </svg>
        <h3 className="text-lg font-medium text-zinc-900 mb-1">ยังไม่มีข้อมูลข่าวสำหรับหุ้นตัวนี้ในระบบ</h3>
        <p className="text-zinc-500 max-w-md text-sm">
          สามารถสั่งงานผ่านผู้จัดการ (Manager Agent) ในระบบเพื่อเริ่มต้นดึงข่าวและวิเคราะห์หุ้นตัวนี้ได้
        </p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex flex-col justify-center items-center h-48 bg-red-50 rounded-xl border border-red-200 p-6 text-center text-red-600" role="alert">
        <h3 className="text-base font-medium mb-1">เกิดข้อผิดพลาด</h3>
        <p className="text-sm">{error}</p>
      </div>
    )
  }

  const hasNews = Boolean(news && news.items && news.items.length > 0)
  const hasDiscovery = Boolean(discoveryNews && discoveryNews.items && discoveryNews.items.length > 0)

  if (!hasNews && !hasDiscovery) {
    return (
      <div className="bg-surface rounded-xl border border-edge p-6 text-center text-zinc-500 text-sm">
        ไม่พบรายการข่าวสารในระบบ
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Primary / Analyzed News Items */}
      {hasNews && (
        <div className="space-y-4">
          <div className="flex items-center justify-between border-b border-edge pb-3">
            <div className="text-xs font-semibold uppercase tracking-wider text-sky-600">
              Latest News ({news!.items.length})
            </div>
            {news!.last_updated && (
              <span className="text-xs text-zinc-400">
                อัปเดตล่าสุด {news!.last_updated}
              </span>
            )}
          </div>

          <div className="grid grid-cols-1 gap-4">
            {news!.items.map((item, idx) => (
              <div
                key={idx}
                className="flex flex-col justify-between rounded-xl border border-edge bg-panel p-4 shadow-sm shadow-black/5 hover:border-sky-200 transition-colors"
              >
                <div className="space-y-2">
                  <div className="flex items-start justify-between gap-3">
                    <h4 className="font-semibold text-zinc-900 text-base leading-snug">
                      {item.title}
                    </h4>
                    {item.is_stale && (
                      <span className="shrink-0 px-2 py-0.5 rounded text-[11px] font-medium bg-amber-50 text-amber-700 border border-amber-200">
                        ⚠️ STALE
                      </span>
                    )}
                  </div>

                  <div className="flex flex-wrap items-center gap-3 text-xs text-zinc-500">
                    <span className="font-medium text-zinc-700">ที่มา: {item.source}</span>
                    {item.sources_count && item.sources_count > 1 && (
                      <span className="bg-surface-strong px-2 py-0.5 rounded text-zinc-600">
                        {item.sources_count} sources
                      </span>
                    )}
                    <span className="text-zinc-400">•</span>
                    <span>{item.freshness_reason || `${item.age_hours} ชั่วโมงที่แล้ว`}</span>
                  </div>
                </div>

                {item.link && item.link !== 'N/A' && (
                  <div className="mt-3 pt-3 border-t border-edge flex justify-end">
                    <a
                      href={item.link}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 text-xs font-medium text-sky-600 hover:text-sky-700 hover:underline"
                    >
                      อ่านต่อบน {item.source}
                      <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                      </svg>
                    </a>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Terminal V2 Phase 4: Secondary RSS News Discovery Candidates */}
      {hasDiscovery && (
        <div className="space-y-4 rounded-xl border border-sky-100 bg-sky-50/20 p-5">
          <div className="flex flex-wrap items-baseline justify-between gap-2 border-b border-sky-200/60 pb-3">
            <div>
              <h4 className="text-xs font-bold uppercase tracking-wider text-sky-800 flex items-center gap-2">
                <span>RSS News Discovery Candidates</span>
                <span className="rounded-full bg-sky-100 text-sky-800 text-[10px] font-bold px-2 py-0.5 border border-sky-200">
                  SECONDARY RSS
                </span>
              </h4>
              <p className="mt-0.5 text-[11px] text-zinc-500">
                News discovery from aggregator/RSS feeds — secondary discovery source, not guaranteed real-time or primary news wire.
              </p>
            </div>
            <span className="text-xs text-zinc-500">
              {discoveryNews!.items.length} candidates
            </span>
          </div>

          <div className="grid grid-cols-1 gap-3">
            {discoveryNews!.items.map((cand, idx) => (
              <div
                key={idx}
                className="flex flex-col justify-between rounded-lg border border-edge bg-panel p-3.5 shadow-2xs hover:border-sky-300 transition-colors"
              >
                <div className="space-y-1.5">
                  <div className="flex items-start justify-between gap-2">
                    <h5 className="font-semibold text-zinc-900 text-sm leading-snug">
                      {cand.headline}
                    </h5>
                    {cand.is_stale && (
                      <span className="shrink-0 px-1.5 py-0.5 rounded text-[10px] font-medium bg-amber-50 text-amber-700 border border-amber-200">
                        STALE
                      </span>
                    )}
                  </div>
                  <div className="flex flex-wrap items-center gap-2 text-[11px] text-zinc-500">
                    <span className="font-semibold text-zinc-700">{cand.publisher}</span>
                    <span className="text-zinc-400">•</span>
                    <span>{cand.source_type || 'RSS'}</span>
                    {cand.published_at && (
                      <>
                        <span className="text-zinc-400">•</span>
                        <span>{cand.published_at}</span>
                      </>
                    )}
                  </div>
                </div>

                {cand.article_url && (
                  <div className="mt-2 pt-2 border-t border-edge/60 flex justify-end">
                    <a
                      href={cand.article_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 text-[11px] font-medium text-sky-600 hover:text-sky-700 hover:underline"
                    >
                      ดูบทความต้นฉบับ ↗
                    </a>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
