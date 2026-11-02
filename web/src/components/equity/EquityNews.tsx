import React, { useEffect, useState } from 'react'
import { api } from '../../api/client'
import type { EquityNewsDTO, NewsDiscoverySnapshotDTO } from '../../api/types'
import { NewsCardListSkeleton } from '../ui/FlowSkeleton'

interface EquityNewsProps {
  ticker: string
}

export const EquityNews: React.FC<EquityNewsProps> = ({ ticker }) => {
  const [news, setNews] = useState<EquityNewsDTO | null>(null)
  const [discoveryNews, setDiscoveryNews] = useState<NewsDiscoverySnapshotDTO | null>(null)
  const [loadingPrimary, setLoadingPrimary] = useState<boolean>(true)
  const [loadingDiscovery, setLoadingDiscovery] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)
  const [is404, setIs404] = useState<boolean>(false)

  useEffect(() => {
    let isMounted = true
    setNews(null)
    setDiscoveryNews(null)
    setLoadingPrimary(true)
    setLoadingDiscovery(true)
    setError(null)
    setIs404(false)

    // Stage 1: Load Vault / Pre-indexed Digest News (~20ms) decoupled
    api.getEquityNews(ticker)
      .then((data) => {
        if (!isMounted) return
        setNews(data)
      })
      .catch((err) => {
        if (!isMounted) return
        if (err?.status === 404 || err?.message?.includes('404')) {
          setIs404(true)
        } else {
          setError(err?.message || 'ไม่สามารถโหลดข้อมูลข่าวได้')
        }
      })
      .finally(() => {
        if (isMounted) setLoadingPrimary(false)
      })

    // Stage 2: Load RSS News Discovery (1-4s) decoupled
    if (api.getEquityNewsDiscovery) {
      api.getEquityNewsDiscovery(ticker)
        .then((res) => {
          if (!isMounted) return
          if (res) setDiscoveryNews(res)
        })
        .catch(() => {
          // Soft fail for discovery
        })
        .finally(() => {
          if (isMounted) setLoadingDiscovery(false)
        })
    } else {
      setLoadingDiscovery(false)
    }

    return () => {
      isMounted = false
    }
  }, [ticker])

  const hasNews = Boolean(news && news.items && news.items.length > 0)
  const hasDiscovery = Boolean(discoveryNews && discoveryNews.items && discoveryNews.items.length > 0)
  const hasYouTubeMentions = Boolean(news && news.youtube_mentions && news.youtube_mentions.length > 0)

  // While primary news is still loading, or while discovery is loading without any primary news
  if ((loadingPrimary && !hasDiscovery) || (loadingDiscovery && !hasNews && !hasDiscovery && !error && is404)) {
    return <NewsCardListSkeleton count={3} label="กำลังโหลดข่าวสาร..." />
  }

  if (!hasNews && !hasDiscovery && !hasYouTubeMentions) {
    if (loadingDiscovery) {
      return <NewsCardListSkeleton count={3} label="กำลังโหลดข่าวสาร..." />
    }

    if (is404 || (!news && !error)) {
      return (
        <div className="flex flex-col justify-center items-center h-64 bg-surface rounded-xl border border-dashed border-edge p-8 text-center">
          <svg className="w-12 h-12 text-zinc-400 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1m2 13a2 2 0 01-2-2V7m2 13a2 2 0 002-2V9a2 2 0 00-2-2h-2m-4-3H9M7 16h6M7 8h6v4H7V8z" />
          </svg>
          <h3 className="text-lg font-medium text-zinc-900 mb-1">ยังไม่มีข้อมูลข่าวสำหรับหุ้นตัวนี้ในระบบ</h3>
          <p className="text-zinc-500 max-w-md text-sm">
            สามารถสั่งงานผ่าน Trinity Advisor (Manager) หรือสั่ง Research Swarm เพื่อเริ่มต้นดึงข่าวและวิเคราะห์หุ้นตัวนี้ได้
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

    return (
      <div className="bg-surface rounded-xl border border-edge p-6 text-center text-zinc-500 text-sm">
        ไม่พบรายการข่าวสารในระบบ
      </div>
    )
  }

  const renderThesisBadge = (signal?: string) => {
    switch (signal) {
      case 'thesis_breaker':
        return (
          <span className="shrink-0 px-2 py-0.5 rounded text-[11px] font-bold bg-rose-50 text-rose-700 border border-rose-200">
            🚨 THESIS BREAKER
          </span>
        )
      case 'buy_on_dip':
        return (
          <span className="shrink-0 px-2 py-0.5 rounded text-[11px] font-bold bg-emerald-50 text-emerald-700 border border-emerald-200">
            💎 BUY ON DIP
          </span>
        )
      case 'thesis_reinforcer':
        return (
          <span className="shrink-0 px-2 py-0.5 rounded text-[11px] font-bold bg-sky-50 text-sky-700 border border-sky-200">
            ✅ THESIS REINFORCER
          </span>
        )
      case 'noise':
        return (
          <span className="shrink-0 px-2 py-0.5 rounded text-[11px] font-medium bg-zinc-100 text-zinc-500 border border-zinc-200">
            💤 NOISE
          </span>
        )
      default:
        return null
    }
  }

  return (
    <div className="space-y-6">
      {/* Primary / Pre-Indexed Digest News Items */}
      {hasNews && (
        <div className="space-y-4">
          <div className="flex items-center justify-between border-b border-edge pb-3">
            <div className="text-xs font-semibold uppercase tracking-wider text-sky-600">
              Analyzed Intelligence & News ({news!.items.length})
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
                <div className="space-y-2.5">
                  <div className="flex flex-wrap items-start justify-between gap-2">
                    <h4 className="font-semibold text-zinc-900 text-base leading-snug flex-1">
                      {item.title}
                    </h4>
                    <div className="flex items-center gap-1.5 shrink-0">
                      {renderThesisBadge(item.thesis_signal)}
                      {item.is_stale && (
                        <span className="px-2 py-0.5 rounded text-[11px] font-medium bg-amber-50 text-amber-700 border border-amber-200">
                          ⚠️ STALE
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Structured Analysis: What Happened & Why it Matters */}
                  {item.what_happened && (
                    <div className="text-xs text-zinc-700 leading-relaxed">
                      <span className="font-semibold text-zinc-900">เหตุการณ์: </span>
                      {item.what_happened}
                    </div>
                  )}

                  {item.why_it_matters && (
                    <div className="rounded-lg bg-sky-50/70 border border-sky-100 p-2.5 text-xs text-sky-950 leading-relaxed">
                      <span className="font-bold text-sky-800">ผลกระทบต่อพอร์ต & วิทยานิพนธ์ (Why It Matters): </span>
                      {item.why_it_matters}
                    </div>
                  )}

                  {/* Fallback to summary if no what_happened */}
                  {!item.what_happened && item.summary && (
                    <p className="text-xs text-zinc-600 line-clamp-3 leading-relaxed">
                      {item.summary}
                    </p>
                  )}

                  <div className="flex flex-wrap items-center gap-3 text-xs text-zinc-500 pt-1">
                    <span className="font-medium text-zinc-700">ที่มา: {item.source}</span>
                    {item.sources_count && item.sources_count > 1 && (
                      <span className="bg-surface-strong px-2 py-0.5 rounded text-zinc-600">
                        {item.sources_count} sources
                      </span>
                    )}
                    <span className="text-zinc-400">•</span>
                    <span>{item.freshness_reason || `${item.age_hours} ชั่วโมงที่แล้ว`}</span>
                    {item.macro_impact_score && (
                      <>
                        <span className="text-zinc-400">•</span>
                        <span className="font-medium text-sky-700">Macro Score: {item.macro_impact_score}/10</span>
                      </>
                    )}
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
                    </a>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* YouTube Channel Mentions & Takeaways */}
      {hasYouTubeMentions && (
        <div className="space-y-4 rounded-xl border border-red-100 bg-red-50/20 p-5">
          <div className="flex items-center justify-between border-b border-red-200/60 pb-3">
            <div>
              <h4 className="text-xs font-bold uppercase tracking-wider text-red-900 flex items-center gap-2">
                <span className="text-red-600 text-sm">▶</span>
                <span>YouTube Insights & Video Mentions ({news!.youtube_mentions!.length})</span>
              </h4>
              <p className="mt-0.5 text-[11px] text-zinc-500">
                สรุปประเด็นจากวิดีโอรายการและช่องการเงินชั้นนำที่พูดถึงหุ้น {ticker}
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-3">
            {news!.youtube_mentions!.map((yt, idx) => (
              <div
                key={idx}
                className="flex flex-col justify-between rounded-lg border border-edge bg-panel p-4 shadow-2xs hover:border-red-200 transition-colors"
              >
                <div className="space-y-2">
                  <div className="flex flex-wrap items-baseline justify-between gap-2">
                    <h5 className="font-semibold text-zinc-900 text-sm leading-snug">
                      {yt.thai_title || yt.title}
                    </h5>
                    <div className="flex items-center gap-1.5 shrink-0">
                      <span className="rounded bg-red-50 text-red-700 px-2 py-0.5 text-[11px] font-semibold border border-red-100">
                        {yt.channel}
                      </span>
                      {yt.alignment_with_news === 'confirmed' && (
                        <span className="rounded bg-emerald-50 text-emerald-700 px-2 py-0.5 text-[10px] font-bold border border-emerald-200">
                          สอดคล้องกับข่าว
                        </span>
                      )}
                      {yt.alignment_with_news === 'contrarian' && (
                        <span className="rounded bg-amber-50 text-amber-700 px-2 py-0.5 text-[10px] font-bold border border-amber-200">
                          มุมมองสวนทาง (Contrarian)
                        </span>
                      )}
                    </div>
                  </div>

                  {yt.key_takeaways && yt.key_takeaways.length > 0 && (
                    <ul className="list-disc pl-4 space-y-1 text-xs text-zinc-700 leading-relaxed">
                      {yt.key_takeaways.map((point, pIdx) => (
                        <li key={pIdx}>{point}</li>
                      ))}
                    </ul>
                  )}

                  {yt.alignment_note && (
                    <p className="text-xs text-zinc-500 italic">
                      หมายเหตุ: {yt.alignment_note}
                    </p>
                  )}
                </div>

                {yt.link && (
                  <div className="mt-3 pt-2.5 border-t border-edge flex justify-end">
                    <a
                      href={yt.link}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 text-xs font-medium text-red-600 hover:text-red-700 hover:underline"
                    >
                      รับชมบน YouTube ↗
                    </a>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* RSS Discovery Loading Shimmer while Vault news is already displayed */}
      {hasNews && loadingDiscovery && (
        <div className="space-y-4 rounded-xl border border-sky-100 bg-sky-50/20 p-5 animate-fade-in">
          <div className="flex flex-wrap items-baseline justify-between gap-2 border-b border-sky-200/60 pb-3">
            <div>
              <h4 className="text-xs font-bold uppercase tracking-wider text-sky-800 flex items-center gap-2">
                <span>RSS News Discovery Candidates</span>
                <span className="rounded-full bg-sky-100 text-sky-800 text-[10px] font-bold px-2 py-0.5 border border-sky-200">
                  SECONDARY RSS
                </span>
              </h4>
              <p className="mt-0.5 text-[11px] text-sky-700/80 font-medium animate-pulse">
                กำลังดึงข่าวเพิ่มเติมจาก RSS Feeds...
              </p>
            </div>
          </div>
          <div className="grid grid-cols-1 gap-3">
            <div className="h-16 rounded-lg border border-edge bg-panel/60 p-3.5 animate-shimmer" />
            <div className="h-16 rounded-lg border border-edge bg-panel/60 p-3.5 animate-shimmer" />
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
