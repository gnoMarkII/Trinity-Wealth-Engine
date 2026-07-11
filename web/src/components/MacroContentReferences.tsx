import { useState } from 'react'
import type { MacroReferenceDTO } from '../api/types'

function youtubeVideoId(url: string): string | null {
  try {
    const parsed = new URL(url)
    if (!/(^|\.)youtube\.com$|(^|\.)youtu\.be$/i.test(parsed.hostname)) return null
    const candidate = parsed.hostname.endsWith('youtu.be') ? parsed.pathname.slice(1) : parsed.searchParams.get('v')
    return candidate && /^[\w-]{11}$/.test(candidate) ? candidate : null
  } catch {
    return null
  }
}

function referenceTime(reference: MacroReferenceDTO): string {
  if (reference.published_at) return reference.published_at
  if (reference.age_hours !== null) return `${reference.age_hours} ชั่วโมงที่ผ่านมา`
  return ''
}

interface Props {
  references: MacroReferenceDTO[]
}

export default function MacroContentReferences({ references }: Props) {
  const [activeYoutube, setActiveYoutube] = useState<MacroReferenceDTO | null>(null)
  const visibleReferences = references.slice(0, 6)
  const activeVideoId = activeYoutube ? youtubeVideoId(activeYoutube.url) : null

  if (visibleReferences.length === 0) return null

  return (
    <section className="rounded-2xl border border-zinc-200/80 bg-white p-5 shadow-sm shadow-black/5">
      <div className="mb-4">
        <p className="text-xs font-bold uppercase tracking-wider text-zinc-500">Related Content</p>
        <h2 className="mt-1 text-lg font-semibold text-zinc-900">ข่าวและวิดีโอที่ใช้ประกอบรายงาน</h2>
      </div>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
        {visibleReferences.map((reference) => {
          const isYoutube = reference.kind === 'youtube'
          return (
            <article key={reference.reference_id} className="overflow-hidden rounded-xl border border-zinc-200 bg-zinc-50/50">
              {isYoutube && reference.thumbnail_url && (
                <img src={reference.thumbnail_url} alt="" className="h-28 w-full object-cover" loading="lazy" />
              )}
              <div className="space-y-2 p-3.5">
                <div className="flex items-center justify-between gap-2 text-[11px] font-semibold uppercase tracking-wide text-zinc-500">
                  <span>{isYoutube ? 'YouTube' : 'News'}</span>
                  {reference.is_stale && <span className="rounded bg-amber-100 px-1.5 py-0.5 text-amber-800">Stale</span>}
                </div>
                <h3 className="line-clamp-2 text-sm font-semibold text-zinc-900">{reference.title}</h3>
                <p className="text-xs text-zinc-500">{[reference.publisher, referenceTime(reference)].filter(Boolean).join(' · ')}</p>
                {reference.summary && <p className="line-clamp-3 text-xs leading-relaxed text-zinc-600">{reference.summary}</p>}
                {isYoutube && youtubeVideoId(reference.url) ? (
                  <button onClick={() => setActiveYoutube(reference)} className="text-xs font-semibold text-zinc-900 underline underline-offset-2 hover:text-amber-700">ดูตัวอย่างวิดีโอ</button>
                ) : (
                  <a href={reference.url} target="_blank" rel="noopener noreferrer" className="text-xs font-semibold text-zinc-900 underline underline-offset-2 hover:text-amber-700">เปิดแหล่งข้อมูล</a>
                )}
              </div>
            </article>
          )
        })}
      </div>

      {activeYoutube && activeVideoId && (
        <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/60 p-4" role="dialog" aria-modal="true" aria-label="YouTube preview">
          <div className="w-full max-w-3xl overflow-hidden rounded-xl bg-white shadow-2xl">
            <div className="flex items-center justify-between gap-4 border-b border-zinc-200 p-4">
              <h3 className="text-sm font-semibold text-zinc-900">{activeYoutube.title}</h3>
              <button onClick={() => setActiveYoutube(null)} className="rounded-md px-2 py-1 text-sm text-zinc-600 hover:bg-zinc-100">ปิด</button>
            </div>
            <div className="aspect-video bg-black">
              <iframe
                className="h-full w-full"
                src={`https://www.youtube-nocookie.com/embed/${activeVideoId}?rel=0`}
                title={activeYoutube.title}
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
              />
            </div>
            <div className="p-4 text-right">
              <a href={activeYoutube.url} target="_blank" rel="noopener noreferrer" className="text-xs font-semibold text-zinc-700 underline underline-offset-2">เปิดใน YouTube</a>
            </div>
          </div>
        </div>
      )}
    </section>
  )
}
