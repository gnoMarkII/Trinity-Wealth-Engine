import React from 'react'

export type ProvenanceOrigin = 'provider' | 'deterministic' | 'ai' | 'external'

interface SourceProvenanceBadgeProps {
  origin: ProvenanceOrigin
  sourceName?: string
  observedAt?: string
  publishedAt?: string
  fetchedAt?: string
  evaluatedAt?: string
  className?: string
  compact?: boolean
}

const ORIGIN_CONFIG: Record<
  ProvenanceOrigin,
  { label: string; badgeClass: string; icon: string; desc: string }
> = {
  provider: {
    label: 'ข้อมูลจากผู้ให้บริการ',
    badgeClass: 'bg-sky-50 text-sky-800 border-sky-200/80',
    icon: '🏛️',
    desc: 'ข้อมูลดิบดึงตรงจากผู้ให้บริการ (Provider Feed เช่น FRED, SET, GTA หรือตลาดการเงิน) ไม่ผ่านการแปลงโดย AI',
  },

  deterministic: {
    label: 'ระบบคำนวณ',
    badgeClass: 'bg-indigo-50 text-indigo-800 border-indigo-200/80',
    icon: '📐',
    desc: 'ผลการคำนวณเชิงตัวเลขแน่นอนทางสถิติ (Deterministic Math) โดยระบบ Python',
  },
  ai: {
    label: 'AI วิเคราะห์',
    badgeClass: 'bg-amber-50 text-amber-800 border-amber-200/80',
    icon: '🧠',
    desc: 'การประเมินเชิงสมมติฐานและฉากทัศน์โดย Agent AI (อาจมีความไม่แน่นอน)',
  },
  external: {
    label: 'กราฟภายนอก',
    badgeClass: 'bg-zinc-100 text-zinc-800 border-zinc-200',
    icon: '🌐',
    desc: 'วิดเจ็ตแสดงผลจากภายนอก เวลาการแสดงผลอาจเหลื่อมจาก Snapshot ทางการ',
  },
}

export const SourceProvenanceBadge: React.FC<SourceProvenanceBadgeProps> = ({
  origin,
  sourceName,
  observedAt,
  publishedAt,
  fetchedAt,
  evaluatedAt,
  className = '',
  compact = false,
}) => {
  const config = ORIGIN_CONFIG[origin]

  return (
    <div className={`inline-flex flex-wrap items-center gap-2 text-xs ${className}`}>
      <span
        title={config.desc}
        className={`inline-flex items-center gap-1 rounded-md border px-2 py-0.5 font-semibold text-[11px] shadow-2xs ${config.badgeClass}`}
      >
        <span>{config.icon}</span>
        <span>{config.label}</span>
        {sourceName && <span className="font-mono font-normal opacity-75">• {sourceName}</span>}
      </span>

      {!compact && (
        <div className="flex flex-wrap items-center gap-2 font-mono text-[11px] text-zinc-500">
          {observedAt && (
            <span title="วันของข้อมูลที่สังเกตการณ์จริงในตลาด">
              ข้อมูล ณ: <strong className="text-zinc-700">{observedAt}</strong>
            </span>
          )}
          {publishedAt && (
            <span title="วันที่ผู้ให้บริการเผยแพร่รายงานจริง (เช่น ข้อมูลมี Lag)">
              เผยแพร่: <span className="text-zinc-600">{publishedAt}</span>
            </span>
          )}
          {fetchedAt && (
            <span title="เวลาที่ระบบดึงข้อมูลล่าสุดจาก API">
              ดึงล่าสุด: <span className="text-zinc-600">{fetchedAt}</span>
            </span>
          )}
          {evaluatedAt && (
            <span title="เวลาที่ AI ทำการประเมินภาพรวมล่าสุด">
              วิเคราะห์เมื่อ: <strong className="text-amber-800">{evaluatedAt}</strong>
            </span>
          )}
        </div>
      )}
    </div>
  )
}
