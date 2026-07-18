import { useState, useEffect } from 'react'
import SegmentedControl from '../ui/SegmentedControl'

interface DateFilters {
  mode: 'lookback' | 'range'
  lookbackDays: number
  fromDate: string
  toDate: string
}

export function parsePitchPromptDate(prompt: string): DateFilters {
  if (!prompt) {
    return { mode: 'lookback', lookbackDays: 7, fromDate: '', toDate: '' }
  }
  const fromMatch = prompt.match(/from_date\s*=\s*([0-9]{4}-[0-9]{2}-[0-9]{2})/i)
  const toMatch = prompt.match(/to_date\s*=\s*([0-9]{4}-[0-9]{2}-[0-9]{2})/i)
  if (fromMatch || toMatch) {
    return {
      mode: 'range',
      lookbackDays: 7,
      fromDate: fromMatch ? (fromMatch[1] ?? '') : '',
      toDate: toMatch ? (toMatch[1] ?? '') : '',
    }
  }
  const lookbackMatch = prompt.match(/lookback_days\s*=\s*([0-9]+)/i)
  if (lookbackMatch) {
    const days = parseInt(lookbackMatch[1] ?? '', 10)
    return {
      mode: 'lookback',
      lookbackDays: !isNaN(days) && days > 0 ? days : 7,
      fromDate: '',
      toDate: '',
    }
  }
  return { mode: 'lookback', lookbackDays: 7, fromDate: '', toDate: '' }
}

export function updatePitchPromptDate(
  prompt: string,
  mode: 'lookback' | 'range',
  lookbackDays: number,
  fromDate: string,
  toDate: string
): string {
  const cleaned = prompt
    .replace(/\s*\[[^\]]*(?:from_date|to_date|lookback_days)[^\]]*\]/gi, '')
    .trim()

  let tag = ''
  if (mode === 'lookback') {
    const days = lookbackDays && lookbackDays > 0 ? lookbackDays : 7
    tag = `[lookback_days=${days}]`
  } else {
    if (fromDate || toDate) {
      tag = `[from_date=${fromDate || ''}, to_date=${toDate || ''}]`
    }
  }

  if (!cleaned) return tag
  if (!tag) return cleaned
  return `${cleaned} ${tag}`
}

interface Props {
  prompt: string
  onChange: (updatedPrompt: string) => void
  disabled?: boolean
  className?: string
}

export default function YoutubePitchDateControls({
  prompt,
  onChange,
  disabled = false,
  className = '',
}: Props) {
  const [filters, setFilters] = useState<DateFilters>(() => parsePitchPromptDate(prompt))

  useEffect(() => {
    setFilters(parsePitchPromptDate(prompt))
  }, [prompt])

  function handleModeChange(newMode: string) {
    const mode = newMode as 'lookback' | 'range'
    const nextFilters = { ...filters, mode }
    setFilters(nextFilters)
    onChange(updatePitchPromptDate(prompt, mode, nextFilters.lookbackDays, nextFilters.fromDate, nextFilters.toDate))
  }

  function handleLookbackChange(days: number) {
    const validDays = isNaN(days) || days <= 0 ? 7 : days
    const nextFilters = { ...filters, lookbackDays: validDays }
    setFilters(nextFilters)
    onChange(updatePitchPromptDate(prompt, filters.mode, validDays, filters.fromDate, filters.toDate))
  }

  function handleRangeChange(from: string, to: string) {
    const nextFilters = { ...filters, fromDate: from, toDate: to }
    setFilters(nextFilters)
    onChange(updatePitchPromptDate(prompt, filters.mode, filters.lookbackDays, from, to))
  }

  const PRESET_DAYS = [3, 7, 14, 30]

  return (
    <div className={`rounded-xl border border-purple-200/80 bg-gradient-to-br from-purple-50/70 to-purple-50/30 p-3.5 shadow-sm ${className}`}>
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-purple-100 pb-2.5">
        <div className="flex items-center gap-1.5">
          <span className="text-base">🕒</span>
          <span id="youtube-pitch-time-label" className="text-xs font-semibold text-purple-900">
            ช่วงเวลา / วันที่ค้นหาไอเดีย YouTube
          </span>
        </div>
        <SegmentedControl
          options={[
            { key: 'lookback', label: 'ย้อนหลัง (วัน)' },
            { key: 'range', label: 'ระบุช่วงวันที่' },
          ]}
          value={filters.mode}
          onChange={handleModeChange}
          ariaLabelledby="youtube-pitch-time-label"
        />
      </div>

      <div className="mt-3">
        {filters.mode === 'lookback' ? (
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs font-medium text-purple-800">ค้นหาย้อนหลัง:</span>
            <div className="flex flex-wrap items-center gap-1.5">
              {PRESET_DAYS.map((days) => {
                const isActive = filters.lookbackDays === days
                return (
                  <button
                    key={days}
                    type="button"
                    disabled={disabled}
                    onClick={() => handleLookbackChange(days)}
                    className={`rounded-lg px-2.5 py-1 text-xs font-medium transition-all ${
                      isActive
                        ? 'bg-purple-600 text-white shadow-sm shadow-purple-500/20'
                        : 'border border-purple-200/80 bg-white text-purple-700 hover:bg-purple-100/60 hover:text-purple-900'
                    }`}
                  >
                    {days} วัน
                  </button>
                )
              })}
            </div>
            <div className="flex items-center gap-1 ml-auto sm:ml-2">
              <span className="text-xs text-purple-600">กำหนดเอง:</span>
              <input
                type="number"
                min={1}
                max={365}
                disabled={disabled}
                value={filters.lookbackDays || ''}
                onChange={(e) => handleLookbackChange(parseInt(e.target.value, 10))}
                className="w-16 rounded-lg border border-purple-200 bg-white px-2 py-1 text-center text-xs font-semibold text-purple-900 outline-none transition-colors focus:border-purple-500 focus:ring-1 focus:ring-purple-500/30 disabled:opacity-50"
                placeholder="วัน"
              />
              <span className="text-xs font-medium text-purple-700">วัน</span>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
            <div>
              <label htmlFor="pitch-from-date" className="mb-1 block text-[11px] font-semibold text-purple-800">
                ตั้งแต่วันที่ (From Date)
              </label>
              <input
                id="pitch-from-date"
                type="date"
                disabled={disabled}
                value={filters.fromDate}
                onChange={(e) => handleRangeChange(e.target.value, filters.toDate)}
                className="w-full rounded-lg border border-purple-200 bg-white px-2.5 py-1.5 text-xs font-medium text-purple-900 outline-none transition-colors focus:border-purple-500 focus:ring-1 focus:ring-purple-500/30 disabled:opacity-50"
              />
            </div>
            <div>
              <label htmlFor="pitch-to-date" className="mb-1 block text-[11px] font-semibold text-purple-800">
                ถึงวันที่ (To Date)
              </label>
              <input
                id="pitch-to-date"
                type="date"
                disabled={disabled}
                value={filters.toDate}
                onChange={(e) => handleRangeChange(filters.fromDate, e.target.value)}
                className="w-full rounded-lg border border-purple-200 bg-white px-2.5 py-1.5 text-xs font-medium text-purple-900 outline-none transition-colors focus:border-purple-500 focus:ring-1 focus:ring-purple-500/30 disabled:opacity-50"
              />
            </div>
          </div>
        )}
      </div>

      <div className="mt-2.5 flex items-center justify-between border-t border-purple-100/60 pt-2 text-[11px] text-purple-600/90">
        <span>💡 ระบบจะแนบการตั้งค่าเวลานี้เข้าไปใน Prompt ให้อัตโนมัติ</span>
        <span className="font-mono font-medium text-purple-800 bg-purple-100/80 px-1.5 py-0.5 rounded">
          {filters.mode === 'lookback'
            ? `[lookback_days=${filters.lookbackDays}]`
            : `[from_date=${filters.fromDate || '...'}, to_date=${filters.toDate || '...'}]`}
        </span>
      </div>
    </div>
  )
}
