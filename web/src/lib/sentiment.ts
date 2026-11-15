export type SentimentCategory = 'bullish' | 'bearish' | 'neutral'

const SENTIMENT_CLASS: Record<SentimentCategory, string> = {
  bullish: 'border-emerald-200 bg-emerald-50 text-emerald-700',
  bearish: 'border-red-200 bg-red-50 text-red-700',
  neutral: 'border-edge bg-surface-strong text-slate-700',
}

export function sentimentClass(sentiment: string): string {
  const s = sentiment.toLowerCase()
  if (s === 'bullish') return SENTIMENT_CLASS.bullish
  if (s === 'bearish') return SENTIMENT_CLASS.bearish
  return SENTIMENT_CLASS.neutral
}
