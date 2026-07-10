export type StanceCategory = 'overweight' | 'underweight' | 'neutral'

export function stanceCategory(stance: string): StanceCategory {
  const s = stance.toLowerCase()
  if (s.includes('overweight')) return 'overweight'
  if (s.includes('underweight')) return 'underweight'
  return 'neutral'
}
