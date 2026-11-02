/**
 * Agent pipeline stage definitions and stage detection helpers.
 */

export interface PipelineStage {
  id: string
  label: string
  sublabel: string
  icon: string
  nodeKeys: string[]
}

export const MACRO_PIPELINE: PipelineStage[] = [
  {
    id: 'macro_quant',
    label: 'Macro Quant',
    sublabel: 'คำนวณ Observable Matrix & Yields',
    icon: '📊',
    nodeKeys: ['macro_quant', 'post_quant', 'supervisor', 'manager'],
  },
  {
    id: 'macro_economist',
    label: 'Macro Economist',
    sublabel: 'วิเคราะห์นโยบายการเงิน & Policy Stance',
    icon: '🌐',
    nodeKeys: ['macro_economist', 'post_economist'],
  },
  {
    id: 'strategic_allocator',
    label: 'Strategic Allocator',
    sublabel: 'ประเมิน Regime & จัดสรรสินทรัพย์ 3 ถัง',
    icon: '⚖️',
    nodeKeys: ['strategic_allocator', 'post_macro_intel'],
  },
  {
    id: 'archivist',
    label: 'Archivist',
    sublabel: 'บันทึกบทวิเคราะห์เข้า Knowledge Vault',
    icon: '📚',
    nodeKeys: ['archivist', 'prepare_archivist', 'post_archivist'],
  },
]

export const EQUITY_PIPELINE: PipelineStage[] = [
  {
    id: 'equity_quant',
    label: 'Equity Quant',
    sublabel: 'วิเคราะห์งบการเงิน, Multiples & DCF',
    icon: '📊',
    nodeKeys: ['equity_quant', 'post_equity_quant', 'supervisor', 'manager'],
  },
  {
    id: 'equity_narrative',
    label: 'Equity Narrative',
    sublabel: 'สรุป Earnings Call & Moat แข่งขัน',
    icon: '📰',
    nodeKeys: ['equity_narrative', 'post_equity_narrative'],
  },
  {
    id: 'equity_synthesizer',
    label: 'Equity Synthesizer',
    sublabel: 'ประเมิน Fair Value & Verdict สรุป',
    icon: '📈',
    nodeKeys: ['equity_synthesizer', 'post_equity_intel'],
  },
  {
    id: 'archivist',
    label: 'Archivist',
    sublabel: 'จัดเก็บรายงานเข้า Obsidian Vault',
    icon: '📚',
    nodeKeys: ['archivist', 'prepare_archivist', 'post_archivist'],
  },
]

export function getPipelineFor(jobType?: string, ticker?: string): PipelineStage[] {
  const isMacro = jobType?.includes('macro') || (!ticker && !jobType?.includes('equity'))
  return isMacro ? MACRO_PIPELINE : EQUITY_PIPELINE
}

export function determineStageIndex(
  pipeline: PipelineStage[],
  currentNode?: string | null
): number {
  if (!currentNode) return 0
  const normalized = currentNode.toLowerCase()
  const foundIdx = pipeline.findIndex((stage) =>
    stage.nodeKeys.some((k) => normalized.includes(k.toLowerCase()))
  )
  return foundIdx >= 0 ? foundIdx : 0
}
