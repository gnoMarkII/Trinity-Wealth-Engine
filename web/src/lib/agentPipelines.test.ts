import { describe, expect, it } from 'vitest'
import {
  determineStageIndex,
  EQUITY_PIPELINE,
  getPipelineFor,
  MACRO_PIPELINE,
} from './agentPipelines'

describe('agentPipelines', () => {
  it('returns MACRO_PIPELINE for macro jobs', () => {
    expect(getPipelineFor('macro_intel')).toBe(MACRO_PIPELINE)
    expect(getPipelineFor(undefined, '')).toBe(MACRO_PIPELINE)
  })

  it('returns EQUITY_PIPELINE for ticker or equity jobs', () => {
    expect(getPipelineFor('equity_intel', 'NVDA')).toBe(EQUITY_PIPELINE)
    expect(getPipelineFor(undefined, 'AAPL')).toBe(EQUITY_PIPELINE)
  })

  it('determines correct stage index based on current node', () => {
    expect(determineStageIndex(MACRO_PIPELINE, 'macro_quant')).toBe(0)
    expect(determineStageIndex(MACRO_PIPELINE, 'macro_economist')).toBe(1)
    expect(determineStageIndex(MACRO_PIPELINE, 'strategic_allocator')).toBe(2)
    expect(determineStageIndex(MACRO_PIPELINE, 'archivist')).toBe(3)
    expect(determineStageIndex(MACRO_PIPELINE, 'supervisor')).toBe(0)
    expect(determineStageIndex(MACRO_PIPELINE, 'manager')).toBe(0)
    expect(determineStageIndex(MACRO_PIPELINE, null)).toBe(0)
    expect(determineStageIndex(MACRO_PIPELINE, 'unknown_node')).toBe(0)
  })
})

