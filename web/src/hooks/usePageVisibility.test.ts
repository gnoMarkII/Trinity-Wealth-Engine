import { act, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import { usePageVisibility } from './usePageVisibility'

function setDocumentHidden(hidden: boolean) {
  Object.defineProperty(document, 'hidden', { configurable: true, get: () => hidden })
  document.dispatchEvent(new Event('visibilitychange'))
}

afterEach(() => {
  Object.defineProperty(document, 'hidden', { configurable: true, get: () => false })
})

describe('usePageVisibility', () => {
  it('เริ่มต้นตามสถานะจริงของ document (jsdom = visible)', () => {
    const { result } = renderHook(() => usePageVisibility())
    expect(result.current).toBe(true)
  })

  it('สลับตาม visibilitychange event', () => {
    const { result } = renderHook(() => usePageVisibility())

    act(() => setDocumentHidden(true))
    expect(result.current).toBe(false)

    act(() => setDocumentHidden(false))
    expect(result.current).toBe(true)
  })
})
