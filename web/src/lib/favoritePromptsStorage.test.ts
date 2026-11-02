import { describe, it, expect, beforeEach, vi } from 'vitest'
import {
  loadFavoritePrompts,
  togglePinFavoritePrompt,
  addCustomFavoritePrompt,
  updateFavoritePrompt,
  deleteFavoritePrompt,
  resetDefaultFavoritePrompts,
  resolvePromptWithTicker,
  DEFAULT_FAVORITE_PROMPTS,
  FAVORITES_UPDATED_EVENT,
} from './favoritePromptsStorage'

describe('favoritePromptsStorage', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.restoreAllMocks()
  })

  it('loads default 28 prompts covering 4 categories when storage is empty', () => {
    const prompts = loadFavoritePrompts()
    expect(prompts.length).toBe(28)

    const macroPrompts = prompts.filter((p) => p.category === 'macro')
    const equityPrompts = prompts.filter((p) => p.category === 'equity')
    const portfolioPrompts = prompts.filter((p) => p.category === 'portfolio')
    const workflowPrompts = prompts.filter((p) => p.category === 'workflows')

    expect(macroPrompts.length).toBe(8)
    expect(equityPrompts.length).toBe(9)
    expect(portfolioPrompts.length).toBe(6)
    expect(workflowPrompts.length).toBe(5)

    // Check pinned items default
    const pinned = prompts.filter((p) => p.isPinned)
    expect(pinned.length).toBeGreaterThanOrEqual(4)
  })

  it('resolves [TICKER] in title, prompt and contextRefs correctly', () => {
    const equityItem = DEFAULT_FAVORITE_PROMPTS.find((p) => p.id === 'equity-5tier-full')!
    expect(equityItem).toBeDefined()

    const resolved = resolvePromptWithTicker(equityItem, 'aapl')
    expect(resolved.title).toContain('AAPL')
    expect(resolved.prompt).toContain('AAPL')
    expect(resolved.title).not.toContain('[TICKER]')
    expect(resolved.prompt).not.toContain('[TICKER]')
    expect(resolved.contextRefs).toContain('$AAPL')
  })

  it('defaults to NVDA when ticker input is empty or whitespace', () => {
    const equityItem = DEFAULT_FAVORITE_PROMPTS.find((p) => p.id === 'equity-valuation-dcf')!
    const resolved = resolvePromptWithTicker(equityItem, '   ')
    expect(resolved.title).toContain('NVDA')
    expect(resolved.prompt).toContain('NVDA')
    expect(resolved.contextRefs).toContain('$NVDA')
  })

  it('toggles pin status and dispatches CustomEvent', () => {
    const listener = vi.fn()
    window.addEventListener(FAVORITES_UPDATED_EVENT, listener)

    const targetId = 'macro-us-yield-fed'
    const initial = loadFavoritePrompts().find((p) => p.id === targetId)!
    expect(initial.isPinned).toBe(false)

    const afterToggle = togglePinFavoritePrompt(targetId)
    const toggledItem = afterToggle.find((p) => p.id === targetId)!
    expect(toggledItem.isPinned).toBe(true)
    expect(listener).toHaveBeenCalled()

    // Untoggle
    const afterUntoggle = togglePinFavoritePrompt(targetId)
    expect(afterUntoggle.find((p) => p.id === targetId)!.isPinned).toBe(false)

    window.removeEventListener(FAVORITES_UPDATED_EVENT, listener)
  })

  it('adds, updates and deletes a custom prompt', () => {
    const custom = addCustomFavoritePrompt({
      title: 'Custom Prompt Test',
      prompt: 'วิเคราะห์ [TICKER] ด้วยเกณฑ์พิเศษ',
      category: 'equity',
      contextRefs: ['@Portfolio'],
      dimensionBadge: 'Custom Dimension',
    })

    expect(custom.length).toBe(29)
    const created = custom.find((p) => p.title === 'Custom Prompt Test')!
    expect(created).toBeDefined()
    expect(created.isCustom).toBe(true)
    expect(created.isPinned).toBe(true)
    expect(created.requiresTicker).toBe(true)

    // Update
    const updatedList = updateFavoritePrompt(created.id, {
      title: 'Updated Prompt Test',
      prompt: 'วิเคราะห์แบบใหม่',
      category: 'portfolio',
      contextRefs: ['@Macro'],
      dimensionBadge: 'Updated Badge',
    })

    const updated = updatedList.find((p) => p.id === created.id)!
    expect(updated.title).toBe('Updated Prompt Test')
    expect(updated.category).toBe('portfolio')

    // Delete
    const deletedList = deleteFavoritePrompt(created.id)
    expect(deletedList.length).toBe(28)
    expect(deletedList.find((p) => p.id === created.id)).toBeUndefined()
  })

  it('merges stored customized pins with default prompts on reload', () => {
    // Pin an unpinned prompt and save
    togglePinFavoritePrompt('macro-thailand-flow')

    const reloaded = loadFavoritePrompts()
    const thPrompt = reloaded.find((p) => p.id === 'macro-thailand-flow')!
    expect(thPrompt.isPinned).toBe(true)

    // Reset works
    const reset = resetDefaultFavoritePrompts()
    const resetThPrompt = reset.find((p) => p.id === 'macro-thailand-flow')!
    expect(resetThPrompt.isPinned).toBe(false)
  })
})

