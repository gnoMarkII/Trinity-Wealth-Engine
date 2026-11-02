/**
 * Utility for robust extraction of Job IDs from text and markdown messages.
 */

export function extractPrimaryJobId(content: string): string | null {
  if (!content) return null

  // 1. Explicit "Job ID" label followed by optional markdown formatting (*, `, space, :) and UUID or known prefix
  const explicitMatch = content.match(
    /Job ID[^\w\n]*[`*]*([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|(?:job_mgr|sim_job|sim)_[a-zA-Z0-9_\-]+)[`*]*/i
  )
  if (explicitMatch && explicitMatch[1]) {
    return explicitMatch[1]
  }

  // 2. Any standard UUID: 8-4-4-4-12 hex characters
  const uuidMatch = content.match(
    /\b([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b/i
  )
  if (uuidMatch && uuidMatch[1]) {
    return uuidMatch[1]
  }

  // 3. Known agent job prefixes: job_mgr_... or sim_job_... or sim_...
  const prefixMatch = content.match(/\b((?:job_mgr|sim_job|sim)_[a-zA-Z0-9_\-]+)\b/i)
  if (prefixMatch && prefixMatch[1]) {
    return prefixMatch[1]
  }

  return null
}

export function extractAllJobIds(content: string): string[] {
  if (!content) return []
  const ids = new Set<string>()

  // 1. Explicit matches first
  const explicitMatches = content.matchAll(
    /Job ID[^\w\n]*[`*]*([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|(?:job_mgr|sim_job|sim)_[a-zA-Z0-9_\-]+)[`*]*/gi
  )
  for (const m of explicitMatches) {
    if (m[1]) ids.add(m[1])
  }

  // 2. Standard UUIDs
  const uuidMatches = content.matchAll(
    /\b([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b/gi
  )
  for (const m of uuidMatches) {
    if (m[1]) ids.add(m[1])
  }

  // 3. Prefixes
  const prefixMatches = content.matchAll(/\b((?:job_mgr|sim_job|sim)_[a-zA-Z0-9_\-]+)\b/gi)
  for (const m of prefixMatches) {
    if (m[1]) ids.add(m[1])
  }

  return Array.from(ids)
}

