// map ชื่อ node จริงในกราฟ (ดู agents/manager_agent.py, agents/news_youtube_flow.py)
// ไปเป็นชื่อที่อ่านง่ายสำหรับ UI — ไม่ครบก็ไม่เป็นไร เพราะ fallback คืนชื่อดิบเสมอ ไม่มีทาง "หาย"
const NODE_DISPLAY_NAMES: Record<string, string> = {
  supervisor: 'Manager',
  macro_quant: 'Macro Quant',
  macro_economist: 'Macro Economist',
  strategic_allocator: 'Strategic Allocator',
  researcher: 'Researcher',
  archivist: 'Archivist',
  bookkeeper: 'Bookkeeper',
  prepare_archivist: 'Archivist',
  post_macro_intel: 'Macro Intel',
  fetch_news: 'ดึงข่าว',
  fetch_youtube: 'ดึงคลิป YouTube',
  gate: 'News/YouTube Gate',
  ingest: 'News/YouTube Ingest',
}

export function nodeDisplayName(node: string | null | undefined): string {
  if (!node) return 'Agent'
  return NODE_DISPLAY_NAMES[node] ?? node
}
