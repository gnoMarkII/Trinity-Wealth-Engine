export interface LogLine {
  node: string | null
  content: string
  role: 'instruction' | 'reply'
  label: string | null
}

export interface StepBlock {
  key: string
  node: string | null
  messages: LogLine[]
}

// รวมบรรทัด log ที่ node เดียวกันติดกันเป็น step เดียว — เปลี่ยน node เมื่อไหร่ขึ้น step ใหม่
// ต่อท้ายด้านล่างทันที (รองรับ supervisor วนกลับมาเรียก node เดิมซ้ำ เพราะกราฟจริงเป็นแบบ
// dynamic dispatch ไม่มีลำดับตายตัว — ดู agents/manager_agent.py supervisor_node)
export function groupIntoSteps(lines: LogLine[]): StepBlock[] {
  const blocks: StepBlock[] = []
  for (const line of lines) {
    const last = blocks[blocks.length - 1]
    if (last && last.node === line.node) {
      last.messages.push(line)
    } else {
      blocks.push({ key: `${blocks.length}-${line.node ?? 'system'}`, node: line.node, messages: [line] })
    }
  }
  return blocks
}
