import { useEffect, useState } from 'react'
import { api } from '../../api/client'
import type { HermesSyncStatusDTO, SandboxStateSnapshotDTO } from '../../api/types'

interface ManagerMemoryDrawerProps {
  isOpen: boolean
  onClose: () => void
}

export default function ManagerMemoryDrawer({ isOpen, onClose }: ManagerMemoryDrawerProps) {
  const [snapshot, setSnapshot] = useState<SandboxStateSnapshotDTO | null>(null)
  const [syncStatus, setSyncStatus] = useState<HermesSyncStatusDTO | null>(null)
  const [isSyncing, setIsSyncing] = useState(false)
  const [syncMsg, setSyncMsg] = useState<string | null>(null)
  const [promotingSkill, setPromotingSkill] = useState<string | null>(null)
  const [promoteMsg, setPromoteMsg] = useState<string | null>(null)

  useEffect(() => {
    if (!isOpen) return

    api.getSandboxState().then(setSnapshot).catch(() => {})
    api.getHermesSyncStatus().then(setSyncStatus).catch(() => {})
  }, [isOpen])

  if (!isOpen) return null

  async function handleTriggerSync() {
    setIsSyncing(true)
    setSyncMsg(null)
    try {
      const res = await api.triggerHermesSync()
      setSyncMsg(res.success ? 'อัปเดตตามต้นน้ำสำเร็จ' : `ข้อผิดพลาด: ${res.output}`)
      const updated = await api.getHermesSyncStatus()
      setSyncStatus(updated)
    } catch (err: any) {
      setSyncMsg(err.message || 'เกิดข้อผิดพลาดในการดึงอัปเดต')
    } finally {
      setIsSyncing(false)
    }
  }

  async function handlePromoteSkill(skillName: string) {
    setPromotingSkill(skillName)
    setPromoteMsg(null)
    try {
      const res = await api.promoteSandboxSkill(skillName)
      setPromoteMsg(res.message || 'อนุมัติเปิดใช้งานสกิลสำเร็จ')
      const updated = await api.getSandboxState()
      setSnapshot(updated)
    } catch (err: any) {
      setPromoteMsg(err.message || 'ไม่สามารถอนุมัติสกิลได้')
    } finally {
      setPromotingSkill(null)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-sky-950/20 backdrop-blur-sm animate-fade-in">
      <div
        className="h-full w-full max-w-md border-l border-sky-200 bg-panel p-6 shadow-2xl backdrop-blur-2xl animate-drawer-in flex flex-col custom-scrollbar overflow-y-auto"
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-sky-100 pb-4">
          <div className="flex items-center gap-2">
            <span className="flex h-8 w-8 items-center justify-center rounded-xl bg-sky-100 text-sky-800 font-bold">
              🛡️
            </span>
            <div>
              <h3 className="text-sm font-bold text-sky-950">Hermes Sandbox & Memory</h3>
              <p className="text-[11px] text-zinc-500">พื้นที่ความจำและวิวัฒนาการเฉพาะของ Manager</p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-xl p-2 text-zinc-400 hover:bg-sky-50 hover:text-sky-900"
          >
            ✕
          </button>
        </div>

        <div className="mt-6 space-y-6 flex-1">
          {/* Hermes Agent Upstream Status */}
          <div className="rounded-2xl border border-sky-100 bg-surface p-4 shadow-sm">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-bold text-sky-900">🔄 NousResearch/hermes-agent</span>
              <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-[10px] font-semibold text-emerald-800">
                Isolated Clone
              </span>
            </div>

            <div className="space-y-1.5 text-xs text-zinc-600">
              <div className="flex justify-between">
                <span>Commit ล่าสุด:</span>
                <code className="font-mono font-medium text-sky-800">
                  {syncStatus?.current_commit || 'กำลังโหลด...'}
                </code>
              </div>
              <div className="flex justify-between">
                <span>โฟลเดอร์รัน:</span>
                <code className="font-mono text-zinc-500">services/hermes_agent/</code>
              </div>
            </div>

            <div className="mt-4 pt-3 border-t border-sky-100/60 flex items-center justify-between">
              <button
                type="button"
                disabled={isSyncing}
                onClick={handleTriggerSync}
                className="rounded-xl bg-white border border-sky-200 px-3 py-1.5 text-xs font-medium text-sky-700 shadow-sm hover:bg-sky-50 hover:text-sky-900 transition-colors disabled:opacity-50"
              >
                {isSyncing ? 'กำลังตรวจสอบ...' : 'ดึงอัปเดตต้นน้ำ (git pull)'}
              </button>
              {syncMsg && <span className="text-[11px] text-sky-800 font-medium">{syncMsg}</span>}
            </div>
          </div>

          {/* Dual Memory Files */}
          <div className="rounded-2xl border border-sky-100 bg-surface p-4 shadow-sm">
            <h4 className="text-xs font-bold text-sky-900 mb-2">🧠 ความจำระยะยาว (manager_sandbox/memory/)</h4>
            <div className="space-y-2">
              {snapshot?.memory_files && snapshot.memory_files.length > 0 ? (
                snapshot.memory_files.map((file) => (
                  <div
                    key={file}
                    className="flex items-center justify-between rounded-xl border border-sky-100 bg-white p-2.5 text-xs"
                  >
                    <span className="font-medium text-sky-950">📄 {file}</span>
                    <span className="text-[10px] text-zinc-400">Sanitized</span>
                  </div>
                ))
              ) : (
                <p className="text-xs text-zinc-400 italic">ยังไม่มีไฟล์ความจำที่บันทึก</p>
              )}
            </div>
          </div>

          {/* Quarantined Skills Pending Review */}
          {snapshot?.quarantined_skills && snapshot.quarantined_skills.length > 0 && (
            <div className="rounded-2xl border border-amber-200/80 bg-amber-50/50 p-4 shadow-sm">
              <div className="flex items-center justify-between mb-2">
                <h4 className="text-xs font-bold text-amber-900">🛡️ สกิลรอตรวจสอบความปลอดภัย (_quarantine)</h4>
                <span className="rounded-full bg-amber-100 px-2 py-0.5 text-[10px] font-semibold text-amber-800">
                  {snapshot.quarantined_skills.length} รายการ
                </span>
              </div>
              <div className="space-y-2">
                {snapshot.quarantined_skills.map((skill) => (
                  <div
                    key={skill.name}
                    className="flex items-center justify-between rounded-xl border border-amber-200 bg-white p-2.5 text-xs"
                  >
                    <div className="flex-1 min-w-0 pr-2">
                      <div className="font-semibold text-sky-950">📦 {skill.name}</div>
                      <div className="text-[10px] text-zinc-500 truncate">{skill.description}</div>
                    </div>
                    <button
                      type="button"
                      disabled={promotingSkill === skill.name}
                      onClick={() => handlePromoteSkill(skill.name)}
                      className="shrink-0 rounded-lg bg-emerald-50 border border-emerald-200 px-2.5 py-1 text-[11px] font-semibold text-emerald-700 shadow-sm hover:bg-emerald-100 transition-colors disabled:opacity-50"
                    >
                      {promotingSkill === skill.name ? 'กำลังอนุมัติ...' : 'อนุมัติ (Promote)'}
                    </button>
                  </div>
                ))}
              </div>
              {promoteMsg && <div className="mt-2 text-[11px] text-emerald-800 font-medium">{promoteMsg}</div>}
            </div>
          )}

          {/* Active Skills */}
          <div className="rounded-2xl border border-sky-100 bg-surface p-4 shadow-sm">
            <h4 className="text-xs font-bold text-sky-900 mb-2">📚 สกิลที่เปิดใช้งาน (manager_sandbox/skills/)</h4>
            <div className="space-y-2">
              {snapshot?.active_skills && snapshot.active_skills.length > 0 ? (
                snapshot.active_skills.map((skill) => (
                  <div
                    key={skill.name}
                    className="rounded-xl border border-sky-100 bg-white p-2.5 text-xs"
                  >
                    <div className="font-medium text-sky-950">✨ {skill.name}</div>
                    <div className="text-[11px] text-zinc-500 line-clamp-1">{skill.description}</div>
                  </div>
                ))
              ) : (
                <p className="text-xs text-zinc-400 italic">ไม่มีสกิลเพิ่มเติมในขณะนี้</p>
              )}
            </div>
          </div>
        </div>

        <div className="pt-4 border-t border-sky-100 text-center">
          <p className="text-[11px] text-zinc-400">
            ระบบทำงานภายใต้การคุ้มกันของ AST Whitelist Guard และ Atomic File Locks
          </p>
        </div>
      </div>
    </div>
  )
}
