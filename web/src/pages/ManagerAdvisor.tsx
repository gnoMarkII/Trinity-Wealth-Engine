import { useEffect, useState, useRef, useMemo } from 'react'
import { useSearchParams } from 'react-router-dom'
import { api, streamAdvisorChat } from '../api/client'
import type { ActiveAgentStatusDTO, AdvisorChatMessageDTO, AdvisorSessionDTO } from '../api/types'
import { usePageVisibility } from '../hooks/usePageVisibility'
import { extractAllJobIds, extractPrimaryJobId } from '../lib/jobDetection'
import ManagerChatStream from '../components/manager/ManagerChatStream'
import ManagerPromptComposer from '../components/manager/ManagerPromptComposer'
import ManagerMemoryDrawer from '../components/manager/ManagerMemoryDrawer'
import ManagerRightPanel, { type ManagerRightPanelTab } from '../components/manager/ManagerRightPanel'

export default function ManagerAdvisor() {
  const [sessions, setSessions] = useState<AdvisorSessionDTO[]>([])
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)
  const [messages, setMessages] = useState<AdvisorChatMessageDTO[]>([])
  const [isThinking, setIsThinking] = useState(false)
  const [isStreaming, setIsStreaming] = useState(false)
  const [streamingText, setStreamingText] = useState('')
  const [drawerOpen, setDrawerOpen] = useState(false)

  // URL Search Params (e.g. ?view=kanban)
  const [searchParams] = useSearchParams()
  const initialView = searchParams.get('view')

  // Right-Hand Tab Panel States (Morning Review, Kanban Board & Trinity Swarm, default: kanban)
  const [rightPanelOpen, setRightPanelOpen] = useState(true)
  const [activeRightTab, setActiveRightTab] = useState<ManagerRightPanelTab>(
    initialView === 'swarm' ? 'swarm' : initialView === 'review' || initialView === 'morning-review' ? 'review' : 'kanban'
  )
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null)
  const [activeAgentStatus, setActiveAgentStatus] = useState<ActiveAgentStatusDTO | null>(null)
  const [kanbanRefreshTrigger, setKanbanRefreshTrigger] = useState(0)

  // Sync tab with URL view parameter if it updates
  useEffect(() => {
    const view = searchParams.get('view')
    if (view === 'kanban') {
      setActiveRightTab('kanban')
      setRightPanelOpen(true)
    } else if (view === 'review' || view === 'morning-review') {
      setActiveRightTab('review')
      setRightPanelOpen(true)
    }
  }, [searchParams])

  const isPageVisible = usePageVisibility()

  const abortControllerRef = useRef<AbortController | null>(null)
  const skipNextFetchRef = useRef(false)

  // 1. Load sessions on mount
  useEffect(() => {
    loadSessions()
  }, [])

  async function loadSessions() {
    try {
      const data = await api.listAdvisorSessions(20)
      setSessions(data)
      if (data.length > 0 && !activeSessionId && data[0]) {
        setActiveSessionId(data[0].id)
      } else if (data.length === 0) {
        // Create initial session if none exists
        handleNewSession()
      }
    } catch (err) {
      console.error('Failed to load advisor sessions:', err)
    }
  }

  // 2. Load messages when active session changes (unless skipped by auto-session creation during send)
  useEffect(() => {
    if (!activeSessionId) return
    if (skipNextFetchRef.current) {
      skipNextFetchRef.current = false
      return
    }
    api
      .getAdvisorMessages(activeSessionId)
      .then(setMessages)
      .catch((err) => console.error('Failed to load session messages:', err))
  }, [activeSessionId])

  // 3. Poll Active Agent Status (4s interval, paused when page hidden)
  useEffect(() => {
    if (!isPageVisible) return
    let cancelled = false

    function pollActive() {
      api
        .getActiveAgentStatus()
        .then((st) => {
          if (!cancelled) setActiveAgentStatus(st)
        })
        .catch(() => {})
    }

    pollActive()
    const timer = window.setInterval(pollActive, 4000)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [isPageVisible])

  // 4. Extract all Job IDs mentioned in session messages + active job
  const detectedJobIds = useMemo(() => {
    const ids = new Set<string>()
    if (activeAgentStatus?.job_id) {
      ids.add(activeAgentStatus.job_id)
    }
    for (const msg of messages) {
      for (const id of extractAllJobIds(msg.content)) {
        ids.add(id)
      }
    }
    return Array.from(ids)
  }, [messages, activeAgentStatus?.job_id])

  // 5. Automatically select newest or active job and focus right panel when job starts
  useEffect(() => {
    if (activeAgentStatus?.running && activeAgentStatus.job_id) {
      setSelectedJobId(activeAgentStatus.job_id)
      setActiveRightTab('kanban')
      setRightPanelOpen(true)
    }
  }, [activeAgentStatus?.running, activeAgentStatus?.job_id])

  async function handleNewSession() {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      abortControllerRef.current = null
    }
    try {
      const session = await api.createAdvisorSession('บทสนทนาใหม่')
      setSessions((prev) => [session, ...prev])
      skipNextFetchRef.current = true
      setActiveSessionId(session.id)
      setMessages([])
      setStreamingText('')
      setIsThinking(false)
      setIsStreaming(false)
    } catch (err) {
      console.error('Failed to create new session:', err)
    }
  }

  function handleSelectSession(sessionId: string) {
    if (sessionId === activeSessionId) return
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      abortControllerRef.current = null
    }
    setStreamingText('')
    setIsThinking(false)
    setIsStreaming(false)
    setActiveSessionId(sessionId)
  }

  async function handleSendMessage(prompt: string, contextRefs: string[]) {
    if (!prompt.trim() || isStreaming || isThinking) return

    let currentSessionId = activeSessionId
    if (!currentSessionId) {
      try {
        const session = await api.createAdvisorSession(prompt.slice(0, 30))
        setSessions((prev) => [session, ...prev])
        currentSessionId = session.id
        skipNextFetchRef.current = true
        setActiveSessionId(session.id)
      } catch {
        return
      }
    }

    // Optimistically append user message
    const tempUserMsg: AdvisorChatMessageDTO = {
      id: `temp_user_${Date.now()}`,
      session_id: currentSessionId,
      role: 'user',
      content: prompt,
      citations: contextRefs,
      created_at: new Date().toISOString(),
    }
    setMessages((prev) => [...prev, tempUserMsg])

    // Enter thinking state (showing ManagerFluidOrb)
    setIsThinking(true)
    setIsStreaming(false)
    setStreamingText('')

    const abortController = new AbortController()
    abortControllerRef.current = abortController

    let accumulatedText = ''

    await streamAdvisorChat(
      currentSessionId,
      prompt,
      contextRefs,
      (delta) => {
        // First token received: transition from thinking to smooth streaming
        setIsThinking(false)
        setIsStreaming(true)
        accumulatedText += delta
        setStreamingText(accumulatedText)
      },
      (donePayload) => {
        // On Done
        setIsThinking(false)
        setIsStreaming(false)
        if (accumulatedText) {
          const assistantMsg: AdvisorChatMessageDTO = {
            id: donePayload?.message_id || `temp_assistant_${Date.now()}`,
            session_id: currentSessionId!,
            role: 'assistant',
            content: accumulatedText,
            citations: donePayload?.citations || contextRefs,
            created_at: new Date().toISOString(),
          }
          setMessages((prev) => [...prev, assistantMsg])

          // Automatically detect and focus on new Job ID if returned
          const primaryJobId = extractPrimaryJobId(accumulatedText)
          if (primaryJobId) {
            setSelectedJobId(primaryJobId)
            setActiveRightTab('kanban')
            setRightPanelOpen(true)
          }
        }
        setStreamingText('')
        setKanbanRefreshTrigger((prev) => prev + 1)
        // Refresh session list to update counts
        api.listAdvisorSessions(20).then(setSessions).catch(() => {})
      },
      (error) => {
        console.error('Stream error:', error)
        setIsThinking(false)
        setIsStreaming(false)
        const errorMsg: AdvisorChatMessageDTO = {
          id: `temp_err_${Date.now()}`,
          session_id: currentSessionId!,
          role: 'assistant',
          content: `ขออภัยครับ เกิดข้อผิดพลาดในการประมวลผล: ${error}`,
          citations: [],
          created_at: new Date().toISOString(),
        }
        setMessages((prev) => [...prev, errorMsg])
        setStreamingText('')
      },
      abortController.signal,
      (job) => {
        if (job?.job_id) {
          setSelectedJobId(job.job_id)
          setActiveRightTab('kanban')
          setRightPanelOpen(true)
          setKanbanRefreshTrigger((prev) => prev + 1)
        }
      }
    )
  }

  function handleStopStreaming() {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      abortControllerRef.current = null
    }
    setIsThinking(false)
    setIsStreaming(false)
  }

  return (
    <div className="flow-theme flex h-full w-full flex-col min-h-0 overflow-hidden bg-transparent">
      {/* Top Navigation Bar — Minimal & Calm */}
      <header className="flex h-14 shrink-0 items-center justify-between border-b border-sky-100/70 bg-white/80 px-4 backdrop-blur-xl sm:px-6">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <h1 className="text-base font-bold tracking-tight text-sky-950 flex items-center gap-2.5">
              Trinity AI Advisor
              <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-50/90 px-2.5 py-0.5 text-xs font-medium text-emerald-700">
                <span className="h-2 w-2 rounded-full bg-emerald-500" />
                พร้อมให้คำปรึกษา
              </span>
            </h1>
          </div>

          <div className="h-4 w-px bg-sky-200/60 mx-1 hidden sm:block" />

          {/* Minimal Session Switcher + New Session */}
          <div className="flex items-center gap-1.5">
            <select
              value={activeSessionId || ''}
              onChange={(e) => handleSelectSession(e.target.value)}
              aria-label="เลือกบทสนทนา"
              className="max-w-[200px] truncate rounded-lg border border-transparent bg-sky-50/70 px-3 py-1.5 text-sm font-medium text-sky-950 hover:border-sky-200/70 hover:bg-sky-50 focus:border-sky-300 focus:outline-none transition-colors"
            >
              {sessions.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.title} ({s.message_count})
                </option>
              ))}
            </select>

            <button
              type="button"
              onClick={handleNewSession}
              title="เริ่มหัวข้อสนทนาใหม่"
              className="flex items-center gap-1 rounded-lg px-2.5 py-1.5 text-sm font-medium text-sky-700 hover:bg-sky-50 transition-colors"
            >
              <span className="text-base leading-none">+</span>
              <span className="hidden md:inline">เรื่องใหม่</span>
            </button>
          </div>
        </div>

        {/* Right Action Controls — Grouped & Low-Noise */}
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setDrawerOpen(true)}
            title="ตั้งค่า Hermes Sandbox & Memory"
            className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium text-zinc-600 hover:bg-sky-50/80 hover:text-sky-900 transition-colors"
          >
            <span>🛡️</span>
            <span className="hidden lg:inline">Sandbox & Memory</span>
          </button>

          <button
            type="button"
            onClick={() => {
              setActiveRightTab('review')
              setRightPanelOpen(true)
            }}
            title="เปิด Morning Review & Delta Watchdog"
            className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${
              rightPanelOpen && activeRightTab === 'review'
                ? 'bg-sky-100/80 text-sky-950 font-semibold'
                : 'text-zinc-600 hover:bg-sky-50/80 hover:text-sky-900'
            }`}
          >
            <span>🏛️</span>
            <span className="hidden md:inline">Morning Review</span>
          </button>

          {/* Toggle Right-Hand Swarm & Execution Panel */}
          <button
            type="button"
            onClick={() => setRightPanelOpen((v) => !v)}
            title={rightPanelOpen ? 'ย่อแถบ Swarm & งานวิจัย' : 'เปิดแถบ Swarm & งานวิจัย'}
            className={`flex items-center gap-2 rounded-xl border px-3.5 py-1.5 text-sm font-semibold transition-all ${
              rightPanelOpen
                ? 'border-sky-200 bg-sky-50/90 text-sky-950'
                : activeAgentStatus?.running
                  ? 'border-emerald-300 bg-emerald-50 text-emerald-800 animate-pulse'
                  : 'border-sky-100 bg-white text-sky-800 hover:bg-sky-50'
            }`}
          >
            <span className="relative flex h-2 w-2">
              {activeAgentStatus?.running ? (
                <>
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
                  <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-500" />
                </>
              ) : (
                <span className="inline-flex h-2 w-2 rounded-full bg-sky-400" />
              )}
            </span>
            <span className="hidden sm:inline">
              {activeAgentStatus?.running ? 'กำลังรันงาน...' : 'Swarm & Jobs'}
            </span>
          </button>
        </div>
      </header>

      {/* Body Layout: Chat Area + Right-Hand Swarm & Execution Panel */}
      <div className="flex flex-1 min-h-0 overflow-hidden">
        {/* Left Main Chat Column */}
        <div className="flex flex-1 min-w-0 flex-col min-h-0 overflow-hidden">
          <main className="flex-1 min-h-0 overflow-hidden relative">
            <ManagerChatStream
              messages={messages}
              streamingText={streamingText}
              isThinking={isThinking}
              isStreaming={isStreaming}
              onQuickPromptClick={(prompt, contextRefs) =>
                handleSendMessage(prompt, contextRefs ?? ['@Portfolio'])
              }
              onSelectJob={(jobId) => {
                setSelectedJobId(jobId)
                setActiveRightTab('kanban')
                setRightPanelOpen(true)
              }}
              onOpenKanban={() => {
                setActiveRightTab('kanban')
                setRightPanelOpen(true)
              }}
            />
          </main>

          <footer className="shrink-0">
            <ManagerPromptComposer
              onSubmit={handleSendMessage}
              isStreaming={isStreaming || isThinking}
              onStop={handleStopStreaming}
            />
          </footer>
        </div>

        {/* Right-Hand Panel: Kanban Board, Live Job Execution & Trinity Swarm */}
        <ManagerRightPanel
          isOpen={rightPanelOpen}
          onClose={() => setRightPanelOpen(false)}
          activeTab={activeRightTab}
          onTabChange={setActiveRightTab}
          selectedJobId={selectedJobId}
          onSelectJob={setSelectedJobId}
          detectedJobIds={detectedJobIds}
          activeAgentStatus={activeAgentStatus}
          refreshTrigger={kanbanRefreshTrigger}
          onTriggerQuickJob={(prompt) => handleSendMessage(prompt, ['@Portfolio', '@Macro'])}
          onAskAiWithCard={(prompt) => handleSendMessage(prompt, ['@Portfolio', '@Macro'])}
        />
      </div>

      {/* Slide-over Sandbox & Memory Drawer */}
      <ManagerMemoryDrawer
        isOpen={drawerOpen}
        onClose={() => setDrawerOpen(false)}
      />
    </div>
  )
}
