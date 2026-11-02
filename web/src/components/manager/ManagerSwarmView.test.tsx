import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import ManagerSwarmView from './ManagerSwarmView'

describe('ManagerSwarmView', () => {
  it('renders all 5 specialist agents with standby badges', () => {
    render(<ManagerSwarmView activeStatus={{ running: false, flow: null, node: null, job_id: null }} />)

    expect(screen.getByText('Trinity Swarm')).toBeInTheDocument()
    expect(screen.getByText('(5 Specialists)')).toBeInTheDocument()
    expect(screen.getByText('Standby (พร้อมทำงาน)')).toBeInTheDocument()

    expect(screen.queryByText('Supervisor')).not.toBeInTheDocument()
    expect(screen.getByText('Macro Quant')).toBeInTheDocument()
    expect(screen.getByText('Macro Economist')).toBeInTheDocument()
    expect(screen.getByText('Strategic Allocator')).toBeInTheDocument()
    expect(screen.getByText('Equity Intel')).toBeInTheDocument()
    expect(screen.getByText('Archivist')).toBeInTheDocument()
  })

  it('highlights the active agent when running', () => {
    render(
      <ManagerSwarmView
        activeStatus={{
          running: true,
          flow: 'manager',
          node: 'macro_quant',
          job_id: 'j1',
        }}
      />
    )

    expect(screen.getByText('Active Swarm')).toBeInTheDocument()
    expect(screen.getByText('กำลังทำงาน')).toBeInTheDocument()
    expect(screen.getByText('Node: Macro Quant')).toBeInTheDocument()
  })

  it('renders workflow indicator for other flows like news_youtube', () => {
    render(
      <ManagerSwarmView
        activeStatus={{
          running: true,
          flow: 'news_youtube',
          node: 'fetch_news',
          job_id: 'j2',
        }}
      />
    )

    expect(screen.getByText(/กำลังรัน Workflow: News\/YouTube/)).toBeInTheDocument()
  })
})
