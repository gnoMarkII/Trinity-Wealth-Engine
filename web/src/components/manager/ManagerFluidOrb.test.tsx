import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import ManagerFluidOrb from './ManagerFluidOrb'

describe('ManagerFluidOrb', () => {
  it('renders gentle thinking label and wave dots without technical noise', () => {
    render(<ManagerFluidOrb />)

    // Peaceful Thai label
    expect(screen.getByText('กำลังประมวลผลมุมมองสำหรับคุณ...')).toBeInTheDocument()

    // No technical noise
    expect(screen.queryByText(/ast_guard/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/tool_call/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/subprocess/i)).not.toBeInTheDocument()
  })

  it('allows custom label override', () => {
    render(<ManagerFluidOrb label="กำลังวิเคราะห์พอร์ต..." />)
    expect(screen.getByText('กำลังวิเคราะห์พอร์ต...')).toBeInTheDocument()
  })
})

