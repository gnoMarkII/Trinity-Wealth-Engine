import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, expect, it, vi, beforeEach } from 'vitest'
import DimePdfViewerModal from './DimePdfViewerModal'
import { api } from '../../../api/client'

vi.mock('../../../api/client', () => ({
  api: {
    getDimePdfUrl: vi.fn((msgId, attId, _pass, decrypt) => `/api/portfolio/dime/pdf?msg=${msgId}&att=${attId}&decrypt=${decrypt}`),
    getDimePdfText: vi.fn(),
    getWealthXPdfUrl: vi.fn((msgId, attId, _pass, decrypt) => `/api/portfolio/wealthx/pdf?msg=${msgId}&att=${attId}&decrypt=${decrypt}`),
    getWealthXPdfText: vi.fn(),
  },
}))

describe('DimePdfViewerModal', () => {
  const onClose = vi.fn()
  const sampleWarning = {
    message_id: 'msg_999',
    attachment_id: 'att_888',
    subject: '[Dime!] Confirmation Note',
    filename: 'confirmation_note_2026.pdf',
    received_at: '2026-09-01 14:30',
    reason: 'Reconciliation mismatch: Units (36.662240) * Price (7.2300) = 265.07 != Gross (265.16), diff=0.09',
    can_preview: true,
  }

  beforeEach(() => {
    vi.clearAllMocks()
    Object.assign(navigator, {
      clipboard: {
        writeText: vi.fn().mockResolvedValue(undefined),
      },
    })
  })

  it('renders filename, status badge, and exact quarantine reason', () => {
    render(<DimePdfViewerModal warning={sampleWarning} password="password123" onClose={onClose} />)

    expect(screen.getByText('confirmation_note_2026.pdf')).toBeInTheDocument()
    expect(screen.getByText('พักรายการ (Quarantined)')).toBeInTheDocument()
    expect(screen.getByText(/สาเหตุที่ระบบไม่นำเข้าอัตโนมัติ/)).toBeInTheDocument()
    expect(screen.getByText(sampleWarning.reason)).toBeInTheDocument()
  })

  it('renders PDF viewer iframe and action links', () => {
    render(<DimePdfViewerModal warning={sampleWarning} password="password123" onClose={onClose} />)

    const iframe = screen.getByTitle(/Dime PDF - confirmation_note_2026.pdf/) as HTMLIFrameElement
    expect(iframe).toBeInTheDocument()
    expect(iframe.src).toContain('/api/portfolio/dime/pdf?msg=msg_999&att=att_888&decrypt=true')

    const openInNewTab = screen.getByText('↗️ เปิดในแท็บใหม่') as HTMLAnchorElement
    expect(openInNewTab.href).toContain('/api/portfolio/dime/pdf?msg=msg_999&att=att_888&decrypt=true')

    const downloadLink = screen.getByText('📥 ดาวน์โหลด') as HTMLAnchorElement
    expect(downloadLink.href).toContain('/api/portfolio/dime/pdf?msg=msg_999&att=att_888&decrypt=false')
  })

  it('switches to text tab and loads raw extracted pages', async () => {
    vi.mocked(api.getDimePdfText).mockResolvedValue({
      message_id: 'msg_999',
      attachment_id: 'att_888',
      filename: 'confirmation_note_2026.pdf',
      page_count: 2,
      pages: [
        { page_number: 1, text: 'ACCOUNT: 12345\nSYMBOL: NOK\nUNITS: 36.662240\nPRICE: 7.2300' },
        { page_number: 2, text: 'TERMS AND CONDITIONS\nFEE SCHEDULE' },
      ],
    })

    render(<DimePdfViewerModal warning={sampleWarning} password="password123" onClose={onClose} />)

    // Switch to Raw Text tab
    fireEvent.click(screen.getByText(/ข้อความในเอกสาร \(Raw Text\)/))

    await waitFor(() => {
      expect(api.getDimePdfText).toHaveBeenCalledWith('msg_999', 'att_888', 'password123')
      expect(screen.getByText(/ACCOUNT: 12345/)).toBeInTheDocument()
      expect(screen.getByText(/TERMS AND CONDITIONS/)).toBeInTheDocument()
    })

    // Test Copy text button
    const copyBtn = screen.getByText('📋 คัดลอกข้อความ')
    fireEvent.click(copyBtn)
    expect(navigator.clipboard.writeText).toHaveBeenCalled()
    await waitFor(() => {
      expect(screen.getByText('✓ คัดลอกแล้ว')).toBeInTheDocument()
    })
  })

  it('calls onClose when close button or header cross is clicked', () => {
    render(<DimePdfViewerModal warning={sampleWarning} onClose={onClose} />)

    fireEvent.click(screen.getByLabelText('ปิดหน้าต่าง'))
    expect(onClose).toHaveBeenCalledTimes(1)

    fireEvent.click(screen.getByText('ปิดหน้าต่าง'))
    expect(onClose).toHaveBeenCalledTimes(2)
  })
})
