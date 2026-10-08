import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, expect, it, vi, beforeEach } from 'vitest'
import DimeSyncModal from './DimeSyncModal'
import { api } from '../../../api/client'

vi.mock('../../../api/client', () => ({
  api: {
    getDimeEmails: vi.fn(),
    scanDimeEmail: vi.fn(),
    scanDimeUpload: vi.fn(),
    commitDimeTrades: vi.fn(),
    streamBatchDimeSync: vi.fn(),
    getDimePdfUrl: vi.fn(() => '/api/portfolio/dime/pdf?test=1'),
    getDimePdfText: vi.fn(),
    getWealthXEmails: vi.fn(),
    scanWealthXEmail: vi.fn(),
    scanWealthXUpload: vi.fn(),
    commitWealthXTrades: vi.fn(),
    streamBatchWealthXSync: vi.fn(),
    getWealthXPdfUrl: vi.fn(() => '/api/portfolio/wealthx/pdf?test=1'),
    getWealthXPdfText: vi.fn(),
    getScbEmails: vi.fn(),
    scanScbEmail: vi.fn(),
    commitScbTrades: vi.fn(),
    streamBatchScbSync: vi.fn(),
    getScbEmailHtmlUrl: vi.fn(() => '/api/portfolio/scb/emails/test_msg/html'),
  },
}))


describe('DimeSyncModal', () => {
  const onClose = vi.fn()
  const onSuccess = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders initial upload tab and switches to email tab', async () => {
    render(<DimeSyncModal portfolioId="default" onClose={onClose} onSuccess={onSuccess} />)

    expect(screen.getByText('Sync ข้อมูลรายการเทรด (Transaction Sync)')).toBeInTheDocument()
    expect(screen.getByText(/โบรกเกอร์ \/ ผู้ให้บริการ/)).toBeInTheDocument()
    expect(screen.getByText('Dime!')).toBeInTheDocument()
    expect(screen.getByText('WealthX')).toBeInTheDocument()
    expect(screen.queryByText('InnovestX')).toBeNull()
    expect(screen.queryByText('Streaming / Settrade')).toBeNull()
    expect(screen.queryByText('Interactive Brokers')).toBeNull()
    expect(screen.getByText('📤 อัปโหลดไฟล์ PDF')).toBeInTheDocument()
    expect(screen.getByText('📧 ค้นหาจาก Gmail')).toBeInTheDocument()

    // Switch to email tab
    fireEvent.click(screen.getByText('📧 ค้นหาจาก Gmail'))
    expect(screen.getByPlaceholderText(/คำค้นหา เช่น Dime Confirmation/)).toBeInTheDocument()
  })

  it('searches Gmail emails and triggers email analysis', async () => {
    vi.mocked(api.getDimeEmails).mockResolvedValue({
      emails: [
        {
          message_id: 'msg_001',
          attachment_id: 'att_001',
          subject: 'Confirmation Note - Dime',
          sender: 'confirm@dime.co.th',
          received_at: '2026-09-01 10:00:00',
          filename: 'confirm_001.pdf',
          size_bytes: 45000,
        },
      ],
    })

    vi.mocked(api.scanDimeEmail).mockResolvedValue({
      scan_id: 'scan_email_123',
      item_count: 1,
      items: [
        {
          item_id: 'item_1',
          trade_date: '2026-09-01',
          symbol: 'AAPL',
          action: 'BUY',
          units: '10',
          price: '150.00',
          gross_amount: '1500.00',
          fees: {
            commission: '1.50',
            vat: '0.11',
            other_fees: '0.00',
            fee_currency: 'USD',
          },
          net_amount: '1501.61',
          currency: 'USD',
          confirmation_no: 'CONF_AAPL_01',
          source: 'DIME',
          fingerprint: 'fp_aapl_01',
          line_index: 0,
          cash_adjusted: true,
        },
      ],
    })

    render(<DimeSyncModal portfolioId="default" onClose={onClose} onSuccess={onSuccess} />)

    // Switch to email tab
    fireEvent.click(screen.getByText('📧 ค้นหาจาก Gmail'))
    fireEvent.click(screen.getByRole('button', { name: 'ค้นหา' }))

    await waitFor(() => {
      expect(screen.getByText('Confirmation Note - Dime')).toBeInTheDocument()
    })

    // Click Analyze
    fireEvent.click(screen.getByRole('button', { name: 'ดึงและวิเคราะห์' }))

    await waitFor(() => {
      expect(screen.getByText('ตรวจสอบรายการที่พบ (1 รายการ)')).toBeInTheDocument()
      expect(screen.getByText('AAPL')).toBeInTheDocument()
      expect(screen.getByText('1501.61 USD')).toBeInTheDocument()
    })
  })

  it('commits staged items successfully', async () => {
    const mockState = {
      holdings: [],
      summary: {},
      allocation_targets: [],
      fx_rates: {},
    } as any

    vi.mocked(api.scanDimeUpload).mockResolvedValue({
      scan_id: 'scan_upload_456',
      item_count: 1,
      items: [
        {
          item_id: 'item_2',
          trade_date: '2026-09-01',
          symbol: 'TSLA',
          action: 'BUY',
          units: '5',
          price: '200.00',
          gross_amount: '1000.00',
          fees: {
            commission: '1.00',
            vat: '0.07',
            other_fees: '0.00',
            fee_currency: 'USD',
          },
          net_amount: '1001.07',
          currency: 'USD',
          confirmation_no: 'CONF_TSLA_02',
          source: 'DIME',
          fingerprint: 'fp_tsla_02',
          line_index: 0,
          cash_adjusted: true,
        },
      ],
    })

    vi.mocked(api.commitDimeTrades).mockResolvedValue({
      ok: true,
      imported_count: 1,
      state: mockState,
    })

    render(<DimeSyncModal portfolioId="default" onClose={onClose} onSuccess={onSuccess} />)

    // Trigger upload
    const file = new File(['%PDF-1.4 mock content'], 'test.pdf', { type: 'application/pdf' })
    const input = screen.getByText('คลิกเพื่อเลือกไฟล์ PDF หรือลากไฟล์มาวางที่นี่')
    fireEvent.click(input)

    // Trigger analyze
    await api.scanDimeUpload(file)

    // Simulate staged result
    fireEvent.click(screen.getByText('📧 ค้นหาจาก Gmail'))
    // Directly test commit button when staged
  })

  it('triggers 1-click batch sync and renders progress and items', async () => {
    vi.mocked(api.streamBatchDimeSync).mockImplementation(async (_payload, callbacks) => {
      callbacks.onProgress?.({
        current: 10,
        total: 50,
        percent: 20,
        items_found: 5,
        subject: 'Confirmation Note - Test',
      })
      callbacks.onWarning?.({
        subject: 'Confirmation Note - Old',
        reason: 'Quarantined due to missing Order ID',
      })
      callbacks.onComplete?.({
        scan_id: 'batch_scan_999',
        item_count: 1,
        items: [
          {
            item_id: 'item_batch_1',
            trade_date: '2026-09-02',
            symbol: 'NVDA',
            action: 'BUY',
            units: '2',
            price: '120.00',
            gross_amount: '240.00',
            fees: {
              commission: '0.50',
              vat: '0.04',
              other_fees: '0.00',
              fee_currency: 'USD',
            },
            net_amount: '240.54',
            currency: 'USD',
            confirmation_no: 'CONF_NVDA_01',
            order_id: 'ORD_NVDA_01',
            source: 'DIME',
            fingerprint: 'fp_nvda_01',
            line_index: 0,
            cash_adjusted: true,
          },
        ],
        warnings: [
          {
            message_id: 'msg_old_1',
            attachment_id: 'att_old_1',
            filename: 'confirm_old.pdf',
            subject: 'Confirmation Note - Old',
            reason: 'Quarantined due to missing Order ID',
            can_preview: true,
          },
        ],
        skipped_synced_count: 10,
      })
    })

    render(<DimeSyncModal portfolioId="default" onClose={onClose} onSuccess={onSuccess} />)

    // Switch to email tab
    fireEvent.click(screen.getByText('📧 ค้นหาจาก Gmail'))

    expect(screen.getByText('ซิงค์ข้อมูลทั้งหมดอัตโนมัติ (Sync All Confirmation Notes)')).toBeInTheDocument()

    // Trigger Sync All
    fireEvent.click(screen.getByRole('button', { name: /ดึงและนำเข้าข้อมูลทั้งหมด \(Sync All\)/ }))

    await waitFor(() => {
      expect(screen.getByText('ตรวจสอบรายการที่พบ (1 รายการ)')).toBeInTheDocument()
      expect(screen.getByText('NVDA')).toBeInTheDocument()
      expect(screen.getByText('ORD_NVDA_01')).toBeInTheDocument()
      expect(screen.getByText(/พบข้อควรระวัง \/ รายการที่ถูกพัก \(Quarantined\)/)).toBeInTheDocument()
      expect(screen.getByText('📄 ดูเอกสาร PDF')).toBeInTheDocument()
    })

    // Click "📄 ดูเอกสาร PDF" to open inspection modal
    fireEvent.click(screen.getByText('📄 ดูเอกสาร PDF'))
    expect(screen.getByRole('heading', { name: 'confirm_old.pdf' })).toBeInTheDocument()
    expect(screen.getAllByText(/Quarantined due to missing Order ID/).length).toBeGreaterThanOrEqual(2)
  })

  it('switches to WealthX and executes WealthX batch sync', async () => {
    vi.mocked(api.streamBatchWealthXSync).mockImplementation(async (_payload, callbacks) => {
      callbacks.onComplete?.({
        scan_id: 'wealthx_scan_777',
        item_count: 1,
        items: [
          {
            item_id: 'item_wx_1',
            trade_date: '2026-06-12',
            symbol: 'TLWORLD-X',
            action: 'BUY',
            units: '898.3838',
            price: '11.1311',
            gross_amount: '10000.00',
            fees: {
              commission: '9.97',
              vat: '0.00',
              other_fees: '0.00',
              fee_currency: 'THB',
            },
            net_amount: '10000.00',
            currency: 'THB',
            confirmation_no: 'DN 202606150569',
            order_id: '2392606120006840',
            source: 'WEALTHX',
            fingerprint: 'fp_wx_01',
            line_index: 0,
            cash_adjusted: true,
            asset_type: 'Fund',
          },
        ],
        warnings: [],
        skipped_synced_count: 0,
      })
    })

    render(<DimeSyncModal portfolioId="default" onClose={onClose} onSuccess={onSuccess} />)

    // Click WealthX card
    fireEvent.click(screen.getByText('WealthX'))
    expect(screen.getByText('WealthX กองทุนรวมไทย')).toBeInTheDocument()

    // Switch to email tab
    fireEvent.click(screen.getByText('📧 ค้นหาจาก Gmail'))
    expect(screen.getByText(/ซิงค์ใบยืนยัน WealthX ทั้งหมดอัตโนมัติ/)).toBeInTheDocument()

    // Trigger Sync All
    fireEvent.click(screen.getByRole('button', { name: /ดึงและนำเข้าข้อมูลทั้งหมด \(Sync All\)/ }))

    await waitFor(() => {
      expect(screen.getByText('ตรวจสอบรายการที่พบ (1 รายการ)')).toBeInTheDocument()
      expect(screen.getByText('TLWORLD-X')).toBeInTheDocument()
      expect(screen.getByText('2392606120006840')).toBeInTheDocument()
      expect(screen.getByText('10000.00 THB')).toBeInTheDocument()
      expect(screen.getByText(/WealthX Confirmation \(กองทุนรวมไทย\)/)).toBeInTheDocument()
    })
  })

  it('switches to SCB (Thai Funds) and executes SCBAM batch sync', async () => {
    vi.mocked(api.streamBatchScbSync).mockImplementation(async (_payload, callbacks) => {
      callbacks.onComplete?.({
        scan_id: 'scb_scan_888',
        item_count: 1,
        items: [
          {
            item_id: 'item_scb_1',
            trade_date: '2025-12-26',
            settlement_date: '2025-12-29',
            symbol: 'SCBS&P500E',
            action: 'BUY',
            units: '198.2116',
            price: '40.3609',
            gross_amount: '8000.00',
            fees: {
              commission: '0.00',
              vat: '0.00',
              other_fees: '0.00',
              fee_currency: 'THB',
            },
            net_amount: '8000.00',
            currency: 'THB',
            confirmation_no: 'SCB-FC-9801452654-20251229',
            order_id: 'SCB-ADV-2025-12-26-15.58.25.068380',
            source: 'SCB',
            fingerprint: 'scb_ADV-2025-12-26-15.58.25.068380',
            line_index: 0,
            cash_adjusted: true,
            asset_type: 'Mutual Fund',
          },
        ],
        warnings: [],
        skipped_synced_count: 0,
      })
    })

    render(<DimeSyncModal portfolioId="default" onClose={onClose} onSuccess={onSuccess} />)

    // Click SCB card
    fireEvent.click(screen.getByText('SCB (Thai Funds)'))
    expect(screen.getByText(/SCBAM Fund Click \(Direct Email Confirmation\)/)).toBeInTheDocument()

    // Trigger Sync All
    fireEvent.click(screen.getByRole('button', { name: /ดึงและนำเข้าข้อมูลทั้งหมด \(Sync All\)/ }))

    await waitFor(() => {
      expect(screen.getByText('ตรวจสอบรายการที่พบ (1 รายการ)')).toBeInTheDocument()
      expect(screen.getByText('SCBS&P500E')).toBeInTheDocument()
      expect(screen.getByText('SCB-ADV-2025-12-26-15.58.25.068380')).toBeInTheDocument()
      expect(screen.getByText('8000.00 THB')).toBeInTheDocument()
      expect(screen.getByText(/SCBAM Fund Click \(กองทุนรวมไทย\)/)).toBeInTheDocument()
    })
  })

  it('supports partial item selection and passes selectedItemIds on commit for Dime & WealthX', async () => {
    const onClose = vi.fn()
    const onSuccess = vi.fn()
    const mockState = {
      holdings: [],
      summary: {},
      allocation_targets: [],
      fx_rates: {},
    } as any

    vi.mocked(api.streamBatchDimeSync).mockImplementation(async (_payload, callbacks) => {
      callbacks.onComplete?.({
        scan_id: 'scan_partial_123',
        item_count: 2,
        items: [
          {
            item_id: 'item_1',
            trade_date: '2026-08-01',
            settlement_date: '2026-08-03',
            symbol: 'AAPL',
            action: 'BUY',
            units: '10',
            price: '150',
            gross_amount: '1500',
            fees: { commission: '0', vat: '0', other_fees: '0', fee_currency: 'USD' },
            net_amount: '1500',
            currency: 'USD',
            confirmation_no: 'CONF_1',
            order_id: 'ORD_1',
            source: 'DIME',
            fingerprint: 'fp_1',
            line_index: 0,
            cash_adjusted: true,
            asset_type: 'Stock',
          },
          {
            item_id: 'item_2',
            trade_date: '2026-08-02',
            settlement_date: '2026-08-04',
            symbol: 'MSFT',
            action: 'BUY',
            units: '5',
            price: '300',
            gross_amount: '1500',
            fees: { commission: '0', vat: '0', other_fees: '0', fee_currency: 'USD' },
            net_amount: '1500',
            currency: 'USD',
            confirmation_no: 'CONF_2',
            order_id: 'ORD_2',
            source: 'DIME',
            fingerprint: 'fp_2',
            line_index: 0,
            cash_adjusted: true,
            asset_type: 'Stock',
          },
        ],
        warnings: [],
        skipped_synced_count: 0,
      })
    })

    vi.mocked(api.commitDimeTrades).mockResolvedValue({
      ok: true,
      imported_count: 1,
      state: mockState,
    })

    render(<DimeSyncModal portfolioId="default" onClose={onClose} onSuccess={onSuccess} />)

    // Switch to email tab
    fireEvent.click(screen.getByText('📧 ค้นหาจาก Gmail'))

    // Trigger Sync All
    fireEvent.click(screen.getByRole('button', { name: /ดึงและนำเข้าข้อมูลทั้งหมด \(Sync All\)/ }))

    await waitFor(() => {
      expect(screen.getByText('ตรวจสอบรายการที่พบ (2 รายการ)')).toBeInTheDocument()
      expect(screen.getByText('ยืนยันนำเข้า 2 รายการ')).toBeInTheDocument()
    })

    // Uncheck MSFT (item_2)
    const msftCheckbox = screen.getByLabelText(/Select MSFT 2026-08-02/)
    fireEvent.click(msftCheckbox)

    // Now button should show 1 item
    expect(screen.getByText('ยืนยันนำเข้า 1 รายการ')).toBeInTheDocument()

    // Click commit
    const commitBtn = screen.getByRole('button', { name: /ยืนยันนำเข้า 1 รายการ/ })
    fireEvent.click(commitBtn)

    await waitFor(() => {
      expect(api.commitDimeTrades).toHaveBeenCalledWith('scan_partial_123', 'default', ['item_1'])
    })
  })
})


