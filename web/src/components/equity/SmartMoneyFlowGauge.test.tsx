import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import SmartMoneyFlowGauge from './SmartMoneyFlowGauge'
import type { SmartMoneyFlagsDTO } from '../../api/types'

describe('SmartMoneyFlowGauge', () => {
  const mockSmartMoney: SmartMoneyFlagsDTO = {
    insider_signal: 'neutral',
    insider_buy_count_90d: 0,
    insider_sell_count_90d: 0,
    institutional_ownership_pct: 84.2,
    insider_ownership_pct: 1.5,
    short_interest_pct: 2.1,
    short_squeeze_risk: false,
    overall_smart_money_flag: 'bullish_signal',
  }

  it('renders institutional ownership, short interest, and smart money badge', () => {
    render(<SmartMoneyFlowGauge smartMoney={mockSmartMoney} />)

    expect(screen.getByTestId('smart-money-flow-gauge')).toBeInTheDocument()
    expect(screen.getByText('84.2%')).toBeInTheDocument()
    expect(screen.getByText('2.10%')).toBeInTheDocument()
    expect(screen.getByText('🟢 Smart Money Accumulation')).toBeInTheDocument()
  })
})
