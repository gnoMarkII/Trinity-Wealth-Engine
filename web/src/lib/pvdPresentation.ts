import type {
  ActualHoldingDTO,
  PvdStatementMetadataDTO,
} from '../api/types'

type PvdHoldingLike = Pick<ActualHoldingDTO, 'symbol' | 'asset_type' | 'units' | 'market_value_thb' | 'status' | 'archived_at' | 'pvd_valuation'>

export interface PvdValuationContext {
  hasPvd: boolean
  hasMarketHoldings: boolean
  hasNonPvdValue: boolean
  dates: string[]
  hasUnknownDate: boolean
  hasUnknownValuation: boolean
  latestRecordedStatementDate: string | null
}

export interface PriceRefreshPartition {
  marketEntries: Array<[string, string]>
  pvdEntries: Array<[string, string]>
}

export function isPvdSymbol(symbol: string | null | undefined): boolean {
  return Boolean(symbol?.trim().toUpperCase().startsWith('PVD_'))
}

export function isPvdHolding(holding: Pick<ActualHoldingDTO, 'symbol' | 'asset_type' | 'pvd_valuation'>): boolean {
  return holding.asset_type.trim().toUpperCase() === 'PVD' ||
    isPvdSymbol(holding.symbol) ||
    Boolean(holding.pvd_valuation)
}

export function isValidIsoDate(value: string | null | undefined): value is string {
  if (!value || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return false
  const year = Number(value.slice(0, 4))
  const month = Number(value.slice(5, 7))
  const day = Number(value.slice(8, 10))
  if (!year || month < 1 || month > 12 || day < 1) return false
  const leapYear = year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0)
  const daysInMonth = [31, leapYear ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
  return day <= daysInMonth[month - 1]!
}

function isArchived(holding: Pick<PvdHoldingLike, 'status' | 'archived_at'>): boolean {
  return Boolean(holding.archived_at) || holding.status?.trim().toLowerCase() === 'archived'
}

function hasCurrentPvdValue(holding: Pick<PvdHoldingLike, 'units' | 'market_value_thb' | 'status' | 'archived_at'>): boolean {
  return !isArchived(holding) && (holding.units > 0 || holding.market_value_thb > 0)
}

export function getPvdValuationContext(
  holdings: PvdHoldingLike[],
  statements: Pick<PvdStatementMetadataDTO, 'as_of_date'>[] = [],
): PvdValuationContext {
  const pvdHoldings = holdings.filter(isPvdHolding)
  const activePvdHoldings = pvdHoldings.filter(hasCurrentPvdValue)
  const dates = [...new Set(activePvdHoldings
    .map((holding) => holding.pvd_valuation?.as_of_date)
    .filter(isValidIsoDate))].sort()
  const latestRecordedStatementDate = statements
    .map((statement) => statement.as_of_date)
    .filter(isValidIsoDate)
    .sort()
    .at(-1) ?? null

  return {
    hasPvd: pvdHoldings.length > 0 || statements.length > 0,
    hasMarketHoldings: holdings.some((holding) =>
      !isPvdHolding(holding) &&
      holding.asset_type.trim().toLowerCase() !== 'cash' &&
      !isArchived(holding) &&
      holding.units > 0
    ),
    hasNonPvdValue: holdings.some((holding) =>
      !isPvdHolding(holding) && holding.market_value_thb > 0
    ),
    dates,
    hasUnknownDate: activePvdHoldings.some((holding) => !isValidIsoDate(holding.pvd_valuation?.as_of_date)),
    hasUnknownValuation: activePvdHoldings.some((holding) => !holding.pvd_valuation),
    latestRecordedStatementDate,
  }
}

export function partitionPriceRefreshInfo(
  info: Record<string, string> | null | undefined,
  holdings: Pick<ActualHoldingDTO, 'symbol' | 'asset_type' | 'pvd_valuation'>[],
): PriceRefreshPartition {
  const pvdSymbols = new Set(holdings.filter(isPvdHolding).map((holding) => holding.symbol.trim().toUpperCase()))
  const marketEntries: Array<[string, string]> = []
  const pvdEntries: Array<[string, string]> = []

  Object.entries(info ?? {}).forEach(([symbol, status]) => {
    const isPvdStatus = /^\s*statement\s+nav\b/i.test(status)
    if (isPvdStatus || isPvdSymbol(symbol) || pvdSymbols.has(symbol.trim().toUpperCase())) {
      pvdEntries.push([symbol, status])
    } else {
      marketEntries.push([symbol, status])
    }
  })

  return { marketEntries, pvdEntries }
}
