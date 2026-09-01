import React, { useState } from 'react'

export interface EvidenceItemMetadataDTO {
  source_as_of: string
  retrieved_at: string
  fiscal_period_end?: string | null
  reported_at?: string | null
  exchange_timezone?: string
  currency?: string
  unit?: string
  source_uri: string
  payload_hash: string
  provider_tier?: string
  status: string
  stale_reason?: string | null
}

export interface EvidenceManifestItemDTO {
  item_id: string
  metadata: EvidenceItemMetadataDTO
  storage_ref?: string | null
  query_slice?: Record<string, any> | null
}

export interface CorporateActionsEvidenceDTO {
  metadata: EvidenceItemMetadataDTO
  chart_price_basis: string
  valuation_price_basis: string
}

export interface SnapshotMetadataDTO {
  analysis_run_id: string
  schema_version: string
  as_of_date: string
  generated_at: string
  snapshot_sha256: string
  data_quality_flags: string[]
  coverage_pct: number
}

export interface AnalysisEvidenceSnapshotDTO {
  metadata: SnapshotMetadataDTO
  manifest_items: Record<string, EvidenceManifestItemDTO>
  corporate_actions?: CorporateActionsEvidenceDTO | null
  derived_features?: Record<string, any>
}

interface EvidenceProvenanceDrawerProps {
  snapshot?: AnalysisEvidenceSnapshotDTO | null
}

export const EvidenceProvenanceDrawer: React.FC<EvidenceProvenanceDrawerProps> = ({ snapshot }) => {
  const [isOpen, setIsOpen] = useState(false)

  if (!snapshot) {
    return null
  }

  const { metadata, manifest_items, corporate_actions } = snapshot

  return (
    <div className="mt-4 border-t border-edge/60 pt-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="inline-flex h-2 w-2 rounded-full bg-emerald-500 shadow-[0_0_6px_rgba(16,185,129,0.6)] animate-pulse" />
          <span className="text-xs font-mono text-zinc-500">
            CAS Snapshot: <strong className="text-zinc-900 font-semibold">{metadata.analysis_run_id}</strong> (v{metadata.schema_version})
          </span>
        </div>
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="px-2.5 py-1 text-xs font-semibold bg-surface hover:bg-surface-strong text-sky-700 rounded-lg border border-edge transition-colors flex items-center gap-1.5 shadow-2xs"
          data-testid="toggle-evidence-drawer"
        >
          <span>{isOpen ? 'Hide Evidence Provenance' : 'View Evidence Provenance'}</span>
          <span className="font-mono text-[10px] text-zinc-400">[{metadata.snapshot_sha256.slice(0, 8)}]</span>
        </button>
      </div>

      {isOpen && (
        <div className="mt-3 p-4 bg-surface/90 border border-edge rounded-xl space-y-4 text-xs font-mono shadow-xs">
          {/* Header Summary */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pb-3 border-b border-edge/60 text-[11px]">
            <div>
              <span className="text-zinc-400">As-Of Date:</span>
              <div className="text-zinc-800 font-semibold">{metadata.as_of_date}</div>
            </div>
            <div>
              <span className="text-zinc-400">Generated UTC:</span>
              <div className="text-zinc-800">{new Date(metadata.generated_at).toLocaleString()}</div>
            </div>
            <div>
              <span className="text-zinc-400">Manifest SHA256:</span>
              <div className="text-sky-700 font-bold truncate" title={metadata.snapshot_sha256}>
                {metadata.snapshot_sha256.slice(0, 16)}...
              </div>
            </div>
            <div>
              <span className="text-zinc-400">Price Basis:</span>
              <div className="text-zinc-700">
                Chart: {corporate_actions?.chart_price_basis || 'adj_close'} | Val: {corporate_actions?.valuation_price_basis || 'unadj_close'}
              </div>
            </div>
          </div>

          {/* Manifest Items Table */}
          <div>
            <div className="text-[11px] font-bold text-zinc-700 mb-2 uppercase tracking-wider">
              Item-Level Manifest Provenance (CAS Verified)
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-edge text-zinc-500 text-[10px]">
                    <th className="pb-1.5 font-semibold">Item ID</th>
                    <th className="pb-1.5 font-semibold">Source URI</th>
                    <th className="pb-1.5 font-semibold">Retrieved At</th>
                    <th className="pb-1.5 font-semibold">Status</th>
                    <th className="pb-1.5 font-semibold">Payload SHA256</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-edge/40 text-[11px]">
                  {Object.entries(manifest_items || {}).map(([key, item]) => (
                    <tr key={key} className="hover:bg-surface-strong/60 transition-colors">
                      <td className="py-1.5 text-sky-800 font-semibold">{item.item_id}</td>
                      <td className="py-1.5 text-zinc-600 truncate max-w-[200px]" title={item.metadata.source_uri}>
                        {item.metadata.source_uri}
                      </td>
                      <td className="py-1.5 text-zinc-500">{item.metadata.retrieved_at?.slice(0, 19)}</td>
                      <td className="py-1.5">
                        <span
                          className={`px-1.5 py-0.5 rounded text-[10px] font-bold border ${
                            item.metadata.status === 'available'
                              ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                              : 'bg-amber-50 text-amber-700 border-amber-200'
                          }`}
                        >
                          {item.metadata.status}
                        </span>
                      </td>
                      <td className="py-1.5 text-zinc-400 font-mono text-[10px]" title={item.metadata.payload_hash}>
                        {item.metadata.payload_hash?.slice(0, 12)}...
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
