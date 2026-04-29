import { useEffect, useMemo, useState } from 'react'
import { parse as parseYaml } from 'yaml'
import type { RunRecord, Snapshot } from '../types'

export function useSelectedRun(
  selectedRunId: string | null,
  refreshKey: number,
  snapshot: Snapshot | null,
) {
  const [selectedRunDetail, setSelectedRunDetail] = useState<RunRecord | null>(null)
  const [config, setConfig] = useState<Record<string, unknown> | null>(null)

  // Fetch full run detail when selection changes
  useEffect(() => {
    if (!selectedRunId) { setSelectedRunDetail(null); return }
    let closed = false
    fetch(`/api/runs/${encodeURIComponent(selectedRunId)}`)
      .then((r) => { if (!r.ok) throw new Error(`run ${r.status}`); return r.json() as Promise<RunRecord> })
      .then((payload) => { if (!closed) setSelectedRunDetail(payload) })
      .catch(() => { if (!closed) setSelectedRunDetail(null) })
    return () => { closed = true }
  }, [selectedRunId, refreshKey])

  // Prefer the freshly-fetched detail; fall back to snapshot summary
  const selectedRun = useMemo(() => {
    const snapshotRun = snapshot?.runs?.find((r) => r.manifest.run_id === selectedRunId) ?? null
    return selectedRunDetail?.manifest.run_id === selectedRunId ? selectedRunDetail : snapshotRun
  }, [selectedRunId, snapshot, selectedRunDetail])

  // Fetch config.yaml for the selected run
  useEffect(() => {
    if (!selectedRun?.manifest.run_id) { setConfig(null); return }
    let closed = false
    fetch(`/runs/${encodeURIComponent(selectedRun.manifest.run_id)}/meta/config.yaml`)
      .then((r) => { if (!r.ok) throw new Error(`config ${r.status}`); return r.text() })
      .then((text) => {
        const parsed = parseYaml(text)
        if (!closed && parsed && typeof parsed === 'object') setConfig(parsed as Record<string, unknown>)
      })
      .catch(() => { if (!closed) setConfig(null) })
    return () => { closed = true }
  }, [selectedRun?.manifest.run_id])

  return { selectedRun, selectedRunDetail, setSelectedRunDetail, config }
}
