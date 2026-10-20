import { useCallback, useEffect, useState } from 'react'
import type { Snapshot } from '../types'
import { REFRESH_EVENTS } from '../lib/constants'

// The viewer backend watches the run folders and emits an event whenever files
// change; every event simply triggers a refetch. A slow poll covers dropped
// connections.
export function useDashboardSnapshot() {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [refreshKey, setRefreshKey] = useState(0)
  const [liveTick, setLiveTick] = useState(Date.now())
  const [sseConnected, setSseConnected] = useState(false)

  const refresh = useCallback(async () => {
    try {
      const response = await fetch('/api/state')
      if (!response.ok) throw new Error(`state ${response.status}`)
      setSnapshot((await response.json()) as Snapshot)
      setRefreshKey((v) => v + 1)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load runs')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void refresh()
    const source = new EventSource('/api/events')
    const onEvent = () => {
      setSseConnected(true)
      void refresh()
    }
    for (const eventName of REFRESH_EVENTS) source.addEventListener(eventName, onEvent)
    source.onerror = () => setSseConnected(false)
    return () => source.close()
  }, [refresh])

  useEffect(() => {
    const timer = window.setInterval(() => { void refresh() }, 30_000)
    return () => window.clearInterval(timer)
  }, [refresh])

  // Clock tick for relative timestamps
  useEffect(() => {
    const timer = window.setInterval(() => setLiveTick(Date.now()), 30_000)
    return () => window.clearInterval(timer)
  }, [])

  return { snapshot, setSnapshot, loading, error, refreshKey, liveTick, sseConnected, refresh }
}
