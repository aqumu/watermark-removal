import { useCallback, useEffect, useRef, useState } from 'react'
import type { Snapshot } from '../types'
import { REFRESH_EVENTS } from '../lib/constants'

export function useDashboardSnapshot() {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [refreshKey, setRefreshKey] = useState(0)
  const [liveTick, setLiveTick] = useState(Date.now())
  const [sseConnected, setSseConnected] = useState(false)

  // Stable fetch function, available outside the SSE effect
  const manualRefresh = useCallback(async () => {
    try {
      const response = await fetch('/api/state')
      if (!response.ok) throw new Error(`snapshot ${response.status}`)
      const payload = (await response.json()) as Snapshot
      setSnapshot(payload)
      setRefreshKey((v) => v + 1)
      setError(null)
      setLoading(false)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load dashboard state')
      setLoading(false)
    }
  }, [])

  // Track whether we have an SSE connection so the fallback poll backs off
  const sseConnectedRef = useRef(false)

  // SSE subscription + initial snapshot load
  useEffect(() => {
    let closed = false

    const loadSnapshot = async () => {
      const response = await fetch('/api/state')
      if (!response.ok) throw new Error(`snapshot ${response.status}`)
      const payload = (await response.json()) as Snapshot
      if (closed) return
      setSnapshot(payload)
      setRefreshKey((v) => v + 1)
      setError(null)
      setLoading(false)
    }

    loadSnapshot().catch((err) => {
      if (!closed) {
        setError(err instanceof Error ? err.message : 'Failed to load dashboard state')
        setLoading(false)
      }
    })

    const source = new EventSource('/api/events')
    const refresh = () => {
      setSseConnected(true)
      sseConnectedRef.current = true
      loadSnapshot().catch(() => {})
    }
    for (const eventName of REFRESH_EVENTS) source.addEventListener(eventName, refresh)
    source.onopen = () => {
      setSseConnected(true)
      sseConnectedRef.current = true
      loadSnapshot().catch(() => {})
    }
    source.onerror = () => {
      setSseConnected(false)
      sseConnectedRef.current = false
    }

    return () => { closed = true; source.close() }
  }, [])

  // Fallback polling: refresh every 20 s regardless of SSE, so subprocess
  // training (which doesn't push live events) still updates the dashboard.
  useEffect(() => {
    const timer = window.setInterval(() => { manualRefresh().catch(() => {}) }, 20_000)
    return () => window.clearInterval(timer)
  }, [manualRefresh])

  // Clock tick for relative timestamps
  useEffect(() => {
    const timer = window.setInterval(() => setLiveTick(Date.now()), 30_000)
    return () => window.clearInterval(timer)
  }, [])

  return { snapshot, setSnapshot, loading, error, refreshKey, liveTick, sseConnected, manualRefresh }
}
