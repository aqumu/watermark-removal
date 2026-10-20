import { useEffect, useRef, useState } from 'react'
import { ScrollText } from 'lucide-react'
import { Card, PanelCardContent, PanelCardHeader, PanelCardTitle } from '@/components/ui/card'
import type { LogTail } from '../../types'
import { EmptyState } from '../shared/EmptyState'

interface RunLogsPanelProps {
  runId: string | null
  refreshKey: number  // bumps whenever the backend reports changed files
}

export function RunLogsPanel({ runId, refreshKey }: RunLogsPanelProps) {
  const [loaded, setLoaded] = useState<{ runId: string; log: LogTail | null } | null>(null)
  const log = loaded && loaded.runId === runId ? loaded.log : null
  const scrollRef = useRef<HTMLDivElement>(null)
  const stickToBottom = useRef(true)

  useEffect(() => {
    if (!runId) return
    let cancelled = false
    fetch(`/api/runs/${encodeURIComponent(runId)}/log`)
      .then((r) => (r.ok ? (r.json() as Promise<LogTail>) : null))
      .then((payload) => { if (!cancelled) setLoaded({ runId, log: payload }) })
      .catch(() => { if (!cancelled) setLoaded({ runId, log: null }) })
    return () => { cancelled = true }
  }, [runId, refreshKey])

  useEffect(() => { stickToBottom.current = true }, [runId])

  // Follow the end of the log unless the user scrolled up to read.
  useEffect(() => {
    const el = scrollRef.current
    if (el && stickToBottom.current) el.scrollTop = el.scrollHeight
  }, [log])

  const onScroll = () => {
    const el = scrollRef.current
    if (el) stickToBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40
  }

  const text = log?.text ?? ''

  return (
    <Card className="border-border/50">
      <PanelCardHeader>
        <PanelCardTitle>
          <ScrollText className="h-3.5 w-3.5 text-muted-foreground" />
          Training Log
          {log?.truncated && <span className="text-xs font-normal text-muted-foreground">(last 256 KB)</span>}
        </PanelCardTitle>
      </PanelCardHeader>
      <PanelCardContent>
        {!text ? (
          <EmptyState text={log?.exists ? 'The log is empty so far.' : 'No training.log for this run.'} />
        ) : (
          <div
            ref={scrollRef}
            onScroll={onScroll}
            className="max-h-[620px] overflow-auto rounded-md border border-border/50 bg-card p-3"
          >
            <pre className="whitespace-pre-wrap break-words font-mono text-[11px] leading-5 text-foreground/80">
              {text}
            </pre>
          </div>
        )}
      </PanelCardContent>
    </Card>
  )
}
