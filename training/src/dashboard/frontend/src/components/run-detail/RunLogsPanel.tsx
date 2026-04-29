import { useEffect, useRef, useState } from 'react'
import { ScrollText } from 'lucide-react'
import { Card, PanelCardContent, PanelCardHeader, PanelCardTitle } from '@/components/ui/card'
import { EmptyState } from '../shared/EmptyState'

interface RunLogsPanelProps {
  lines: string[]          // live in-memory logs from training_manager
  runId: string | null     // selected run id (for fetching persisted log file)
  activeRunId: string | null  // currently active run id
}

export function RunLogsPanel({ lines, runId, activeRunId }: RunLogsPanelProps) {
  const [fileLines, setFileLines] = useState<string[] | null>(null)
  const [fetchError, setFetchError] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)

  // Fetch the persisted log file when viewing a non-active or completed run
  const isLiveRun = runId != null && runId === activeRunId && lines.length > 0
  useEffect(() => {
    if (!runId || isLiveRun) {
      setFileLines(null)
      setFetchError(false)
      return
    }
    let cancelled = false
    fetch(`/runs/${encodeURIComponent(runId)}/training.log`)
      .then(r => {
        if (!r.ok) throw new Error('not found')
        return r.text()
      })
      .then(text => {
        if (!cancelled) {
          setFileLines(text.split('\n').filter(Boolean))
          setFetchError(false)
        }
      })
      .catch(() => {
        if (!cancelled) {
          setFileLines(null)
          setFetchError(true)
        }
      })
    return () => { cancelled = true }
  }, [runId, isLiveRun])

  // Auto-scroll to bottom when live logs update
  useEffect(() => {
    if (isLiveRun && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [lines, isLiveRun])

  const displayLines = isLiveRun ? lines : (fileLines ?? [])
  const isEmpty = displayLines.length === 0

  return (
    <Card className="border-border/50">
      <PanelCardHeader>
        <PanelCardTitle>
          <ScrollText className="h-3.5 w-3.5 text-muted-foreground" />
          {isLiveRun ? 'Live Logs' : 'Training Logs'}
        </PanelCardTitle>
      </PanelCardHeader>
      <PanelCardContent>
        {isEmpty ? (
          <EmptyState text={
            fetchError
              ? 'No log file found for this run.'
              : isLiveRun
              ? 'No output yet — training is starting up.'
              : 'No log file found for this run.'
          } />
        ) : (
          <div
            ref={scrollRef}
            className="max-h-[620px] overflow-auto rounded-md border border-border/50 bg-card p-3"
          >
            <pre className="whitespace-pre-wrap break-words font-mono text-[11px] leading-5 text-foreground/80">
              {displayLines.join('\n')}
            </pre>
          </div>
        )}
      </PanelCardContent>
    </Card>
  )
}
