import { useEffect, useMemo, useState } from 'react'
import {
  Activity,
  Clock,
  FolderOpen,
  History,
  LayoutDashboard,
  Layers,
  Settings2,
  Star,
  Trash2,
  Zap,
} from 'lucide-react'

import { Badge } from '@/components/ui/badge'
import { Card, PanelCardContent, PanelCardHeader, PanelCardTitle } from '@/components/ui/card'

import type { ActivePage, RunRecord } from './types'
import { formatDurationSince } from './lib/format'
import {
  buildHyperparameters,
  buildLossWeights,
  buildMetricCards,
  buildModelOverviewEntries,
  buildRunMetricSummary,
  buildTrainSeries,
  buildValSeries,
  formatProgress,
  resolveRunStatus,
} from './lib/run-utils'

import { useDashboardSnapshot } from './hooks/useDashboardSnapshot'
import { useSelectedRun } from './hooks/useSelectedRun'

import { NavButton } from './components/shared/NavButton'
import { RunStatusBadge } from './components/shared/RunStatusBadge'
import { MetricRailCard } from './components/shared/MetricRailCard'
import { EmptyState } from './components/shared/EmptyState'
import { ErrorMessage } from './components/shared/ErrorMessage'

import { RunDashboardWorkspace } from './components/run-detail/RunDashboardWorkspace'
import { RunSettingsPanel } from './components/run-detail/RunSettingsPanel'
import { RunArtifactsPanel } from './components/run-detail/RunArtifactsPanel'
import { RunLogsPanel } from './components/run-detail/RunLogsPanel'

import './App.css'

// The selected run lives in the URL hash so a reload or a shared link keeps it.
function readRunIdFromHash(): string | null {
  const match = window.location.hash.match(/^#\/run\/(.+)$/)
  return match ? decodeURIComponent(match[1]) : null
}

function App() {
  // ─── Data layer ─────────────────────────────────────────────────────────────
  const { snapshot, setSnapshot, loading, error, refreshKey, liveTick, sseConnected, refresh } =
    useDashboardSnapshot()

  const [activePage, setActivePage] = useState<ActivePage>('dashboard')
  const [selectedRunId, setSelectedRunIdState] = useState<string | null>(readRunIdFromHash)

  const selectRun = (runId: string | null) => {
    setSelectedRunIdState(runId)
    const hash = runId ? `#/run/${encodeURIComponent(runId)}` : ''
    if (window.location.hash !== hash) window.history.replaceState(null, '', hash || window.location.pathname)
  }

  useEffect(() => {
    const onHashChange = () => setSelectedRunIdState(readRunIdFromHash())
    window.addEventListener('hashchange', onHashChange)
    return () => window.removeEventListener('hashchange', onHashChange)
  }, [])

  const { selectedRun, setSelectedRunDetail, config } = useSelectedRun(selectedRunId, refreshKey, snapshot)

  // Starred runs float to the top; within each group, newest first (server order).
  const runs = useMemo(() => {
    const list = snapshot?.runs ?? []
    return [...list].sort((a, b) => (b.manifest.starred ? 1 : 0) - (a.manifest.starred ? 1 : 0))
  }, [snapshot?.runs])

  // Pick a run when nothing (or a run that no longer exists) is selected:
  // the newest running run, otherwise the newest run.
  useEffect(() => {
    if (!snapshot) return
    const all = snapshot.runs ?? []
    if (selectedRunId && all.some((r) => r.manifest.run_id === selectedRunId)) return
    const fallback = all.find((r) => resolveRunStatus(r) === 'running') ?? all[0]
    selectRun(fallback?.manifest.run_id ?? null)
  }, [snapshot, selectedRunId])

  // ─── Derived values ─────────────────────────────────────────────────────────
  const trainSeries = useMemo(() => buildTrainSeries(selectedRun?.train_metrics ?? []), [selectedRun?.train_metrics])
  const valSeries = useMemo(() => buildValSeries(selectedRun?.val_metrics ?? []), [selectedRun?.val_metrics])
  const metricCards = useMemo(
    () => buildMetricCards(selectedRun, trainSeries, valSeries, selectedRun?.preview_history ?? []),
    [selectedRun, trainSeries, valSeries],
  )
  const modelOverviewEntries = useMemo(
    () => buildModelOverviewEntries(selectedRun?.model_overview),
    [selectedRun?.model_overview],
  )
  const hyperparameters = useMemo(() => buildHyperparameters(config), [config])
  const lossWeights = useMemo(() => buildLossWeights(config, selectedRun), [config, selectedRun])

  const runStatus = resolveRunStatus(selectedRun)
  const hasSelectedRun = Boolean(selectedRun?.manifest.run_id)

  // ─── Actions (viewer annotations only) ──────────────────────────────────────
  const handleHideRun = async (runId: string) => {
    setSnapshot((current) =>
      current ? { ...current, runs: (current.runs ?? []).filter((r) => r.manifest.run_id !== runId) } : current,
    )
    if (selectedRunId === runId) {
      setSelectedRunDetail(null)
      selectRun(null)
    }
    try {
      await fetch(`/api/runs/${encodeURIComponent(runId)}`, { method: 'DELETE' })
    } finally {
      await refresh()
    }
  }

  // ─── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b border-border bg-card/80 backdrop-blur-sm">
        <div className="flex flex-col gap-3 px-4 py-3 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex items-center gap-3 min-w-0">
            <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-primary/10">
              <Activity className="h-4 w-4 text-primary" />
            </div>

            <div className="space-y-1 min-w-0">
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-sm font-semibold tracking-tight truncate">
                  {selectedRun?.manifest.run_id ?? 'No run selected'}
                </span>
                {hasSelectedRun && <RunStatusBadge status={runStatus} />}
                {selectedRun?.manifest.lineage?.start_mode === 'load_weights' && (
                  <span className="rounded-full border border-violet-500/30 bg-violet-500/10 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-violet-500">
                    from ckpt
                  </span>
                )}
                <Badge
                  variant="secondary"
                  className={sseConnected ? 'bg-primary/10 text-primary' : 'bg-secondary text-muted-foreground'}
                >
                  {sseConnected ? 'Live' : 'Reconnecting'}
                </Badge>
              </div>
              {hasSelectedRun && (
                <div className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
                  <span className="flex items-center gap-1.5">
                    <Clock className="h-3.5 w-3.5" />
                    {formatDurationSince(selectedRun?.manifest.created_at, liveTick)}
                  </span>
                  <span className="flex items-center gap-1.5">
                    <Layers className="h-3.5 w-3.5" />
                    {formatProgress(selectedRun)}
                  </span>
                  <span>{selectedRun?.manifest.task_name ?? 'Unknown task'}</span>
                  {runStatus === 'stopped' && selectedRun?.manifest.last_seen && (
                    <span>last activity {formatDurationSince(selectedRun.manifest.last_seen, liveTick)} ago</span>
                  )}
                </div>
              )}
            </div>
          </div>

          <div className="flex items-center gap-0.5 rounded-md bg-secondary/50 p-0.5 self-start lg:self-auto">
            <NavButton active={activePage === 'dashboard'} icon={LayoutDashboard} label="Dashboard" onClick={() => setActivePage('dashboard')} />
            <NavButton active={activePage === 'settings'} icon={Settings2} label="Settings" onClick={() => setActivePage('settings')} />
            <NavButton active={activePage === 'artifacts'} icon={FolderOpen} label="Artifacts" onClick={() => setActivePage('artifacts')} />
            <NavButton active={activePage === 'logs'} icon={Clock} label="Logs" onClick={() => setActivePage('logs')} />
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="p-4">
        {error && (
          <div className="mb-4">
            <ErrorMessage text={`Failed to load runs: ${error}`} />
          </div>
        )}

        <div className="grid grid-cols-1 gap-4 xl:grid-cols-12">
          <div className="space-y-4 xl:col-span-9">
            {hasSelectedRun ? (
              activePage === 'settings' ? (
                <RunSettingsPanel
                  modelOverviewEntries={modelOverviewEntries}
                  hyperparameters={hyperparameters}
                  lossWeights={lossWeights}
                />
              ) : activePage === 'artifacts' ? (
                <RunArtifactsPanel artifacts={selectedRun?.artifacts ?? []} />
              ) : activePage === 'logs' ? (
                <RunLogsPanel runId={selectedRunId} refreshKey={refreshKey} />
              ) : (
                <RunDashboardWorkspace
                  trainSeries={trainSeries}
                  valSeries={valSeries}
                  selectedRun={selectedRun}
                  refreshKey={refreshKey}
                />
              )
            ) : (
              <Card className="border-border/50">
                <PanelCardContent className="space-y-3 py-8 text-center text-sm text-muted-foreground">
                  {loading ? (
                    <p>Loading runs…</p>
                  ) : (
                    <>
                      <p>No runs found in {(snapshot?.roots ?? []).join(', ') || 'the run folders'}.</p>
                      <p>
                        Start one from a terminal, e.g.{' '}
                        <code className="rounded bg-secondary/50 px-1.5 py-0.5 font-mono text-xs">
                          cd training &amp;&amp; python train.py --config configs/train_512.yaml
                        </code>
                        . It shows up here on its own.
                      </p>
                    </>
                  )}
                </PanelCardContent>
              </Card>
            )}
          </div>

          {/* Sidebar */}
          <div className="space-y-3 xl:col-span-3">
            {hasSelectedRun && activePage === 'dashboard' && (
              <Card className="border-border/50">
                <PanelCardHeader>
                  <PanelCardTitle className="text-xs">
                    <Zap className="h-3.5 w-3.5 text-muted-foreground" />
                    Metrics
                  </PanelCardTitle>
                </PanelCardHeader>
                <PanelCardContent className="space-y-2">
                  {metricCards.map((metric) => (
                    <MetricRailCard key={metric.label} metric={metric} />
                  ))}
                </PanelCardContent>
              </Card>
            )}

            <Card className="border-border/50">
              <PanelCardHeader>
                <PanelCardTitle className="text-xs">
                  <History className="h-3.5 w-3.5 text-muted-foreground" />
                  Runs
                </PanelCardTitle>
              </PanelCardHeader>
              <PanelCardContent className="space-y-1.5">
                {runs.length ? (
                  runs.map((run) => (
                    <RunListItem
                      key={run.manifest.run_id}
                      run={run}
                      selected={run.manifest.run_id === selectedRun?.manifest.run_id}
                      liveTick={liveTick}
                      onSelect={() => selectRun(run.manifest.run_id)}
                      onHide={() => { void handleHideRun(run.manifest.run_id) }}
                      onStarChange={(starred, note) => {
                        setSnapshot((current) => {
                          if (!current) return current
                          return {
                            ...current,
                            runs: (current.runs ?? []).map((r) =>
                              r.manifest.run_id === run.manifest.run_id
                                ? { ...r, manifest: { ...r.manifest, starred, note } }
                                : r,
                            ),
                          }
                        })
                      }}
                    />
                  ))
                ) : (
                  <EmptyState text={loading ? 'Loading runs…' : 'No runs found yet.'} />
                )}
              </PanelCardContent>
            </Card>
          </div>
        </div>
      </main>
    </div>
  )
}

// ── RunListItem ───────────────────────────────────────────────────────────────

function RunListItem({
  run,
  selected,
  liveTick,
  onSelect,
  onHide,
  onStarChange,
}: {
  run: RunRecord
  selected: boolean
  liveTick: number
  onSelect: () => void
  onHide: () => void
  onStarChange: (starred: boolean, note: string) => void
}) {
  return (
    <div
      role="button"
      tabIndex={0}
      onClick={onSelect}
      onKeyDown={(e) => e.key === 'Enter' && onSelect()}
      className={[
        'group w-full cursor-pointer rounded-md border px-2.5 py-2 text-left transition-colors',
        selected ? 'border-primary/30 bg-primary/10' : 'border-border/40 bg-secondary/15 hover:bg-secondary/30',
      ].join(' ')}
    >
      <div className="mb-1 flex items-center justify-between gap-2">
        <span className="truncate font-mono text-xs text-muted-foreground">{run.manifest.run_id}</span>
        <div className="flex items-center gap-1.5 shrink-0">
          <DeleteRunButton runId={run.manifest.run_id} onDelete={onHide} />
          <StarButton run={run} onStarChange={onStarChange} />
          <RunStatusBadge status={resolveRunStatus(run)} compact />
        </div>
      </div>
      <div className="text-sm font-medium truncate">{buildRunMetricSummary(run)}</div>
      {run.manifest.starred && run.manifest.note && (
        <div className="mt-1 truncate text-xs text-amber-400/80 italic">{run.manifest.note}</div>
      )}
      <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-xs text-muted-foreground">
        <span>{formatProgress(run)}</span>
        <span>{formatDurationSince(run.manifest.created_at, liveTick)}</span>
        <span>{run.manifest.task_name ?? 'Unknown task'}</span>
        {run.manifest.lineage?.start_mode === 'load_weights' && (
          <span className="rounded-full border border-violet-500/30 bg-violet-500/10 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-violet-500">
            from ckpt
          </span>
        )}
      </div>
    </div>
  )
}

// ── StarButton ────────────────────────────────────────────────────────────────

function StarButton({
  run,
  onStarChange,
}: {
  run: RunRecord
  onStarChange?: (starred: boolean, note: string) => void
}) {
  const [saving, setSaving] = useState(false)
  const [noteOpen, setNoteOpen] = useState(false)
  const [noteValue, setNoteValue] = useState(run.manifest.note ?? '')
  const isStarred = run.manifest.starred ?? false

  function toggleStar(e: React.MouseEvent) {
    e.stopPropagation()
    if (saving) return
    if (isStarred) {
      // unstar immediately (keep note in case they re-star)
      const note = run.manifest.note ?? ''
      onStarChange?.(false, note)
      patchStar(run.manifest.run_id, false, note, setSaving)
    } else {
      // open note dialog before starring
      setNoteValue(run.manifest.note ?? '')
      setNoteOpen(true)
    }
  }

  function confirmStar(e: React.FormEvent) {
    e.preventDefault()
    e.stopPropagation()
    setNoteOpen(false)
    onStarChange?.(true, noteValue)
    patchStar(run.manifest.run_id, true, noteValue, setSaving)
  }

  return (
    <>
      <button
        type="button"
        title={isStarred ? 'Remove from favorites' : 'Add to favorites'}
        onClick={toggleStar}
        disabled={saving}
        className={[
          'rounded p-0.5 transition-colors',
          isStarred
            ? 'text-amber-400 hover:text-amber-300'
            : 'text-muted-foreground/40 hover:text-amber-400 opacity-0 group-hover:opacity-100',
        ].join(' ')}
      >
        <Star className={`h-3.5 w-3.5 ${isStarred ? 'fill-amber-400' : ''}`} />
      </button>

      {noteOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-background/70 backdrop-blur-sm"
          onClick={(e) => { e.stopPropagation(); setNoteOpen(false) }}
        >
          <form
            className="w-80 rounded-lg border border-border bg-card p-4 shadow-xl"
            onClick={(e) => e.stopPropagation()}
            onSubmit={confirmStar}
          >
            <p className="mb-3 text-sm font-medium">Add a note (optional)</p>
            <input
              autoFocus
              type="text"
              maxLength={200}
              value={noteValue}
              onChange={(e) => setNoteValue(e.target.value)}
              placeholder="e.g. best PSNR so far"
              className="mb-3 w-full rounded-md border border-border bg-secondary/30 px-3 py-1.5 text-sm outline-none focus:border-primary/50"
            />
            <div className="flex justify-end gap-2">
              <button
                type="button"
                onClick={() => setNoteOpen(false)}
                className="rounded px-3 py-1.5 text-xs text-muted-foreground hover:bg-secondary/50"
              >
                Cancel
              </button>
              <button
                type="submit"
                className="rounded bg-amber-500/20 px-3 py-1.5 text-xs font-medium text-amber-400 hover:bg-amber-500/30"
              >
                <Star className="mr-1 inline h-3 w-3 fill-amber-400" />
                Star
              </button>
            </div>
          </form>
        </div>
      )}
    </>
  )
}

function patchStar(runId: string, starred: boolean, note: string, setSaving: (v: boolean) => void) {
  setSaving(true)
  fetch(`/api/runs/${encodeURIComponent(runId)}/star`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ starred, note }),
  }).finally(() => setSaving(false))
}

// ── DeleteRunButton ───────────────────────────────────────────────────────────

function DeleteRunButton({ runId, onDelete }: { runId: string; onDelete: () => void }) {
  const [confirming, setConfirming] = useState(false)

  function handleClick(e: React.MouseEvent) {
    e.stopPropagation()
    setConfirming(true)
  }

  function handleConfirm(e: React.MouseEvent) {
    e.stopPropagation()
    setConfirming(false)
    onDelete()
  }

  function handleCancel(e: React.MouseEvent) {
    e.stopPropagation()
    setConfirming(false)
  }

  if (confirming) {
    return (
      // eslint-disable-next-line jsx-a11y/click-events-have-key-events, jsx-a11y/no-static-element-interactions
      <span
        className="flex items-center gap-0.5"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          type="button"
          onClick={handleConfirm}
          className="rounded px-1.5 py-0.5 text-[10px] font-medium text-red-400 hover:bg-red-500/15 transition-colors"
        >
          Hide
        </button>
        <button
          type="button"
          onClick={handleCancel}
          className="rounded px-1.5 py-0.5 text-[10px] text-muted-foreground hover:bg-secondary/50 transition-colors"
        >
          ×
        </button>
      </span>
    )
  }

  return (
    <button
      type="button"
      title={`Hide ${runId} from the list (files stay on disk)`}
      onClick={handleClick}
      className="rounded p-0.5 text-muted-foreground/40 opacity-0 transition-colors hover:text-red-400 group-hover:opacity-100"
    >
      <Trash2 className="h-3.5 w-3.5" />
    </button>
  )
}

export default App
