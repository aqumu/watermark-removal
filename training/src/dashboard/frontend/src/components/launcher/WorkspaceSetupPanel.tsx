import { useEffect, useState } from 'react'
import { parse as parseYaml } from 'yaml'
import { ChevronLeft, FolderGit2, FolderOpen, PlayCircle, Rocket, Star } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, SectionCardContent, SectionCardHeader, SectionCardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { useLauncherContext } from '../../context/LauncherContext'
import type { CheckpointRecord, DraftTaskType } from '../../types'
import { buildCheckpointLabel, buildRunMetricSummary, buildRunSummary, resolveRunStatus } from '../../lib/run-utils'
import { getNestedValue, normalizeDraftConfig } from '../../lib/draft-utils'
import { formatDurationSince } from '../../lib/format'
import { ErrorMessage } from '../shared/ErrorMessage'
import { EmptyState } from '../shared/EmptyState'
import { RunStatusBadge } from '../shared/RunStatusBadge'
import { DraftQuickSetupEditor } from './DraftQuickSetupEditor'
import { LauncherColumnHint } from './LauncherColumnHint'

// Phases inside the "Start From Checkpoint" column
type CkptPhase = 'idle' | 'picking' | 'editing'

export function WorkspaceSetupPanel() {
  const {
    launcherSelection,
    continueSelection,
    continueFamilies,
    orchestration,
    resumableFamilies,
    canResumeAnotherRun,
    lifecycleActionLoading,
    startLaunching,
    startLaunchError,
    draftNameError,
    draftNameExists,
    setDraftConfig,
    setDraftTaskType,
    setDraftFamilyName,
    onSelectFresh,
    onSelectCheckpoint,
    onStartSelected,
    onActivateFlow,
    onInspectRun,
    onSelectContinueFamily,
    onBackToContinueFamilies,
    onResumeRun,
    onGoBack,
  } = useLauncherContext()

  const [ckptPhase, setCkptPhase] = useState<CkptPhase>('idle')
  const [ckptStep, setCkptStep] = useState(1)
  const [ckptLoadingConfig, setCkptLoadingConfig] = useState(false)
  const [ckptConfigError, setCkptConfigError] = useState<string | null>(null)

  // Type-based checkpoint picker (panel 3 local state — independent of context startPointOptions)
  const [ckptTaskType, setCkptTaskType] = useState<DraftTaskType | null>(null)
  const [ckptTypedOptions, setCkptTypedOptions] = useState<CheckpointRecord[]>([])
  const [ckptTypeLoading, setCkptTypeLoading] = useState(false)
  const [ckptTypeError, setCkptTypeError] = useState<string | null>(null)

  const activeFlow = launcherSelection.kind === 'draft' ? launcherSelection.flow : null
  const selectedContinueFamily =
    continueSelection.kind === 'family'
      ? continueFamilies.find((f) => f.familyName === continueSelection.familyName) ?? null
      : null

  const canStart = !draftNameError && !draftNameExists

  // Reset checkpoint phase when the flow is deactivated (e.g. user clicks Reset)
  useEffect(() => {
    if (activeFlow !== 'checkpoint') {
      setCkptPhase('idle')
      setCkptStep(1)
      setCkptConfigError(null)
      setCkptTaskType(null)
      setCkptTypedOptions([])
      setCkptTypeError(null)
    }
  }, [activeFlow])

  // Fetch checkpoints for the selected type (sorted most-recent first)
  useEffect(() => {
    if (!ckptTaskType || ckptPhase !== 'picking') return
    let cancelled = false
    setCkptTypeLoading(true)
    setCkptTypeError(null)
    fetch(`/api/checkpoints?task_type=${encodeURIComponent(ckptTaskType)}`)
      .then((r) => {
        if (!r.ok) throw new Error(`checkpoints ${r.status}`)
        return r.json() as Promise<CheckpointRecord[]>
      })
      .then((data) => {
        if (cancelled) return
        // Sort: highest epoch first, then 'best' kind first within same epoch
        const sorted = [...data].sort((a, b) => {
          const ae = a.epoch ?? -1
          const be = b.epoch ?? -1
          if (be !== ae) return be - ae
          const ak = a.kind === 'best' ? 1 : 0
          const bk = b.kind === 'best' ? 1 : 0
          if (bk !== ak) return bk - ak
          return (b.run_id ?? '').localeCompare(a.run_id ?? '')
        })
        setCkptTypedOptions(sorted)
      })
      .catch((err) => {
        if (!cancelled) setCkptTypeError(err instanceof Error ? err.message : 'Failed to load checkpoints')
      })
      .finally(() => { if (!cancelled) setCkptTypeLoading(false) })
    return () => { cancelled = true }
  }, [ckptTaskType, ckptPhase])

  async function handleCheckpointSelect(checkpoint: CheckpointRecord) {
    setCkptLoadingConfig(true)
    setCkptConfigError(null)
    try {
      const text = await fetch(`/runs/${encodeURIComponent(checkpoint.run_id)}/meta/config.yaml`).then((r) => {
        if (!r.ok) throw new Error(`config ${r.status}`)
        return r.text()
      })
      const parsed = parseYaml(text) as Record<string, unknown>
      // Align draft task type with the type-button the user picked, otherwise
      // the editor keeps rendering a stale task (e.g. segmentation UI for a
      // removal checkpoint) because draftTaskType survives across flows.
      if (ckptTaskType) setDraftTaskType(ckptTaskType)
      const normalized = normalizeDraftConfig(parsed)
      setDraftConfig(normalized)
      // Suggest a fresh family name based on the source run's actual task +
      // resolution. Without this, draftFamilyName keeps whatever stale value
      // the user had before entering the checkpoint flow (e.g. "segmentation"
      // even when loading a 512×256 removal checkpoint).
      if (ckptTaskType) {
        const w = getNestedValue(normalized, ['dataset', 'image_width'])
        const s = getNestedValue(normalized, ['dataset', 'image_size'])
        const dim = typeof w === 'number' ? w : typeof s === 'number' ? s : null
        const resSuffix = dim === 512 ? '512' : dim === 256 ? '256' : null
        setDraftFamilyName(resSuffix ? `${ckptTaskType}_${resSuffix}_v2` : `${ckptTaskType}_v2`)
      }
      onSelectCheckpoint(checkpoint)
      setCkptPhase('editing')
      setCkptStep(1)
    } catch (err) {
      setCkptConfigError(err instanceof Error ? err.message : 'Failed to load source config')
    } finally {
      setCkptLoadingConfig(false)
    }
  }

  function handleCkptBack() {
    onSelectFresh()
    setCkptPhase('picking')
    setCkptStep(1)
    setCkptConfigError(null)
  }

  function handleCkptTypeBack() {
    setCkptTaskType(null)
    setCkptTypedOptions([])
    setCkptTypeError(null)
  }

  return (
    <section className="space-y-4 animate-in-up">
      {/* Header */}
      <div className="rounded-xl border border-border/50 bg-card px-5 py-4">
        <div className="flex flex-col gap-2 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <p className="label-caps">Launcher</p>
            <h2 className="mt-1 text-xl font-semibold tracking-tight">Choose how to enter training</h2>
          </div>
          {launcherSelection.kind === 'draft' && (
            <Button variant="outline" size="sm" className="gap-2 self-start lg:self-auto" onClick={onGoBack}>
              <ChevronLeft className="h-3.5 w-3.5" />
              Reset
            </Button>
          )}
        </div>
      </div>

      <div className="grid gap-4 xl:grid-cols-3">
        {/* ── Column 1: Train From Scratch ────────────────────────────── */}
        <Card className={`min-h-[780px] border-border/50 ${activeFlow === 'scratch' ? 'ring-1 ring-primary/30' : ''}`}>
          <SectionCardHeader>
            <SectionCardTitle>
              <PlayCircle className="h-4 w-4 text-muted-foreground" />
              Train From Scratch
            </SectionCardTitle>
          </SectionCardHeader>
          <SectionCardContent className="space-y-4">
            <p className="text-xs text-muted-foreground">
              Start a new family with no checkpoint. Shortest path for a clean run lineage.
            </p>

            {activeFlow !== 'scratch' && (
              <Button className="w-full gap-2" onClick={() => onActivateFlow('scratch')}>
                <Rocket className="h-4 w-4" />
                Configure scratch run
              </Button>
            )}

            {activeFlow === 'scratch' && (
              <DraftQuickSetupEditor flow="scratch" />
            )}

            {activeFlow !== 'scratch' && (
              <LauncherColumnHint
                title="Fastest path"
                text="Select a template, name the family, review key parameters, and start."
              />
            )}
          </SectionCardContent>
        </Card>

        {/* ── Column 2: Continue Existing ─────────────────────────────── */}
        <Card className="min-h-[780px] border-border/50">
          <SectionCardHeader>
            <div className="flex items-center justify-between gap-3">
              <SectionCardTitle>
                <FolderGit2 className="h-4 w-4 text-muted-foreground" />
                {selectedContinueFamily ? selectedContinueFamily.familyName : 'Continue Existing'}
              </SectionCardTitle>
              {selectedContinueFamily && (
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="h-7 gap-1.5 px-2 text-xs"
                  onClick={onBackToContinueFamilies}
                >
                  <ChevronLeft className="h-3.5 w-3.5" />
                  Back
                </Button>
              )}
            </div>
          </SectionCardHeader>
          <SectionCardContent className="space-y-3">
            {selectedContinueFamily ? (
              <>
                <p className="text-xs text-muted-foreground">
                  Inspect runs inside this family. Training only resumes after an explicit action.
                </p>
                <div className="space-y-2">
                  {selectedContinueFamily.runs.map((run) => {
                    const status = resolveRunStatus(run, orchestration)
                    const canResume =
                      status === 'paused' &&
                      resumableFamilies.has(run.manifest.project_run ?? run.manifest.run_id)
                    return (
                      <button
                        key={run.manifest.run_id}
                        type="button"
                        onClick={() => onInspectRun(run.manifest.run_id)}
                        className="w-full rounded-lg border border-border/40 bg-secondary/15 px-3 py-3 text-left transition-colors hover:bg-secondary/30"
                      >
                        <div className="flex items-center justify-between gap-2">
                          <p className="truncate font-mono text-xs text-muted-foreground">
                            {run.manifest.run_id}
                          </p>
                          <div className="flex items-center gap-2">
                            {run.manifest.starred && (
                              <Star className="h-3.5 w-3.5 shrink-0 fill-amber-400 text-amber-400" />
                            )}
                            {canResume && (
                              <Button
                                type="button"
                                size="sm"
                                className="h-6 gap-1 px-2 text-xs"
                                disabled={!canResumeAnotherRun || lifecycleActionLoading !== null}
                                onClick={(e) => { e.stopPropagation(); void onResumeRun(run) }}
                              >
                                <Rocket className="h-3 w-3" />
                                {lifecycleActionLoading === 'resume' ? 'Continuing…' : 'Continue'}
                              </Button>
                            )}
                            <RunStatusBadge status={status} compact />
                          </div>
                        </div>
                        {run.manifest.starred && run.manifest.note && (
                          <p className="mt-1 truncate text-xs text-amber-400/80 italic">{run.manifest.note}</p>
                        )}
                        <div className="mt-1 flex items-center justify-between gap-3 text-xs">
                          <span className="truncate">{buildRunMetricSummary(run)}</span>
                          <span className="truncate text-muted-foreground">{buildRunSummary(run)}</span>
                        </div>
                        <div className="mt-2 flex items-center justify-between gap-3 text-xs text-muted-foreground">
                          <span>{run.manifest.task_name ?? 'Unknown task'}</span>
                          <span>{formatDurationSince(run.manifest.created_at, Date.now())}</span>
                        </div>
                      </button>
                    )
                  })}
                </div>
              </>
            ) : (
              <>
                <p className="text-xs text-muted-foreground">
                  Browse by family. Opening a family only changes this column and does not commit the launcher.
                </p>
                <div className="space-y-2">
                  {continueFamilies.length ? (
                    continueFamilies.map((family) => {
                      const hasStarred = family.runs.some((r) => r.manifest.starred)
                      return (
                        <button
                          key={family.familyName}
                          type="button"
                          onClick={() => onSelectContinueFamily(family.familyName)}
                          className="w-full rounded-lg border border-border/40 bg-secondary/15 px-3 py-3 text-left transition-colors hover:bg-secondary/30"
                        >
                          <div className="flex items-center justify-between gap-2">
                            <div className="flex min-w-0 items-center gap-1.5">
                              {hasStarred && (
                                <Star className="h-3.5 w-3.5 shrink-0 fill-amber-400 text-amber-400" />
                              )}
                              <p className="truncate text-sm font-medium">{family.familyName}</p>
                            </div>
                            <RunStatusBadge status={family.latestStatus} compact />
                          </div>
                          <p className="mt-1 text-xs text-muted-foreground">
                            {family.runs.length} run{family.runs.length === 1 ? '' : 's'} · {family.taskLabel}
                          </p>
                          <div className="mt-2 flex items-center justify-between text-xs text-muted-foreground">
                            <span>{buildRunSummary(family.latestRun)}</span>
                            <span>{formatDurationSince(family.latestRun.manifest.created_at, Date.now())}</span>
                          </div>
                          <div className="mt-2 flex items-center justify-between gap-3 text-xs">
                            <span className="truncate">{family.bestMetricSummary}</span>
                            <span className="truncate font-mono text-muted-foreground">
                              {family.latestRun.manifest.run_id}
                            </span>
                          </div>
                        </button>
                      )
                    })
                  ) : (
                    <EmptyState text="No run families found yet." />
                  )}
                </div>
              </>
            )}
          </SectionCardContent>
        </Card>

        {/* ── Column 3: Start From Checkpoint ─────────────────────────── */}
        <Card className={`min-h-[780px] border-border/50 ${activeFlow === 'checkpoint' ? 'ring-1 ring-primary/30' : ''}`}>
          <SectionCardHeader>
            <SectionCardTitle>
              <FolderOpen className="h-4 w-4 text-muted-foreground" />
              Start From Checkpoint
            </SectionCardTitle>
          </SectionCardHeader>
          <SectionCardContent className="space-y-4">

            {/* ── idle: entry button ─────────────────────────────────── */}
            {ckptPhase === 'idle' && (
              <>
                <p className="text-xs text-muted-foreground">
                  Pick a compatible checkpoint, carry over its settings, and start a new family with tweaked loss weights or epoch count.
                </p>
                <Button
                  className="w-full gap-2"
                  variant="secondary"
                  onClick={() => { onActivateFlow('checkpoint'); setCkptPhase('picking') }}
                >
                  <FolderGit2 className="h-4 w-4" />
                  Choose a checkpoint
                </Button>
                <LauncherColumnHint
                  title="What happens"
                  text="Loads model weights only — optimizer and epoch counter start fresh. Use Column 2 to resume a paused run instead."
                />
              </>
            )}

            {/* ── picking: type selector → checkpoint list ───────────── */}
            {ckptPhase === 'picking' && (
              <>
                {!ckptTaskType ? (
                  /* Step 1 — choose a checkpoint type */
                  <>
                    <p className="text-xs text-muted-foreground">
                      Choose the type of model you want to initialize from, then pick a checkpoint.
                    </p>
                    <div className="space-y-2">
                      {(
                        [
                          { type: 'removal', label: 'Removal', desc: 'Watermark removal models' },
                          { type: 'restoration', label: 'Restoration', desc: 'Image restoration models' },
                          { type: 'segmentation', label: 'Segmentation', desc: 'Mask segmentation models' },
                        ] as const
                      ).map(({ type, label, desc }) => (
                        <button
                          key={type}
                          type="button"
                          onClick={() => setCkptTaskType(type)}
                          className="w-full rounded-lg border border-border/40 bg-background/40 px-3 py-2.5 text-left transition-colors hover:border-primary/40 hover:bg-primary/5"
                        >
                          <p className="text-sm font-medium">{label}</p>
                          <p className="text-xs text-muted-foreground">{desc}</p>
                        </button>
                      ))}
                    </div>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="w-full gap-1.5 text-muted-foreground"
                      onClick={() => { setCkptPhase('idle'); onActivateFlow('scratch') }}
                    >
                      <ChevronLeft className="h-3.5 w-3.5" />
                      Cancel
                    </Button>
                  </>
                ) : (
                  /* Step 2 — checkpoint list for the chosen type */
                  <>
                    <div className="flex items-center gap-1.5">
                      <button
                        type="button"
                        onClick={handleCkptTypeBack}
                        className="flex items-center gap-1 text-xs text-muted-foreground transition-colors hover:text-foreground"
                      >
                        <ChevronLeft className="h-3.5 w-3.5" />
                        <span className="capitalize">{ckptTaskType}</span>
                      </button>
                    </div>

                    <p className="text-xs text-muted-foreground">
                      Select a checkpoint — settings will be pre-filled from that run.
                    </p>

                    {ckptConfigError && <ErrorMessage text={ckptConfigError} />}

                    {ckptTypeLoading || ckptLoadingConfig ? (
                      <EmptyState text={ckptLoadingConfig ? 'Loading source config…' : 'Loading checkpoints…'} />
                    ) : ckptTypeError ? (
                      <ErrorMessage text={ckptTypeError} />
                    ) : ckptTypedOptions.length ? (
                      <div className="space-y-1.5">
                        {ckptTypedOptions.map((checkpoint) => (
                          <button
                            key={checkpoint.checkpoint_path}
                            type="button"
                            disabled={ckptLoadingConfig}
                            onClick={() => void handleCheckpointSelect(checkpoint)}
                            className="w-full rounded-md border border-border/40 bg-background/40 px-3 py-2 text-left transition-colors hover:border-primary/40 hover:bg-primary/5 disabled:pointer-events-none disabled:opacity-50"
                          >
                            <div className="mb-1 flex items-center justify-between gap-2">
                              <p className="truncate text-sm font-medium">{checkpoint.checkpoint_name}</p>
                              <Badge
                                variant={checkpoint.kind === 'best' ? 'default' : 'secondary'}
                                className="shrink-0 text-xs"
                              >
                                {checkpoint.kind ?? 'checkpoint'}
                              </Badge>
                            </div>
                            <p className="truncate text-xs text-muted-foreground">
                              {buildCheckpointLabel(checkpoint)}
                            </p>
                            <p className="truncate font-mono text-xs text-muted-foreground">
                              {checkpoint.run_id}
                            </p>
                          </button>
                        ))}
                      </div>
                    ) : (
                      <EmptyState text={`No ${ckptTaskType} checkpoints found. Train a run first.`} />
                    )}

                    <Button
                      variant="ghost"
                      size="sm"
                      className="w-full gap-1.5 text-muted-foreground"
                      onClick={handleCkptTypeBack}
                    >
                      <ChevronLeft className="h-3.5 w-3.5" />
                      Back to type selection
                    </Button>
                  </>
                )}
              </>
            )}

            {/* ── editing: editor + start button ─────────────────────── */}
            {ckptPhase === 'editing' && (
              <>
                <DraftQuickSetupEditor flow="checkpoint" onStepChange={setCkptStep} />

                {ckptStep === 2 && (
                  <>
                    {startLaunchError && <ErrorMessage text={startLaunchError} />}
                    <div className="pt-1 space-y-2">
                      <Button
                        className="w-full gap-2"
                        disabled={!canStart || startLaunching}
                        onClick={() => void onStartSelected()}
                      >
                        <Rocket className="h-4 w-4" />
                        {startLaunching ? 'Starting…' : 'Load weights & start'}
                      </Button>
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        className="w-full gap-1.5 text-muted-foreground"
                        onClick={handleCkptBack}
                      >
                        <ChevronLeft className="h-3.5 w-3.5" />
                        Back to checkpoint list
                      </Button>
                    </div>
                  </>
                )}
              </>
            )}

          </SectionCardContent>
        </Card>
      </div>
    </section>
  )
}
