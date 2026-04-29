import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Activity,
  ChevronLeft,
  Clock,
  FolderOpen,
  History,
  LayoutDashboard,
  Layers,
  Rocket,
  Settings2,
  Star,
  Trash2,
  XCircle,
  Zap,
} from 'lucide-react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, PanelCardContent, PanelCardHeader, PanelCardTitle } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'

import type {
  ActivePage,
  BrowsingContext,
  CheckpointRecord,
  ContinueBrowseSelection,
  DraftRecord,
  DraftTaskType,
  DraftWorkspace,
  LauncherSelection,
  RunRecord,
  UXState,
} from './types'
import { FALLBACK_TEMPLATES } from './lib/constants'
import { formatApiError, formatDurationSince, humanizeKey } from './lib/format'
import {
  buildCheckpointLabel,
  buildContinueFamilies,
  buildHyperparameters,
  buildLossWeights,
  buildMetricCards,
  buildModelOverviewEntries,
  buildRunMetricSummary,
  buildTrainSeries,
  buildValSeries,
  extractStorePath,
  formatStep,
  isRemovalConfig,
  resolveRunStatus,
} from './lib/run-utils'
import {
  collectDraftFields,
  normalizeDraftConfig,
  pickScratchQuickFields,
  sanitizeRunFamilyName,
  validateRunFamilyName,
} from './lib/draft-utils'

import { useDashboardSnapshot } from './hooks/useDashboardSnapshot'
import { useSelectedRun } from './hooks/useSelectedRun'
import { LauncherContext } from './context/LauncherContext'
import type { LauncherContextValue } from './context/LauncherContext'

import { NavButton } from './components/shared/NavButton'
import { RunStatusBadge } from './components/shared/RunStatusBadge'
import { WorkspaceModeBadge } from './components/shared/WorkspaceModeBadge'
import { MetricRailCard } from './components/shared/MetricRailCard'
import { EmptyState } from './components/shared/EmptyState'
import { ErrorMessage } from './components/shared/ErrorMessage'

import { RunDashboardWorkspace } from './components/run-detail/RunDashboardWorkspace'
import { RunSettingsPanel } from './components/run-detail/RunSettingsPanel'
import { RunArtifactsPanel } from './components/run-detail/RunArtifactsPanel'
import { RunLogsPanel } from './components/run-detail/RunLogsPanel'
import { WorkspaceSetupPanel } from './components/launcher/WorkspaceSetupPanel'

import './App.css'

function App() {
  // ─── Data layer ─────────────────────────────────────────────────────────────
  const { snapshot, setSnapshot, loading, error, refreshKey, liveTick, sseConnected, manualRefresh } =
    useDashboardSnapshot()

  // ─── Navigation state ───────────────────────────────────────────────────────
  const [uxState, setUxState] = useState<UXState>({
    mode: 'browsing',
    launcher: { kind: 'home' },
    continueSelection: { kind: 'families' },
  })
  const [activePage, setActivePage] = useState<ActivePage>('dashboard')
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null)
  const [autoNavigated, setAutoNavigated] = useState(false)

  // ─── Selected run data ──────────────────────────────────────────────────────
  const { selectedRun, setSelectedRunDetail, config } = useSelectedRun(
    selectedRunId,
    refreshKey,
    snapshot,
  )

  // ─── Draft state ────────────────────────────────────────────────────────────
  const [draftTaskType, setDraftTaskType] = useState<DraftTaskType>('removal')
  const [draftFamilyName, setDraftFamilyName] = useState('removal_256')
  const [draftTemplateId, setDraftTemplateId] = useState('blank')
  const [draftConfig, setDraftConfig] = useState<Record<string, unknown>>({})
  const lastAppliedTemplateId = useRef<string | null>(null)
  const [draftSaving, setDraftSaving] = useState(false)
  const [draftSaveError, setDraftSaveError] = useState<string | null>(null)
  const [draftDeleteError, setDraftDeleteError] = useState<string | null>(null)
  const [deletingDraftName, setDeletingDraftName] = useState<string | null>(null)

  // ─── Lifecycle / start-point state ──────────────────────────────────────────
  const [startPointLoading, setStartPointLoading] = useState(false)
  const [startPointError, setStartPointError] = useState<string | null>(null)
  const [startPointOptions, setStartPointOptions] = useState<CheckpointRecord[]>([])
  const [startPointSelection, setStartPointSelection] = useState<
    { kind: 'fresh' } | { kind: 'checkpoint'; checkpoint_path: string; checkpoint_name: string; run_id: string; label: string }
  >({ kind: 'fresh' })
  const [startLaunching, setStartLaunching] = useState(false)
  const [startLaunchError, setStartLaunchError] = useState<string | null>(null)
  const [lifecycleActionLoading, setLifecycleActionLoading] = useState<'pause' | 'resume' | null>(null)
  const [lifecycleActionError, setLifecycleActionError] = useState<string | null>(null)

  // ─── Derived values ─────────────────────────────────────────────────────────
  const availableTemplates = useMemo(() => {
    const backendTemplates = snapshot?.training_manager?.templates ?? []
    const mapped = backendTemplates.map((t) => ({
      id: t.template_id,
      label: t.label,
      description: `${humanizeKey(t.task_type)} template from ${t.path.replace(/.*[\\/]/, '')}`,
      taskType: t.task_type,
      suggestedFamilyName:
        t.task_type === 'segmentation'
          ? 'segmentation'
          : t.task_type === 'restoration'
            ? t.template_id.replace(/^train_/, 'restoration_')
            : t.template_id.replace(/^train_/, 'removal_'),
      data: t.data,
    }))
    return mapped.length ? mapped : FALLBACK_TEMPLATES
  }, [snapshot?.training_manager?.templates])

  const savedDrafts = snapshot?.training_manager?.drafts ?? []
  const existingDraftNames = useMemo(
    () => new Set((snapshot?.training_manager?.drafts ?? []).map((d) => d.family_name)),
    [snapshot?.training_manager?.drafts],
  )
  const existingRunFamilies = useMemo(
    () => new Set((snapshot?.runs ?? []).map((r) => r.manifest.project_run ?? r.manifest.run_id)),
    [snapshot?.runs],
  )
  const continueFamilies = useMemo(
    () => buildContinueFamilies(snapshot?.runs ?? [], snapshot?.orchestration),
    [snapshot?.runs, snapshot?.orchestration],
  )

  const currentBrowsingContext = useMemo<BrowsingContext>(() => {
    if (uxState.mode === 'browsing') return { launcher: uxState.launcher, continueSelection: uxState.continueSelection }
    if (uxState.mode === 'inspect') return uxState.returnTo
    return { launcher: { kind: 'home' }, continueSelection: { kind: 'families' } }
  }, [uxState])

  const launcherSelection: LauncherSelection = currentBrowsingContext.launcher
  const continueSelection: ContinueBrowseSelection = currentBrowsingContext.continueSelection
  const draftWorkspace: DraftWorkspace | null = launcherSelection.kind === 'draft' ? launcherSelection : null

  const sanitizedDraftFamilyName = sanitizeRunFamilyName(draftFamilyName)
  const draftNameError = validateRunFamilyName(draftFamilyName)
  const draftNameExists = sanitizedDraftFamilyName ? existingDraftNames.has(sanitizedDraftFamilyName) : false
  const draftFamilyExists = sanitizedDraftFamilyName ? existingRunFamilies.has(sanitizedDraftFamilyName) : false
  const resumableFamilies = existingDraftNames
  const orchestrationPhase = snapshot?.orchestration?.phase ?? 'idle'
  const canResumeAnotherRun =
    orchestrationPhase !== 'running' && orchestrationPhase !== 'starting' && orchestrationPhase !== 'pausing'

  const draftFields = useMemo(() => collectDraftFields(draftConfig), [draftConfig])
  const scratchQuickFields = useMemo(() => pickScratchQuickFields(draftFields), [draftFields])

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
  const runStorePath = useMemo(() => extractStorePath(config), [config])
  const runIsRemoval = useMemo(() => isRemovalConfig(config), [config])

  const runStatus = draftWorkspace ? 'unknown' : resolveRunStatus(selectedRun, snapshot?.orchestration)
  const runModeLabel =
    uxState.mode === 'inspect'
      ? runStatus === 'paused' || runStatus === 'failed' ? 'Resume available' : 'Inspect mode'
      : uxState.mode === 'active' ? 'Active workspace'
      : draftWorkspace ? 'Draft setup'
      : null
  const workspaceModeTone =
    uxState.mode === 'inspect'
      ? runStatus === 'paused' || runStatus === 'failed' ? 'resume' : 'inspect'
      : uxState.mode === 'active' ? 'active'
      : 'draft'

  const stepLabel = formatStep(selectedRun)
  const durationLabel = formatDurationSince(selectedRun?.manifest.created_at, liveTick)
  const selectedRunLabel = draftWorkspace
    ? sanitizedDraftFamilyName || draftWorkspace.familyName || 'Configured workspace'
    : selectedRun?.manifest.run_id ?? 'No run selected'
  const taskLabel = draftWorkspace
    ? humanizeKey(draftWorkspace.taskType)
    : selectedRun?.manifest.task_name ?? 'Unknown task'
  const hasSelectedRun = Boolean(selectedRun?.manifest.run_id)
  // Starred runs float to the top; within each group, preserve server order.
  const latestRuns = useMemo(() => {
    const runs = snapshot?.runs ?? []
    return [...runs].sort((a, b) => {
      const as = a.manifest.starred ? 1 : 0
      const bs = b.manifest.starred ? 1 : 0
      return bs - as
    })
  }, [snapshot?.runs])

  // ─── Navigation helpers ─────────────────────────────────────────────────────
  const commitActiveWorkspace = (runId: string) => {
    setSelectedRunId(runId)
    setSelectedRunDetail(null)
    setUxState({ mode: 'active', runId })
    setActivePage('dashboard')
    setStartPointSelection({ kind: 'fresh' })
    setStartPointOptions([])
    setStartPointError(null)
    setStartLaunchError(null)
    setStartLaunching(false)
    setLifecycleActionLoading(null)
  }

  const returnToLauncher = () => {
    setSelectedRunId(null)
    setSelectedRunDetail(null)
    setUxState({ mode: 'browsing', launcher: { kind: 'home' }, continueSelection: { kind: 'families' } })
    setActivePage('dashboard')
    setStartPointSelection({ kind: 'fresh' })
    setStartPointOptions([])
    setStartPointError(null)
    setStartLaunchError(null)
  }

  const enterInspectMode = (runId: string) => {
    setSelectedRunId(runId)
    setUxState({ mode: 'inspect', runId, returnTo: currentBrowsingContext })
    setStartPointSelection({ kind: 'fresh' })
    setStartPointOptions([])
    setStartPointError(null)
    setActivePage('dashboard')
  }

  const exitInspectMode = () => {
    if (uxState.mode !== 'inspect') return
    setSelectedRunId(null)
    setSelectedRunDetail(null)
    setUxState({
      mode: 'browsing',
      launcher: uxState.returnTo.launcher,
      continueSelection: uxState.returnTo.continueSelection,
    })
  }

  const selectedRunIsResumable =
    selectedRun !== null &&
    (runStatus === 'paused' || runStatus === 'failed') &&
    canResumeAnotherRun

  // ─── Cross-cutting effects ──────────────────────────────────────────────────

  // Sync template when available templates change
  useEffect(() => {
    if (!availableTemplates.length) return
    if (!availableTemplates.some((t) => t.id === draftTemplateId)) {
      // Current template no longer available — fall back to first
      const fallback = availableTemplates[0]
      setDraftTemplateId(fallback.id)
      setDraftTaskType(fallback.taskType)
      setDraftFamilyName(fallback.suggestedFamilyName)
      setDraftConfig(normalizeDraftConfig(fallback.data ?? {}))
      lastAppliedTemplateId.current = fallback.id
      return
    }
    // Only reset config when the template ID itself changed (user picked a new template),
    // not on every poll cycle that refreshes availableTemplates from the backend snapshot.
    if (draftTemplateId !== lastAppliedTemplateId.current) {
      const current = availableTemplates.find((t) => t.id === draftTemplateId)
      if (current) {
        setDraftConfig(normalizeDraftConfig(current.data ?? {}))
        setDraftTaskType(current.taskType)
      }
      lastAppliedTemplateId.current = draftTemplateId
    }
  }, [availableTemplates, draftTemplateId])

  // Evict stale run selection
  useEffect(() => {
    const runIds = new Set((snapshot?.runs ?? []).map((r) => r.manifest.run_id))
    if (selectedRunId && !runIds.has(selectedRunId)) setSelectedRunId(null)
  }, [snapshot, selectedRunId])

  // Auto-commit when orchestration picks up an active run (or when active run changes due to resume).
  // Only fires for genuinely active phases — 'paused' is excluded so the user can open the draft
  // launcher after pausing without being immediately bounced back to the dashboard.
  useEffect(() => {
    const activeRunId = snapshot?.orchestration?.active_run_id
    const activePhase = snapshot?.orchestration?.phase
    const isLivePhase = activePhase === 'running' || activePhase === 'starting' || activePhase === 'pausing'
    if (!activeRunId || !isLivePhase) return
    const shouldAutoCommit =
      draftWorkspace != null ||                                        // started from draft launcher
      (uxState.mode === 'active' && selectedRunId !== activeRunId) || // active run changed
      lifecycleActionLoading === 'resume'                             // resume initiated from browsing
    if (shouldAutoCommit) commitActiveWorkspace(activeRunId)
  }, [draftWorkspace, selectedRunId, snapshot?.orchestration?.active_run_id, snapshot?.orchestration?.phase, uxState.mode, lifecycleActionLoading])

  // Auto-navigate on first snapshot load — only commit to a live run, stay on launcher otherwise
  useEffect(() => {
    if (autoNavigated || !snapshot) return
    const activeRunId = snapshot.orchestration?.active_run_id
    const activePhase = snapshot.orchestration?.phase
    const isActivePhase = activePhase === 'running' || activePhase === 'starting' || activePhase === 'pausing'
    // If we're in an active phase but haven't received the run ID yet, keep waiting
    if (isActivePhase && !activeRunId) return
    setAutoNavigated(true)
    if (activeRunId && isActivePhase) commitActiveWorkspace(activeRunId)
  }, [snapshot?.orchestration?.active_run_id, autoNavigated])

  // Clear in-flight loading indicators when orchestration reaches a terminal phase
  useEffect(() => {
    const phase = snapshot?.orchestration?.phase
    if (phase === 'paused' || phase === 'idle' || phase === 'failed') {
      setLifecycleActionLoading(null)
      if (phase !== 'paused') setStartLaunching(false)
    }
  }, [snapshot?.orchestration?.phase])

  // Clear checkpoint options when draft is closed
  useEffect(() => {
    if (!draftWorkspace) {
      setStartPointOptions([])
      setStartPointError(null)
      setStartPointLoading(false)
    }
  }, [draftWorkspace])

  // Load compatible checkpoints when a draft workspace is active
  useEffect(() => {
    if (!draftWorkspace) return
    let closed = false
    const load = async () => {
      setStartPointLoading(true)
      setStartPointError(null)
      try {
        const response = await fetch(
          `/api/checkpoints?compatible_for=${encodeURIComponent(draftWorkspace.familyName)}`,
        )
        if (!response.ok) throw new Error(`checkpoint inventory ${response.status}`)
        const payload = (await response.json()) as CheckpointRecord[]
        if (closed) return
        setStartPointOptions(payload)
        setStartPointSelection((current) =>
          current.kind === 'checkpoint' && !payload.some((e) => e.checkpoint_path === current.checkpoint_path)
            ? { kind: 'fresh' }
            : current,
        )
      } catch (err) {
        if (!closed) {
          setStartPointOptions([])
          setStartPointError(err instanceof Error ? err.message : 'Failed to load compatible checkpoints')
        }
      } finally {
        if (!closed) setStartPointLoading(false)
      }
    }
    load()
    return () => { closed = true }
  }, [draftWorkspace])

  // ─── Action handlers ────────────────────────────────────────────────────────
  const persistCurrentDraft = async (familyName: string) => {
    const response = await fetch(`/api/draft-configs/${encodeURIComponent(familyName)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ...draftConfig,
        dashboard: {
          ...(typeof draftConfig.dashboard === 'object' && draftConfig.dashboard && !Array.isArray(draftConfig.dashboard)
            ? (draftConfig.dashboard as Record<string, unknown>)
            : {}),
          task_type: draftTaskType,
        },
        template_id: draftTemplateId,
        family_name: familyName,
        task_type: draftTaskType,
      }),
    })
    if (!response.ok) {
      const payload = (await response.json().catch(() => null)) as { error?: string } | null
      throw new Error(payload?.error ?? `draft save ${response.status}`)
    }
  }

  const handleStartSelected = async () => {
    if (!draftWorkspace) return
    // Always use the current family name from the input, not the stale snapshot
    // in draftWorkspace (which was frozen when the user clicked "Configure").
    const familyName = sanitizedDraftFamilyName
    const validationError = validateRunFamilyName(familyName)
    if (validationError) { setStartLaunchError(validationError); return }
    setAutoNavigated(false)
    setStartLaunching(true)
    setStartLaunchError(null)
    try {
      await persistCurrentDraft(familyName)
      const body: Record<string, unknown> = {
        family_name: familyName,
        mode: startPointSelection.kind === 'checkpoint' ? 'load_weights' : 'fresh',
      }
      if (startPointSelection.kind === 'checkpoint') {
        body.checkpoint_path = startPointSelection.checkpoint_path
        body.checkpoint_run_id = startPointSelection.run_id
        body.checkpoint_name = startPointSelection.checkpoint_name
      }
      const response = await fetch('/api/runs/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!response.ok) {
        const payload = (await response.json().catch(() => null)) as { error?: string } | null
        throw new Error(formatApiError(response.status, payload?.error, 'Failed to start run'))
      }
      // Delete the draft — the run is now the source of truth.
      // Fire-and-forget: a failed delete is non-fatal.
      fetch(`/api/draft-configs/${encodeURIComponent(familyName)}`, { method: 'DELETE' }).catch(() => undefined)

      // Keep startLaunching=true — it will be cleared by commitActiveWorkspace when
      // the orchestration reports the run is active.  Refresh immediately so the
      // 'starting' phase shows up in the UI without waiting for the next SSE tick.
      await manualRefresh()
    } catch (err) {
      setStartLaunchError(err instanceof Error ? err.message : 'Failed to start training')
      setStartLaunching(false)
    }
  }

  const handlePause = async () => {
    setLifecycleActionLoading('pause')
    setLifecycleActionError(null)
    try {
      const response = await fetch('/api/runs/pause', { method: 'POST' })
      if (!response.ok) {
        const payload = (await response.json().catch(() => null)) as { error?: string } | null
        throw new Error(formatApiError(response.status, payload?.error, 'Failed to pause'))
      }
      // Refresh immediately so the UI transitions to the 'pausing' phase without
      // waiting for the next SSE tick.  lifecycleActionLoading stays set until the
      // snapshot confirms the phase changed (cleared by commitActiveWorkspace or
      // the snapshot update triggering a re-render through the existing render logic).
      await manualRefresh()
    } catch (err) {
      setLifecycleActionError(err instanceof Error ? err.message : 'Failed to pause training')
      setLifecycleActionLoading(null)
    }
  }

  const handleResume = async () => {
    setLifecycleActionLoading('resume')
    setLifecycleActionError(null)
    try {
      const response = await fetch('/api/runs/resume', { method: 'POST' })
      if (!response.ok) {
        const payload = (await response.json().catch(() => null)) as { error?: string } | null
        throw new Error(formatApiError(response.status, payload?.error, 'Failed to resume'))
      }
      // Keep lifecycleActionLoading set — commitActiveWorkspace clears it when the
      // new run ID becomes active.  Refresh now to pick up the 'starting' phase.
      await manualRefresh()
    } catch (err) {
      setLifecycleActionError(err instanceof Error ? err.message : 'Failed to resume training')
      setLifecycleActionLoading(null)
    }
  }

  const handleResumeTarget = async (run: RunRecord) => {
    const familyName = run.manifest.project_run ?? run.manifest.run_id
    setLifecycleActionLoading('resume')
    setLifecycleActionError(null)
    try {
      const response = await fetch('/api/runs/resume', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ family_name: familyName, run_id: run.manifest.run_id }),
      })
      if (!response.ok) {
        const payload = (await response.json().catch(() => null)) as { error?: string } | null
        throw new Error(formatApiError(response.status, payload?.error, 'Failed to resume'))
      }
      // Keep lifecycleActionLoading set — commitActiveWorkspace clears it.
      await manualRefresh()
    } catch (err) {
      setLifecycleActionError(err instanceof Error ? err.message : 'Failed to resume training')
      setLifecycleActionLoading(null)
    }
  }

  const handleSaveDraft = async () => {
    const familyName = sanitizeRunFamilyName(draftFamilyName)
    const validationError = validateRunFamilyName(familyName)
    if (validationError) { setDraftSaveError(validationError); return }
    if (existingDraftNames.has(sanitizeRunFamilyName(familyName))) {
      setDraftSaveError(`Draft '${sanitizeRunFamilyName(familyName)}' already exists.`)
      return
    }
    setDraftSaving(true)
    setDraftSaveError(null)
    setDraftDeleteError(null)
    try {
      await persistCurrentDraft(familyName)
      setSelectedRunId(null)
      setSelectedRunDetail(null)
      setStartPointSelection({ kind: 'fresh' })
      setStartPointOptions([])
      setStartPointError(null)
      setUxState({
        mode: 'browsing',
        launcher: { kind: 'draft', taskType: draftTaskType, familyName, templateId: draftTemplateId, flow: 'scratch' },
        continueSelection,
      })
      setActivePage('dashboard')
    } catch (err) {
      setDraftSaveError(err instanceof Error ? err.message : 'Failed to save draft config')
    } finally {
      setDraftSaving(false)
    }
  }

  const openExistingDraft = (draft: DraftRecord) => {
    setDraftTemplateId(draft.template_id)
    setDraftTaskType(draft.task_type)
    setDraftFamilyName(draft.family_name)
    if (draft.data) setDraftConfig(normalizeDraftConfig(draft.data))
    setSelectedRunId(null)
    setSelectedRunDetail(null)
    setStartPointSelection({ kind: 'fresh' })
    setStartPointOptions([])
    setStartPointError(null)
    setStartLaunchError(null)
    setDraftSaveError(null)
    setDraftDeleteError(null)
    setUxState({
      mode: 'browsing',
      launcher: {
        kind: 'draft',
        taskType: draft.task_type,
        familyName: draft.family_name,
        templateId: draft.template_id,
        flow: launcherSelection.kind === 'draft' ? launcherSelection.flow : 'scratch',
      },
      continueSelection,
    })
    setActivePage('dashboard')
  }

  const handleDeleteDraft = async (familyName: string) => {
    setDeletingDraftName(familyName)
    setDraftDeleteError(null)
    try {
      const response = await fetch(`/api/draft-configs/${encodeURIComponent(familyName)}`, { method: 'DELETE' })
      if (!response.ok) {
        const payload = (await response.json().catch(() => null)) as { error?: string } | null
        throw new Error(payload?.error ?? `draft delete ${response.status}`)
      }
      setSnapshot((current) => {
        if (!current) return current
        const tm = current.training_manager ?? {}
        return { ...current, training_manager: { ...tm, drafts: (tm.drafts ?? []).filter((d) => d.family_name !== familyName) } }
      })
      if (sanitizeRunFamilyName(draftFamilyName) === familyName) setDraftSaveError(null)
    } catch (err) {
      setDraftDeleteError(err instanceof Error ? err.message : 'Failed to delete draft')
    } finally {
      setDeletingDraftName(null)
    }
  }

  const handleDeleteRun = async (runId: string) => {
    // Optimistically remove from the sidebar immediately.
    setSnapshot((current) => {
      if (!current) return current
      return { ...current, runs: (current.runs ?? []).filter((r) => r.manifest.run_id !== runId) }
    })
    // If the deleted run is currently displayed, navigate away.
    if (selectedRunId === runId) returnToLauncher()
    try {
      await fetch(`/api/runs/${encodeURIComponent(runId)}`, { method: 'DELETE' })
    } catch {
      // On network error, re-sync from the server.
      await manualRefresh()
    }
  }

  // ─── Launcher context value ─────────────────────────────────────────────────
  const launcherContextValue: LauncherContextValue = {
    templates: availableTemplates,
    draftTaskType,
    draftFamilyName,
    draftTemplateId,
    draftConfig,
    draftFields,
    scratchQuickFields,
    draftNameError,
    draftNameExists,
    draftFamilyExists,
    sanitizedDraftFamilyName,
    draftSaving,
    draftSaveError,
    savedDrafts,
    draftDeleteError,
    deletingDraftName,
    startPointSelection,
    startPointOptions,
    startPointLoading,
    startPointError,
    startLaunching,
    startLaunchError,
    launcherSelection,
    continueSelection,
    continueFamilies,
    resumableFamilies,
    canResumeAnotherRun,
    lifecycleActionLoading,
    orchestration: snapshot?.orchestration,
    setDraftTaskType,
    setDraftFamilyName,
    setDraftTemplateId,
    setDraftConfig,
    onSelectFresh: () => { setStartPointSelection({ kind: 'fresh' }); setStartLaunchError(null) },
    onSelectCheckpoint: (checkpoint) => {
      setStartLaunchError(null)
      setStartPointSelection({
        kind: 'checkpoint',
        checkpoint_path: checkpoint.checkpoint_path,
        checkpoint_name: checkpoint.checkpoint_name,
        run_id: checkpoint.run_id,
        label: buildCheckpointLabel(checkpoint),
      })
    },
    onStartSelected: handleStartSelected,
    onSaveDraft: handleSaveDraft,
    onOpenDraft: openExistingDraft,
    onDeleteDraft: handleDeleteDraft,
    onActivateFlow: (flow) => {
      setSelectedRunId(null)
      setSelectedRunDetail(null)
      setUxState({
        mode: 'browsing',
        launcher: {
          kind: 'draft',
          taskType: draftTaskType,
          familyName: sanitizeRunFamilyName(draftFamilyName) || draftFamilyName,
          templateId: draftTemplateId,
          flow,
        },
        continueSelection,
      })
      if (flow === 'scratch') setStartPointSelection({ kind: 'fresh' })
      setStartPointError(null)
      setStartLaunchError(null)
    },
    onInspectRun: enterInspectMode,
    onSelectContinueFamily: (familyName) =>
      setUxState({ mode: 'browsing', launcher: launcherSelection, continueSelection: { kind: 'family', familyName } }),
    onBackToContinueFamilies: () =>
      setUxState({ mode: 'browsing', launcher: launcherSelection, continueSelection: { kind: 'families' } }),
    onResumeRun: handleResumeTarget,
    onGoBack: () => {
      setUxState({ mode: 'browsing', launcher: { kind: 'home' }, continueSelection: { kind: 'families' } })
      setStartPointSelection({ kind: 'fresh' })
      setStartPointOptions([])
      setStartPointError(null)
      setStartLaunchError(null)
    },
  }

  // ─── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b border-border bg-card/80 backdrop-blur-sm">
        <div className="flex flex-col gap-3 px-4 py-3 lg:flex-row lg:items-center lg:justify-between">
          {/* Left: run identity */}
          <div className="flex items-center gap-3">
            <div className="flex min-w-[68px] justify-start">
              {uxState.mode === 'inspect' ? (
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="h-8 gap-2 px-2 text-xs"
                  onClick={exitInspectMode}
                >
                  <ChevronLeft className="h-3.5 w-3.5" />
                  Back
                </Button>
              ) : (
                <div aria-hidden className="h-8 w-[68px]" />
              )}
            </div>

            <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-primary/10">
              <Activity className="h-4 w-4 text-primary" />
            </div>

            <div className="space-y-1 min-w-0">
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-sm font-semibold tracking-tight truncate">{selectedRunLabel}</span>
                <RunStatusBadge status={runStatus} />
                {runModeLabel && <WorkspaceModeBadge label={runModeLabel} tone={workspaceModeTone} />}
                {selectedRun?.manifest.lineage?.start_mode === 'load_weights' && (
                  <span className="rounded-full border border-violet-500/30 bg-violet-500/10 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-violet-500">
                    from ckpt
                  </span>
                )}
                <Badge
                  variant="secondary"
                  className={
                    sseConnected
                      ? 'bg-primary/10 text-primary'
                      : 'bg-secondary text-muted-foreground'
                  }
                >
                  {sseConnected ? 'Live' : 'Reconnecting'}
                </Badge>
              </div>
              <div className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
                <span className="flex items-center gap-1.5">
                  <Clock className="h-3.5 w-3.5" />
                  {durationLabel}
                </span>
                <span className="flex items-center gap-1.5">
                  <Layers className="h-3.5 w-3.5" />
                  {stepLabel}
                </span>
                <span>{taskLabel}</span>
              </div>
            </div>
          </div>

          {/* Right: nav tabs + lifecycle controls */}
          <div className="flex flex-wrap items-center gap-3">
            <div className="flex items-center gap-0.5 rounded-md bg-secondary/50 p-0.5">
              <NavButton active={activePage === 'dashboard'} icon={LayoutDashboard} label="Dashboard" onClick={() => setActivePage('dashboard')} />
              <NavButton active={activePage === 'settings'} icon={Settings2} label="Settings" onClick={() => setActivePage('settings')} />
              <NavButton active={activePage === 'artifacts'} icon={FolderOpen} label="Artifacts" onClick={() => setActivePage('artifacts')} />
              <NavButton active={activePage === 'logs'} icon={Clock} label="Logs" onClick={() => setActivePage('logs')} />
            </div>

            <Separator orientation="vertical" className="hidden h-6 sm:block" />

            <div className="flex items-center gap-1.5">
              {selectedRunIsResumable && selectedRun ? (
                <>
                  {(uxState.mode === 'active' || orchestrationPhase === 'paused' || orchestrationPhase === 'failed') && (
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-8 gap-1.5 px-3 text-xs"
                      onClick={returnToLauncher}
                      disabled={lifecycleActionLoading !== null}
                    >
                      <ChevronLeft className="h-3.5 w-3.5" />
                      Launcher
                    </Button>
                  )}
                  <Button
                    variant="secondary"
                    size="sm"
                    className="h-8 gap-1.5 px-3 text-xs"
                    disabled={lifecycleActionLoading !== null}
                    onClick={() => {
                      if ((orchestrationPhase === 'paused' || orchestrationPhase === 'failed') && snapshot?.orchestration?.active_run_id === selectedRun.manifest.run_id) {
                        void handleResume()
                      } else {
                        void handleResumeTarget(selectedRun)
                      }
                    }}
                  >
                    <Rocket className="h-3.5 w-3.5" />
                    {lifecycleActionLoading === 'resume' ? 'Resuming…' : 'Resume'}
                  </Button>
                </>
              ) : orchestrationPhase === 'running' || orchestrationPhase === 'starting' ? (
                <Button
                  variant="secondary"
                  size="sm"
                  className="h-8 gap-1.5 px-3 text-xs"
                  disabled={lifecycleActionLoading !== null || orchestrationPhase === 'starting'}
                  onClick={handlePause}
                >
                  <XCircle className="h-3.5 w-3.5" />
                  {lifecycleActionLoading === 'pause' ? 'Pausing…' : 'Pause'}
                </Button>
              ) : orchestrationPhase === 'pausing' ? (
                <Button variant="secondary" size="sm" className="h-8 gap-1.5 px-3 text-xs" disabled>
                  <XCircle className="h-3.5 w-3.5" />
                  Pausing…
                </Button>
              ) : orchestrationPhase === 'paused' || orchestrationPhase === 'failed' ? (
                <>
                  <Button variant="ghost" size="sm" className="h-8 gap-1.5 px-3 text-xs" onClick={returnToLauncher}>
                    <ChevronLeft className="h-3.5 w-3.5" />
                    Launcher
                  </Button>
                  <Button
                    variant="secondary"
                    size="sm"
                    className="h-8 gap-1.5 px-3 text-xs"
                    disabled={lifecycleActionLoading !== null}
                    onClick={handleResume}
                  >
                    <Rocket className="h-3.5 w-3.5" />
                    {lifecycleActionLoading === 'resume' ? 'Resuming…' : 'Resume'}
                  </Button>
                </>
              ) : null}
            </div>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="p-4">
        {(error || lifecycleActionError) && (
          <div className="mb-4 space-y-2">
            {error && <ErrorMessage text={`Failed to load dashboard state: ${error}`} />}
            {lifecycleActionError && <ErrorMessage text={lifecycleActionError} />}
          </div>
        )}

        <div className="grid grid-cols-1 gap-4 xl:grid-cols-12">
          {/* Primary content area */}
          <div className="space-y-4 xl:col-span-9">
            {hasSelectedRun ? (
              activePage === 'settings' ? (
                <RunSettingsPanel
                  modelOverviewEntries={modelOverviewEntries}
                  hyperparameters={hyperparameters}
                  lossWeights={lossWeights}
                  storePath={runStorePath}
                  isRemoval={runIsRemoval}
                />
              ) : activePage === 'artifacts' ? (
                <RunArtifactsPanel artifacts={selectedRun?.artifacts ?? []} />
              ) : activePage === 'logs' ? (
                <RunLogsPanel
                  lines={snapshot?.training_manager?.recent_logs ?? []}
                  runId={selectedRunId}
                  activeRunId={snapshot?.orchestration?.active_run_id ?? null}
                />
              ) : (
                <RunDashboardWorkspace
                  trainSeries={trainSeries}
                  valSeries={valSeries}
                  selectedRun={selectedRun}
                  refreshKey={refreshKey}
                />
              )
            ) : (
              <LauncherContext.Provider value={launcherContextValue}>
                <WorkspaceSetupPanel />
              </LauncherContext.Provider>
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
                  Latest Runs
                </PanelCardTitle>
              </PanelCardHeader>
              <PanelCardContent className="space-y-1.5">
                {latestRuns.length ? (
                  latestRuns.map((run) => (
                    <div
                      key={run.manifest.run_id}
                      role="button"
                      tabIndex={0}
                      onClick={() => enterInspectMode(run.manifest.run_id)}
                      onKeyDown={(e) => e.key === 'Enter' && enterInspectMode(run.manifest.run_id)}
                      className={[
                        'group w-full cursor-pointer rounded-md border px-2.5 py-2 text-left transition-colors',
                        run.manifest.run_id === selectedRun?.manifest.run_id
                          ? 'border-primary/30 bg-primary/10'
                          : 'border-border/40 bg-secondary/15 hover:bg-secondary/30',
                      ].join(' ')}
                    >
                      <div className="mb-1 flex items-center justify-between gap-2">
                        <span className="truncate font-mono text-xs text-muted-foreground">
                          {run.manifest.run_id}
                        </span>
                        <div className="flex items-center gap-1.5 shrink-0">
                          <DeleteRunButton
                            runId={run.manifest.run_id}
                            onDelete={() => { void handleDeleteRun(run.manifest.run_id) }}
                          />
                          <StarButton
                            run={run}
                            onStarChange={(starred, note) => {
                              setSnapshot((current) => {
                                if (!current) return current
                                return {
                                  ...current,
                                  runs: (current.runs ?? []).map((r) =>
                                    r.manifest.run_id === run.manifest.run_id
                                      ? { ...r, manifest: { ...r.manifest, starred, note } }
                                      : r
                                  ),
                                }
                              })
                            }}
                          />
                          <RunStatusBadge status={resolveRunStatus(run, snapshot?.orchestration)} compact />
                        </div>
                      </div>
                      <div className="text-sm font-medium truncate">{buildRunMetricSummary(run)}</div>
                      {run.manifest.starred && run.manifest.note && (
                        <div className="mt-1 truncate text-xs text-amber-400/80 italic">{run.manifest.note}</div>
                      )}
                      <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-xs text-muted-foreground">
                        <span>{formatStep(run)}</span>
                        <span>{formatDurationSince(run.manifest.created_at, liveTick)}</span>
                        <span>{run.manifest.task_name ?? 'Unknown task'}</span>
                        {run.manifest.lineage?.start_mode === 'load_weights' && (
                          <span className="rounded-full border border-violet-500/30 bg-violet-500/10 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-violet-500">
                            from ckpt
                          </span>
                        )}
                      </div>
                    </div>
                  ))
                ) : loading ? (
                  <EmptyState text="Loading runs…" />
                ) : (
                  <EmptyState text="No runs found yet." />
                )}
              </PanelCardContent>
            </Card>
          </div>
        </div>
      </main>
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
          Remove
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
      title={`Delete run ${runId}`}
      onClick={handleClick}
      className="rounded p-0.5 text-muted-foreground/40 opacity-0 transition-colors hover:text-red-400 group-hover:opacity-100"
    >
      <Trash2 className="h-3.5 w-3.5" />
    </button>
  )
}

export default App
