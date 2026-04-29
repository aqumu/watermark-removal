export type RunStatus = 'running' | 'paused' | 'completed' | 'failed' | 'unknown'

export type RunLineage = {
  start_mode: string
  history_mode: string
  parent_run_id?: string | null
  parent_timeline_id?: string | null
  parent_checkpoint?: string | null
  resumed_from_epoch?: number | null
}

export type Manifest = {
  run_id: string
  task_name?: string
  created_at?: string
  status?: string
  device?: string
  timeline_id?: string
  project_run?: string
  lineage?: RunLineage
  starred?: boolean
  note?: string
}

export type MetricRow = Record<string, string | number | boolean | null | undefined>

export type PreviewEntry = {
  step: number | string
  label: string
  previews: Record<string, string>
}

export type ArtifactEntry = {
  name: string
  category: string
  url: string
}

export type ModelOverview = {
  model_name?: string
  parameter_count?: number
  optimizer?: string
  scheduler?: string
  extra?: Record<string, unknown>
}

export type RunRecord = {
  manifest: Manifest
  model_overview?: ModelOverview | null
  train_metrics?: MetricRow[]
  val_metrics?: MetricRow[]
  previews?: Record<string, string>
  preview_history?: PreviewEntry[]
  artifacts?: ArtifactEntry[]
}

export type OrchestrationSnapshot = {
  phase?: string
  selected_family?: string | null
  draft_config_path?: string | null
  active_run_id?: string | null
  active_checkpoint?: string | null
  last_error?: string | null
  metadata?: Record<string, unknown>
}

export type TemplateRecord = {
  template_id: string
  label: string
  task_type: DraftTaskType
  path: string
  data?: Record<string, unknown>
}

export type DraftRecord = {
  family_name: string
  template_id: string
  task_type: DraftTaskType
  path: string
  data?: Record<string, unknown>
}

export type TrainingManagerSnapshot = {
  templates?: TemplateRecord[]
  drafts?: DraftRecord[]
  active_job?: Record<string, unknown> | null
  recent_logs?: string[]
  run_id_ready?: boolean
}

export type Snapshot = {
  family_root?: string
  current_run_id?: string | null
  runs?: RunRecord[]
  current_run?: RunRecord | null
  training_manager?: TrainingManagerSnapshot
  orchestration?: OrchestrationSnapshot
}

export type TrendMetric = {
  label: string
  value: string
  trend?: string
  positive?: boolean | null
}

export type ChartPoint = {
  step: number
  total?: number
  lr?: number
  gradNorm?: number
  terms: Record<string, number>
}

export type ValPoint = {
  step: number
  psnr?: number
  psnrMasked?: number
  iou?: number
  trainPsnr?: number
}

export type PreviewCard = {
  key: string
  label: string
  url?: string
}

export type FlatEntry = {
  label: string
  value: string
}

export type ActivePage = 'dashboard' | 'settings' | 'artifacts' | 'logs'

export type DraftField = {
  path: string[]
  label: string
  value: string | number | boolean | null
  kind: 'string' | 'number' | 'boolean' | 'null'
}

export type NavButtonProps = {
  active: boolean
  icon: React.ComponentType<{ className?: string }>
  label: string
}

export type DraftTaskType = 'removal' | 'restoration' | 'segmentation'

export type CheckpointRecord = {
  run_id: string
  project_run?: string
  task_name?: string
  checkpoint_name: string
  checkpoint_path: string
  checkpoint_url: string
  kind?: string
  epoch?: number | null
  status?: string
  family_root?: string
  task_type?: string
  signature?: Record<string, unknown>
}

export type StartPointSelection =
  | { kind: 'fresh' }
  | {
      kind: 'checkpoint'
      checkpoint_path: string
      checkpoint_name: string
      run_id: string
      label: string
    }

export type DraftWorkspace = {
  taskType: DraftTaskType
  familyName: string
  templateId: string
  flow: 'scratch' | 'checkpoint'
}

export type LauncherSelection = { kind: 'home' } | ({ kind: 'draft' } & DraftWorkspace)

export type ContinueBrowseSelection = { kind: 'families' } | { kind: 'family'; familyName: string }

export type BrowsingContext = {
  launcher: LauncherSelection
  continueSelection: ContinueBrowseSelection
}

export type UXState =
  | {
      mode: 'browsing'
      launcher: LauncherSelection
      continueSelection: ContinueBrowseSelection
    }
  | {
      mode: 'inspect'
      runId: string
      returnTo: BrowsingContext
    }
  | {
      mode: 'active'
      runId: string
    }

export type TemplateOption = {
  id: string
  label: string
  description: string
  taskType: DraftTaskType
  suggestedFamilyName: string
  data?: Record<string, unknown>
}

export type ContinueFamilySummary = {
  familyName: string
  taskLabel: string
  latestStatus: RunStatus
  latestRun: RunRecord
  runs: RunRecord[]
  bestMetricSummary: string
}
