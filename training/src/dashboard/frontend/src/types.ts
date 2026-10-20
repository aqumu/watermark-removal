export type RunStatus = 'running' | 'paused' | 'completed' | 'failed' | 'stopped' | 'unknown'

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
  last_seen?: string | null
  family_dir?: string
  progress?: RunProgress
}

export type RunProgress = {
  epoch?: number
  step?: number
  global_step?: number
  epochs_total?: number
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

export type Snapshot = {
  roots?: string[]
  runs?: RunRecord[]
}

export type LogTail = {
  exists: boolean
  truncated: boolean
  text: string
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

export type NavButtonProps = {
  active: boolean
  icon: React.ComponentType<{ className?: string }>
  label: string
}
