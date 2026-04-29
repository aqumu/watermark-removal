import type { EChartsOption } from 'echarts'
import { graphic } from 'echarts'
import type {
  RunRecord,
  RunStatus,
  OrchestrationSnapshot,
  MetricRow,
  ChartPoint,
  ValPoint,
  PreviewEntry,
  PreviewCard,
  TrendMetric,
  FlatEntry,
  ModelOverview,
  CheckpointRecord,
  StartPointSelection,
  ContinueFamilySummary,
} from '../types'
import {
  IGNORED_TERM_KEYS,
  IGNORED_TERM_PREFIX,
  PREVIEW_TITLES,
  TERM_COLORS,
} from './constants'
import {
  num,
  formatDecimal,
  formatChartNumber,
  formatInteger,
  formatUnknown,
  formatDelta,
  humanizeKey,
  isPrimitive,
  previewOrder,
  flattenObject,
} from './format'

export function normalizeStatus(value: string | undefined): RunStatus {
  if (value === 'running' || value === 'paused' || value === 'completed' || value === 'failed') {
    return value
  }
  return 'unknown'
}

export function resolveRunStatus(
  run: RunRecord | undefined | null,
  orchestration: OrchestrationSnapshot | undefined,
): RunStatus {
  const status = normalizeStatus(run?.manifest.status)
  if (status === 'running') {
    const phase = orchestration?.phase
    const isActivePhase = phase === 'running' || phase === 'starting' || phase === 'pausing'
    if (!isActivePhase || run?.manifest.run_id !== orchestration?.active_run_id) {
      return 'unknown'
    }
  }
  return status
}

export function compareRunsByCreatedAtDesc(left: RunRecord, right: RunRecord): number {
  const leftTime = Date.parse(left.manifest.created_at ?? '')
  const rightTime = Date.parse(right.manifest.created_at ?? '')
  const safeLeft = Number.isNaN(leftTime) ? 0 : leftTime
  const safeRight = Number.isNaN(rightTime) ? 0 : rightTime
  return safeRight - safeLeft || right.manifest.run_id.localeCompare(left.manifest.run_id)
}

export function buildTrainSeries(rows: MetricRow[]): ChartPoint[] {
  const points: ChartPoint[] = []
  for (const row of rows) {
    const step = num(row.global_step) ?? num(row.step)
    if (step === null) {
      continue
    }

    // Collect raw term values and their corresponding weights in one pass
    const rawValues: Record<string, number> = {}
    const weights: Record<string, number> = {}
    for (const [key, value] of Object.entries(row)) {
      if (IGNORED_TERM_KEYS.has(key)) continue
      const parsed = num(value)
      if (parsed === null) continue
      if (key.startsWith(IGNORED_TERM_PREFIX)) {
        weights[key.slice(IGNORED_TERM_PREFIX.length)] = parsed
      } else {
        rawValues[key] = parsed
      }
    }

    // Apply weights so each term reflects its actual contribution to the total loss
    const terms: Record<string, number> = {}
    for (const [key, value] of Object.entries(rawValues)) {
      if (key === 'total') continue  // total is already the weighted sum; shown in overview chart
      terms[key] = value * (weights[key] ?? 1)
    }

    points.push({
      step,
      total: num(row.total) ?? undefined,
      lr: num(row.lr) ?? undefined,
      gradNorm: num(row.grad_norm) ?? undefined,
      terms,
    })
  }
  return points
}

export function buildValSeries(rows: MetricRow[]): ValPoint[] {
  const points: ValPoint[] = []
  for (const row of rows) {
    const step = num(row.global_step) ?? num(row.step)
    if (step === null) {
      continue
    }
    points.push({
      step,
      psnr: num(row.psnr) ?? undefined,
      psnrMasked: num(row.psnr_masked) ?? undefined,
      iou: num(row.iou) ?? undefined,
      trainPsnr: num(row.train_psnr) ?? undefined,
    })
  }
  return points
}

export function formatStep(run: RunRecord | null | undefined): string {
  const latestTrain = buildTrainSeries(run?.train_metrics ?? []).at(-1)
  const latestVal = buildValSeries(run?.val_metrics ?? []).at(-1)
  const step = latestTrain?.step ?? latestVal?.step
  return step !== undefined ? `Step ${formatInteger(step)}` : 'No steps yet'
}

export function buildRunSummary(run: RunRecord): string {
  const latestVal = buildValSeries(run.val_metrics ?? []).at(-1)
  if (latestVal?.psnrMasked !== undefined) {
    return `${formatDecimal(latestVal.psnrMasked, 2)} dB`
  }
  if (latestVal?.iou !== undefined) {
    return `IoU ${formatDecimal(latestVal.iou, 4)}`
  }
  const latestTrain = buildTrainSeries(run.train_metrics ?? []).at(-1)
  if (latestTrain?.total !== undefined) {
    return `loss ${formatDecimal(latestTrain.total)}`
  }
  return run.manifest.task_name ?? 'run'
}

export function buildRunMetricSummary(run: RunRecord): string {
  const latestVal = buildValSeries(run.val_metrics ?? []).at(-1)
  if (latestVal?.psnrMasked !== undefined) {
    return `Best PSNR: ${formatDecimal(latestVal.psnrMasked, 2)} dB`
  }
  if (latestVal?.psnr !== undefined) {
    return `Best PSNR: ${formatDecimal(latestVal.psnr, 2)} dB`
  }
  if (latestVal?.iou !== undefined) {
    return `Best IoU: ${formatDecimal(latestVal.iou, 4)}`
  }
  const latestTrain = buildTrainSeries(run.train_metrics ?? []).at(-1)
  if (latestTrain?.total !== undefined) {
    return `Best Loss: ${formatDecimal(latestTrain.total)}`
  }
  return 'No metrics yet'
}

export function buildFamilyMetricSummary(runs: RunRecord[]): string {
  let bestLabel: string | null = null
  let bestValue: number | null = null
  let higherIsBetter = true

  for (const run of runs) {
    const latestVal = buildValSeries(run.val_metrics ?? []).at(-1)
    if (latestVal?.psnrMasked !== undefined) {
      if (bestValue === null || !higherIsBetter || latestVal.psnrMasked > bestValue) {
        bestLabel = 'Best PSNR'
        bestValue = latestVal.psnrMasked
        higherIsBetter = true
      }
      continue
    }
    if (latestVal?.psnr !== undefined) {
      if (bestValue === null || !higherIsBetter || latestVal.psnr > bestValue) {
        bestLabel = 'Best PSNR'
        bestValue = latestVal.psnr
        higherIsBetter = true
      }
      continue
    }
    if (latestVal?.iou !== undefined) {
      if (bestValue === null || !higherIsBetter || latestVal.iou > bestValue) {
        bestLabel = 'Best IoU'
        bestValue = latestVal.iou
        higherIsBetter = true
      }
      continue
    }

    const latestTrain = buildTrainSeries(run.train_metrics ?? []).at(-1)
    if (latestTrain?.total !== undefined) {
      if (bestValue === null || higherIsBetter || latestTrain.total < bestValue) {
        bestLabel = 'Best Loss'
        bestValue = latestTrain.total
        higherIsBetter = false
      }
    }
  }

  if (bestLabel === null || bestValue === null) {
    return 'No metrics yet'
  }
  if (bestLabel === 'Best PSNR') {
    return `${bestLabel}: ${formatDecimal(bestValue, 2)} dB`
  }
  if (bestLabel === 'Best IoU') {
    return `${bestLabel}: ${formatDecimal(bestValue, 4)}`
  }
  return `${bestLabel}: ${formatDecimal(bestValue)}`
}

export function buildContinueFamilies(
  runs: RunRecord[],
  orchestration: OrchestrationSnapshot | undefined,
): ContinueFamilySummary[] {
  const grouped = new Map<string, RunRecord[]>()
  for (const run of runs) {
    const familyName = run.manifest.project_run ?? run.manifest.run_id
    const familyRuns = grouped.get(familyName) ?? []
    familyRuns.push(run)
    grouped.set(familyName, familyRuns)
  }

  return [...grouped.entries()]
    .map(([familyName, familyRuns]) => {
      const sortedRuns = [...familyRuns].sort(compareRunsByCreatedAtDesc)
      const latestRun = sortedRuns[0]
      return {
        familyName,
        taskLabel: latestRun?.manifest.task_name ?? 'Unknown task',
        latestStatus: resolveRunStatus(latestRun, orchestration),
        latestRun,
        runs: sortedRuns,
        bestMetricSummary: buildFamilyMetricSummary(sortedRuns),
      }
    })
    .sort((left, right) => compareRunsByCreatedAtDesc(left.latestRun, right.latestRun))
}

export function buildMetricCards(
  run: RunRecord | null | undefined,
  trainSeries: ChartPoint[],
  valSeries: ValPoint[],
  previewHistory: PreviewEntry[],
): TrendMetric[] {
  const latestTrain = trainSeries.at(-1)
  const previousTrain = trainSeries.at(-2)
  const latestVal = valSeries.at(-1)
  const previousVal = valSeries.at(-2)
  const validationMetric = latestVal?.psnrMasked ?? latestVal?.psnr ?? latestVal?.iou ?? null
  const previousValidationMetric = previousVal?.psnrMasked ?? previousVal?.psnr ?? previousVal?.iou ?? null

  return [
    {
      label: 'Total Loss',
      value: latestTrain?.total !== undefined ? formatDecimal(latestTrain.total) : 'n/a',
      trend: formatDelta(latestTrain?.total, previousTrain?.total),
      positive: latestTrain?.total !== undefined && previousTrain?.total !== undefined ? latestTrain.total < previousTrain.total : null,
    },
    {
      label: latestVal?.iou !== undefined ? 'Validation IoU' : 'PSNR (Masked)',
      value:
        validationMetric !== null
          ? latestVal?.iou !== undefined
            ? formatDecimal(validationMetric, 4)
            : `${formatDecimal(validationMetric, 2)} dB`
          : 'n/a',
      trend: validationMetric !== null && previousValidationMetric !== null ? formatDelta(validationMetric, previousValidationMetric) : undefined,
      positive: validationMetric !== null && previousValidationMetric !== null ? validationMetric >= previousValidationMetric : null,
    },
    {
      label: 'Learning Rate',
      value: latestTrain?.lr !== undefined ? latestTrain.lr.toExponential(2) : 'n/a',
      trend: latestTrain?.lr !== undefined ? 'schedule' : undefined,
      positive: null,
    },
    {
      label: 'Grad Norm',
      value: latestTrain?.gradNorm !== undefined ? formatDecimal(latestTrain.gradNorm, 3) : 'n/a',
      positive: null,
    },
    {
      label: 'Preview Steps',
      value: String(previewHistory.length),
      trend: run?.artifacts?.length ? `${run.artifacts.length} artifacts` : undefined,
      positive: null,
    },
  ]
}

export function buildPreviewCards(
  run: RunRecord | null | undefined,
  currentPreview: PreviewEntry | undefined,
  refreshKey: number,
): PreviewCard[] {
  const previews = currentPreview?.previews ?? run?.previews ?? {}
  const keys = Object.keys(previews)
  if (!keys.length) {
    return []
  }

  const ordered = [...keys].sort((left, right) => previewOrder(left) - previewOrder(right) || left.localeCompare(right))
  return ordered.map((key) => ({
    key,
    label: PREVIEW_TITLES[key] ?? humanizeKey(key),
    url: `${previews[key]}?t=${encodeURIComponent(currentPreview?.label ?? String(refreshKey))}`,
  }))
}

export function buildModelOverviewEntries(modelOverview: ModelOverview | null | undefined): FlatEntry[] {
  if (!modelOverview) {
    return []
  }

  const entries: FlatEntry[] = []
  if (modelOverview.model_name) {
    entries.push({ label: 'Model', value: modelOverview.model_name })
  }
  if (typeof modelOverview.parameter_count === 'number') {
    entries.push({ label: 'Parameters', value: formatInteger(modelOverview.parameter_count) })
  }
  if (modelOverview.optimizer) {
    entries.push({ label: 'Optimizer', value: modelOverview.optimizer })
  }
  if (modelOverview.scheduler) {
    entries.push({ label: 'Scheduler', value: modelOverview.scheduler })
  }

  if (modelOverview.extra && typeof modelOverview.extra === 'object') {
    for (const [key, value] of Object.entries(modelOverview.extra)) {
      entries.push({ label: humanizeKey(key), value: formatUnknown(value) })
      if (entries.length >= 8) {
        break
      }
    }
  }

  return entries.slice(0, 8)
}

export function buildHyperparameters(config: Record<string, unknown> | null): FlatEntry[] {
  if (!config) {
    return []
  }
  return flattenObject(config, ['training', 'dataset', 'model'])
    .filter((entry) => !entry.label.includes('Loss.Weights'))
    .filter((entry) => entry.label !== 'Dataset.Preprocessed Store Dir')
    .slice(0, 18)
}

/** Extract the resolved preprocessed_store_dir from a config, or null if absent. */
export function extractStorePath(config: Record<string, unknown> | null): string | null {
  if (!config) return null
  const dataset = config.dataset
  if (!dataset || typeof dataset !== 'object' || Array.isArray(dataset)) return null
  const val = (dataset as Record<string, unknown>).preprocessed_store_dir
  return typeof val === 'string' && val.length > 0 ? val : null
}

/** Return true if this looks like a removal (not segmentation) config. */
export function isRemovalConfig(config: Record<string, unknown> | null): boolean {
  if (!config) return false
  const dataset = config.dataset
  if (!dataset || typeof dataset !== 'object' || Array.isArray(dataset)) return false
  const d = dataset as Record<string, unknown>
  return typeof d.image_width === 'number' && typeof d.image_height === 'number'
}

export function buildLossWeights(config: Record<string, unknown> | null, run: RunRecord | null | undefined): FlatEntry[] {
  if (config) {
    const lossSection = ((config.loss as Record<string, unknown> | undefined)?.weights ??
      (config.loss_weights as Record<string, unknown> | undefined)) as Record<string, unknown> | undefined
    if (lossSection && typeof lossSection === 'object') {
      return Object.entries(lossSection)
        .map(([key, value]) => ({ label: humanizeKey(key), value: formatUnknown(value) }))
        .slice(0, 16)
    }
  }

  const termNames = collectTopTermNames(run?.train_metrics ?? [])
  return termNames.map((term) => ({ label: humanizeKey(term), value: 'tracked' })).slice(0, 16)
}

export function collectTopTermNames(rows: MetricRow[]): string[] {
  const counts = new Map<string, number>()
  for (const row of rows) {
    for (const [key, value] of Object.entries(row)) {
      if (IGNORED_TERM_KEYS.has(key) || num(value) === null) {
        continue
      }
      counts.set(key, (counts.get(key) ?? 0) + 1)
    }
  }

  return [...counts.entries()]
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
    .map(([key]) => key)
}

export function buildCheckpointLabel(checkpoint: CheckpointRecord): string {
  const parts = [
    checkpoint.task_name,
    checkpoint.kind === 'best' ? 'best checkpoint' : null,
    checkpoint.epoch !== null && checkpoint.epoch !== undefined ? `epoch ${checkpoint.epoch}` : null,
  ].filter(Boolean)
  return parts.length ? parts.join(' • ') : 'Compatible checkpoint'
}

export function buildCheckpointLabelFromSelection(selection: StartPointSelection): string {
  if (selection.kind === 'checkpoint') {
    return `${selection.checkpoint_name} from ${selection.run_id}`
  }
  return 'Fresh start'
}

function localAverageAtStep(series: ValPoint[], step: number, key: 'psnr' | 'psnrMasked' | 'iou' | 'trainPsnr'): number | null {
  if (!series.length) {
    return null
  }

  let nearestIndex = 0
  let nearestDistance = Infinity
  for (let index = 0; index < series.length; index += 1) {
    const distance = Math.abs(series[index].step - step)
    if (distance < nearestDistance) {
      nearestDistance = distance
      nearestIndex = index
    }
  }

  const values: number[] = []
  for (let index = Math.max(0, nearestIndex - 1); index <= Math.min(series.length - 1, nearestIndex + 1); index += 1) {
    const value = series[index][key]
    if (typeof value === 'number' && Number.isFinite(value)) {
      values.push(value)
    }
  }

  if (!values.length) {
    return null
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length
}

function dotMarker(color: string): string {
  return `<span style="display:inline-block;margin-right:4px;border-radius:50%;width:10px;height:10px;background-color:${color};border:2px solid #0f172a;"></span>`
}

// Series names prefixed with this string are parent-phase series: excluded from legend
// and filtered from the tooltip so the user only sees current-phase values in the popup.
const PARENT_SERIES_PREFIX = '\u200b'

function formatOverviewTooltip(
  params: unknown,
  valSeries: ValPoint[],
  flags: { hasPsnr: boolean; hasIou: boolean; hasTrainPsnr: boolean },
): string {
  // Filter out parent-phase helper series from the tooltip
  const rows = (Array.isArray(params) ? params : []).filter(
    (row) => !(typeof row?.seriesName === 'string' && row.seriesName.startsWith(PARENT_SERIES_PREFIX)),
  )
  if (!rows.length) return ''

  const axisValue = typeof rows[0]?.axisValue === 'number' ? rows[0].axisValue : num(rows[0]?.axisValue)
  // Always use the numeric value (rounded to int) — axisValueLabel from ECharts carries ".00" noise
  const header = axisValue !== null ? formatInteger(Math.round(axisValue)) : ''
  const lines = rows.map((row) => {
    const label = typeof row.seriesName === 'string' ? row.seriesName : ''
    const marker = typeof row.marker === 'string' ? row.marker : ''
    const rawValue = Array.isArray(row.value) ? row.value[1] : row.value
    return `${marker}${label}: ${formatChartNumber(rawValue)}`
  })

  // When cursor is between sparse val checkpoints, inject nearest val values
  // with proper colored markers so tooltip looks identical to the on-point case.
  const seriesNames = new Set(rows.map((r) => (typeof r.seriesName === 'string' ? r.seriesName : '')))

  if (axisValue !== null && flags.hasPsnr) {
    if (!seriesNames.has('PSNR')) {
      const v = localAverageAtStep(valSeries, axisValue, 'psnr')
      if (v !== null) lines.push(`${dotMarker('#fb7185')}PSNR: ${formatChartNumber(v)}`)
    }
    if (!seriesNames.has('PSNR (mask)')) {
      const v = localAverageAtStep(valSeries, axisValue, 'psnrMasked')
      if (v !== null) lines.push(`${dotMarker('#fbbf24')}PSNR (mask): ${formatChartNumber(v)}`)
    }
  } else if (axisValue !== null && flags.hasIou && !flags.hasPsnr && !seriesNames.has('IoU')) {
    const v = localAverageAtStep(valSeries, axisValue, 'iou')
    if (v !== null) lines.push(`${dotMarker('#fbbf24')}IoU: ${formatChartNumber(v)}`)
  }
  if (axisValue !== null && flags.hasTrainPsnr && !seriesNames.has('PSNR (train)')) {
    const v = localAverageAtStep(valSeries, axisValue, 'trainPsnr')
    if (v !== null) lines.push(`${dotMarker('#34d399')}PSNR (train): ${formatChartNumber(v)}`)
  }

  return [header, ...lines].join('<br/>')
}

/** Tight data bounds for log-scale axes (only positive finite values). */
function dataBounds(values: (number | undefined)[]): { min: number; max: number } | null {
  const pos = values.filter((v): v is number => v != null && Number.isFinite(v) && v > 0)
  if (!pos.length) return null
  return { min: Math.min(...pos), max: Math.max(...pos) }
}

/** EMA alpha that gives ~2*sqrt(n) point window, clamped for stability. */
function smoothingAlpha(nPoints: number): number {
  const window = Math.max(3, 2 * Math.sqrt(nPoints))
  return 2 / (window + 1)
}

function applyEma(
  data: Array<[number, number | undefined]>,
  alpha: number,
): Array<[number, number | undefined]> {
  let ema: number | undefined
  return data.map(([step, value]) => {
    if (value === undefined || !Number.isFinite(value)) return [step, value]
    ema = ema === undefined ? value : alpha * value + (1 - alpha) * ema
    return [step, ema]
  })
}

/**
 * Gradient for a PSNR line whose tail glows over exactly the EMA memory window.
 * xMin/xMax and the window are all in global optimizer steps, so no epoch
 * conversion is needed. Falls back to the solid brightColor when ema_decay is
 * zero (EMA disabled) or there isn't enough x-range to show anything useful.
 */
function psnrTailGradient(
  xMin: number,
  xMax: number,
  emaDecay: number,
  dimColor: string,
  brightColor: string,
): graphic.LinearGradient | string {
  if (emaDecay <= 0 || emaDecay >= 1 || xMax <= xMin) return brightColor
  // Steps back until cumulative EMA weight exceeds 95 %
  const nEff = Math.ceil(Math.log(0.05) / Math.log(emaDecay))
  const tailOffset = Math.max(0, Math.min(0.9, (xMax - nEff - xMin) / (xMax - xMin)))
  return new graphic.LinearGradient(0, 0, 1, 0, [
    { offset: 0,           color: dimColor },
    { offset: tailOffset,  color: dimColor },
    { offset: 1,           color: brightColor },
  ])
}

/**
 * Maps [step, value] pairs to per-point ECharts data objects, marking each
 * point bright if it is a new running maximum, dim otherwise.
 * Pass initialBest to carry over the best from a parent run in continuous view.
 */
function markNewBests(
  data: [number, number][],
  brightColor: string,
  dimColor: string,
  initialBest = -Infinity,
): { value: [number, number]; itemStyle: { color: string; borderColor: string; borderWidth: number } }[] {
  let best = initialBest
  return data.map(([step, val]) => {
    const isNewBest = val > best
    if (isNewBest) best = val
    return {
      value: [step, val],
      itemStyle: { color: isNewBest ? brightColor : dimColor, borderColor: '#0f172a', borderWidth: 2 },
    }
  })
}

export function makeOverviewOption(
  trainSeries: ChartPoint[],
  valSeries: ValPoint[],
  logScale = false,
  parentTrainSeries: ChartPoint[] = [],
  parentValSeries: ValPoint[] = [],
  continuousView = true,
  emaDecay = 0,
): EChartsOption {
  const hasPsnr = valSeries.some((p) => p.psnr !== undefined || p.psnrMasked !== undefined)
    || parentValSeries.some((p) => p.psnr !== undefined || p.psnrMasked !== undefined)
  const hasIou = valSeries.some((p) => p.iou !== undefined)
    || parentValSeries.some((p) => p.iou !== undefined)
  const hasTrainPsnr = valSeries.some((p) => p.trainPsnr !== undefined)
    || parentValSeries.some((p) => p.trainPsnr !== undefined)

  const hasParent = parentTrainSeries.length > 0 || parentValSeries.length > 0

  // ── Continuous view: concatenate parent history + current run offset ──────────
  if (hasParent && continuousView) {
    const parentLastTrainStep = parentTrainSeries.at(-1)?.step ?? 0
    const parentLastValStep = parentValSeries.at(-1)?.step ?? 0
    const parentLastStep = Math.max(parentLastTrainStep, parentLastValStep)
    const stepOffset = parentLastStep + 1

    const currentTrainOffset = trainSeries.map((p) => ({ ...p, step: p.step + stepOffset }))
    const currentValOffset = valSeries.map((p) => ({ ...p, step: p.step + stepOffset }))

    // EMA applied to the full concatenated train series so the smoothed line
    // transitions continuously across the phase boundary.
    const alpha = smoothingAlpha(parentTrainSeries.length + currentTrainOffset.length)
    const combinedRawLoss = [
      ...parentTrainSeries.map((p): [number, number | undefined] => [p.step, p.total]),
      ...currentTrainOffset.map((p): [number, number | undefined] => [p.step, p.total]),
    ]
    const combinedSmoothedLoss = applyEma(combinedRawLoss, alpha)
    const parentSmoothedLoss = combinedSmoothedLoss.slice(0, parentTrainSeries.length)
    const currentSmoothedLoss = combinedSmoothedLoss.slice(parentTrainSeries.length)

    const allTrainValues = combinedSmoothedLoss.map(([, v]) => v)
    const lossBounds = logScale ? dataBounds(allTrainValues) : null

    const xMin = Math.min(
      parentTrainSeries[0]?.step ?? Infinity,
      parentValSeries[0]?.step ?? Infinity,
    )
    const xMax = Math.max(
      currentTrainOffset.at(-1)?.step ?? 0,
      currentValOffset.at(-1)?.step ?? 0,
      // if current is empty, show at least to end of parent
      parentLastStep,
    )

    // Combined val series (in display coordinate space) for tooltip interpolation
    const effectiveValSeries = [...parentValSeries, ...currentValOffset]
    const psnrSecondaryLabel = hasIou && !hasPsnr ? 'IoU' : 'PSNR (mask)'

    // Running bests from the parent run, so new-best marking carries over correctly
    const parentBestPsnr = Math.max(-Infinity, ...parentValSeries.filter((p) => p.psnr !== undefined).map((p) => p.psnr!))
    const parentBestSecondary = hasIou && !hasPsnr
      ? Math.max(-Infinity, ...parentValSeries.filter((p) => p.iou !== undefined).map((p) => p.iou!))
      : Math.max(-Infinity, ...parentValSeries.filter((p) => p.psnrMasked !== undefined).map((p) => p.psnrMasked!))

    return {
      animation: true,
      animationDuration: 1500,
      animationEasing: 'quinticOut',
      animationDurationUpdate: 0,
      backgroundColor: 'transparent',
      textStyle: { color: '#cbd5e1', fontFamily: 'ui-sans-serif, system-ui, sans-serif' },
      grid: { left: 48, right: 52, top: 36, bottom: 36 },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'line', lineStyle: { type: 'solid', color: 'rgba(148, 163, 184, 0.4)', width: 1 } },
        backgroundColor: 'rgba(11, 18, 32, 0.72)',
        borderColor: 'rgba(71, 85, 105, 0.4)',
        borderRadius: 10,
        extraCssText: 'backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);',
        textStyle: { color: '#e2e8f0' },
        valueFormatter: (value) => formatChartNumber(value),
        formatter: (params) => formatOverviewTooltip(params, effectiveValSeries, { hasPsnr, hasIou, hasTrainPsnr }),
      },
      legend: {
        orient: 'horizontal',
        left: 'center',
        top: 4,
        itemGap: 16,
        textStyle: { color: '#94a3b8', fontSize: 11 },
        // Only list current-phase series; parent-phase series use the PARENT_SERIES_PREFIX
        // so they are automatically excluded because they don't match any legend entry.
        data: ['Loss', 'PSNR', psnrSecondaryLabel, ...(hasTrainPsnr ? ['PSNR (train)'] : [])],
      },
      xAxis: {
        type: 'value',
        min: Number.isFinite(xMin) ? xMin : undefined,
        max: xMax || undefined,
        axisLabel: { color: '#94a3b8', fontSize: 10 },
        splitLine: { lineStyle: { color: 'rgba(71, 85, 105, 0.22)' } },
      },
      yAxis: [
        {
          type: logScale ? 'log' : 'value',
          name: 'Loss',
          scale: true,
          min: lossBounds ? lossBounds.min * 0.95 : undefined,
          max: lossBounds ? lossBounds.max * 1.05 : undefined,
          nameTextStyle: { color: '#94a3b8', fontSize: 10 },
          axisLabel: { color: '#94a3b8', fontSize: 10, formatter: (value: number) => formatChartNumber(value) },
          splitLine: { lineStyle: { color: 'rgba(71, 85, 105, 0.22)' } },
        },
        {
          type: 'value',
          name: hasIou && !hasPsnr ? 'IoU' : 'PSNR / IoU',
          scale: true,
          nameTextStyle: { color: '#94a3b8', fontSize: 10 },
          axisLabel: { color: '#94a3b8', fontSize: 10, formatter: (value: number) => formatChartNumber(value) },
          splitLine: { show: false },
        },
      ],
      series: [
        // ── Phase 1 (parent) – faded ghost series ────────────────────────────
        {
          name: `${PARENT_SERIES_PREFIX}Loss`,
          type: 'line',
          yAxisIndex: 0,
          showSymbol: false,
          smooth: 0.2,
          animation: false,
          lineStyle: { width: 1.5, color: 'rgba(248,250,252,0.18)' },
          itemStyle: { color: 'rgba(248,250,252,0.18)' },
          data: parentSmoothedLoss,
          zlevel: 0,
        },
        {
          name: `${PARENT_SERIES_PREFIX}PSNR`,
          type: 'line',
          yAxisIndex: 1,
          showSymbol: false,
          smooth: 0.2,
          animation: false,
          lineStyle: { width: 1.5, color: 'rgba(251,113,133,0.22)' },
          itemStyle: { color: 'rgba(251,113,133,0.22)' },
          data: parentValSeries
            .filter((p) => p.psnr !== undefined)
            .map((p) => [p.step, p.psnr]),
          zlevel: 0,
        },
        {
          name: `${PARENT_SERIES_PREFIX}${psnrSecondaryLabel}`,
          type: 'line',
          yAxisIndex: 1,
          showSymbol: false,
          smooth: 0.2,
          animation: false,
          lineStyle: { width: 1.5, color: 'rgba(251,191,36,0.22)' },
          itemStyle: { color: 'rgba(251,191,36,0.22)' },
          data: parentValSeries
            .filter((p) => (hasIou && !hasPsnr ? p.iou !== undefined : p.psnrMasked !== undefined))
            .map((p) => [p.step, hasIou && !hasPsnr ? p.iou : p.psnrMasked]),
          zlevel: 0,
        },
        // ── Phase 2 (current run) – full brightness ──────────────────────────
        {
          name: 'Loss',
          type: 'line',
          yAxisIndex: 0,
          symbol: 'circle',
          symbolSize: 8,
          showSymbol: false,
          smooth: 0.2,
          animationEasing: 'cubicOut',
          animationDuration: 1500,
          lineStyle: { width: 2, color: '#f8fafc' },
          itemStyle: { color: '#f8fafc', borderColor: '#0f172a', borderWidth: 2 },
          data: currentSmoothedLoss,
          markLine: {
            silent: true,
            symbol: ['none', 'none'] as ['none', 'none'],
            animation: false,
            label: {
              show: true,
              position: 'insideStartTop' as const,
              formatter: 'Phase 2',
              fontSize: 10,
              color: '#64748b',
            },
            lineStyle: { type: 'dashed' as const, color: 'rgba(71,85,105,0.55)', width: 1 },
            data: [{ xAxis: stepOffset }],
          },
        },
        {
          name: 'PSNR',
          type: 'line',
          yAxisIndex: 1,
          symbol: 'circle',
          symbolSize: 6,
          showSymbol: true,
          smooth: 0.2,
          animationEasing: 'cubicOut' as const,
          animationDuration: 1500,
          lineStyle: { width: 2, color: psnrTailGradient(xMin, xMax, emaDecay, 'rgba(251,113,133,0.2)', '#fb7185') },
          itemStyle: { color: '#fb7185', borderColor: '#0f172a', borderWidth: 2 },
          data: markNewBests(
            currentValOffset.filter((p) => p.psnr !== undefined).map((p) => [p.step, p.psnr!]),
            '#fb7185', 'rgba(251,113,133,0.35)', parentBestPsnr,
          ),
        },
        {
          name: psnrSecondaryLabel,
          type: 'line',
          yAxisIndex: 1,
          symbol: 'circle',
          symbolSize: 6,
          showSymbol: true,
          smooth: 0.2,
          animationEasing: 'cubicOut' as const,
          animationDuration: 1500,
          lineStyle: { width: 2, color: psnrTailGradient(xMin, xMax, emaDecay, 'rgba(251,191,36,0.2)', '#fbbf24') },
          itemStyle: { color: '#fbbf24', borderColor: '#0f172a', borderWidth: 2 },
          data: markNewBests(
            currentValOffset
              .filter((p) => (hasIou && !hasPsnr ? p.iou !== undefined : p.psnrMasked !== undefined))
              .map((p) => [p.step, (hasIou && !hasPsnr ? p.iou : p.psnrMasked)!]),
            '#fbbf24', 'rgba(251,191,36,0.35)', parentBestSecondary,
          ),
        },
        ...(hasTrainPsnr ? [{
          name: 'PSNR (train)',
          type: 'line' as const,
          yAxisIndex: 1,
          showSymbol: false,
          smooth: 0.2,
          animationEasing: 'cubicOut' as const,
          animationDuration: 1500,
          lineStyle: { width: 1.5, type: 'dashed' as const, color: '#34d399' },
          itemStyle: { color: '#34d399' },
          data: [
            ...parentValSeries
              .filter((p) => p.trainPsnr !== undefined)
              .map((p): [number, number] => [p.step, p.trainPsnr!]),
            ...currentValOffset
              .filter((p) => p.trainPsnr !== undefined)
              .map((p): [number, number] => [p.step, p.trainPsnr!]),
          ],
        }] : []),
      ],
    }
  }

  // ── Reference line view (toggle OFF): show current run + horizontal baselines ─
  const alpha = smoothingAlpha(trainSeries.length)
  const rawLossData = trainSeries.map((p): [number, number | undefined] => [p.step, p.total])
  const smoothLossData = applyEma(rawLossData, alpha)
  const lossBounds = logScale ? dataBounds(smoothLossData.map(([, v]) => v)) : null
  const xMin = Math.min(trainSeries[0]?.step ?? Infinity, valSeries[0]?.step ?? Infinity)
  const xMax = Math.max(trainSeries.at(-1)?.step ?? 0, valSeries.at(-1)?.step ?? 0)
  const psnrSecondaryLabel = hasIou && !hasPsnr ? 'IoU' : 'PSNR (mask)'

  // Seed new-best tracking with parent run's best so the dot shading is correct
  // when showing current run alongside a reference line from a parent.
  const refBestPsnr = Math.max(-Infinity, ...parentValSeries.filter((p) => p.psnr !== undefined).map((p) => p.psnr!))
  const refBestSecondary = hasIou && !hasPsnr
    ? Math.max(-Infinity, ...parentValSeries.filter((p) => p.iou !== undefined).map((p) => p.iou!))
    : Math.max(-Infinity, ...parentValSeries.filter((p) => p.psnrMasked !== undefined).map((p) => p.psnrMasked!))

  // Reference lines: parent run's final values shown as dashed horizontal rules
  const parentFinalLoss = parentTrainSeries.at(-1)?.total
  const parentFinalPsnr = parentValSeries.at(-1)?.psnrMasked ?? parentValSeries.at(-1)?.psnr
  const parentFinalIou = parentValSeries.at(-1)?.iou

  const lossMarkLine =
    hasParent && parentFinalLoss != null && Number.isFinite(parentFinalLoss)
      ? {
          silent: true,
          symbol: ['none', 'none'] as ['none', 'none'],
          animation: false,
          label: {
            show: true,
            position: 'insideEndTop' as const,
            formatter: `Source: ${formatChartNumber(parentFinalLoss)}`,
            fontSize: 9,
            color: '#64748b',
          },
          lineStyle: { type: 'dashed' as const, color: 'rgba(248,250,252,0.28)', width: 1 },
          data: [{ yAxis: parentFinalLoss }],
        }
      : undefined

  const psnrRefValue = hasPsnr
    ? (parentFinalPsnr ?? null)
    : (!hasPsnr && hasIou ? (parentFinalIou ?? null) : null)
  const psnrMarkLine =
    hasParent && psnrRefValue != null && Number.isFinite(psnrRefValue)
      ? {
          silent: true,
          symbol: ['none', 'none'] as ['none', 'none'],
          animation: false,
          label: {
            show: true,
            position: 'insideEndTop' as const,
            formatter: `Source: ${formatChartNumber(psnrRefValue)}`,
            fontSize: 9,
            color: '#64748b',
          },
          lineStyle: { type: 'dashed' as const, color: 'rgba(251,191,36,0.45)', width: 1 },
          data: [{ yAxis: psnrRefValue }],
        }
      : undefined

  return {
    animation: true,
    animationDuration: 1500,
    animationEasing: 'quinticOut',
    animationDurationUpdate: 0,
    backgroundColor: 'transparent',
    textStyle: { color: '#cbd5e1', fontFamily: 'ui-sans-serif, system-ui, sans-serif' },
    grid: { left: 48, right: 52, top: 36, bottom: 36 },
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'line', lineStyle: { type: 'solid', color: 'rgba(148, 163, 184, 0.4)', width: 1 } },
      backgroundColor: 'rgba(11, 18, 32, 0.72)',
      borderColor: 'rgba(71, 85, 105, 0.4)',
      borderRadius: 10,
      extraCssText: 'backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);',
      textStyle: { color: '#e2e8f0' },
      valueFormatter: (value) => formatChartNumber(value),
      formatter: (params) => formatOverviewTooltip(params, valSeries, { hasPsnr, hasIou, hasTrainPsnr }),
    },
    legend: {
      orient: 'horizontal',
      left: 'center',
      top: 4,
      itemGap: 16,
      textStyle: { color: '#94a3b8', fontSize: 11 },
    },
    xAxis: {
      type: 'value',
      min: Number.isFinite(xMin) ? xMin : undefined,
      max: xMax || undefined,
      axisLabel: { color: '#94a3b8', fontSize: 10 },
      splitLine: { lineStyle: { color: 'rgba(71, 85, 105, 0.22)' } },
    },
    yAxis: [
      {
        type: logScale ? 'log' : 'value',
        name: 'Loss',
        scale: true,
        min: lossBounds ? lossBounds.min * 0.95 : undefined,
        max: lossBounds ? lossBounds.max * 1.05 : undefined,
        nameTextStyle: { color: '#94a3b8', fontSize: 10 },
        axisLabel: { color: '#94a3b8', fontSize: 10, formatter: (value: number) => formatChartNumber(value) },
        splitLine: { lineStyle: { color: 'rgba(71, 85, 105, 0.22)' } },
      },
      {
        type: 'value',
        name: hasIou && !hasPsnr ? 'IoU' : 'PSNR / IoU',
        scale: true,
        nameTextStyle: { color: '#94a3b8', fontSize: 10 },
        axisLabel: { color: '#94a3b8', fontSize: 10, formatter: (value: number) => formatChartNumber(value) },
        splitLine: { show: false },
      },
    ],
    series: [
      {
        name: 'Loss',
        type: 'line',
        yAxisIndex: 0,
        symbol: 'circle',
        symbolSize: 8,
        showSymbol: false,
        smooth: 0.2,
        animationEasing: 'cubicOut',
        animationDuration: 1500,
        lineStyle: { width: 2, color: '#f8fafc' },
        itemStyle: { color: '#f8fafc', borderColor: '#0f172a', borderWidth: 2 },
        data: smoothLossData,
        ...(lossMarkLine ? { markLine: lossMarkLine } : {}),
      },
      {
        name: 'PSNR',
        type: 'line',
        yAxisIndex: 1,
        symbol: 'circle',
        symbolSize: 6,
        showSymbol: true,
        smooth: 0.2,
        animationEasing: 'cubicOut',
        animationDuration: 1500,
        lineStyle: { width: 2, color: psnrTailGradient(xMin, xMax, emaDecay, 'rgba(251,113,133,0.2)', '#fb7185') },
        itemStyle: { color: '#fb7185', borderColor: '#0f172a', borderWidth: 2 },
        data: markNewBests(
          valSeries.filter((p) => p.psnr !== undefined).map((p) => [p.step, p.psnr!]),
          '#fb7185', 'rgba(251,113,133,0.35)', refBestPsnr,
        ),
      },
      {
        name: psnrSecondaryLabel,
        type: 'line',
        yAxisIndex: 1,
        symbol: 'circle',
        symbolSize: 6,
        showSymbol: true,
        smooth: 0.2,
        animationEasing: 'cubicOut',
        animationDuration: 1500,
        lineStyle: { width: 2, color: psnrTailGradient(xMin, xMax, emaDecay, 'rgba(251,191,36,0.2)', '#fbbf24') },
        itemStyle: { color: '#fbbf24', borderColor: '#0f172a', borderWidth: 2 },
        data: markNewBests(
          valSeries
            .filter((p) => (hasIou && !hasPsnr ? p.iou !== undefined : p.psnrMasked !== undefined))
            .map((p) => [p.step, (hasIou && !hasPsnr ? p.iou : p.psnrMasked)!]),
          '#fbbf24', 'rgba(251,191,36,0.35)', refBestSecondary,
        ),
        ...(psnrMarkLine ? { markLine: psnrMarkLine } : {}),
      },
      ...(hasTrainPsnr ? [{
        name: 'PSNR (train)',
        type: 'line' as const,
        yAxisIndex: 1,
        showSymbol: false,
        smooth: 0.2,
        animationEasing: 'cubicOut' as const,
        animationDuration: 1500,
        lineStyle: { width: 1.5, type: 'dashed' as const, color: '#34d399' },
        itemStyle: { color: '#34d399' },
        data: valSeries
          .filter((p) => p.trainPsnr !== undefined)
          .map((p): [number, number] => [p.step, p.trainPsnr!]),
      }] : []),
    ],
  }
}

export function makeLossTermOption(
  trainSeries: ChartPoint[],
  logScale = false,
  parentTrainSeries: ChartPoint[] = [],
  continuousView = true,
): EChartsOption {
  const hasParent = parentTrainSeries.length > 0

  // ── Continuous view ──────────────────────────────────────────────────────────
  if (hasParent && continuousView) {
    const parentLastStep = parentTrainSeries.at(-1)?.step ?? 0
    const stepOffset = parentLastStep + 1
    const currentTrainOffset = trainSeries.map((p) => ({ ...p, step: p.step + stepOffset }))

    // Union of term names across both phases (capped at 10)
    const termNames = Array.from(
      new Set([
        ...parentTrainSeries.flatMap((p) => Object.keys(p.terms)),
        ...currentTrainOffset.flatMap((p) => Object.keys(p.terms)),
      ]),
    ).slice(0, 10)

    const alpha = smoothingAlpha(parentTrainSeries.length + currentTrainOffset.length)

    // For each term: compute continuous EMA across the phase boundary
    const termSeries = termNames.map((term) => {
      const parentRaw = parentTrainSeries
        .filter((p) => p.terms[term] !== undefined)
        .map((p): [number, number | undefined] => [p.step, p.terms[term]])
      const currentRaw = currentTrainOffset
        .filter((p) => p.terms[term] !== undefined)
        .map((p): [number, number | undefined] => [p.step, p.terms[term]])

      // Apply EMA to parent first so the state carries over to current
      let ema: number | undefined
      const parentSmoothed: Array<[number, number | undefined]> = parentRaw.map(([step, value]) => {
        if (value === undefined || !Number.isFinite(value)) return [step, value]
        ema = ema === undefined ? value : alpha * value + (1 - alpha) * ema
        return [step, ema]
      })
      const currentSmoothed: Array<[number, number | undefined]> = currentRaw.map(([step, value]) => {
        if (value === undefined || !Number.isFinite(value)) return [step, value]
        ema = ema === undefined ? value : alpha * value + (1 - alpha) * ema
        return [step, ema]
      })

      return { term, parentData: parentSmoothed, currentData: currentSmoothed }
    })

    const allValues = termSeries.flatMap(({ parentData, currentData }) =>
      [...parentData, ...currentData].map(([, v]) => v),
    )
    const termBounds = logScale ? dataBounds(allValues) : null

    const xMin = parentTrainSeries[0]?.step ?? currentTrainOffset[0]?.step
    const xMax = Math.max(
      currentTrainOffset.at(-1)?.step ?? 0,
      parentLastStep,
    )

    const tooltipFormatter = (params: unknown) => {
      const rows = (Array.isArray(params) ? params : []).filter(
        (r) => !(typeof r?.seriesName === 'string' && r.seriesName.startsWith(PARENT_SERIES_PREFIX)),
      )
      if (!rows.length) return ''
      const step = typeof rows[0]?.axisValue === 'number' ? rows[0].axisValue : num(rows[0]?.axisValue)
      const header = step !== null ? formatInteger(Math.round(step ?? 0)) : ''
      const sorted = [...rows]
        .map((r) => ({
          marker: typeof r.marker === 'string' ? r.marker : '',
          name: typeof r.seriesName === 'string' ? r.seriesName : '',
          value: Array.isArray(r.value) ? (r.value[1] as number) : (r.value as number),
        }))
        .filter((r) => r.value != null && Number.isFinite(r.value))
        .sort((a, b) => b.value - a.value)
      const lines = sorted.map((r) => `${r.marker}${r.name}: <b>${formatChartNumber(r.value)}</b>`)
      return [header, ...lines].join('<br/>')
    }

    return {
      animation: true,
      animationDuration: 1500,
      animationEasing: 'quinticOut',
      animationDurationUpdate: 0,
      backgroundColor: 'transparent',
      textStyle: { color: '#cbd5e1', fontFamily: 'ui-sans-serif, system-ui, sans-serif' },
      grid: { left: 48, right: 18, top: 26, bottom: 36 },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'line', lineStyle: { type: 'solid', color: 'rgba(148, 163, 184, 0.4)', width: 1 } },
        backgroundColor: 'rgba(11, 18, 32, 0.72)',
        borderColor: 'rgba(71, 85, 105, 0.4)',
        borderRadius: 10,
        extraCssText: 'backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);',
        textStyle: { color: '#e2e8f0' },
        formatter: tooltipFormatter,
      },
      legend: {
        orient: 'horizontal',
        left: 'center',
        top: 4,
        itemGap: 16,
        textStyle: { color: '#94a3b8', fontSize: 11 },
        data: termNames.map((t) => humanizeKey(t)),
      },
      xAxis: {
        type: 'value',
        min: xMin ?? undefined,
        max: xMax || undefined,
        axisLabel: { color: '#94a3b8', fontSize: 10 },
        splitLine: { lineStyle: { color: 'rgba(71, 85, 105, 0.22)' } },
      },
      yAxis: {
        type: logScale ? 'log' : 'value',
        scale: true,
        min: termBounds ? termBounds.min * 0.9 : undefined,
        max: termBounds ? termBounds.max * 1.1 : undefined,
        axisLabel: { color: '#94a3b8', fontSize: 10, formatter: (value: number) => formatChartNumber(value) },
        splitLine: { lineStyle: { color: 'rgba(71, 85, 105, 0.22)' } },
      },
      series: [
        // Phase 1 ghost series (faded, filtered from tooltip + legend)
        ...termSeries.map(({ term, parentData }) => ({
          name: `${PARENT_SERIES_PREFIX}${humanizeKey(term)}`,
          type: 'line' as const,
          showSymbol: false,
          smooth: 0.15,
          animation: false,
          lineStyle: { width: 1.2, color: `${TERM_COLORS[term] ?? '#cbd5e1'}33` },
          itemStyle: { color: `${TERM_COLORS[term] ?? '#cbd5e1'}33` },
          data: parentData,
          zlevel: 0,
        })),
        // Phase 2 current-run series (bright)
        ...termSeries.map(({ term, currentData }, index) => ({
          name: humanizeKey(term),
          type: 'line' as const,
          symbol: 'circle',
          symbolSize: 8,
          showSymbol: false,
          smooth: 0.15,
          animationEasing: 'quinticOut' as const,
          animationDuration: 1500,
          lineStyle: { width: 1.8, color: TERM_COLORS[term] ?? '#cbd5e1' },
          itemStyle: { color: TERM_COLORS[term] ?? '#cbd5e1', borderColor: '#0f172a', borderWidth: 2 },
          data: currentData,
          // Only the first visible series carries the phase boundary marker
          ...(index === 0
            ? {
                markLine: {
                  silent: true,
                  symbol: ['none', 'none'] as ['none', 'none'],
                  animation: false,
                  label: {
                    show: true,
                    position: 'insideStartTop' as const,
                    formatter: 'Phase 2',
                    fontSize: 10,
                    color: '#64748b',
                  },
                  lineStyle: { type: 'dashed' as const, color: 'rgba(71,85,105,0.55)', width: 1 },
                  data: [{ xAxis: stepOffset }],
                },
              }
            : {}),
        })),
      ],
    }
  }

  // ── Standard view (no parent, or continuous view off) ───────────────────────
  const xMin = trainSeries[0]?.step
  const xMax = trainSeries.at(-1)?.step
  const termNames = Array.from(new Set(trainSeries.flatMap((point) => Object.keys(point.terms)))).slice(0, 10)
  const alpha = smoothingAlpha(trainSeries.length)

  // Pre-compute smoothed series so we can derive tight log-scale bounds
  const smoothedSeries = termNames.map((term) => {
    const rawData = trainSeries
      .filter((point) => point.terms[term] !== undefined)
      .map((point): [number, number | undefined] => [point.step, point.terms[term]])
    return { term, data: applyEma(rawData, alpha) }
  })
  const allTermValues = smoothedSeries.flatMap(({ data }) => data.map(([, v]) => v))
  const termBounds = logScale ? dataBounds(allTermValues) : null

  // Reference final values from parent (when continuous view is off but parent exists)
  // Not shown for loss terms to avoid clutter — the overview chart covers that.

  return {
    animation: true,
    animationDuration: 1500,
    animationEasing: 'quinticOut',
    animationDurationUpdate: 0,
    backgroundColor: 'transparent',
    textStyle: { color: '#cbd5e1', fontFamily: 'ui-sans-serif, system-ui, sans-serif' },
    grid: { left: 48, right: 18, top: 26, bottom: 36 },
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'line', lineStyle: { type: 'solid', color: 'rgba(148, 163, 184, 0.4)', width: 1 } },
      backgroundColor: 'rgba(11, 18, 32, 0.72)',
      borderColor: 'rgba(71, 85, 105, 0.4)',
      borderRadius: 10,
      extraCssText: 'backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);',
      textStyle: { color: '#e2e8f0' },
      formatter: (params: unknown) => {
        const rows = Array.isArray(params) ? params : []
        if (!rows.length) return ''
        const step = typeof rows[0]?.axisValue === 'number' ? rows[0].axisValue : num(rows[0]?.axisValue)
        const header = step !== null ? formatInteger(Math.round(step ?? 0)) : ''
        const sorted = [...rows]
          .map((r) => ({
            marker: typeof r.marker === 'string' ? r.marker : '',
            name: typeof r.seriesName === 'string' ? r.seriesName : '',
            value: Array.isArray(r.value) ? (r.value[1] as number) : (r.value as number),
          }))
          .filter((r) => r.value != null && Number.isFinite(r.value))
          .sort((a, b) => b.value - a.value)
        const lines = sorted.map((r) => `${r.marker}${r.name}: <b>${formatChartNumber(r.value)}</b>`)
        return [header, ...lines].join('<br/>')
      },
    },
    xAxis: {
      type: 'value',
      min: xMin ?? undefined,
      max: xMax ?? undefined,
      axisLabel: { color: '#94a3b8', fontSize: 10 },
      splitLine: { lineStyle: { color: 'rgba(71, 85, 105, 0.22)' } },
    },
    yAxis: {
      type: logScale ? 'log' : 'value',
      scale: true,
      min: termBounds ? termBounds.min * 0.9 : undefined,
      max: termBounds ? termBounds.max * 1.1 : undefined,
      axisLabel: { color: '#94a3b8', fontSize: 10, formatter: (value: number) => formatChartNumber(value) },
      splitLine: { lineStyle: { color: 'rgba(71, 85, 105, 0.22)' } },
    },
    series: smoothedSeries.map(({ term, data }) => ({
      name: humanizeKey(term),
      type: 'line',
      symbol: 'circle',
      symbolSize: 8,
      showSymbol: false,
      smooth: 0.15,
      animationEasing: 'quinticOut',
      animationDuration: 1500,
      lineStyle: { width: 1.8, color: TERM_COLORS[term] ?? '#cbd5e1' },
      itemStyle: { color: TERM_COLORS[term] ?? '#cbd5e1', borderColor: '#0f172a', borderWidth: 2 },
      data,
    })),
  }
}

export function isPrimitiveValue(value: unknown): value is string | number | boolean | null {
  return isPrimitive(value)
}
