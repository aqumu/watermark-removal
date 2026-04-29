import { useEffect, useMemo, useState } from 'react'
import { Image, Layers, TrendingDown } from 'lucide-react'
import ReactECharts from 'echarts-for-react'
import { Badge } from '@/components/ui/badge'
import { Card, PanelCardContent, PanelCardHeader, PanelCardTitle } from '@/components/ui/card'
import type { ChartPoint, RunRecord, ValPoint } from '../../types'
import { buildPreviewCards, buildTrainSeries, buildValSeries, makeOverviewOption, makeLossTermOption } from '../../lib/run-utils'
import { EmptyState } from '../shared/EmptyState'
import { InferencePreviewPanel } from './InferencePreviewPanel'

export function RunDashboardWorkspace({
  trainSeries,
  valSeries,
  selectedRun,
  refreshKey,
}: {
  trainSeries: ChartPoint[]
  valSeries: ValPoint[]
  selectedRun: RunRecord | null
  refreshKey: number
}) {
  const [logScale, setLogScale] = useState(false)
  const [continuousView, setContinuousView] = useState(true)
  const [previewIndex, setPreviewIndex] = useState(0)

  const previewHistory = selectedRun?.preview_history ?? []

  // The backend caps preview_history at 50 entries — once full, new previews push out
  // the oldest without changing the array length. Track the last step explicitly so
  // we still snap to the newest entry even when the length no longer changes.
  const lastPreviewStep = previewHistory.at(-1)?.step ?? 0
  // Jump to latest preview when the run changes, history grows, or a new preview step arrives.
  useEffect(() => {
    setPreviewIndex(Math.max(previewHistory.length - 1, 0))
  }, [selectedRun?.manifest.run_id, previewHistory.length, lastPreviewStep])

  const currentPreview = previewHistory[Math.min(previewIndex, Math.max(previewHistory.length - 1, 0))]

  // Fetch parent run data when this run was started via load_weights from a checkpoint
  const parentRunId =
    selectedRun?.manifest.lineage?.start_mode === 'load_weights'
      ? (selectedRun.manifest.lineage.parent_run_id ?? null)
      : null

  const [parentRecord, setParentRecord] = useState<RunRecord | null>(null)
  useEffect(() => {
    if (!parentRunId) {
      setParentRecord(null)
      return
    }
    let cancelled = false
    fetch(`/api/runs/${encodeURIComponent(parentRunId)}`)
      .then((r) => (r.ok ? (r.json() as Promise<RunRecord>) : Promise.resolve(null)))
      .then((data) => { if (!cancelled) setParentRecord(data) })
      .catch(() => { if (!cancelled) setParentRecord(null) })
    return () => { cancelled = true }
  }, [parentRunId])

  const parentTrainSeries = useMemo(
    () => buildTrainSeries(parentRecord?.train_metrics ?? []),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [parentRecord?.train_metrics],
  )
  const parentValSeries = useMemo(
    () => buildValSeries(parentRecord?.val_metrics ?? []),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [parentRecord?.val_metrics],
  )

  // Show the toggle only when the parent run has actual metric data
  const hasParentData = parentTrainSeries.length > 0 || parentValSeries.length > 0

  const emaDecay = Number(selectedRun?.model_overview?.extra?.ema_decay ?? 0)

  const chartOption = useMemo(
    () => makeOverviewOption(trainSeries, valSeries, logScale, parentTrainSeries, parentValSeries, continuousView, emaDecay),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [trainSeries, valSeries, logScale, parentTrainSeries, parentValSeries, continuousView, emaDecay],
  )
  const lossTermOption = useMemo(
    () => makeLossTermOption(trainSeries, logScale, parentTrainSeries, continuousView),
    [trainSeries, logScale, parentTrainSeries, continuousView],
  )
  const previewCards = useMemo(
    () => buildPreviewCards(selectedRun, currentPreview, refreshKey),
    [currentPreview, refreshKey, selectedRun],
  )

  const hasChartHistory = trainSeries.length > 0 || valSeries.length > 0
  const hasPreviewHistory = previewCards.length > 0

  return (
    <div className="space-y-4 animate-in-up">
      {/* Chart toggles */}
      <div className="flex items-center justify-end gap-2">
        {hasParentData && (
          <button
            type="button"
            onClick={() => setContinuousView((v) => !v)}
            className={[
              'flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-xs font-medium transition-colors',
              continuousView
                ? 'border-primary/50 bg-primary/15 text-primary'
                : 'border-border/50 text-muted-foreground hover:border-border hover:text-foreground',
            ].join(' ')}
          >
            <span
              className={[
                'inline-block h-1.5 w-1.5 rounded-full transition-colors',
                continuousView ? 'bg-primary' : 'bg-muted-foreground/40',
              ].join(' ')}
            />
            Continuous view
          </button>
        )}
        <button
          type="button"
          onClick={() => setLogScale((v) => !v)}
          className={[
            'flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-xs font-medium transition-colors',
            logScale
              ? 'border-primary/50 bg-primary/15 text-primary'
              : 'border-border/50 text-muted-foreground hover:border-border hover:text-foreground',
          ].join(' ')}
        >
          <span
            className={[
              'inline-block h-1.5 w-1.5 rounded-full transition-colors',
              logScale ? 'bg-primary' : 'bg-muted-foreground/40',
            ].join(' ')}
          />
          Log scale
        </button>
      </div>

      {/* Loss charts */}
      <div className="grid grid-cols-1 gap-4 2xl:grid-cols-2">
        <Card className="border-border/50">
          <PanelCardHeader>
            <PanelCardTitle>
              <TrendingDown className="h-3.5 w-3.5 text-muted-foreground" />
              Loss & Validation
            </PanelCardTitle>
          </PanelCardHeader>
          <PanelCardContent>
            <div className="h-[620px]">
              {hasChartHistory ? (
                <ReactECharts option={chartOption} style={{ height: '100%', width: '100%' }} />
              ) : (
                <div className="flex h-full items-center justify-center">
                  <EmptyState text="No training or validation metrics yet." />
                </div>
              )}
            </div>
          </PanelCardContent>
        </Card>

        <Card className="border-border/50">
          <PanelCardHeader>
            <PanelCardTitle>
              <Layers className="h-3.5 w-3.5 text-muted-foreground" />
              Loss Terms
            </PanelCardTitle>
          </PanelCardHeader>
          <PanelCardContent>
            <div className="h-[620px]">
              {hasChartHistory ? (
                <ReactECharts option={lossTermOption} style={{ height: '100%', width: '100%' }} />
              ) : (
                <div className="flex h-full items-center justify-center">
                  <EmptyState text="No loss-term metrics yet." />
                </div>
              )}
            </div>
          </PanelCardContent>
        </Card>
      </div>

      {/* Inference previews */}
      <Card className="border-border/50">
        <PanelCardHeader>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <PanelCardTitle>
              <Image className="h-3.5 w-3.5 text-muted-foreground" />
              Inference Preview
            </PanelCardTitle>
            <Badge variant="secondary" className="font-mono text-xs tabular-nums">
              {currentPreview ? `Epoch ${parseInt(currentPreview.label, 10)}` : 'No previews'}
            </Badge>
          </div>
        </PanelCardHeader>
        <PanelCardContent>
          {hasPreviewHistory ? (
            <InferencePreviewPanel
              previewCards={previewCards}
              previewHistory={previewHistory}
              previewIndex={previewIndex}
              onPreviewIndexChange={setPreviewIndex}
              currentPreview={currentPreview}
            />
          ) : (
            <div className="flex min-h-[260px] items-center justify-center">
              <EmptyState text="No preview history yet." />
            </div>
          )}
        </PanelCardContent>
      </Card>
    </div>
  )
}
