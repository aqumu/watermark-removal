import type { TrendMetric } from '../../types'

export function MetricRailCard({ metric }: { metric: TrendMetric }) {
  const trendClass =
    metric.positive === null || metric.positive === undefined
      ? 'bg-secondary text-muted-foreground'
      : metric.positive
        ? 'bg-success/10 text-success'
        : 'bg-destructive/10 text-destructive'

  return (
    <div className="flex items-center justify-between gap-3 rounded-md border border-border/40 bg-secondary/20 px-3 py-2.5 transition-colors hover:bg-secondary/30">
      <div className="min-w-0">
        <p className="label-caps">{metric.label}</p>
        <p className="mt-1 truncate font-mono text-sm font-semibold tabular-nums tracking-tight">
          {metric.value}
        </p>
      </div>
      {metric.trend ? (
        <span className={`shrink-0 rounded-full px-1.5 py-0.5 font-mono text-xs font-medium tabular-nums ${trendClass}`}>
          {metric.trend}
        </span>
      ) : null}
    </div>
  )
}
