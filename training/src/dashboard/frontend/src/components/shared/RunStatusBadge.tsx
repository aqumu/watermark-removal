import { Activity, CheckCircle2, PlayCircle, Timer, XCircle } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import type { RunStatus } from '../../types'

const STATUS_CONFIG: Record<RunStatus, { icon: typeof PlayCircle; className: string; label: string }> = {
  running: { icon: PlayCircle, className: 'bg-success/10 text-success', label: 'Running' },
  paused: { icon: Timer, className: 'bg-warning/10 text-warning', label: 'Paused' },
  completed: { icon: CheckCircle2, className: 'bg-success/10 text-success', label: 'Done' },
  failed: { icon: XCircle, className: 'bg-destructive/10 text-destructive', label: 'Failed' },
  unknown: { icon: Activity, className: 'bg-secondary text-muted-foreground', label: 'Unknown' },
}

export function RunStatusBadge({ status, compact = false }: { status: RunStatus; compact?: boolean }) {
  const { icon: Icon, className, label } = STATUS_CONFIG[status]
  return (
    <Badge
      variant="secondary"
      className={`${className} ${compact ? 'h-4 gap-0.5 px-1 text-[10px]' : 'h-5 gap-1 text-xs'}`}
    >
      <Icon className={compact ? 'h-2.5 w-2.5' : 'h-3 w-3'} />
      {label}
    </Badge>
  )
}
