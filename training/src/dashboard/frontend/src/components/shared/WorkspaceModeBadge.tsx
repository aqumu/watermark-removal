import { Badge } from '@/components/ui/badge'

type Tone = 'inspect' | 'resume' | 'active' | 'draft'

const TONE_CLASS: Record<Tone, string> = {
  active: 'border-success/30 bg-success/10 text-success',
  resume: 'border-warning/30 bg-warning/10 text-warning',
  inspect: 'border-primary/30 bg-primary/10 text-primary',
  draft: 'border-border/60 bg-secondary/40 text-muted-foreground',
}

export function WorkspaceModeBadge({ label, tone }: { label: string; tone: Tone }) {
  return (
    <Badge variant="outline" className={`h-5 px-2 text-xs ${TONE_CLASS[tone]}`}>
      {label}
    </Badge>
  )
}
