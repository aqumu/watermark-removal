export type StoreCacheType = 'aligned' | 'legacy' | 'missing' | 'loading'

const CONFIG: Record<StoreCacheType, { label: string; className: string }> = {
  aligned:  { label: 'Aligned',  className: 'bg-emerald-500/15 text-emerald-500 border-emerald-500/30' },
  legacy:   { label: 'Simple',    className: 'bg-amber-500/15  text-amber-500  border-amber-500/30'  },
  missing:  { label: 'Missing',  className: 'bg-rose-500/15   text-rose-500   border-rose-500/30'   },
  loading:  { label: '…',        className: 'bg-muted/30      text-muted-foreground border-border/30' },
}

export function StoreCacheBadge({ type }: { type: StoreCacheType }) {
  const { label, className } = CONFIG[type]
  return (
    <span className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${className}`}>
      {label}
    </span>
  )
}
