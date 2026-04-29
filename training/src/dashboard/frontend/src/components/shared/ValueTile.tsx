export function ValueTile({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="rounded-md border border-border/30 bg-secondary/30 px-3 py-2.5">
      <p className="label-caps mb-1">{label}</p>
      <p className={`truncate text-sm font-medium ${mono ? 'font-mono tabular-nums' : ''}`}>{value}</p>
    </div>
  )
}
