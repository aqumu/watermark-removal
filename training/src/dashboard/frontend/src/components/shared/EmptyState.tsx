export function EmptyState({ text }: { text: string }) {
  return (
    <div className="rounded-md border border-dashed border-border/50 bg-secondary/10 px-3 py-3 text-center text-xs text-muted-foreground">
      {text}
    </div>
  )
}
