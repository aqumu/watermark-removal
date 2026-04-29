export function LauncherColumnHint({ title, text }: { title: string; text: string }) {
  return (
    <div className="rounded-lg border border-dashed border-border/50 bg-secondary/10 p-4">
      <p className="label-caps mb-1.5">{title}</p>
      <p className="text-xs leading-relaxed text-muted-foreground">{text}</p>
    </div>
  )
}
