import { AlertCircle } from 'lucide-react'

export function ErrorMessage({ text }: { text: string }) {
  return (
    <div className="flex items-start gap-2 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
      <AlertCircle className="mt-px h-3.5 w-3.5 shrink-0" />
      <span>{text}</span>
    </div>
  )
}
