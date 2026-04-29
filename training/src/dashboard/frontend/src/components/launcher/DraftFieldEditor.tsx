import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import type { DraftField } from '../../types'
import { coerceDraftFieldValue } from '../../lib/draft-utils'

export function DraftFieldEditor({
  field,
  value,
  onChange,
}: {
  field: DraftField
  value: unknown
  onChange: (value: string | number | boolean | null) => void
}) {
  const currentValue = value ?? field.value
  return (
    <div className="grid gap-1.5 sm:grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)] sm:items-center">
      <div>
        <p className="label-caps">{field.label}</p>
        <p className="font-mono text-[10px] text-muted-foreground/70">{field.path.join('.')}</p>
      </div>
      {field.kind === 'boolean' ? (
        <Select
          value={String(Boolean(currentValue))}
          onValueChange={(nextValue) => onChange(nextValue === 'true')}
        >
          <SelectTrigger className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="true">true</SelectItem>
            <SelectItem value="false">false</SelectItem>
          </SelectContent>
        </Select>
      ) : (
        <Input
          value={currentValue === null || currentValue === undefined ? '' : String(currentValue)}
          onChange={(e) => onChange(coerceDraftFieldValue(e.target.value, field.kind))}
          placeholder={field.kind === 'number' ? '0' : 'value'}
          type={field.kind === 'number' ? 'number' : 'text'}
        />
      )}
    </div>
  )
}
