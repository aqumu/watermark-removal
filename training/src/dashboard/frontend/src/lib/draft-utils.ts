import type { DraftField } from '../types'

export function humanizeKey(value: string): string {
  return value.replace(/_/g, ' ').replace(/\b\w/g, (char) => char.toUpperCase())
}

export function sanitizeRunFamilyName(value: string): string {
  return value.trim().replace(/\s+/g, '_')
}

export function validateRunFamilyName(value: string): string | null {
  const sanitized = sanitizeRunFamilyName(value)
  if (!sanitized) {
    return 'Run family name is required.'
  }
  if (sanitized.length > 64) {
    return 'Run family name must be 64 characters or fewer.'
  }
  if (!/^[A-Za-z0-9][A-Za-z0-9_-]*$/.test(sanitized)) {
    return 'Use letters, digits, underscores, or dashes, and start with a letter or digit.'
  }
  return null
}

export function deepClone<T>(value: T): T {
  if (Array.isArray(value)) {
    return value.map((entry) => deepClone(entry)) as T
  }
  if (value && typeof value === 'object') {
    const result: Record<string, unknown> = {}
    for (const [key, nested] of Object.entries(value as Record<string, unknown>)) {
      result[key] = deepClone(nested)
    }
    return result as T
  }
  return value
}

export function normalizeDraftConfig(source: Record<string, unknown>): Record<string, unknown> {
  const copy = deepClone(source)
  delete copy.seed
  if (copy.logging && typeof copy.logging === 'object' && !Array.isArray(copy.logging)) {
    const logging = copy.logging as Record<string, unknown>
    delete logging.dir
    delete logging.family_dir
    delete logging.run_id
    delete logging.timeline_id
    delete logging.project_run
    delete logging.config_path
    if (!Object.keys(logging).length) {
      delete copy.logging
    }
  }
  if (copy.checkpointing && typeof copy.checkpointing === 'object' && !Array.isArray(copy.checkpointing)) {
    const checkpointing = copy.checkpointing as Record<string, unknown>
    delete checkpointing.dir
    delete checkpointing.resume_from
    if (!Object.keys(checkpointing).length) {
      delete copy.checkpointing
    }
  }
  return copy
}

export function setNestedValue(source: Record<string, unknown>, path: string[], value: unknown): Record<string, unknown> {
  const copy = deepClone(source)
  if (!path.length) {
    return copy
  }
  let cursor: Record<string, unknown> = copy
  for (let index = 0; index < path.length - 1; index += 1) {
    const segment = path[index]
    const next = cursor[segment]
    if (!next || typeof next !== 'object' || Array.isArray(next)) {
      cursor[segment] = {}
    }
    cursor = cursor[segment] as Record<string, unknown>
  }
  cursor[path[path.length - 1]] = value
  return copy
}

export function getNestedValue(source: Record<string, unknown>, path: string[]): unknown {
  let cursor: unknown = source
  for (const segment of path) {
    if (!cursor || typeof cursor !== 'object' || Array.isArray(cursor)) {
      return undefined
    }
    cursor = (cursor as Record<string, unknown>)[segment]
  }
  return cursor
}

export function inferDraftFieldKind(value: unknown): DraftField['kind'] {
  if (typeof value === 'boolean') {
    return 'boolean'
  }
  if (typeof value === 'number') {
    return 'number'
  }
  if (value === null) {
    return 'null'
  }
  return 'string'
}

export function coerceDraftFieldValue(value: string, kind: DraftField['kind']): string | number | boolean | null {
  if (kind === 'boolean') {
    return value === 'true'
  }
  if (kind === 'number') {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : 0
  }
  if (kind === 'null') {
    return null
  }
  return value
}

function isPrimitive(value: unknown): value is string | number | boolean | null {
  return value === null || ['string', 'number', 'boolean'].includes(typeof value)
}

function collectDraftFieldsFromObject(
  target: DraftField[],
  path: string[],
  source: Record<string, unknown>,
): void {
  for (const [key, value] of Object.entries(source)) {
    if (key === 'dir' || key === 'root' || key === 'seed' || key === 'logging' || key === 'checkpointing') {
      continue
    }
    const nextPath = [...path, key]
    if (value && typeof value === 'object' && !Array.isArray(value)) {
      collectDraftFieldsFromObject(target, nextPath, value as Record<string, unknown>)
    } else if (isPrimitive(value)) {
      target.push({
        path: nextPath,
        label: nextPath.map(humanizeKey).join('. '),
        value,
        kind: inferDraftFieldKind(value),
      })
    }
  }
}

export function collectDraftFields(source: Record<string, unknown>): DraftField[] {
  const fields: DraftField[] = []
  const sections = ['dataset', 'model', 'training', 'loss']
  for (const section of sections) {
    const value = source[section]
    if (value && typeof value === 'object' && !Array.isArray(value)) {
      collectDraftFieldsFromObject(fields, [section], value as Record<string, unknown>)
    }
  }

  for (const [key, value] of Object.entries(source)) {
    if (sections.includes(key) || key === 'logging' || key === 'checkpointing' || key === 'dashboard' || key === 'seed') {
      continue
    }
    if (isPrimitive(value)) {
      fields.push({
        path: [key],
        label: humanizeKey(key),
        value: value,
        kind: inferDraftFieldKind(value),
      })
    }
  }
  return fields
}

export function pickScratchQuickFields(fields: DraftField[]): DraftField[] {
  const preferredPatterns = [
    'dataset.image_size',
    'dataset.image_width',
    'dataset.image_height',
    'training.batch_size',
    'training.epochs',
    'training.num_epochs',
    'training.max_epochs',
    'training.lr',
    'training.learning_rate',
    'training.accumulate',
    'model.name',
    'model.arch',
    'model.backbone',
    'loss.weights',
  ]

  const selected: DraftField[] = []
  const seen = new Set<string>()

  for (const pattern of preferredPatterns) {
    const match = fields.find((field) => field.path.join('.').toLowerCase().includes(pattern))
    if (match && !seen.has(match.path.join('.'))) {
      selected.push(match)
      seen.add(match.path.join('.'))
    }
    if (selected.length >= 6) {
      return selected
    }
  }

  for (const field of fields) {
    const key = field.path.join('.')
    if (seen.has(key)) {
      continue
    }
    selected.push(field)
    if (selected.length >= 6) {
      break
    }
  }

  return selected
}
