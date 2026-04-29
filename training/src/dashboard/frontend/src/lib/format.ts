import { PREFERRED_PREVIEW_ORDER } from './constants'

export function num(value: unknown): number | null {
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : null
  }
  if (typeof value === 'string' && value.trim() !== '') {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : null
  }
  return null
}

export function formatDecimal(value: number, digits = 4): string {
  return value.toFixed(digits)
}

export function formatChartNumber(value: unknown): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return value === null || value === undefined ? '' : String(value)
  }
  return new Intl.NumberFormat(undefined, {
    minimumFractionDigits: 0,
    maximumFractionDigits: 2,
  }).format(value)
}

export function formatInteger(value: number): string {
  return new Intl.NumberFormat().format(value)
}

export function formatUnknown(value: unknown): string {
  if (typeof value === 'number') {
    return Number.isInteger(value) ? String(value) : value.toPrecision(4)
  }
  if (typeof value === 'boolean') {
    return value ? 'true' : 'false'
  }
  if (value === null) {
    return 'null'
  }
  return String(value)
}

export function formatDelta(current: number | undefined, previous: number | undefined): string | undefined {
  if (current === undefined || previous === undefined || previous === 0) {
    return undefined
  }
  const delta = ((current - previous) / Math.abs(previous)) * 100
  const sign = delta > 0 ? '+' : ''
  return `${sign}${delta.toFixed(1)}%`
}

export function formatApiError(status: number, backendMessage: string | undefined, fallback: string): string {
  const prefix =
    status === 400 ? 'Invalid request' :
    status === 409 ? 'Conflict' :
    status >= 500 ? 'Server error' :
    `Error ${status}`
  const detail = backendMessage ? `: ${backendMessage}` : ''
  return `${prefix}${detail}` || fallback
}

export function paddedMin(min: number, max: number): number {
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return 0
  }
  if (min === max) {
    const pad = Math.abs(min || 1) * 0.1
    return min - pad
  }
  return min - (max - min) * 0.08
}

export function paddedMax(min: number, max: number): number {
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return 1
  }
  if (min === max) {
    const pad = Math.abs(max || 1) * 0.1
    return max + pad
  }
  return max + (max - min) * 0.08
}

export function humanizeKey(value: string): string {
  return value.replace(/_/g, ' ').replace(/\b\w/g, (char) => char.toUpperCase())
}

export function isPrimitive(value: unknown): value is string | number | boolean | null {
  return value === null || ['string', 'number', 'boolean'].includes(typeof value)
}

export function previewOrder(key: string): number {
  const index = PREFERRED_PREVIEW_ORDER.indexOf(key)
  return index === -1 ? PREFERRED_PREVIEW_ORDER.length + 1 : index
}

export function formatDurationSince(timestamp: string | undefined, now: number): string {
  if (!timestamp) {
    return 'n/a'
  }

  const createdAt = Date.parse(timestamp)
  if (Number.isNaN(createdAt)) {
    return timestamp
  }

  const diffMinutes = Math.max(0, Math.floor((now - createdAt) / 60000))
  const days = Math.floor(diffMinutes / 1440)
  const hours = Math.floor((diffMinutes % 1440) / 60)
  const minutes = diffMinutes % 60

  if (days > 0) {
    return `${days}d ${hours}h`
  }
  if (hours > 0) {
    return `${hours}h ${minutes}m`
  }
  return `${minutes}m`
}

export function flattenObject(source: Record<string, unknown>, preferredSections: string[]): { label: string; value: string }[] {
  const entries: { label: string; value: string }[] = []
  for (const sectionName of preferredSections) {
    const section = source[sectionName]
    if (section && typeof section === 'object' && !Array.isArray(section)) {
      for (const [key, value] of Object.entries(section as Record<string, unknown>)) {
        if (isPrimitive(value)) {
          entries.push({ label: `${humanizeKey(sectionName)}.${humanizeKey(key)}`, value: formatUnknown(value) })
        }
      }
    }
  }

  for (const [key, value] of Object.entries(source)) {
    if (preferredSections.includes(key)) {
      continue
    }
    if (isPrimitive(value)) {
      entries.push({ label: humanizeKey(key), value: formatUnknown(value) })
    }
  }
  return entries
}
