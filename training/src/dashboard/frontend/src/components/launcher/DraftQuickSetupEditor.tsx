import React, { useState, useRef, useEffect, useCallback } from 'react'
import { XCircle, Rocket, Square } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Tooltip, TooltipTrigger, TooltipContent } from '@/components/ui/tooltip'
import type { DraftTaskType } from '../../types'
import { getNestedValue, normalizeDraftConfig, setNestedValue } from '../../lib/draft-utils'
import { humanizeKey } from '../../lib/format'
import { useLauncherContext } from '../../context/LauncherContext'
import { ErrorMessage } from '../shared/ErrorMessage'
import { EmptyState } from '../shared/EmptyState'
import { StoreCacheBadge } from '../shared/StoreCacheBadge'
import type { StoreCacheType } from '../shared/StoreCacheBadge'

// ── Types ──────────────────────────────────────────────────────────────────

type Resolution = '256' | '512'

type HardwareInfo = {
  device_type: 'cuda' | 'cpu'
  primary_name: string
  primary_vram_mb: number
  devices: { index: number; name: string; vram_mb: number }[]
}

// ── Presets ────────────────────────────────────────────────────────────────

const REMOVAL_PRESETS: Record<Resolution, {
  templateId: string
  image_width: number
  image_height: number
  ema_decay: number
}> = {
  '256': { templateId: 'train_256', image_width: 256, image_height: 128, ema_decay: 0.999 },
  '512': { templateId: 'train_512', image_width: 512, image_height: 256, ema_decay: 0.99  },
}

const RESTORATION_PRESETS: Record<Resolution, {
  templateId: string
  image_width: number
  image_height: number
  ema_decay: number
}> = {
  '256': { templateId: 'train_restoration_512', image_width: 256, image_height: 128, ema_decay: 0.99 },
  '512': { templateId: 'train_restoration_512', image_width: 512, image_height: 256, ema_decay: 0.99 },
}

const SEG_PRESETS: Record<Resolution, { image_size: number }> = {
  '256': { image_size: 256 },
  '512': { image_size: 512 },
}

const SEG_TEMPLATE_ID = 'seg'

// ── Batch size calculation ─────────────────────────────────────────────────

// Target effective batch size for all configurations
const EFFECTIVE_BATCH = 64

// Empirically calibrated batch sizes that fit safely in 8 GB VRAM.
// These are the ground truth — the presets already account for headroom,
// so no additional safety factor is applied on top.
const REFERENCE_VRAM_MB = 8192
const REFERENCE_BATCH: Record<string, number> = {
  'removal-256':       32,
  'removal-512':       16,
  'restoration-256':   32,
  'restoration-512':   16,
  'segmentation-256':   4,
  'segmentation-512':   2,
}

function prevPow2(n: number): number {
  if (n <= 1) return 1
  return Math.pow(2, Math.floor(Math.log2(n)))
}

function calcBatch(
  taskType: DraftTaskType,
  resolution: Resolution,
  vramMb: number,
): { batch_size: number; grad_accum_steps: number } {
  if (vramMb <= 0) {
    return { batch_size: 1, grad_accum_steps: EFFECTIVE_BATCH }
  }

  const refBatch = REFERENCE_BATCH[`${taskType}-${resolution}`] ?? 4
  // Scale linearly by VRAM ratio, round to nearest power of 2
  const scaled = refBatch * (vramMb / REFERENCE_VRAM_MB)
  const batch = Math.max(1, Math.min(prevPow2(Math.round(scaled)), EFFECTIVE_BATCH))
  const gradAccum = Math.max(1, Math.round(EFFECTIVE_BATCH / batch))
  return { batch_size: batch, grad_accum_steps: gradAccum }
}

function inferTaskName(taskType: DraftTaskType): string {
  if (taskType === 'segmentation') return 'segmentation'
  if (taskType === 'restoration') return 'restoration'
  return 'removal'
}

// ── Number formatting ──────────────────────────────────────────────────────

function formatNum(v: number | null | undefined): string {
  if (v == null) return ''
  if (v !== 0 && Math.abs(v) < 0.01) {
    return v.toExponential().replace(/\.0+(e)/, '$1').replace(/(\.\d*?)0+(e)/, '$1$2')
  }
  return String(v)
}

function parseNum(s: string): number | null {
  const n = Number(s)
  return Number.isFinite(n) ? n : null
}

// WD formula for removal: lr × 0.02 × √(epochs / 50)
// Anchored to train_256/512 defaults (lr=5e-3, epochs=50 → wd=1e-4).
function inferWeightDecay(lr: number, epochs: number): number {
  const base = lr * 0.02 * Math.sqrt(epochs / 50)
  const mag = Math.pow(10, Math.floor(Math.log10(base)))
  return Math.round(base / mag) * mag
}

// ── Sub-components ─────────────────────────────────────────────────────────

function SegmentedControl({
  options,
  value,
  onChange,
}: {
  options: { value: string; label: string }[]
  value: string
  onChange: (v: string) => void
}) {
  return (
    <div className="flex overflow-hidden rounded-md border border-border/60">
      {options.map((opt, i) => (
        <button
          key={opt.value}
          type="button"
          onClick={() => onChange(opt.value)}
          className={[
            'flex-1 py-1.5 text-sm font-medium transition-colors',
            i > 0 ? 'border-l border-border/60' : '',
            value === opt.value
              ? 'bg-primary text-primary-foreground'
              : 'bg-background text-muted-foreground hover:bg-secondary/50',
          ].join(' ')}
        >
          {opt.label}
        </button>
      ))}
    </div>
  )
}

function HelpTip({ text }: { text: string }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          tabIndex={-1}
          className="mr-1.5 inline-flex h-3.5 w-3.5 shrink-0 items-center justify-center rounded-full border border-border/60 text-[9px] font-bold text-muted-foreground transition-colors hover:border-primary/60 hover:text-primary"
        >
          ?
        </button>
      </TooltipTrigger>
      <TooltipContent side="top" className="max-w-[230px] leading-snug">
        {text}
      </TooltipContent>
    </Tooltip>
  )
}

function FieldRow({ label, hint, help, children }: { label: string; hint?: string; help?: string; children: React.ReactNode }) {
  return (
    <div className="grid gap-1.5 sm:grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)] sm:items-center">
      <div>
        <p className="flex items-center text-sm font-medium">
          {help && <HelpTip text={help} />}
          {label}
        </p>
        {hint && <p className="mt-0.5 text-[10px] leading-tight text-muted-foreground">{hint}</p>}
      </div>
      {children}
    </div>
  )
}

function NumInput({ value, onChange, placeholder }: {
  value: number | null | undefined
  onChange: (v: number) => void
  placeholder?: string
}) {
  const [raw, setRaw] = useState(() => formatNum(value))
  const focused = useRef(false)
  const lastExternal = useRef(value)

  useEffect(() => {
    if (value !== lastExternal.current) {
      lastExternal.current = value
      // Only overwrite what the user is typing when the field is not focused
      if (!focused.current) {
        setRaw(formatNum(value))
      }
    }
  }, [value])

  return (
    <Input
      value={raw}
      placeholder={placeholder ?? '0'}
      className="font-mono text-sm"
      onFocus={() => { focused.current = true }}
      onBlur={() => {
        focused.current = false
        const n = parseNum(raw)
        if (n !== null) { setRaw(formatNum(n)); onChange(n) }
        else setRaw(formatNum(lastExternal.current))
      }}
      onChange={(e) => {
        setRaw(e.target.value)
        const n = parseNum(e.target.value)
        if (n !== null) onChange(n)
      }}
    />
  )
}

// ── Step indicator (bottom) ────────────────────────────────────────────────

function StepNav({
  step,
  canAdvance,
  onBack,
  onNext,
  onStart,
  starting,
  showStart,
}: {
  step: 1 | 2
  canAdvance: boolean
  onBack: () => void
  onNext: () => void
  onStart: () => void
  starting: boolean
  showStart: boolean
}) {
  return (
    <div className="space-y-3 border-t border-border/40 pt-3">
      {/* (1) ── (2) indicator */}
      <div className="flex items-center gap-2 w-fit mx-auto">
        {[1, 2].map((n, i) => (
          <React.Fragment key={n}>
            <div
              className={[
                'flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-xs font-semibold transition-colors',
                step >= n
                  ? 'bg-primary text-primary-foreground'
                  : 'border border-border/60 text-muted-foreground',
              ].join(' ')}
            >
              {n}
            </div>
            {i < 1 && (
              <div className={['h-px w-8 transition-colors', step >= 2 ? 'bg-primary' : 'bg-border/40'].join(' ')} />
            )}
          </React.Fragment>
        ))}
      </div>

      {/* Navigation buttons */}
      {step === 1 ? (
        <Button className="w-full" disabled={!canAdvance} onClick={onNext}>
          Loss Weights →
        </Button>
      ) : (
        <div className="grid grid-cols-2 gap-2">
          <Button variant="outline" onClick={onBack}>
            ← Back
          </Button>
          {showStart && (
            <Button disabled={!canAdvance || starting} onClick={onStart}>
              <Rocket className="mr-1.5 h-3.5 w-3.5" />
              {starting ? 'Starting…' : 'Start'}
            </Button>
          )}
        </div>
      )}
    </div>
  )
}

// ── Store cache info row ───────────────────────────────────────────────────

function StoreCacheRow({ storePath, configPath, imageWidth, imageHeight }: {
  storePath: string | null
  configPath: string | null
  imageWidth: number | null
  imageHeight: number | null
}) {
  const [cacheType, setCacheType] = useState<StoreCacheType>('loading')
  const [isPrecomputing, setIsPrecomputing] = useState(false)
  const [progress, setProgress] = useState<{current: number; total: number} | null>(null)
  const [wantsAligned, setWantsAligned] = useState(false)

  const folderName = storePath
    ? (storePath.replace(/\\/g, '/').split('/').filter(Boolean).at(-1) ?? storePath)
    : null

  const checkStatus = useCallback(() => {
    if (!storePath) { setCacheType('missing'); return }
    const resString = imageWidth && imageHeight ? `${imageWidth}x${imageHeight}` : null
    const url = `/api/store-info?path=${encodeURIComponent(storePath)}${resString ? `&resolution=${resString}` : ''}`
    fetch(url)
      .then((r) => r.json())
      .then((d: { cache_type: string }) => {
        const t = d.cache_type
        if (t === 'aligned' || t === 'legacy' || t === 'missing') setCacheType(t)
        else setCacheType('missing')
      })
      .catch(() => setCacheType('missing'))
  }, [storePath, imageWidth, imageHeight])

  useEffect(() => {
    checkStatus()
  }, [checkStatus])

  useEffect(() => {
    fetch('/api/store-info/precompute')
      .then(r => r.json())
      .then(d => {
        setIsPrecomputing(Boolean(d.running))
      })
      .catch(() => {})
      
    const evs = new EventSource('/api/events')
    evs.onmessage = (e) => {
      try {
        const payload = JSON.parse(e.data)
        if (payload.type === 'precompute_progress') {
          if (payload.data.current < 0) {
            setProgress(null)
            setIsPrecomputing(false)
            setWantsAligned(false)
            checkStatus() // Refresh
          } else {
            setIsPrecomputing(true)
            setProgress({ current: payload.data.current, total: Math.max(1, payload.data.total) })
          }
        }
      } catch (err) {}
    }
    return () => evs.close()
  }, [checkStatus])

  async function togglePrecompute() {
    if (isPrecomputing) {
      await fetch('/api/store-info/precompute', { method: 'DELETE' })
      setWantsAligned(false)
    } else {
      if (!configPath) return // Cant start if we dont have the draft saved
      const r = await fetch('/api/store-info/precompute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config_path: configPath })
      })
      if (r.ok) {
        setWantsAligned(true)
        setProgress({ current: 0, total: 100 }) // placeholder until SSE catches up
      }
    }
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-2">
        <div className="min-w-0">
          <span className="text-sm text-muted-foreground">Mask cache</span>
          {folderName && (
            <span className="ml-2 font-mono text-xs text-foreground/70" title={storePath ?? undefined}>{folderName}</span>
          )}
        </div>
        <StoreCacheBadge type={cacheType} />
      </div>

      {(cacheType === 'legacy' || cacheType === 'missing' || isPrecomputing) && (
        <div className="flex flex-col gap-2 rounded-md bg-background/50 p-2.5 outline outline-1 -outline-offset-1 outline-border/40">
          <label className="flex items-center gap-2 cursor-pointer text-sm">
            <input 
              type="checkbox" 
              className="rounded border-border bg-background"
              checked={wantsAligned || isPrecomputing}
              onChange={(e) => { 
                setWantsAligned(e.target.checked)
                if (e.target.checked) { togglePrecompute() } 
              }}
              disabled={isPrecomputing}
            />
            <span className="font-medium flex items-center">
              Generate 'good' aligned cache
              <HelpTip text="Use the segmentation model to perfectly align cache elements natively. Fixes legacy jitter. Takes a little while." />
            </span>
          </label>
          
          {isPrecomputing && progress && (
            <div className="flex items-center gap-3">
              <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-secondary">
                <div 
                  className="h-full bg-primary transition-all duration-200" 
                  style={{ width: `${Math.min(100, Math.max(0, (progress.current / progress.total) * 100))}%` }} 
                />
              </div>
              <span className="text-[10px] font-mono text-muted-foreground">
                {progress.current}/{progress.total}
              </span>
              <Button type="button" variant="ghost" size="icon" className="h-6 w-6 shrink-0 text-destructive hover:bg-destructive/10" onClick={togglePrecompute}>
                <Square className="h-3.5 w-3.5 fill-current" />
              </Button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ── Main component ─────────────────────────────────────────────────────────

export function DraftQuickSetupEditor({
  flow = 'scratch',
  onStepChange,
}: {
  flow?: 'scratch' | 'checkpoint'
  onStepChange?: (step: number) => void
}) {
  const {
    templates,
    draftTaskType,
    setDraftTaskType,
    draftFamilyName,
    setDraftFamilyName,
    setDraftTemplateId,
    draftConfig,
    setDraftConfig,
    draftNameError,
    draftNameExists,
    draftFamilyExists,
    sanitizedDraftFamilyName,
    savedDrafts,
    draftDeleteError,
    deletingDraftName,
    onOpenDraft,
    onDeleteDraft,
    onSelectFresh,
    onStartSelected,
    startLaunching,
    startLaunchError,
  } = useLauncherContext()

  const [step, setStep] = useState<1 | 2>(1)
  const [resolution, setResolution] = useState<Resolution>('256')
  const [wdOverridden, setWdOverridden] = useState(false)
  // In checkpoint mode the source config already has sensible batch/grad-accum values —
  // don't overwrite them with a hardware-based auto-calculation.
  const [batchOverridden, setBatchOverridden] = useState(flow === 'checkpoint')
  const [hardware, setHardware] = useState<HardwareInfo | null>(null)

  const isSegmentation = draftTaskType === 'segmentation'
  const canStart = !draftNameError && !draftNameExists

  // Fetch hardware info once on mount
  useEffect(() => {
    fetch('/api/hardware')
      .then((r) => r.json())
      .then((d: HardwareInfo) => setHardware(d))
      .catch(() => setHardware({ device_type: 'cpu', primary_name: 'CPU (no CUDA)', primary_vram_mb: 0, devices: [] }))
  }, [])

  function goToStep(s: 1 | 2) {
    setStep(s)
    onStepChange?.(s)
  }

  // ── Config helpers ──

  function getNum(path: string[]): number | null {
    const v = getNestedValue(draftConfig, path)
    return typeof v === 'number' ? v : null
  }

  function setNum(path: string[], v: number) {
    setDraftConfig(setNestedValue(draftConfig, path, v))
  }

  // ── Template application ──

  function applyRemovalPreset(
    res: Resolution,
    overrides?: { lr?: number; epochs?: number; wd?: number; batch?: number; gradAccum?: number },
    hw?: HardwareInfo | null,
  ) {
    const preset = REMOVAL_PRESETS[res]
    const template = templates.find((t) => t.id === preset.templateId) ?? templates[0]
    let cfg = normalizeDraftConfig(template?.data ?? {})
    cfg = setNestedValue(cfg, ['dataset', 'image_width'], preset.image_width)
    cfg = setNestedValue(cfg, ['dataset', 'image_height'], preset.image_height)
    cfg = setNestedValue(cfg, ['training', 'ema_decay'], preset.ema_decay)
    const hwInfo = hw !== undefined ? hw : hardware
    const { batch_size, grad_accum_steps } = calcBatch('removal', res, hwInfo?.primary_vram_mb ?? 0)
    cfg = setNestedValue(cfg, ['training', 'batch_size'], overrides?.batch ?? batch_size)
    cfg = setNestedValue(cfg, ['training', 'grad_accum_steps'], overrides?.gradAccum ?? grad_accum_steps)
    if (overrides?.lr != null)     cfg = setNestedValue(cfg, ['training', 'lr'], overrides.lr)
    if (overrides?.epochs != null) cfg = setNestedValue(cfg, ['training', 'epochs'], overrides.epochs)
    if (overrides?.wd != null)     cfg = setNestedValue(cfg, ['training', 'weight_decay'], overrides.wd)
    setDraftTemplateId(preset.templateId)
    setDraftConfig(cfg)
  }

  function applyRestorationPreset(
    res: Resolution,
    overrides?: { lr?: number; epochs?: number; wd?: number; batch?: number; gradAccum?: number },
    hw?: HardwareInfo | null,
  ) {
    const preset = RESTORATION_PRESETS[res]
    const template = templates.find((t) => t.id === preset.templateId) ?? templates[0]
    let cfg = normalizeDraftConfig(template?.data ?? {})
    cfg = setNestedValue(cfg, ['dataset', 'image_width'], preset.image_width)
    cfg = setNestedValue(cfg, ['dataset', 'image_height'], preset.image_height)
    cfg = setNestedValue(cfg, ['training', 'ema_decay'], preset.ema_decay)
    cfg = setNestedValue(cfg, ['loss', 'bg_tv'], 0)
    cfg = setNestedValue(cfg, ['loss', 'bg_delta'], 0)
    const hwInfo = hw !== undefined ? hw : hardware
    const { batch_size, grad_accum_steps } = calcBatch('restoration', res, hwInfo?.primary_vram_mb ?? 0)
    cfg = setNestedValue(cfg, ['training', 'batch_size'], overrides?.batch ?? batch_size)
    cfg = setNestedValue(cfg, ['training', 'grad_accum_steps'], overrides?.gradAccum ?? grad_accum_steps)
    if (overrides?.lr != null)     cfg = setNestedValue(cfg, ['training', 'lr'], overrides.lr)
    if (overrides?.epochs != null) cfg = setNestedValue(cfg, ['training', 'epochs'], overrides.epochs)
    if (overrides?.wd != null)     cfg = setNestedValue(cfg, ['training', 'weight_decay'], overrides.wd)
    setDraftTemplateId(preset.templateId)
    setDraftConfig(cfg)
  }

  function applySegPreset(
    res: Resolution,
    overrides?: { lr?: number; epochs?: number; batch?: number; gradAccum?: number },
    hw?: HardwareInfo | null,
  ) {
    const preset = SEG_PRESETS[res]
    const template = templates.find((t) => t.id === SEG_TEMPLATE_ID) ?? templates[0]
    let cfg = normalizeDraftConfig(template?.data ?? {})
    cfg = setNestedValue(cfg, ['dataset', 'image_size'], preset.image_size)
    const hwInfo = hw !== undefined ? hw : hardware
    const { batch_size, grad_accum_steps } = calcBatch('segmentation', res, hwInfo?.primary_vram_mb ?? 0)
    cfg = setNestedValue(cfg, ['training', 'batch_size'], overrides?.batch ?? batch_size)
    cfg = setNestedValue(cfg, ['training', 'grad_accum_steps'], overrides?.gradAccum ?? grad_accum_steps)
    if (overrides?.lr != null)     cfg = setNestedValue(cfg, ['training', 'lr'], overrides.lr)
    if (overrides?.epochs != null) cfg = setNestedValue(cfg, ['training', 'epochs'], overrides.epochs)
    setDraftTemplateId(SEG_TEMPLATE_ID)
    setDraftConfig(cfg)
  }

  // Re-apply batch calculation when hardware info arrives (only if not overridden by user).
  // Skipped in checkpoint mode — source config batch values are authoritative.
  useEffect(() => {
    if (!hardware || !templates.length || batchOverridden || flow === 'checkpoint') return
    const lr     = getNum(['training', 'lr'])     ?? undefined
    const epochs = getNum(['training', 'epochs']) ?? undefined
    const wd     = wdOverridden ? (getNum(['training', 'weight_decay']) ?? undefined) : undefined
    if (isSegmentation) applySegPreset(resolution, { lr, epochs }, hardware)
    else if (draftTaskType === 'restoration') applyRestorationPreset(resolution, { lr, epochs, wd }, hardware)
    else applyRemovalPreset(resolution, { lr, epochs, wd }, hardware)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hardware])

  // ── Event handlers ──

  function onTypeChange(newType: string) {
    const t = newType as DraftTaskType
    setDraftTaskType(t)
    setWdOverridden(false)
    setBatchOverridden(false)
    setDraftFamilyName(`${inferTaskName(t)}_${resolution}`)
    if (t === 'segmentation') applySegPreset(resolution)
    else if (t === 'restoration') applyRestorationPreset(resolution)
    else applyRemovalPreset(resolution)
  }

  function onResolutionChange(res: string) {
    const r = res as Resolution
    setResolution(r)
    setWdOverridden(false)
    setBatchOverridden(false)
    setDraftFamilyName(`${inferTaskName(draftTaskType)}_${r}`)
    const lr     = getNum(['training', 'lr'])     ?? undefined
    const epochs = getNum(['training', 'epochs']) ?? undefined
    const wd     = wdOverridden ? (getNum(['training', 'weight_decay']) ?? undefined) : undefined
    if (isSegmentation) applySegPreset(r, { lr, epochs })
    else if (draftTaskType === 'restoration') applyRestorationPreset(r, { lr, epochs, wd })
    else applyRemovalPreset(r, { lr, epochs, wd })
  }

  function onLrChange(lr: number) {
    let cfg = setNestedValue(draftConfig, ['training', 'lr'], lr)
    if (!isSegmentation && !wdOverridden) {
      const epochs = (getNestedValue(cfg, ['training', 'epochs']) as number) ?? 50
      cfg = setNestedValue(cfg, ['training', 'weight_decay'], inferWeightDecay(lr, epochs))
    }
    if (isSegmentation) {
      cfg = setNestedValue(cfg, ['training', 'encoder_lr'], lr / 10)
    }
    setDraftConfig(cfg)
  }

  function onEpochsChange(epochs: number) {
    let cfg = setNestedValue(draftConfig, ['training', 'epochs'], epochs)
    if (!isSegmentation && !wdOverridden) {
      const lr = (getNestedValue(cfg, ['training', 'lr']) as number) ?? 5e-3
      cfg = setNestedValue(cfg, ['training', 'weight_decay'], inferWeightDecay(lr, epochs))
    }
    setDraftConfig(cfg)
  }

  function onWdChange(wd: number) {
    setNum(['training', 'weight_decay'], wd)
    setWdOverridden(true)
  }

  function onBatchChange(batch: number) {
    const gradAccum = Math.max(1, Math.round(EFFECTIVE_BATCH / batch))
    let cfg = setNestedValue(draftConfig, ['training', 'batch_size'], batch)
    cfg = setNestedValue(cfg, ['training', 'grad_accum_steps'], gradAccum)
    setDraftConfig(cfg)
    setBatchOverridden(true)
  }

  function handleStart() {
    onSelectFresh()
    void onStartSelected()
  }

  // ── Hardware hint ──

  const hwHint = hardware
    ? hardware.device_type === 'cuda'
      ? `Auto for ${hardware.primary_name} (${hardware.primary_vram_mb} MB)`
      : 'CPU detected — using minimum batch'
    : 'Detecting hardware…'

  // ── Loss field definitions ──

  const removalLossFields = [
    { path: ['loss', 'l1_masked'],        label: 'L1 Masked',       help: 'Mean absolute pixel error inside the watermark mask. The core term — punishes any brightness/color deviation from ground truth within the masked region.' },
    { path: ['loss', 'border'],           label: 'Border',          help: 'L1 loss on a dilated ring around the mask edge. Punishes color or brightness bleeding at the watermark boundary.' },
    { path: ['loss', 'perceptual'],       label: 'Perceptual',      help: 'VGG feature-space L1 loss between prediction and ground truth. Catches texture and structural mismatches that pure pixel losses miss.' },
    { path: ['loss', 'perceptual_every'], label: 'Perceptual Every', help: 'Run perceptual loss every N steps (e.g. 4 = once per 4 steps). Not a weight — controls compute cost. Set to 1 to run every step.' },
    { path: ['loss', 'saturation'],       label: 'Saturation',      help: 'Matches HSV saturation inside the mask. Punishes desaturation or over-saturation (gray or over-vivid patches in the restored area).' },
    { path: ['loss', 'color_moment'],     label: 'Color Moment',    help: 'Matches mean and variance of RGB channels inside the mask. Punishes overall color cast or contrast drift even when per-pixel loss is low.' },
    { path: ['loss', 'bg_tv'],            label: 'BG TV',           help: 'Total variation on background pixels (outside the mask). Punishes high-frequency noise or grain introduced outside the watermark area.' },
    { path: ['loss', 'bg_delta'],         label: 'BG Delta',        help: 'L1 between output and input pixels outside the mask. Directly punishes the model for touching any pixel it shouldn\'t change.' },
    { path: ['loss', 'edge_coherence'],   label: 'Edge Coherence',  help: 'Matches Sobel edge response at mask boundaries. Punishes blurry or artificially over-sharpened transitions at the watermark border.' },
  ]

  const segLossFields = [
    { path: ['training', 'loss_weights', 'bce'],   label: 'BCE',        help: 'Binary cross-entropy per pixel against the 0/1 mask. Penalizes wrong confidence for every pixel — the primary classification loss.' },
    { path: ['training', 'loss_weights', 'focal'],  label: 'Focal',     help: 'BCE re-weighted so uncertain/hard pixels dominate: each term is multiplied by (1−p)^γ. Punishes confident mistakes on difficult pixels more strongly.' },
    { path: ['training', 'loss_weights', 'l1'],     label: 'L1',        help: 'Mean absolute error between predicted soft mask and ground truth. A direct deviation penalty that complements BCE.' },
    { path: ['training', 'loss_weights', 'ms'],     label: 'Multi-Scale', help: 'BCE computed at ×1, ×½, and ×¼ resolution. Punishes missing large-scale mask structure that fine-detail losses can overlook.' },
    { path: ['training', 'loss_weights', 'dice'],   label: 'Dice',      help: '1 − 2|P∩G| / (|P|+|G|). Punishes overall shape mismatch; robust to class imbalance when watermark pixels are a small fraction of the image.' },
  ]

  const segExtraFields = [
    { path: ['training', 'pos_weight'],  label: 'Pos Weight',  help: 'Upweights the positive class (watermark pixels) in BCE. Raise it if the watermark covers a small area and the model keeps predicting all-background.' },
    { path: ['training', 'focal_gamma'], label: 'Focal Gamma', help: 'Exponent γ in focal loss. γ=0 equals plain BCE; γ=2 means hard pixels get ~4× more weight than easy ones. Typical values: 1–3.' },
  ]

  const lossFields = isSegmentation ? segLossFields : removalLossFields

  const otherFields = isSegmentation ? [
    { path: ['training', 'val_every'],    label: 'Val Every',    kind: 'number'  as const },
    { path: ['logging',  'log_every'],    label: 'Log Every',    kind: 'number'  as const },
    { path: ['training', 'amp'],          label: 'AMP',          kind: 'boolean' as const },
    { path: ['training', 'grad_clip'],    label: 'Grad Clip',    kind: 'number'  as const },
    { path: ['training', 'lr_scheduler'], label: 'LR Scheduler', kind: 'string'  as const },
    { path: ['dataset',  'train_split'],  label: 'Train Split',  kind: 'number'  as const },
    { path: ['dataset',  'num_workers'],  label: 'Num Workers',  kind: 'number'  as const },
    { path: ['dataset',  'max_samples'],  label: 'Dataset Size', kind: 'number'  as const },
    { path: ['model',    'encoder'],      label: 'Encoder',      kind: 'string'  as const },
  ] : [
    { path: ['training', 'val_every'],        label: 'Val Every',       kind: 'number'  as const },
    { path: ['logging',  'log_every'],        label: 'Log Every',       kind: 'number'  as const },
    { path: ['training', 'ema_decay'],        label: 'EMA Decay',       kind: 'number'  as const },
    { path: ['training', 'amp'],              label: 'AMP',             kind: 'boolean' as const },
    { path: ['training', 'grad_clip'],        label: 'Grad Clip',       kind: 'number'  as const },
    { path: ['training', 'lr_scheduler'],     label: 'LR Scheduler',    kind: 'string'  as const },
    { path: ['dataset',  'train_split'],      label: 'Train Split',     kind: 'number'  as const },
    { path: ['dataset',  'num_workers'],      label: 'Num Workers',     kind: 'number'  as const },
    { path: ['dataset',  'max_samples'],      label: 'Dataset Size',    kind: 'number'  as const },
    { path: ['model',    'use_checkpoint'],   label: 'Use Checkpoint',  kind: 'boolean' as const },
    { path: ['model',    'base_channels'],    label: 'Base Channels',   kind: 'number'  as const },
    { path: ['loss',     'loss_mask_blur_pct'], label: 'Mask Blur Pct', kind: 'number'  as const },
    { path: ['loss',     'erosion_kernel'],   label: 'Erosion Kernel',  kind: 'number'  as const },
  ]

  // ── Derived values for locked-settings display (checkpoint mode) ──────

  const lockedResolutionLabel = (() => {
    const w = getNum(['dataset', 'image_width'])
    const h = getNum(['dataset', 'image_height'])
    const s = getNum(['dataset', 'image_size'])
    if (w && h) return `${w} × ${h}`
    if (s) return `${s} × ${s}`
    return '—'
  })()

  // ── Render ─────────────────────────────────────────────────────────────

  return (
    <div className="space-y-4">

      {/* ── Step 1: Setup ─────────────────────────────────────────────── */}
      {step === 1 && (
        <>
          {/* Checkpoint mode: show locked settings as read-only, skip model/resolution pickers */}
          {flow === 'checkpoint' ? (
            <>
              <div className="rounded-md border border-border/40 bg-secondary/10 p-3">
                <p className="label-caps mb-2 text-muted-foreground">Locked (from source run)</p>
                <div className="space-y-1.5 text-sm">
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">Type</span>
                    <span className="font-medium">{humanizeKey(draftTaskType)}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">Resolution</span>
                    <span className="font-mono font-medium">{lockedResolutionLabel}</span>
                  </div>
                </div>
              </div>

              <div className="space-y-3 rounded-md border border-border/40 bg-secondary/15 p-3">
                <p className="label-caps">New Family</p>
                <FieldRow label="Family Name">
                  <Input
                    value={draftFamilyName}
                    onChange={(e) => setDraftFamilyName(e.target.value)}
                    placeholder={isSegmentation ? `segmentation_${lockedResolutionLabel.split(' ')[0] ?? '256'}_v2` : `removal_${lockedResolutionLabel.split(' ')[0] ?? '256'}_v2`}
                  />
                </FieldRow>
                {draftNameError && <ErrorMessage text={draftNameError} />}
                {!draftNameError && draftNameExists && (
                  <ErrorMessage text={`Draft already exists for '${sanitizedDraftFamilyName}'.`} />
                )}
                {!draftNameError && !draftNameExists && draftFamilyExists && (
                  <p className="text-xs text-muted-foreground flex items-center gap-1.5">
                    <Rocket className="h-3.5 w-3.5" />
                    Will be grouped with existing '{sanitizedDraftFamilyName}' runs.
                  </p>
                )}
              </div>
            </>
          ) : (
            <div className="space-y-3 rounded-md border border-border/40 bg-secondary/15 p-3">
              <p className="label-caps">Model</p>

              <FieldRow label="Type">
                <SegmentedControl
                  options={[
                    { value: 'removal', label: 'Removal' },
                    { value: 'restoration', label: 'Restoration' },
                    { value: 'segmentation', label: 'Segmentation' },
                  ]}
                  value={draftTaskType}
                  onChange={onTypeChange}
                />
              </FieldRow>

              <FieldRow label="Resolution" hint="Adjusts batch size and grad accum for effective batch 64">
                <SegmentedControl
                  options={[
                    { value: '256', label: isSegmentation ? '256 × 256' : '256 × 128' },
                    { value: '512', label: isSegmentation ? '512 × 512' : '512 × 256' },
                  ]}
                  value={resolution}
                  onChange={onResolutionChange}
                />
              </FieldRow>

              <FieldRow label="Family Name">
                <Input
                  value={draftFamilyName}
                  onChange={(e) => setDraftFamilyName(e.target.value)}
                  placeholder={isSegmentation ? `segmentation_${resolution}` : `removal_${resolution}`}
                />
              </FieldRow>

              {draftNameError && <ErrorMessage text={draftNameError} />}
              {!draftNameError && draftNameExists && (
                <ErrorMessage text={`Draft already exists for '${sanitizedDraftFamilyName}'.`} />
              )}
              {!draftNameError && !draftNameExists && draftFamilyExists && (
                <p className="text-xs text-muted-foreground flex items-center gap-1.5">
                  <Rocket className="h-3.5 w-3.5" />
                  Will be grouped with existing '{sanitizedDraftFamilyName}' runs.
                </p>
              )}
            </div>
          )}

          {!isSegmentation && (
            <div className="rounded-md border border-border/40 bg-secondary/15 px-3 py-2.5">
              <StoreCacheRow
                storePath={typeof (getNestedValue(draftConfig, ['dataset', 'preprocessed_store_dir'])) === 'string'
                  ? (getNestedValue(draftConfig, ['dataset', 'preprocessed_store_dir']) as string)
                  : null}
                configPath={flow !== 'checkpoint' && draftConfig && draftFamilyName
                  ? `${draftFamilyName}.yaml`
                  : null}
                imageWidth={getNum(['dataset', 'image_width']) ?? REMOVAL_PRESETS[resolution].image_width}
                imageHeight={getNum(['dataset', 'image_height']) ?? REMOVAL_PRESETS[resolution].image_height}
              />
            </div>
          )}

          <div className="space-y-3 rounded-md border border-border/40 bg-secondary/15 p-3">
            <p className="label-caps">Training</p>

            <FieldRow label="Epochs" help="Total passes over the training dataset. More epochs = more training time; diminishing returns after convergence.">
              <NumInput value={getNum(['training', 'epochs'])} onChange={onEpochsChange} />
            </FieldRow>

            <FieldRow label="Learning Rate" help="Step size for each optimizer update. Higher = faster convergence but risk of divergence or instability.">
              <NumInput value={getNum(['training', 'lr'])} onChange={onLrChange} />
            </FieldRow>

            {isSegmentation && (
              <FieldRow label="Encoder LR" hint="Auto-set to LR ÷ 10 — protects pretrained weights" help="Separate, lower LR for the pretrained encoder backbone. Prevents overwriting learned features while the decoder head trains faster.">
                <NumInput value={getNum(['training', 'encoder_lr'])} onChange={(v) => setNum(['training', 'encoder_lr'], v)} />
              </FieldRow>
            )}

            <FieldRow
              label="Weight Decay"
              hint={isSegmentation ? undefined : wdOverridden ? 'Manually set' : 'Auto: LR × 0.02 × √(epochs / 50)'}
              help="L2 regularization: shrinks all weights slightly toward zero each step. Prevents overfitting by penalizing large weights."
            >
              <NumInput value={getNum(['training', 'weight_decay'])} onChange={onWdChange} />
            </FieldRow>

            <FieldRow
              label="Batch Size"
              hint={batchOverridden ? `Grad accum: ${getNum(['training', 'grad_accum_steps']) ?? '—'} → effective ${(getNum(['training', 'batch_size']) ?? 1) * (getNum(['training', 'grad_accum_steps']) ?? 1)}` : hwHint}
              help="Number of samples in each forward/backward pass. Larger = more stable gradients but more VRAM. Halve it if you run out of memory."
            >
              <NumInput value={getNum(['training', 'batch_size'])} onChange={onBatchChange} />
            </FieldRow>

            <FieldRow
              label="Grad Accum"
              hint={`Effective batch: ${(getNum(['training', 'batch_size']) ?? 1) * (getNum(['training', 'grad_accum_steps']) ?? 1)}`}
              help="Accumulate gradients over N mini-batches before each optimizer step. Effective batch = Batch Size × Grad Accum. Lets you simulate a larger batch on limited VRAM."
            >
              <NumInput value={getNum(['training', 'grad_accum_steps'])} onChange={(v) => { setNum(['training', 'grad_accum_steps'], v); setBatchOverridden(true) }} />
            </FieldRow>
          </div>

          {/* Saved drafts — not shown in checkpoint mode (irrelevant) */}
          {flow !== 'checkpoint' && (
          <div className="rounded-md border border-border/40 bg-secondary/15 p-3">
            <div className="mb-2 flex items-center justify-between gap-2">
              <p className="label-caps">Saved Drafts</p>
              {draftDeleteError && <span className="text-xs text-destructive">Delete failed</span>}
            </div>
            {savedDrafts.length ? (
              <div className="space-y-1.5">
                {savedDrafts.slice(0, 6).map((draft) => (
                  <div key={draft.family_name} className="flex items-start justify-between gap-3 rounded-md border border-border/30 bg-background/50 px-3 py-2">
                    <button type="button" onClick={() => onOpenDraft(draft)} className="min-w-0 flex-1 text-left">
                      <p className="truncate text-sm font-medium">{draft.family_name}</p>
                      <p className="truncate text-xs text-muted-foreground">{humanizeKey(draft.task_type)} — {draft.template_id}</p>
                    </button>
                    <div className="flex shrink-0 items-center gap-0.5">
                      <Button type="button" variant="ghost" size="sm" className="h-7 px-2 text-xs" onClick={() => onOpenDraft(draft)}>Use</Button>
                      <Button
                        type="button" variant="ghost" size="icon"
                        className="h-7 w-7 text-muted-foreground hover:bg-destructive/10 hover:text-destructive"
                        onClick={() => void onDeleteDraft(draft.family_name)}
                        disabled={deletingDraftName === draft.family_name}
                      >
                        <XCircle className="h-3.5 w-3.5" />
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <EmptyState text="No saved drafts yet." />
            )}
          </div>
          )}
        </>
      )}

      {/* ── Step 2: Loss weights ───────────────────────────────────────── */}
      {step === 2 && (
        <>
          <div className="space-y-3 rounded-md border border-border/40 bg-secondary/15 p-3">
            <div>
              <p className="label-caps">Loss Weights</p>
              <p className="mt-0.5 text-xs text-muted-foreground">
                {flow === 'checkpoint'
                  ? 'Pre-filled from source run. Adjust to steer the next training phase.'
                  : isSegmentation
                    ? 'Segmentation mask loss components.'
                    : draftTaskType === 'restoration'
                      ? 'Direct restoration loss components. Background delta regularizers are disabled by default.'
                    : 'Removal loss components. Defaults are well-tuned — adjust only when experimenting.'}
              </p>
            </div>
            <div className="space-y-3">
              {lossFields.map((f) => (
                <FieldRow key={f.path.join('.')} label={f.label} help={f.help}>
                  <NumInput value={getNum(f.path)} onChange={(v) => setNum(f.path, v)} />
                </FieldRow>
              ))}
            </div>
          </div>

          {isSegmentation && (
            <div className="space-y-3 rounded-md border border-border/40 bg-secondary/15 p-3">
              <p className="label-caps">Segmentation Tuning</p>
              <div className="space-y-3">
                {segExtraFields.map((f) => (
                  <FieldRow key={f.path.join('.')} label={f.label} help={f.help}>
                    <NumInput value={getNum(f.path)} onChange={(v) => setNum(f.path, v)} />
                  </FieldRow>
                ))}
              </div>
            </div>
          )}

          <details className="rounded-md border border-border/40 bg-secondary/10 p-3">
            <summary className="label-caps cursor-pointer list-none select-none">Other Parameters</summary>
            <div className="mt-3 space-y-3">
              {otherFields.map((f) => {
                const v = getNestedValue(draftConfig, f.path)
                if (f.kind === 'boolean') return (
                  <FieldRow key={f.path.join('.')} label={f.label}>
                    <Select value={String(Boolean(v))} onValueChange={(next) => setDraftConfig(setNestedValue(draftConfig, f.path, next === 'true'))}>
                      <SelectTrigger className="w-full"><SelectValue /></SelectTrigger>
                      <SelectContent>
                        <SelectItem value="true">true</SelectItem>
                        <SelectItem value="false">false</SelectItem>
                      </SelectContent>
                    </Select>
                  </FieldRow>
                )
                if (f.kind === 'string') return (
                  <FieldRow key={f.path.join('.')} label={f.label}>
                    <Input value={typeof v === 'string' ? v : ''} onChange={(e) => setDraftConfig(setNestedValue(draftConfig, f.path, e.target.value))} />
                  </FieldRow>
                )
                return (
                  <FieldRow key={f.path.join('.')} label={f.label}>
                    <NumInput value={typeof v === 'number' ? v : null} onChange={(n) => setNum(f.path, n)} />
                  </FieldRow>
                )
              })}
            </div>
          </details>

          {startLaunchError && <ErrorMessage text={startLaunchError} />}
        </>
      )}

      {/* ── Step indicator + nav (always at bottom) ────────────────────── */}
      <StepNav
        step={step}
        canAdvance={canStart}
        onBack={() => goToStep(1)}
        onNext={() => goToStep(2)}
        onStart={handleStart}
        starting={startLaunching}
        showStart={flow === 'scratch'}
      />
    </div>
  )
}
