import { useCallback, useEffect, useId, useMemo, useReducer, useRef, useState } from 'react'
import { ChevronDown, ChevronLeft, ChevronRight, ChevronUp, Image, RotateCcw, ZoomIn, ZoomOut } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Slider } from '@/components/ui/slider'
import type { PreviewCard, PreviewEntry } from '../../types'

// ─── View state (pan / zoom) ──────────────────────────────────────────────────

type ViewState = { zoom: number; pan: { x: number; y: number } }
type ViewAction =
  | { type: 'wheel'; deltaY: number; originX: number; originY: number }
  | { type: 'pan'; dx: number; dy: number }
  | { type: 'zoom-step'; dir: 1 | -1 }
  | { type: 'reset' }

const DEFAULT_VIEW: ViewState = { zoom: 1, pan: { x: 0, y: 0 } }
const ZOOM_FACTOR = 1.18
const ZOOM_MIN = 0.2
const ZOOM_MAX = 10

function clampZoom(z: number) {
  return Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, z))
}

function viewReducer(state: ViewState, action: ViewAction): ViewState {
  switch (action.type) {
    case 'wheel': {
      const factor = action.deltaY < 0 ? ZOOM_FACTOR : 1 / ZOOM_FACTOR
      const newZoom = clampZoom(state.zoom * factor)
      const scale = newZoom / state.zoom
      return {
        zoom: newZoom,
        pan: {
          x: action.originX - (action.originX - state.pan.x) * scale,
          y: action.originY - (action.originY - state.pan.y) * scale,
        },
      }
    }
    case 'pan':
      return { ...state, pan: { x: state.pan.x + action.dx, y: state.pan.y + action.dy } }
    case 'zoom-step': {
      const newZoom = clampZoom(state.zoom * (action.dir === 1 ? ZOOM_FACTOR : 1 / ZOOM_FACTOR))
      const scale = newZoom / state.zoom
      return {
        zoom: newZoom,
        pan: { x: state.pan.x * scale, y: state.pan.y * scale },
      }
    }
    case 'reset':
      return DEFAULT_VIEW
  }
}

// ─── Component ────────────────────────────────────────────────────────────────

const PREVIEW_DEFAULT_COUNT = 3

export function InferencePreviewPanel({
  previewCards,
  previewHistory,
  previewIndex,
  onPreviewIndexChange,
  currentPreview,
}: {
  previewCards: PreviewCard[]
  previewHistory: PreviewEntry[]
  previewIndex: number
  onPreviewIndexChange: (i: number) => void
  currentPreview: PreviewEntry | undefined
}) {
  // View controls — intentionally NOT reset on epoch change
  const [view, dispatchView] = useReducer(viewReducer, DEFAULT_VIEW)
  const [contrast, setContrast] = useState(1)
  const [sharpness, setSharpness] = useState(0)
  const [previewExpanded, setPreviewExpanded] = useState(false)
  const [isDragging, setIsDragging] = useState(false)

  const uid = useId()
  const filterId = `sharpen${uid.replace(/[^a-zA-Z0-9]/g, '')}`

  // Non-passive wheel listener so we can preventDefault and stop page scroll
  const gridRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const el = gridRef.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const rect = el.getBoundingClientRect()
      dispatchView({
        type: 'wheel',
        deltaY: e.deltaY,
        originX: e.clientX - rect.left - rect.width / 2,
        originY: e.clientY - rect.top - rect.height / 2,
      })
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [])

  // Drag-to-pan
  const dragRef = useRef<{ x: number; y: number } | null>(null)

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button !== 0) return
    dragRef.current = { x: e.clientX, y: e.clientY }
    setIsDragging(true)
    e.preventDefault()
  }, [])

  const onMouseMove = useCallback((e: React.MouseEvent) => {
    if (!dragRef.current) return
    const dx = e.clientX - dragRef.current.x
    const dy = e.clientY - dragRef.current.y
    dragRef.current = { x: e.clientX, y: e.clientY }
    dispatchView({ type: 'pan', dx, dy })
  }, [])

  const stopDrag = useCallback(() => {
    dragRef.current = null
    setIsDragging(false)
  }, [])

  // CSS filter built from contrast + optional SVG sharpness filter
  const cssFilter = useMemo(() => {
    const parts: string[] = []
    if (contrast !== 1) parts.push(`contrast(${contrast.toFixed(3)})`)
    if (sharpness > 0) parts.push(`url(#${filterId})`)
    return parts.join(' ') || 'none'
  }, [contrast, sharpness, filterId])

  // SVG sharpness kernel: 0 -a 0 / -a (1+4a) -a / 0 -a 0
  const kernelMatrix = useMemo(() => {
    const a = sharpness
    return `0 ${-a} 0 ${-a} ${1 + 4 * a} ${-a} 0 ${-a} 0`
  }, [sharpness])

  const imageTransform = `translate(${view.pan.x}px, ${view.pan.y}px) scale(${view.zoom})`

  const visibleCards = previewExpanded ? previewCards : previewCards.slice(0, PREVIEW_DEFAULT_COUNT)
  const hasMore = previewCards.length > PREVIEW_DEFAULT_COUNT

  const atStart = previewIndex <= 0
  const atEnd = previewIndex >= previewHistory.length - 1

  return (
    <div className="space-y-3">
      {/* SVG filter definition for sharpness */}
      {sharpness > 0 && (
        <svg
          aria-hidden="true"
          width="0"
          height="0"
          style={{ position: 'absolute', overflow: 'hidden', pointerEvents: 'none' }}
        >
          <defs>
            <filter id={filterId} x="0%" y="0%" width="100%" height="100%" colorInterpolationFilters="sRGB">
              <feConvolveMatrix
                order="3"
                kernelMatrix={kernelMatrix}
                divisor="1"
                bias="0"
                preserveAlpha="true"
              />
            </filter>
          </defs>
        </svg>
      )}

      {/* Image grid — pan/zoom interaction area */}
      <div
        ref={gridRef}
        className="relative select-none rounded-lg"
        style={{ cursor: isDragging ? 'grabbing' : 'grab' }}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={stopDrag}
        onMouseLeave={stopDrag}
      >
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
          {visibleCards.map((card) => (
            <div key={card.key} className="space-y-1.5">
              <span className="label-caps">{card.label}</span>
              <div className="aspect-video overflow-hidden rounded-md border border-border/50 bg-secondary/30">
                {card.url ? (
                  <img
                    src={card.url}
                    alt={card.label}
                    draggable={false}
                    className="h-full w-full object-cover pointer-events-none"
                    style={{
                      transform: imageTransform,
                      transformOrigin: 'center center',
                      filter: cssFilter,
                      willChange: 'transform',
                    }}
                  />
                ) : (
                  <div className="flex h-full items-center justify-center pointer-events-none">
                    <div className="text-center">
                      <Image className="mx-auto mb-1 h-6 w-6 text-muted-foreground/30" />
                      <span className="text-xs text-muted-foreground">No image</span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Show more / show less */}
      {hasMore && (
        <button
          type="button"
          onClick={() => setPreviewExpanded((v) => !v)}
          className="flex w-full items-center justify-center gap-1.5 rounded-md border border-border/40 py-1.5 text-xs text-muted-foreground transition-colors hover:border-border hover:text-foreground"
        >
          {previewExpanded ? (
            <><ChevronUp className="h-3.5 w-3.5" /> Show less</>
          ) : (
            <><ChevronDown className="h-3.5 w-3.5" /> Show {previewCards.length - PREVIEW_DEFAULT_COUNT} more</>
          )}
        </button>
      )}

      {/* Controls bar */}
      <div className="border-t border-border/50 pt-3 space-y-3">
        {/* Epoch slider row */}
        <div className="flex items-center gap-3">
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7 shrink-0"
            onClick={() => onPreviewIndexChange(Math.max(previewIndex - 1, 0))}
            disabled={atStart}
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <div className="flex-1 space-y-1.5">
            <div className="flex items-center justify-between font-mono text-xs tabular-nums text-muted-foreground">
              <span>Epoch {parseInt(previewHistory[0]?.label ?? '0', 10)}</span>
              <span>Epoch {parseInt(currentPreview?.label ?? '0', 10)}</span>
              <span>Epoch {parseInt(previewHistory.at(-1)?.label ?? '0', 10)}</span>
            </div>
            <Slider
              value={[previewIndex]}
              onValueChange={(value) => onPreviewIndexChange(value[0] ?? 0)}
              max={Math.max(previewHistory.length - 1, 0)}
              step={1}
              disabled={previewHistory.length <= 1}
            />
          </div>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7 shrink-0"
            onClick={() => onPreviewIndexChange(Math.min(previewIndex + 1, Math.max(previewHistory.length - 1, 0)))}
            disabled={atEnd}
          >
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>

        {/* Zoom + image adjustment controls */}
        <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
          {/* Zoom controls */}
          <div className="flex items-center gap-1">
            <button
              type="button"
              className="flex h-6 w-6 items-center justify-center rounded border border-border/50 text-muted-foreground transition-colors hover:border-border hover:text-foreground disabled:opacity-40"
              onClick={() => dispatchView({ type: 'zoom-step', dir: -1 })}
              disabled={view.zoom <= ZOOM_MIN}
              aria-label="Zoom out"
            >
              <ZoomOut className="h-3.5 w-3.5" />
            </button>
            <span className="w-12 text-center font-mono text-xs tabular-nums text-muted-foreground">
              {Math.round(view.zoom * 100)}%
            </span>
            <button
              type="button"
              className="flex h-6 w-6 items-center justify-center rounded border border-border/50 text-muted-foreground transition-colors hover:border-border hover:text-foreground disabled:opacity-40"
              onClick={() => dispatchView({ type: 'zoom-step', dir: 1 })}
              disabled={view.zoom >= ZOOM_MAX}
              aria-label="Zoom in"
            >
              <ZoomIn className="h-3.5 w-3.5" />
            </button>
            <button
              type="button"
              className="flex h-6 w-6 items-center justify-center rounded border border-border/50 text-muted-foreground transition-colors hover:border-border hover:text-foreground disabled:opacity-40"
              onClick={() => dispatchView({ type: 'reset' })}
              disabled={view.zoom === 1 && view.pan.x === 0 && view.pan.y === 0}
              aria-label="Reset view"
            >
              <RotateCcw className="h-3.5 w-3.5" />
            </button>
          </div>

          {/* Contrast slider */}
          <div className="flex min-w-[140px] flex-1 items-center gap-2">
            <span className="label-caps shrink-0">Contrast</span>
            <Slider
              value={[contrast]}
              onValueChange={([v]) => setContrast(v ?? 1)}
              min={1}
              max={3}
              step={0.02}
            />
            <span className="w-10 shrink-0 text-right font-mono text-xs tabular-nums text-muted-foreground">
              {contrast.toFixed(2)}x
            </span>
          </div>

          {/* Sharpness slider */}
          <div className="flex min-w-[140px] flex-1 items-center gap-2">
            <span className="label-caps shrink-0">Sharpness</span>
            <Slider
              value={[sharpness]}
              onValueChange={([v]) => setSharpness(v ?? 0)}
              min={0}
              max={2}
              step={0.05}
            />
            <span className="w-10 shrink-0 text-right font-mono text-xs tabular-nums text-muted-foreground">
              {sharpness.toFixed(2)}
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}
