import { useEffect, useState } from 'react'
import { Activity, Settings } from 'lucide-react'
import { Card, PanelCardContent, PanelCardHeader, PanelCardTitle } from '@/components/ui/card'
import type { FlatEntry } from '../../types'
import { EmptyState } from '../shared/EmptyState'
import { ValueTile } from '../shared/ValueTile'
import { StoreCacheBadge } from '../shared/StoreCacheBadge'

export function RunSettingsPanel({
  modelOverviewEntries,
  hyperparameters,
  lossWeights,
  storePath,
  isRemoval,
}: {
  modelOverviewEntries: FlatEntry[]
  hyperparameters: FlatEntry[]
  lossWeights: FlatEntry[]
  storePath: string | null
  isRemoval: boolean
}) {
  return (
    <div className="space-y-4 animate-in-up">
      <Card className="border-border/50">
        <PanelCardHeader>
          <PanelCardTitle>
            <Activity className="h-3.5 w-3.5 text-muted-foreground" />
            Run Summary
          </PanelCardTitle>
        </PanelCardHeader>
        <PanelCardContent>
          {modelOverviewEntries.length ? (
            <div className="grid grid-cols-1 gap-2 md:grid-cols-2 xl:grid-cols-4">
              {modelOverviewEntries.map((entry) => (
                <ValueTile key={entry.label} label={entry.label} value={entry.value} />
              ))}
            </div>
          ) : (
            <EmptyState text="No model overview data available for this run." />
          )}
        </PanelCardContent>
      </Card>

      <div className="grid grid-cols-1 gap-4 2xl:grid-cols-2">
        <Card className="border-border/50">
          <PanelCardHeader>
            <PanelCardTitle>
              <Settings className="h-3.5 w-3.5 text-muted-foreground" />
              Hyperparameters
            </PanelCardTitle>
          </PanelCardHeader>
          <PanelCardContent>
            {isRemoval && storePath && (
              <div className="mb-3">
                <StoreInfoTile storePath={storePath} />
              </div>
            )}
            {hyperparameters.length ? (
              <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
                {hyperparameters.map((entry) => (
                  <ValueTile key={entry.label} label={entry.label} value={entry.value} />
                ))}
              </div>
            ) : (
              <EmptyState text="No config.yaml data available for this run." />
            )}
          </PanelCardContent>
        </Card>

        <Card className="border-border/50">
          <PanelCardHeader>
            <PanelCardTitle>
              <Activity className="h-3.5 w-3.5 text-muted-foreground" />
              Loss Weights
            </PanelCardTitle>
          </PanelCardHeader>
          <PanelCardContent>
            {lossWeights.length ? (
              <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
                {lossWeights.map((entry) => (
                  <ValueTile key={entry.label} label={entry.label} value={entry.value} mono />
                ))}
              </div>
            ) : (
              <EmptyState text="No loss-weight values found in the parsed config." />
            )}
          </PanelCardContent>
        </Card>
      </div>
    </div>
  )
}

// ── Store info tile (run detail) ───────────────────────────────────────────────

type CacheType = 'aligned' | 'legacy' | 'missing' | 'loading'

function StoreInfoTile({ storePath }: { storePath: string }) {
  const [cacheType, setCacheType] = useState<CacheType>('loading')

  // Extract the last two path segments: signature folder + its parent name.
  // The signature folder looks like: removal-512x256-aab487f60709
  const folderName = storePath.replace(/\\/g, '/').split('/').filter(Boolean).at(-1) ?? storePath

  useEffect(() => {
    setCacheType('loading')
    fetch(`/api/store-info?path=${encodeURIComponent(storePath)}`)
      .then((r) => r.json())
      .then((d: { cache_type: string }) => {
        const t = d.cache_type
        if (t === 'aligned' || t === 'legacy' || t === 'missing') setCacheType(t)
        else setCacheType('missing')
      })
      .catch(() => setCacheType('missing'))
  }, [storePath])

  return (
    <div className="flex items-center justify-between gap-3 rounded-md border border-border/30 bg-secondary/30 px-3 py-2.5">
      <div className="min-w-0">
        <p className="label-caps mb-1">Mask Cache</p>
        <p className="truncate font-mono text-sm font-medium" title={storePath}>{folderName}</p>
      </div>
      <div className="shrink-0">
        <StoreCacheBadge type={cacheType} />
      </div>
    </div>
  )
}
