import { Activity, Settings } from 'lucide-react'
import { Card, PanelCardContent, PanelCardHeader, PanelCardTitle } from '@/components/ui/card'
import type { FlatEntry } from '../../types'
import { EmptyState } from '../shared/EmptyState'
import { ValueTile } from '../shared/ValueTile'

export function RunSettingsPanel({
  modelOverviewEntries,
  hyperparameters,
  lossWeights,
}: {
  modelOverviewEntries: FlatEntry[]
  hyperparameters: FlatEntry[]
  lossWeights: FlatEntry[]
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
