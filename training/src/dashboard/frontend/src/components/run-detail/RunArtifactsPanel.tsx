import { ChevronRight, FolderOpen } from 'lucide-react'
import { Card, PanelCardContent, PanelCardHeader, PanelCardTitle } from '@/components/ui/card'
import type { ArtifactEntry } from '../../types'
import { EmptyState } from '../shared/EmptyState'

export function RunArtifactsPanel({ artifacts }: { artifacts: ArtifactEntry[] }) {
  return (
    <Card className="border-border/50 animate-in-up">
      <PanelCardHeader>
        <PanelCardTitle>
          <FolderOpen className="h-3.5 w-3.5 text-muted-foreground" />
          Artifacts
        </PanelCardTitle>
      </PanelCardHeader>
      <PanelCardContent>
        {artifacts.length ? (
          <div className="space-y-1.5">
            {artifacts.map((artifact) => (
              <a
                key={`${artifact.category}:${artifact.name}:${artifact.url}`}
                href={artifact.url}
                target="_blank"
                rel="noreferrer"
                className="flex items-center justify-between gap-3 rounded-md border border-border/40 bg-secondary/20 px-3 py-2.5 transition-colors hover:bg-secondary/40"
              >
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium">{artifact.name}</p>
                  <p className="label-caps mt-0.5">{artifact.category}</p>
                </div>
                <ChevronRight className="h-4 w-4 shrink-0 text-muted-foreground" />
              </a>
            ))}
          </div>
        ) : (
          <EmptyState text="No staged artifacts are available for this run yet." />
        )}
      </PanelCardContent>
    </Card>
  )
}
