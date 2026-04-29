import { FolderOpen, PlusCircle, Rocket } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, SectionCardContent, SectionCardHeader, SectionCardTitle } from '@/components/ui/card'
import { Dialog, DialogContent } from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import type {
  DraftField,
  DraftRecord,
  DraftTaskType,
  LauncherSelection,
  OrchestrationSnapshot,
  RunRecord,
  TemplateOption,
} from '../../types'
import { getNestedValue, normalizeDraftConfig, setNestedValue } from '../../lib/draft-utils'
import { humanizeKey } from '../../lib/format'
import { resolveRunStatus } from '../../lib/run-utils'
import { EmptyState } from '../shared/EmptyState'
import { RunStatusBadge } from '../shared/RunStatusBadge'
import { DraftFieldEditor } from './DraftFieldEditor'

export function BootstrapDialog({
  open,
  onOpenChange,
  latestRuns,
  templates,
  launcherSelection,
  draftTaskType,
  setDraftTaskType,
  draftFamilyName,
  setDraftFamilyName,
  draftTemplateId,
  setDraftTemplateId,
  draftConfig,
  setDraftConfig,
  draftFields,
  draftSaving,
  draftSaveError,
  draftNameError,
  draftNameExists,
  draftFamilyExists,
  sanitizedDraftFamilyName,
  savedDrafts,
  orchestration,
  onOpenDraft,
  onSelectRun,
  onCreateDraft,
}: {
  open: boolean
  onOpenChange: (nextOpen: boolean) => void
  latestRuns: RunRecord[]
  templates: TemplateOption[]
  launcherSelection: LauncherSelection
  draftTaskType: DraftTaskType
  setDraftTaskType: (value: DraftTaskType) => void
  draftFamilyName: string
  setDraftFamilyName: (value: string) => void
  draftTemplateId: string
  setDraftTemplateId: (value: string) => void
  draftConfig: Record<string, unknown>
  setDraftConfig: (value: Record<string, unknown>) => void
  draftFields: DraftField[]
  scratchQuickFields: DraftField[]
  draftSaving: boolean
  draftSaveError: string | null
  draftNameError: string | null
  draftNameExists: boolean
  draftFamilyExists: boolean
  sanitizedDraftFamilyName: string
  savedDrafts: DraftRecord[]
  orchestration: OrchestrationSnapshot | undefined
  onOpenDraft: (draft: DraftRecord) => void
  onSelectRun: (runId: string) => void
  onCreateDraft: () => void | Promise<void>
}) {
  const template = templates.find((entry) => entry.id === draftTemplateId) ?? templates[0]
  const templateForTaskType = (taskType: DraftTaskType) =>
    templates.find((entry) => entry.taskType === taskType && entry.id !== 'blank')
    ?? templates.find((entry) => entry.taskType === taskType)
    ?? templates[0]

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className="w-[min(calc(100vw-1rem),1800px)] max-w-none sm:max-w-none max-h-[94vh] overflow-y-auto overflow-x-hidden p-4 sm:p-4"
        showCloseButton={launcherSelection.kind !== 'home'}
      >
        <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.25fr)]">
          <Card className="min-w-0 border-border/50">
            <SectionCardHeader>
              <SectionCardTitle>
                <FolderOpen className="h-4 w-4 text-muted-foreground" />
                Inspect existing runs
              </SectionCardTitle>
            </SectionCardHeader>
            <SectionCardContent className="min-w-0 space-y-2">
              {latestRuns.length ? (
                latestRuns.slice(0, 8).map((run) => (
                  <button
                    key={run.manifest.run_id}
                    onClick={() => onSelectRun(run.manifest.run_id)}
                    className="group w-full rounded-md border border-border/40 bg-secondary/20 px-3 py-2 text-left transition-colors hover:bg-secondary/50"
                  >
                    <div className="mb-1 flex items-center justify-between gap-2">
                      <p className="truncate font-mono text-xs text-muted-foreground">{run.manifest.run_id}</p>
                      <RunStatusBadge status={resolveRunStatus(run, orchestration)} compact />
                    </div>
                    <p className="truncate text-xs text-foreground">{run.manifest.task_name ?? 'Unknown task'}</p>
                    <p className="mt-1 text-xs text-muted-foreground">
                      {resolveRunStatus(run, orchestration) === 'paused'
                        ? 'Open to resume this run.'
                        : 'Read-only view.'}
                    </p>
                  </button>
                ))
              ) : (
                <EmptyState text={open ? 'No existing runs indexed yet.' : 'No runs found yet.'} />
              )}
            </SectionCardContent>
          </Card>

          <Card className="min-w-0 border-border/50">
            <SectionCardHeader>
              <SectionCardTitle>
                <PlusCircle className="h-4 w-4 text-muted-foreground" />
                Start a new run
              </SectionCardTitle>
            </SectionCardHeader>
            <SectionCardContent className="min-w-0 space-y-4">
              <div className="space-y-2">
                <p className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Template</p>
                <Select
                  value={draftTemplateId}
                  onValueChange={(value) => {
                    const nextTemplate = templates.find((entry) => entry.id === value) ?? templates[0]
                    setDraftTemplateId(nextTemplate.id)
                    setDraftTaskType(nextTemplate.taskType)
                    setDraftFamilyName(nextTemplate.suggestedFamilyName)
                    setDraftConfig(normalizeDraftConfig(nextTemplate.data ?? {}))
                  }}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Choose a template" />
                  </SelectTrigger>
                  <SelectContent>
                    {templates.map((entry) => (
                      <SelectItem key={entry.id} value={entry.id}>
                        {entry.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground">{template?.description}</p>
              </div>

              <div className="space-y-2">
                <p className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Task type</p>
                <Select
                  value={draftTaskType}
                  onValueChange={(value) => {
                    const nextTaskType = value as DraftTaskType
                    const nextTemplate = templateForTaskType(nextTaskType)
                    setDraftTaskType(nextTaskType)
                    setDraftTemplateId(nextTemplate.id)
                    setDraftFamilyName(nextTemplate.suggestedFamilyName)
                    setDraftConfig(normalizeDraftConfig(nextTemplate.data ?? {}))
                  }}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Choose task type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="removal">Removal</SelectItem>
                    <SelectItem value="restoration">Restoration</SelectItem>
                    <SelectItem value="segmentation">Segmentation</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <p className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Run family name</p>
                <Input
                  value={draftFamilyName}
                  onChange={(event) => setDraftFamilyName(event.target.value)}
                  placeholder="removal_256"
                />
                <p className="text-xs text-muted-foreground">
                  This is the dashboard-facing name. It must be path-safe and unique among saved dashboard drafts.
                </p>
                {draftNameError ? <EmptyState text={draftNameError} /> : null}
                {!draftNameError && draftNameExists ? (
                  <div className="space-y-2">
                    <EmptyState text={`A dashboard draft already exists for '${sanitizedDraftFamilyName}'.`} />
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      className="w-full gap-2"
                      onClick={() => {
                        const existingDraft = savedDrafts.find((draft) => draft.family_name === sanitizedDraftFamilyName)
                        if (existingDraft) {
                          onOpenDraft(existingDraft)
                        }
                      }}
                      disabled={!savedDrafts.some((draft) => draft.family_name === sanitizedDraftFamilyName)}
                    >
                      <FolderOpen className="h-3.5 w-3.5" />
                      Open existing draft
                    </Button>
                  </div>
                ) : null}
                {!draftNameError && !draftNameExists && draftFamilyExists ? (
                  <div className="flex items-center gap-2 rounded-md bg-secondary/30 px-3 py-2 text-xs text-muted-foreground border border-border/40">
                    <Rocket className="h-3.5 w-3.5 shrink-0" />
                    Will be grouped with existing '{sanitizedDraftFamilyName}' runs.
                  </div>
                ) : null}
              </div>

              <div className="space-y-2 rounded-md border border-border/50 bg-secondary/20 p-3">
                <p className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Saved drafts</p>
                {savedDrafts.length ? (
                  <div className="space-y-2">
                    {savedDrafts.slice(0, 6).map((draft) => (
                      <button
                        key={draft.family_name}
                        type="button"
                        onClick={() => onOpenDraft(draft)}
                        className="flex w-full items-start justify-between gap-3 rounded-md border border-border/40 bg-background/60 px-3 py-2 text-left transition-colors hover:bg-background"
                      >
                        <div className="min-w-0">
                          <p className="truncate text-sm font-medium">{draft.family_name}</p>
                          <p className="truncate text-xs text-muted-foreground">
                            {humanizeKey(draft.task_type)} template: {draft.template_id}
                          </p>
                        </div>
                        <FolderOpen className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                      </button>
                    ))}
                  </div>
                ) : (
                  <EmptyState text="No saved drafts yet." />
                )}
              </div>

              <div className="rounded-md border border-border/50 bg-secondary/20 p-3">
                <div className="mb-3 flex items-center justify-between gap-2">
                  <div>
                    <p className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Hyperparameters</p>
                    <p className="text-xs text-muted-foreground">Primitive values only. Infrastructure paths stay hidden.</p>
                  </div>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="h-8 px-3 text-xs"
                    onClick={() => setDraftConfig(normalizeDraftConfig(template?.data ?? {}))}
                  >
                    Reset
                  </Button>
                </div>
                <div className="space-y-3">
                  {draftFields.length ? (
                    draftFields.map((field) => (
                      <DraftFieldEditor
                        key={field.path.join('.')}
                        field={field}
                        value={getNestedValue(draftConfig, field.path)}
                        onChange={(nextValue) => setDraftConfig(setNestedValue(draftConfig, field.path, nextValue))}
                      />
                    ))
                  ) : (
                    <EmptyState text="No editable hyperparameters were found in this template." />
                  )}
                </div>
              </div>

              {draftSaveError ? <EmptyState text={draftSaveError} /> : null}

              <Button
                className="w-full gap-2"
                disabled={Boolean(draftNameError) || draftNameExists || draftSaving}
                onClick={onCreateDraft}
              >
                <Rocket className="h-4 w-4" />
                {draftSaving ? 'Saving...' : 'Continue to dashboard'}
              </Button>
            </SectionCardContent>
          </Card>
        </div>
      </DialogContent>
    </Dialog>
  )
}
