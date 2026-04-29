import { createContext, useContext } from 'react'
import type {
  CheckpointRecord,
  ContinueBrowseSelection,
  ContinueFamilySummary,
  DraftField,
  DraftRecord,
  DraftTaskType,
  DraftWorkspace,
  LauncherSelection,
  OrchestrationSnapshot,
  RunRecord,
  StartPointSelection,
  TemplateOption,
} from '../types'

export type LauncherContextValue = {
  // Templates + draft editing
  templates: TemplateOption[]
  draftTaskType: DraftTaskType
  draftFamilyName: string
  draftTemplateId: string
  draftConfig: Record<string, unknown>
  draftFields: DraftField[]
  scratchQuickFields: DraftField[]
  // Validation
  draftNameError: string | null
  draftNameExists: boolean
  draftFamilyExists: boolean
  sanitizedDraftFamilyName: string
  // Persistence state
  draftSaving: boolean
  draftSaveError: string | null
  savedDrafts: DraftRecord[]
  draftDeleteError: string | null
  deletingDraftName: string | null
  // Start point / checkpoint
  startPointSelection: StartPointSelection
  startPointOptions: CheckpointRecord[]
  startPointLoading: boolean
  startPointError: string | null
  startLaunching: boolean
  startLaunchError: string | null
  // Navigation context
  launcherSelection: LauncherSelection
  continueSelection: ContinueBrowseSelection
  continueFamilies: ContinueFamilySummary[]
  resumableFamilies: Set<string>
  canResumeAnotherRun: boolean
  lifecycleActionLoading: 'pause' | 'resume' | null
  orchestration: OrchestrationSnapshot | undefined
  // Setters
  setDraftTaskType: (v: DraftTaskType) => void
  setDraftFamilyName: (v: string) => void
  setDraftTemplateId: (v: string) => void
  setDraftConfig: (v: Record<string, unknown>) => void
  // Handlers
  onSelectFresh: () => void
  onSelectCheckpoint: (c: CheckpointRecord) => void
  onStartSelected: () => Promise<void>
  onSaveDraft: () => Promise<void>
  onOpenDraft: (draft: DraftRecord) => void
  onDeleteDraft: (familyName: string) => Promise<void>
  onActivateFlow: (flow: DraftWorkspace['flow']) => void
  onInspectRun: (runId: string) => void
  onSelectContinueFamily: (familyName: string) => void
  onBackToContinueFamilies: () => void
  onResumeRun: (run: RunRecord) => Promise<void>
  onGoBack: () => void
}

const LauncherContext = createContext<LauncherContextValue | null>(null)

export function useLauncherContext(): LauncherContextValue {
  const ctx = useContext(LauncherContext)
  if (!ctx) throw new Error('useLauncherContext must be used inside LauncherContext.Provider')
  return ctx
}

export { LauncherContext }
