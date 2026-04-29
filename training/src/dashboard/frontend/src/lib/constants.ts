import type { TemplateOption } from '../types'

export const REFRESH_EVENTS = [
  'snapshot',
  'status',
  'model_overview',
  'train_metrics',
  'val_metrics',
  'preview',
  'artifact',
  'run_registered',
  'star',
  'run_deleted',
  'tick',
] as const

export const PREVIEW_TITLES: Record<string, string> = {
  blended_output: 'Blended Output',
  raw_model_output: 'Model Output',
  weighted_loss: 'Loss Heatmap',
  watermarked: 'Watermarked Input',
  predicted_mask: 'Predicted Mask',
  ground_truth_mask: 'Ground Truth',
  sample: 'Sample',
}

export const PREFERRED_PREVIEW_ORDER = [
  'watermarked',
  'blended_output',
  'raw_model_output',
  'weighted_loss',
  'predicted_mask',
  'ground_truth_mask',
  'sample',
]

export const TERM_COLORS: Record<string, string> = {
  total: '#34d399',
  l1_masked: '#60a5fa',
  perceptual: '#a78bfa',
  saturation: '#f472b6',
  color_moment: '#34d399',
  border: '#fbbf24',
  bg_tv: '#a3a3a3',
  bg_delta: '#818cf8',
  edge_coherence: '#c084fc',
  bce: '#60a5fa',
  focal: '#f59e0b',
  l1: '#a78bfa',
  ms: '#34d399',
  dice: '#f87171',
}

export const IGNORED_TERM_KEYS = new Set([
  'epoch',
  'step',
  'global_step',
  'lr',
  'grad_norm',
  'elapsed_s',
  'psnr',
  'psnr_masked',
  'iou',
])

// Any key starting with this prefix is also ignored (e.g. weight_perceptual, weight_border, …)
export const IGNORED_TERM_PREFIX = 'weight_'

export const FALLBACK_TEMPLATES: TemplateOption[] = [
  {
    id: 'blank',
    label: 'Blank config',
    description: 'Start from scratch with an empty dashboard draft.',
    taskType: 'removal',
    suggestedFamilyName: 'removal_256',
  },
  {
    id: 'train_restoration_512',
    label: 'Restoration 512',
    description: 'Template for direct clean-image restoration runs.',
    taskType: 'restoration',
    suggestedFamilyName: 'restoration_512',
  },
  {
    id: 'train_256',
    label: 'Removal 256',
    description: 'Template for 256px watermark removal runs.',
    taskType: 'removal',
    suggestedFamilyName: 'removal_256',
  },
  {
    id: 'train_512',
    label: 'Removal 512',
    description: 'Template for the larger removal configuration.',
    taskType: 'removal',
    suggestedFamilyName: 'removal_512',
  },
  {
    id: 'train_512_second_stage',
    label: 'Removal 512 second stage',
    description: 'Template for loading a previous 512px removal checkpoint.',
    taskType: 'removal',
    suggestedFamilyName: 'removal_512_second_stage',
  },
  {
    id: 'seg',
    label: 'Segmentation',
    description: 'Template for segmentation-oriented experiments.',
    taskType: 'segmentation',
    suggestedFamilyName: 'segmentation',
  },
]
