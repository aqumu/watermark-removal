// Server-sent events from the viewer backend; any of them means "refetch".
export const REFRESH_EVENTS = ['hello', 'runs_changed'] as const

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
