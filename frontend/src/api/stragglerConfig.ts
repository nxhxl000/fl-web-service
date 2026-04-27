export type StragglerMode = 'none' | 'chunk' | 'epochs' | 'drop'
export type StragglerTarget = 'min' | 'median'

export type StragglerNumericKey =
  | 'tolerance'
  | 'drop-tolerance'
  | 'max-dropped'
  | 'min-chunk'
  | 'min-epochs'

export type StragglerParams = {
  mode: StragglerMode
  target: StragglerTarget
  tolerance: number
  'drop-tolerance': number
  'max-dropped': number
  'min-chunk': number
  'min-epochs': number
}

export const STRAGGLER_MODES: { id: StragglerMode; label: string }[] = [
  { id: 'none', label: 'None' },
  { id: 'chunk', label: 'Chunk' },
  { id: 'epochs', label: 'Epochs' },
  { id: 'drop', label: 'Drop' },
]

export const STRAGGLER_MODE_DESCRIPTION: Record<StragglerMode, string> = {
  none: 'No mitigation — every client does full work each round.',
  chunk: 'Slow clients use a smaller fraction of their local data.',
  epochs: 'Slow clients run fewer local epochs.',
  drop: 'The slowest clients are excluded from the round.',
}

// Numeric fields shown for a given mode (target is rendered separately).
export const MODE_NUMERIC_KEYS: Record<StragglerMode, StragglerNumericKey[]> = {
  none: [],
  chunk: ['tolerance', 'min-chunk'],
  epochs: ['tolerance', 'min-epochs'],
  drop: ['drop-tolerance', 'max-dropped'],
}

export const STRAGGLER_PARAM_LABELS: Record<StragglerNumericKey, string> = {
  tolerance: 'Tolerance band',
  'drop-tolerance': 'Drop tolerance',
  'max-dropped': 'Max dropped clients',
  'min-chunk': 'Min chunk fraction',
  'min-epochs': 'Min local epochs',
}

export const STRAGGLER_PARAM_RANGES: Record<
  StragglerNumericKey,
  { min: number; max: number; step: number }
> = {
  tolerance: { min: 0, max: 1, step: 0.01 },
  'drop-tolerance': { min: 0, max: 2, step: 0.05 },
  'max-dropped': { min: 0, max: 20, step: 1 },
  'min-chunk': { min: 0, max: 1, step: 0.05 },
  'min-epochs': { min: 1, max: 20, step: 1 },
}

export const STRAGGLER_PARAM_DEFAULTS: StragglerParams = {
  mode: 'none',
  target: 'min',
  tolerance: 0.05,
  'drop-tolerance': 0.5,
  'max-dropped': 3,
  'min-chunk': 0.1,
  'min-epochs': 1,
}
