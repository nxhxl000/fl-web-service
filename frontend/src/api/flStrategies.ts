export type StrategyId = 'fedavg' | 'fedavgm' | 'fedprox' | 'fednovam'

export type FLParams = {
  // common
  'num-server-rounds': number
  'local-epochs': number
  'fraction-train': number
  'min-train-nodes': number
  'min-available-nodes': number
  // fedavgm / fednovam
  'server-momentum': number
  'server-lr': number
  // fedprox
  'proximal-mu': number
}

export const FL_STRATEGIES: { id: StrategyId; label: string }[] = [
  { id: 'fedavg', label: 'FedAvg' },
  { id: 'fedavgm', label: 'FedAvgM' },
  { id: 'fedprox', label: 'FedProx' },
  { id: 'fednovam', label: 'FedNovaM' },
]

export const COMMON_FL_KEYS: (keyof FLParams)[] = [
  'num-server-rounds',
  'local-epochs',
  'fraction-train',
  'min-train-nodes',
  'min-available-nodes',
]

export const STRATEGY_SPECIFIC_KEYS: Record<StrategyId, (keyof FLParams)[]> = {
  fedavg: [],
  fedavgm: ['server-momentum', 'server-lr'],
  fedprox: ['proximal-mu'],
  fednovam: ['server-momentum', 'server-lr'],
}

/**
 * The complete list of FL keys that are only valid for *some* strategies. Used
 * by `filterRunConfig` to drop a key when it doesn't belong to the picked
 * strategy (e.g. `proximal-mu` when aggregation = FedAvgM).
 */
export const STRATEGY_GATED_KEYS = [
  'server-momentum',
  'server-lr',
  'proximal-mu',
] as const satisfies ReadonlyArray<keyof FLParams>

export const FL_PARAM_LABELS: Record<keyof FLParams, string> = {
  'num-server-rounds': 'Server rounds',
  'local-epochs': 'Local epochs',
  'fraction-train': 'Fraction of clients per round',
  'min-train-nodes': 'Min training nodes',
  'min-available-nodes': 'Min available nodes',
  'server-momentum': 'Server momentum',
  'server-lr': 'Server learning rate',
  'proximal-mu': 'Proximal μ',
}

export const FL_PARAM_RANGES: Record<
  keyof FLParams,
  { min: number; max: number; step: number }
> = {
  'num-server-rounds': { min: 1, max: 200, step: 1 },
  'local-epochs': { min: 1, max: 20, step: 1 },
  'fraction-train': { min: 0, max: 1, step: 0.05 },
  'min-train-nodes': { min: 1, max: 50, step: 1 },
  'min-available-nodes': { min: 1, max: 50, step: 1 },
  'server-momentum': { min: 0, max: 0.99, step: 0.01 },
  'server-lr': { min: 0, max: 2, step: 0.01 },
  'proximal-mu': { min: 0, max: 0.1, step: 0.0001 },
}

export const FL_PARAM_DEFAULTS: FLParams = {
  'num-server-rounds': 80,
  'local-epochs': 3,
  'fraction-train': 1.0,
  'min-train-nodes': 10,
  'min-available-nodes': 10,
  'server-momentum': 0.5,
  'server-lr': 1.0,
  'proximal-mu': 0.0005,
}

/**
 * Per-strategy overrides applied when (model, strategy) is selected.
 * Mirror of `fl_app/fl_app/models/__init__.py:_CIFAR_PER_STRATEGY`.
 *
 * Keys here may belong to either `FLParams` or `ModelHParams`; the form
 * dispatcher routes each key to the right state slice.
 */
export type StrategyOverrides = Partial<{
  // ModelHParams keys
  'client-lr': number
  'client-momentum': number
  // FLParams keys
  'local-epochs': number
  'server-momentum': number
  'server-lr': number
  'proximal-mu': number
}>

export const CIFAR_PER_STRATEGY: Record<StrategyId, StrategyOverrides> = {
  fedavg: {
    'client-lr': 0.1,
    'client-momentum': 0.9,
    'local-epochs': 3,
  },
  fedavgm: {
    'client-lr': 0.05,
    'client-momentum': 0.9,
    'local-epochs': 3,
    'server-momentum': 0.5,
    'server-lr': 1.0,
  },
  fedprox: {
    'client-lr': 0.1,
    'client-momentum': 0.9,
    'local-epochs': 3,
    'proximal-mu': 0.0005,
  },
  fednovam: {
    'client-lr': 0.05,
    'client-momentum': 0.9,
    'local-epochs': 3,
    'server-momentum': 0.5,
    'server-lr': 1.0,
  },
}

const HPARAM_OVERRIDE_KEYS: readonly string[] = ['client-lr', 'client-momentum']
const FLPARAM_OVERRIDE_KEYS: readonly string[] = [
  'local-epochs',
  'server-momentum',
  'server-lr',
  'proximal-mu',
]

/**
 * Split a `StrategyOverrides` blob into two patches: one for `ModelHParams`
 * state, one for `FLParams` state. Form dispatchers apply each to the
 * corresponding `useState` slice.
 */
export function splitStrategyOverrides(overrides: StrategyOverrides): {
  hparamPatch: Record<string, unknown>
  flParamPatch: Record<string, unknown>
} {
  const hparamPatch: Record<string, unknown> = {}
  const flParamPatch: Record<string, unknown> = {}
  for (const [k, v] of Object.entries(overrides)) {
    if (v === undefined) continue
    if (HPARAM_OVERRIDE_KEYS.includes(k)) {
      hparamPatch[k] = v
    } else if (FLPARAM_OVERRIDE_KEYS.includes(k)) {
      flParamPatch[k] = v
    }
  }
  return { hparamPatch, flParamPatch }
}

/**
 * Strip strategy-gated keys (e.g. `proximal-mu`, `server-momentum`) that are
 * not relevant to the picked strategy. Defense in depth: prevents a stale
 * `flParams` value from leaking into the saved run-config.
 */
export function filterRunConfig<T extends Record<string, unknown>>(
  rc: T,
  strategy: StrategyId,
): T {
  const allowed = new Set<string>(STRATEGY_SPECIFIC_KEYS[strategy])
  const out: Record<string, unknown> = {}
  for (const [k, v] of Object.entries(rc)) {
    if ((STRATEGY_GATED_KEYS as readonly string[]).includes(k) && !allowed.has(k)) {
      continue
    }
    out[k] = v
  }
  return out as T
}
